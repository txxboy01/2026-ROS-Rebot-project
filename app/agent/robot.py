"""명령 모듈과 로봇 제어 모듈 사이의 경계.

로봇 제어 모듈은 `RobotBackend`를 구현하면 된다. 명령 모듈은 LLM이 고른 도구를
`run(이름, 인자)`로 넘기고, 돌려받은 딕셔너리를 그대로 LLM에 보여준다.

결과 딕셔너리 약속
    {"ok": True,  "message": "obj_4를 건넸습니다"}                   성공
    {"ok": False, "message": "작업 영역 밖이라 놓을 수 없습니다"}      실패 (LLM이 사용자에게 이유를 말함)
    그 밖의 키(objects 등)는 자유롭게 붙여도 된다.

`MockRobot`은 로봇 없이 개발·시험하기 위한 가짜 작업대다.
"""
from __future__ import annotations

import copy
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol


class RobotBackend(Protocol):
    capabilities: set[str]   # 로봇 프로필의 능력. 예: {"gripper", "vision.top"}

    def run(self, tool: str, args: dict) -> dict:
        """도구 하나를 끝까지 실행하고 결과를 돌려준다. 시간이 걸려도 된다(블로킹)."""
        ...

    def stop(self) -> None:
        """정지어가 들리면 바로 불린다. 다른 스레드에서 호출되며, 즉시 돌아와야 한다."""
        ...


@dataclass
class MockObject:
    id: str
    name: str
    kind: str
    color: str
    x_mm: float
    y_mm: float
    slot: str | None = None   # 칸 안에 들어가 있으면 칸 이름

    @property
    def side(self) -> str:
        if self.slot:
            return self.slot
        return "왼쪽" if self.y_mm > 60 else "오른쪽" if self.y_mm < -60 else "가운데"

    def view(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "color": self.color,
                "position": self.side, "x_mm": self.x_mm, "y_mm": self.y_mm}


SLOTS = ["1번 칸", "2번 칸", "3번 칸"]
TOOL_KINDS = {"screwdriver", "wrench"}

DEFAULT_WORLD = [
    MockObject("obj_1", "빨간 블록", "block", "red", 200, 120),
    MockObject("obj_2", "빨간 블록", "block", "red", 200, -120),
    MockObject("obj_3", "파란 블록", "block", "blue", 240, 0),
    MockObject("obj_4", "드라이버", "screwdriver", "yellow", 160, 60),
    MockObject("obj_5", "렌치", "wrench", "gray", 280, -60),
    MockObject("obj_6", "초록 공", "ball", "green", 300, 100),
]


class MockRobot:
    """가짜 로봇과 작업대. 동작은 즉시(또는 step_s 초 뒤) 끝난다."""

    def __init__(self, capabilities: set[str] | None = None, world: list[MockObject] | None = None,
                 step_s: float = 0.0):
        self.capabilities = set(capabilities or {"gripper", "vision.top", "vision.front"})
        self.objects = {o.id: o for o in copy.deepcopy(world or DEFAULT_WORLD)}
        self.step_s = step_s
        self.log: list[tuple[str, dict]] = []
        self._stopped = threading.Event()

    # ---- RobotBackend ---------------------------------------------------
    def run(self, tool: str, args: dict) -> dict:
        self.log.append((tool, dict(args)))
        self._stopped.clear()
        handler = getattr(self, f"_{tool}", None)
        if handler is None:
            return {"ok": False, "message": f"로봇이 '{tool}' 도구를 지원하지 않습니다"}
        if self.step_s and self._stopped.wait(self.step_s):
            return {"ok": False, "message": "정지 신호로 멈췄습니다"}
        return handler(**args)

    def stop(self) -> None:
        self._stopped.set()
        self.log.append(("stop", {}))

    # ---- 화면 표시용 --------------------------------------------------------
    def describe(self) -> str:
        return "\n".join(f"  {o.id} {o.name:<6} {o.side}" for o in self.objects.values())

    # ---- 도구 -----------------------------------------------------------
    def _find_objects(self, color: str | None = None, kind: str | None = None) -> dict:
        found = [o.view() for o in self.objects.values()
                 if (color is None or o.color == color) and (kind is None or o.kind == kind)]
        return {"ok": True, "count": len(found), "objects": found}

    def _target(self, target_id: str) -> MockObject | None:
        return self.objects.get(target_id)

    def _resolve_place(self, place: str) -> str | None:
        p = place.replace(" ", "")
        for slot in SLOTS:
            if slot.replace(" ", "") in p:
                return slot
        m = re.search(r"obj_\d+", place)
        if m and m.group() in self.objects:
            return f"{m.group()} 옆"
        return None

    def _pick_and_place(self, target_id: str, place: str) -> dict:
        obj = self._target(target_id)
        if obj is None:
            return {"ok": False, "message": f"{target_id} 물체가 작업대에 없습니다"}
        where = self._resolve_place(place)
        if where is None:
            return {"ok": False, "message": f"'{place}'는 등록된 위치가 아닙니다. 칸 이름: {SLOTS}"}
        obj.slot = where if where in SLOTS else None
        return {"ok": True, "message": f"{obj.name}({obj.id})를 {where}에 놓았습니다"}

    def _hand_over(self, target_id: str, orientation: str = "any") -> dict:
        obj = self._target(target_id)
        if obj is None:
            return {"ok": False, "message": f"{target_id} 물체가 작업대에 없습니다"}
        del self.objects[target_id]
        return {"ok": True, "message": f"{obj.name}({obj.id})를 건넸고 사용자가 받았습니다"}

    def _tidy_up(self, rule: str = "kind") -> dict:
        moved = []
        for obj in self.objects.values():
            if rule == "color":
                slot = {"red": "1번 칸", "blue": "2번 칸"}.get(obj.color, "3번 칸")
            else:
                slot = "1번 칸" if obj.kind == "block" else "2번 칸" if obj.kind in TOOL_KINDS else "3번 칸"
            obj.slot = slot
            moved.append(f"{obj.name}→{slot}")
        return {"ok": True, "message": f"{len(moved)}개 정리: " + ", ".join(moved)}

    def _pick_from_slot(self, slot: int, place: str) -> dict:
        name = f"{slot}번 칸"
        dest = "사용자에게 건넸습니다" if "사용자" in place else f"{place}에 놓았습니다"
        return {"ok": True, "message": f"{name}의 물체를 {dest}"}

    def _push(self, target_id: str, direction: str) -> dict:
        obj = self._target(target_id)
        if obj is None:
            return {"ok": False, "message": f"{target_id} 물체가 작업대에 없습니다"}
        return {"ok": True, "message": f"{obj.name}를 {direction} 방향으로 밀었습니다"}

    def _gesture(self, name: str) -> dict:
        return {"ok": True, "message": f"{name} 몸짓을 했습니다"}

    def _go_home(self) -> dict:
        return {"ok": True, "message": "대기 자세로 돌아왔습니다"}
