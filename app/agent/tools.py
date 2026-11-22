"""LLM에 보여주는 도구 정의와, 로봇 프로필(능력)에 따라 도구 목록을 고르는 toolbox (설계서 5.2).

로봇 제어 모듈은 여기 정의된 이름과 인자로 호출을 받는다. 바꿀 때는 양쪽이 같이 바꾼다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .persona import PERSONAS

COLORS = ["red", "blue", "green", "yellow", "gray"]
KINDS = ["block", "screwdriver", "wrench", "ball"]
GESTURES = ["greet", "nod", "shake_head", "wave"]
DIRECTIONS = ["left", "right", "forward", "backward"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict
    required: list[str] = field(default_factory=list)
    needs: frozenset[str] = frozenset()      # 이 능력이 모두 있어야 노출
    absent: frozenset[str] = frozenset()     # 이 능력이 없을 때만 노출 (대체 도구)
    local: bool = False                       # 로봇이 아니라 명령 모듈이 직접 처리

    def available(self, capabilities: set[str]) -> bool:
        return self.needs <= capabilities and not (self.absent & capabilities)

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": self.required,
                "additionalProperties": False,
            },
        }


def _s(desc: str, **kw) -> dict:
    return {"type": "string", "description": desc, **kw}


TOOLS: list[ToolSpec] = [
    ToolSpec(
        "find_objects",
        "작업대 카메라로 조건에 맞는 물체 후보를 찾는다. 물체를 집거나 옮기기 전에 먼저 호출해서 id를 얻는다. "
        "조건을 비우면 작업대의 모든 물체를 돌려준다.",
        {
            "color": _s("물체 색", enum=COLORS),
            "kind": _s("물체 종류", enum=KINDS),
        },
        needs=frozenset({"vision.top"}),
    ),
    ToolSpec(
        "pick_and_place",
        "물체를 집어 다른 위치에 놓는다.",
        {
            "target_id": _s("옮길 물체 id. find_objects가 돌려준 값만 쓴다. 예: obj_3"),
            "place": _s(
                "놓을 위치. 등록된 칸 이름(예: '2번 칸'), 다른 물체 id(그 옆에 놓음), "
                "또는 '<물체 id>의 왼쪽/오른쪽' 같은 상대 위치. 좌표는 쓰지 않는다."
            ),
        },
        ["target_id", "place"],
        needs=frozenset({"gripper"}),
    ),
    ToolSpec(
        "tidy_up",
        "작업대 위 물체를 정리 규칙에 따라 칸으로 차례로 옮긴다.",
        {"rule": _s("정리 기준. kind는 종류별, color는 색별. 말이 없으면 kind", enum=["kind", "color"])},
        needs=frozenset({"gripper", "vision.top"}),
    ),
    ToolSpec(
        "hand_over",
        "물체를 집어 사용자에게 건넨다. 건네기 위치에서 기다리다가 사용자가 잡으면 놓는다.",
        {
            "target_id": _s("건넬 물체 id. find_objects가 돌려준 값만 쓴다."),
            "orientation": _s("손잡이 방향. 공구는 handle_to_user", enum=["handle_to_user", "any"]),
        },
        ["target_id"],
        needs=frozenset({"gripper"}),
    ),
    ToolSpec(
        "pick_from_slot",
        "카메라가 없을 때 쓰는 도구. 등록된 칸 번호의 물체를 집어 다른 칸에 놓거나 사용자에게 건넨다.",
        {
            "slot": {"type": "integer", "description": "물체가 있는 칸 번호", "enum": [1, 2, 3]},
            "place": _s("놓을 곳. '사용자'면 건네기, 아니면 칸 이름(예: '2번 칸')"),
        },
        ["slot", "place"],
        needs=frozenset({"gripper"}),
        absent=frozenset({"vision.top"}),
    ),
    ToolSpec(
        "push",
        "집게가 없을 때 쓰는 도구. 물체를 한 방향으로 민다.",
        {
            "target_id": _s("밀 물체 id"),
            "direction": _s("미는 방향", enum=DIRECTIONS),
        },
        ["target_id", "direction"],
        needs=frozenset({"vision.top"}),
        absent=frozenset({"gripper"}),
    ),
    ToolSpec(
        "gesture",
        "제자리에서 짧은 몸짓을 한다. greet 인사, nod 끄덕임, shake_head 고개 젓기, wave 손 흔들기.",
        {"name": _s("몸짓 이름", enum=GESTURES)},
        ["name"],
    ),
    ToolSpec("go_home", "대기 자세로 돌아간다.", {}),
    ToolSpec(
        "ask_clarification",
        "명령이 모호할 때 사용자에게 되묻는다. 대상 후보가 여럿인데 말로 구별되지 않을 때 쓴다. "
        "이 도구를 부르면 이번 차례는 끝나고, 사용자의 답이 다음 메시지로 온다.",
        {
            "question": _s("화면에 보여줄 짧은 질문"),
            "options": {"type": "array", "items": {"type": "string"}, "description": "선택지. 2~4개"},
        },
        ["question", "options"],
        local=True,
    ),
    ToolSpec(
        "set_persona",
        "로봇의 페르소나(말투와 몸짓 성격)를 바꾼다.",
        {"name": _s("페르소나 이름", enum=list(PERSONAS))},
        ["name"],
        local=True,
    ),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}
# 로봇을 움직이는 도구 (eval에서 '실제 행동'으로 세는 것)
ACTION_TOOLS = {t.name for t in TOOLS if not t.local and t.name != "find_objects"}


def build_tools(capabilities: set[str]) -> list[ToolSpec]:
    """프로필 능력으로 노출할 도구를 고른다. 순서는 고정이라 프롬프트 캐시가 유지된다."""
    return [t for t in TOOLS if t.available(capabilities)]


def validate_input(spec: ToolSpec, args: object) -> str | None:
    """인자가 스키마에 맞으면 None, 아니면 오류 설명."""
    if not isinstance(args, dict):
        return "인자가 객체가 아님"
    extra = set(args) - set(spec.properties)
    if extra:
        return f"모르는 인자: {sorted(extra)}"
    for key in spec.required:
        if key not in args:
            return f"필수 인자 없음: {key}"
    for key, value in args.items():
        prop = spec.properties[key]
        kind = prop["type"]
        ok = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "array": isinstance(value, list),
        }[kind]
        if not ok:
            return f"{key}의 형식이 {kind}가 아님"
        if "enum" in prop and value not in prop["enum"]:
            return f"{key}는 {prop['enum']} 중 하나여야 함"
    return None
