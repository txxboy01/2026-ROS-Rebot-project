"""네트워크가 없을 때 쓰는 고정 명령어 해석 (설계서 NFR-06).

LLM 없이 정해진 말만 알아듣는다. 정지어는 여기가 아니라 음성 입력 모듈이 처리한다.
"""
from __future__ import annotations

import re

from .persona import PERSONAS

# (정규식, 도구, 인자) — 위에서부터 처음 맞는 것 하나
_RULES: list[tuple[str, str, dict]] = [
    (r"제자리|원위치|원래 ?자리|대기 ?자세|처음 ?자세", "go_home", {}),
    (r"인사", "gesture", {"name": "greet"}),
    (r"끄덕", "gesture", {"name": "nod"}),
    (r"고개.*저", "gesture", {"name": "shake_head"}),
    (r"손.*흔들", "gesture", {"name": "wave"}),
]

_SLOT = re.compile(r"([123]|일|이|삼|한|두|세) ?번 ?칸")
_SLOT_NUM = {"1": 1, "2": 2, "3": 3, "일": 1, "이": 2, "삼": 3, "한": 1, "두": 2, "세": 3}

HELP = "인터넷 연결이 없어 간단한 명령만 할 수 있어요: 제자리로, 인사해, 끄덕여, 1번 칸 거 줘, 집사로 바꿔"


def parse(text: str) -> tuple[str, dict] | None:
    """문장을 (도구, 인자)로 바꾼다. 모르는 문장이면 None."""
    t = text.strip()
    if "바꿔" in t:
        for name in PERSONAS:
            # 이름 전체, 또는 첫·끝 단어 ("츤데레", "고양이")로도 알아듣는다
            words = name.split()
            if any(w in t for w in (name, words[0], words[-1])):
                return "set_persona", {"name": name}
    slots = [_SLOT_NUM[m] for m in _SLOT.findall(t)]
    if slots and re.search(r"줘|건네|가져", t):
        place = f"{slots[1]}번 칸" if len(slots) > 1 else "사용자"
        return "pick_from_slot", {"slot": slots[0], "place": place}
    for pattern, tool, args in _RULES:
        if re.search(pattern, t):
            return tool, dict(args)
    return None
