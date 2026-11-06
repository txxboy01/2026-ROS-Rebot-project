"""인식 문장에서 정지어("멈춰", "정지" 등)를 찾는다."""
from __future__ import annotations

import re

_NON_WORD = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """공백, 문장부호를 지우고 소문자로. "멈춰!!" → "멈춰", "STOP." → "stop"."""
    return _NON_WORD.sub("", text).lower()


class StopWordDetector:
    def __init__(self, words: tuple[str, ...] | list[str]):
        self.words = tuple(dict.fromkeys(normalize(w) for w in words if normalize(w)))

    def match(self, text: str) -> str | None:
        """정지어가 들어 있으면 그 단어를, 없으면 None.

        안전 쪽으로 기울여 부분 일치로 판단한다("멈춰줘", "아 그만해" 모두 정지).
        잘못 멈추는 것은 괜찮지만 못 멈추는 것은 안 된다.
        """
        norm = normalize(text)
        for w in self.words:
            if w in norm:
                return w
        return None
