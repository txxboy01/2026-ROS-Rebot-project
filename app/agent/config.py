"""명령 해석 모듈 설정값."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentConfig:
    model: str = "claude-opus-5-5"
    # 생각 깊이. 명령 한 줄 해석은 low면 충분하고 응답이 빠르다(설계서 NFR-01: 발화 끝 → 동작 3초).
    # 정확도가 부족하면 medium으로 올려서 eval로 비교한다.
    effort: str = "low"
    max_tokens: int = 16000
    timeout_s: float = 20.0
    max_retries: int = 1
    # 모델이 안전 분류기로 요청을 거절하면 서버가 다른 모델로 다시 시도한다
    refusal_fallback: bool = True

    max_tool_calls: int = 5       # 발화 한 번에 도구 호출 상한 (설계서 5.2)
    max_turns: int = 10           # 대화 기억: 최근 10번의 발화까지 (설계서 5.2)
    idle_reset_s: float = 600.0   # 이만큼 명령이 없으면 대화를 새로 시작
    min_confidence: float = 0.5   # STT 신뢰도가 이보다 낮으면 실행하지 않고 다시 묻는다 (설계서 5.1)

    offline: bool = False         # True면 LLM 없이 고정 명령어만 처리 (설계서 NFR-06)
    persona: str = "툴메이트"
