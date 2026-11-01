"""음성 입력 모듈 설정값."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    # 장치 이름 일부(예: "Jabra") 또는 번호. None이면 시스템 기본 마이크.
    # Windows는 재연결 시 번호가 바뀔 수 있으므로 이름으로 지정하는 것을 권장.
    device: str | int | None = None

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


@dataclass
class VADConfig:
    """에너지 기반 발화 구간 검출. 잡음 수준을 계속 추정해 문턱값을 정한다."""

    start_ratio: float = 3.0      # 잡음 대비 이 배수를 넘으면 말소리 후보
    end_ratio: float = 1.8        # 말하는 중 잡음의 이 배수 아래로 떨어지면 침묵 후보
    end_speech_ratio: float = 0.2 # 또는 목소리 크기의 이 비율(약 -14 dB) 아래면 침묵 후보
    smooth_frames: int = 3        # 침묵 판단은 최근 3프레임(90 ms) 평균으로
    min_rms: float = 0.004        # 아주 조용한 방에서도 문턱이 이 값 아래로 내려가지 않음
    calibration_ms: int = 500     # 시작 직후 이 시간은 말소리 판단 없이 잡음 수준만 잰다
    noise_window_ms: int = 3000   # 최근 이만큼의 쉬는 구간으로 잡음 수준을 정한다
    noise_percentile: float = 75  # 그 구간 소리 크기의 상위 25% 지점을 잡음으로 본다
    start_frames: int = 3         # 연속 몇 프레임이 넘어야 발화 시작으로 인정
    end_silence_ms: int = 500     # 이만큼 조용하면 발화 끝
    pre_roll_ms: int = 300        # 발화 시작 전 소리도 붙여서 첫 음절 잘림 방지
    min_speech_ms: int = 250      # 이보다 짧으면 잡음으로 보고 버림
    max_speech_ms: int = 15000    # 이보다 길면 강제로 끊음


# Whisper가 무음이나 잡음에서 지어내는 대표 문장 (한국어 방송 자막 학습 영향).
# 결과 안에 들어 있기만 해도 버린다
DEFAULT_HALLUCINATIONS: tuple[str, ...] = (
    "시청해주셔서감사합니다",
    "시청해주셔서고맙습니다",
    "구독과좋아요",
    "구독좋아요",
    "다음영상에서만나요",
    "mbc뉴스",
    "kbs뉴스",
    "자막제공",
    "시각세계였습니다",   # 뉴스 마무리 멘트. "음" 소리에서 나옴
)

# 실제 말에도 들어갈 수 있는 짧은 것. 결과 전체가 정확히 이것일 때만 버린다
DEFAULT_HALLUCINATIONS_EXACT: tuple[str, ...] = (
    "아멘",        # 무음·저주파 잡음 (no_speech_prob도 0으로 나와 따로 거름)
    "감사합니다",   # 키보드 딸깍 소리
)


@dataclass
class STTConfig:
    model: str = "large-v3-turbo"   # tiny / base / small / medium / large-v3 / large-v3-turbo
    device: str = "auto"            # auto / cuda / cpu
    compute_type: str = "auto"      # auto면 cuda=float16, cpu=int8
    language: str = "ko"
    beam_size: int = 1              # 1이 가장 빠름. 정확도가 부족하면 5
    # 어휘 단서를 주면 "빨간"을 "알간"으로 듣는 식의 실수가 줄어든다.
    # 말소리가 아닌 소리(숨, 딸깍, "음")에서는 모델이 이 단서를 그대로 읊는다 (실측).
    #   - 명령문으로 쓰면 읊은 결과가 실행 가능한 명령이 됨 → 마이크 시험에서
    #     말하지도 않은 "드라이버 좀 건네줘."가 나옴. 그래서 반드시 명령이 아닌 설명문으로 쓴다.
    #   - 쉼표 나열로 쓰면 형식을 따라 하며 말하지 않은 단어("왼쪽")를 붙인다.
    #   - 정지어를 넣으면 잡음에 잘못 정지한다.
    # 읊은 결과는 is_prompt_echo가 버린다. 말하는 중 사전 검사에는 쓰지 않는다.
    initial_prompt: str | None = "작업대 위의 빨간 블록, 파란 블록, 드라이버를 다루는 로봇팔과의 대화입니다."
    # 이보다 낮으면 결과를 버린다. 숨소리·감탄사("헉!", "아...")가 0.2~0.35로 나옴.
    # 0.4~0.5 구간은 넘기되 명령 해석 쪽에서 실행 전에 다시 묻는 것을 권장
    min_confidence: float = 0.4
    stop_words: tuple[str, ...] = ("멈춰", "멈춰요", "멈춰라", "정지", "스톱", "스탑", "그만")
    hallucinations: tuple[str, ...] = DEFAULT_HALLUCINATIONS
    hallucinations_exact: tuple[str, ...] = DEFAULT_HALLUCINATIONS_EXACT
    # 말하는 도중에도 이 간격으로 미리 인식해 정지어를 빨리 잡는다. 0이면 끔
    partial_interval_ms: int = 400
    # 사전 검사는 최근 이 길이만 본다. 긴 문장 끝의 "멈춰"도 말이 끝나기 전에 잡고,
    # 인식 시간도 일정하게 유지된다 (RTX 4060에서 2초 구간 약 0.2초)
    partial_window_ms: int = 2000


@dataclass
class VoiceConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
