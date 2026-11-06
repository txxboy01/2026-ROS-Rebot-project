"""음성 입력 모듈 단위 시험. 마이크와 Whisper 없이 돈다."""
import numpy as np
import pytest

from app.audio.config import VADConfig, VoiceConfig
from app.audio.listener import VoiceListener
from app.audio.stopword import StopWordDetector, normalize
from app.audio.stt import is_prompt_echo
from app.audio.vad import Segmenter

SR = 16000
FRAME = 480  # 30 ms


def tone(seconds: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def noise(seconds: float, amp: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (amp * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def frames(signal: np.ndarray):
    for i in range(0, len(signal) - FRAME + 1, FRAME):
        yield signal[i : i + FRAME]


def run_segmenter(signal, cfg=None):
    seg = Segmenter(cfg or VADConfig(), SR, FRAME)
    out = [u for f in frames(signal) if (u := seg.feed(f)) is not None]
    if (u := seg.flush()) is not None:
        out.append(u)
    return out


# ---- VAD ----------------------------------------------------------------
def test_one_utterance_between_silence():
    sig = np.concatenate([noise(1.0), tone(1.0), noise(1.0)])
    utts = run_segmenter(sig)
    assert len(utts) == 1
    u = utts[0]
    # 시작은 pre-roll(0.3s)만큼 앞당겨지고, 끝은 침묵 일부를 포함
    assert 0.6 <= u.start_s <= 1.0
    assert 1.9 <= u.end_s <= 2.4


def test_two_utterances_split_by_pause():
    sig = np.concatenate([noise(0.5), tone(0.6), noise(1.0), tone(0.6), noise(1.0)])
    assert len(run_segmenter(sig)) == 2


def test_short_click_is_dropped():
    sig = np.concatenate([noise(0.5), tone(0.12), noise(1.0)])
    assert run_segmenter(sig) == []


def test_silence_only_gives_nothing():
    assert run_segmenter(noise(3.0)) == []


def test_long_speech_is_cut_at_max():
    cfg = VADConfig(max_speech_ms=2000)
    utts = run_segmenter(np.concatenate([noise(0.5), tone(5.0), noise(1.0)]), cfg)
    assert len(utts) >= 2
    assert all(u.end_s - u.start_s <= 2.4 for u in utts)


def test_noise_floor_adapts_to_loud_room():
    # 시끄러운 방(잡음 0.02)에서는 잡음만으로 발화가 시작되면 안 된다
    loud_noise = noise(3.0, amp=0.02)
    assert run_segmenter(loud_noise) == []


# ---- 정지어 ---------------------------------------------------------------
@pytest.mark.parametrize("text", ["멈춰", "멈춰!!", "아 멈춰 멈춰", "정지.", "STOP", "스톱", "그만해"])
def test_stop_words_detected(text):
    det = StopWordDetector(("멈춰", "정지", "stop", "스톱", "그만"))
    assert det.match(text) is not None


@pytest.mark.parametrize("text", ["빨간 블록 줘", "드라이버 건네줘", "왼쪽으로 옮겨"])
def test_commands_are_not_stop(text):
    assert StopWordDetector(VoiceConfig().stt.stop_words).match(text) is None


def test_normalize():
    assert normalize(" 시청해 주셔서, 감사합니다! ") == "시청해주셔서감사합니다"


def test_prompt_never_contains_stop_word():
    # 단서 문장에 정지어가 있으면 짧은 잡음에서 단서를 읊어 잘못 정지한다 (실측으로 확인된 문제)
    stt = VoiceConfig().stt
    assert StopWordDetector(stt.stop_words).match(stt.initial_prompt or "") is None


PROMPT = normalize(VoiceConfig().stt.initial_prompt or "")


@pytest.mark.parametrize("echo", [
    "작업대 위의 빨간 블록, 파란 블록, 드라이버를 다루는 로봇팔과의 대화입니다.",
    "그녀를 다루는 로봇팔과의 대화입니다.",          # 실측: "음" 소리에서 나온 일부 읊기
    "빨간 블록, 파란 블록, 드라이버",
])
def test_prompt_echo_dropped(echo):
    assert is_prompt_echo(echo, PROMPT)


@pytest.mark.parametrize("command", [
    "빨간 블록 나한테 건네줘.",
    "십자 드라이버 좀 건네줄래.",
    "파란 블록을 왼쪽 칸으로 옮겨",
    "작업대 위에 있는 거 다 정리해",
    "드라이버 좀 건네줘.",
])
def test_real_commands_not_treated_as_echo(command):
    assert not is_prompt_echo(command, PROMPT)


def test_prompt_is_not_a_command():
    # 단서를 읊어도 실행 가능한 명령이 되지 않아야 한다 (마이크 시험에서 "드라이버 좀 건네줘." 지어냄)
    prompt = VoiceConfig().stt.initial_prompt or ""
    for verb in ("건네", "줘", "옮겨", "정리해", "집어", "넣어"):
        assert verb not in prompt


def test_short_hallucination_only_dropped_when_whole_result():
    from app.audio.stt import WhisperTranscriber

    t = WhisperTranscriber()
    assert t._is_hallucination("감사합니다.")
    assert not t._is_hallucination("감사합니다 빨간 블록 줘")
    assert t._is_hallucination("시청해 주셔서 감사합니다")


# ---- 리스너 (가짜 STT) ------------------------------------------------------
class FakeSTT:
    """호출될 때마다 정해진 문장을 차례로 돌려준다."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, audio, partial=False):
        self.calls += 1
        text = self.texts.pop(0) if self.texts else ""
        return text, 0.9, "ko"


def make_listener(texts, **kw):
    got, stops = [], []
    stt = FakeSTT(texts)
    lis = VoiceListener(on_transcript=got.append, on_stop=stops.append, transcriber=stt, **kw)
    return lis, stt, got, stops


def feed(lis, signal):
    for f in frames(signal):
        lis.process_frame(f)


def test_command_reaches_callback():
    cfg = VoiceConfig()
    cfg.stt.partial_interval_ms = 0  # 사전 검사 끔
    lis, _, got, stops = make_listener(["빨간 블록 줘"], cfg=cfg)
    feed(lis, np.concatenate([noise(0.5), tone(1.0), noise(1.0)]))
    assert [t.text for t in got] == ["빨간 블록 줘"]
    assert stops == []
    assert got[0].confidence == 0.9 and not got[0].partial


def test_stop_word_fires_during_speech_and_only_once():
    # 사전 검사 첫 호출에서 "멈춰"가 나오면 즉시 on_stop, 발화 끝 결과는 중복 호출 안 함
    lis, _, got, stops = make_listener(["멈춰", "멈춰 멈춰"])
    feed(lis, np.concatenate([noise(0.5), tone(1.0), noise(1.0)]))
    assert len(stops) == 1 and stops[0].partial
    assert got == []


def test_muted_blocks_commands_but_not_stop():
    cfg = VoiceConfig()
    cfg.stt.partial_interval_ms = 0
    lis, _, got, stops = make_listener(["안녕하세요 집사입니다", "정지"], cfg=cfg)
    lis.set_muted(True)
    feed(lis, np.concatenate([noise(0.5), tone(0.8), noise(1.0), tone(0.5), noise(1.0)]))
    assert got == []
    assert [t.stop_word for t in stops] == ["정지"]


def test_empty_text_is_ignored():
    cfg = VoiceConfig()
    cfg.stt.partial_interval_ms = 0
    lis, _, got, _ = make_listener([""], cfg=cfg)
    feed(lis, np.concatenate([noise(0.5), tone(1.0), noise(1.0)]))
    assert got == []


def test_push_to_talk_ignores_audio_outside_button():
    lis, stt, got, _ = make_listener(["파란 블록 정리해"], push_to_talk=True)
    lis.cfg.stt.partial_interval_ms = 0
    feed(lis, tone(1.0))            # 버튼 안 누른 상태
    assert stt.calls == 0
    lis.begin_utterance()
    feed(lis, tone(1.0))
    lis.end_utterance()
    lis.process_frame(lis._frames.get())   # 스레드 대신 직접 종료 신호 처리
    assert [t.text for t in got] == ["파란 블록 정리해"]
    assert 0.9 <= got[0].start_s <= 1.1


def test_stop_word_at_end_of_long_utterance_is_caught_early():
    # 3초 넘게 말한 뒤 끝에 "멈춰" → 말이 끝나기(침묵 0.6초) 전에 최근 구간 검사로 잡아야 한다
    texts = ["빨간 블록"] * 6 + ["나한테 건네줘 멈춰"] + ["빨간 블록 나한테 건네줘 멈춰"]
    lis, stt, got, stops = make_listener(texts)
    sig = np.concatenate([noise(0.5), tone(3.5), noise(1.0)])
    speech_end = 0.5 + 3.5
    for i, f in enumerate(frames(sig)):
        lis.process_frame(f)
        if stops:
            caught_at = (i + 1) * FRAME / SR
            break
    assert stops and stops[0].partial
    assert caught_at <= speech_end      # 침묵을 기다리지 않음
    # 사전 검사 구간은 최근 2초로 제한
    assert stops[0].end_s - stops[0].start_s <= 2.01


def test_prompt_has_no_direction_words():
    # 단서에 "왼쪽" 같은 방향어가 있으면 말하지 않은 방향을 붙인다 (마이크 시험에서 "왼쪽"이 붙음)
    prompt = VoiceConfig().stt.initial_prompt or ""
    for word in ("왼쪽", "오른쪽", "위", "아래"):
        assert word not in prompt.replace("위의", "")


def spiky_room(seconds: float, base: float = 0.003, spike: float = 0.02, every: int = 6) -> np.ndarray:
    """헤드셋 마이크의 쉬는 구간: 낮은 잡음에 숨소리·옷 스침 같은 짧은 튐이 섞임."""
    sig = noise(seconds, amp=base)
    rng = np.random.default_rng(7)
    for k in range(0, len(sig) // FRAME, every):
        s = k * FRAME
        sig[s : s + FRAME] += (spike * 1.4 * rng.standard_normal(FRAME)).astype(np.float32)
    return sig


def test_pause_with_spikes_still_splits_utterances():
    # 마이크 시험: 문장 사이에 쉬었는데 하나로 이어 붙여짐 ("되고 있는 거야? 되고 있는 거야? 왜 이래 이거?")
    sig = np.concatenate([spiky_room(1.0), tone(1.0), spiky_room(1.0), tone(1.0), spiky_room(1.0)])
    assert len(run_segmenter(sig)) == 2


def test_end_detected_quickly_after_speech():
    sig = np.concatenate([noise(1.0), tone(1.0), spiky_room(2.0)])
    seg = Segmenter(VADConfig(), SR, FRAME)
    for i, f in enumerate(frames(sig)):
        if seg.feed(f) is not None:
            ended_at = (i + 1) * FRAME / SR
            break
    else:
        pytest.fail("발화 끝을 못 찾음")
    assert ended_at - 2.0 <= 0.65   # 말 끝나고 0.65초 안에 결과로 넘어가야 함


def test_speech_start_and_end_notified():
    events = []
    cfg = VoiceConfig()
    cfg.stt.partial_interval_ms = 0
    lis = VoiceListener(
        on_transcript=lambda t: events.append(("text", t.text)),
        transcriber=FakeSTT(["빨간 블록 줘"]),
        cfg=cfg,
        on_speech_start=lambda: events.append("start"),
        on_speech_end=lambda: events.append("end"),
    )
    feed(lis, np.concatenate([noise(0.6), tone(1.0), noise(1.0)]))
    assert events == ["start", "end", ("text", "빨간 블록 줘")]


def gated_room(seconds: float, seed: int = 11) -> np.ndarray:
    """노이즈 게이트가 있는 헤드셋(G733 실측): 평소엔 0, 가끔 1~3프레임 0.005~0.01 소리가 샘."""
    rng = np.random.default_rng(seed)
    sig = np.zeros(int(SR * seconds), dtype=np.float32)
    n = len(sig) // FRAME
    k = 0
    while k < n:
        if rng.random() < 0.25:
            length = int(rng.integers(1, 4))
            amp = rng.uniform(0.005, 0.01) * 1.4
            s = k * FRAME
            sig[s : s + length * FRAME] = (amp * rng.standard_normal(length * FRAME)).astype(np.float32)[: len(sig[s : s + length * FRAME])]
            k += length
        k += 1
    return sig


def test_gated_headset_idle_is_not_speech():
    assert run_segmenter(gated_room(6.0)) == []


def test_gated_headset_pauses_split_utterances():
    sig = np.concatenate([gated_room(1.5, 1), tone(1.0, 0.05), gated_room(1.0, 2), tone(1.0, 0.05), gated_room(1.5, 3)])
    assert len(run_segmenter(sig)) == 2
