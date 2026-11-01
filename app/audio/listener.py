"""마이크 → 발화 구간 → 텍스트까지 묶은 음성 입력 모듈.

다른 모듈은 이 클래스만 쓰면 된다.

    listener = VoiceListener(on_transcript=..., on_stop=...)
    listener.start()
    ...
    listener.set_muted(True)   # TTS 재생 중: 명령은 무시, 정지어는 계속 감지
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

from .config import VoiceConfig
from .stopword import StopWordDetector
from .stt import Transcriber, Transcript, WhisperTranscriber
from .vad import Segmenter, Utterance

log = logging.getLogger(__name__)

TranscriptCallback = Callable[[Transcript], None]


class VoiceListener:
    def __init__(
        self,
        on_transcript: TranscriptCallback,
        on_stop: TranscriptCallback | None = None,
        cfg: VoiceConfig | None = None,
        transcriber: Transcriber | None = None,
        mic=None,
        push_to_talk: bool = False,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[], None] | None = None,
    ):
        """
        on_transcript: 일반 명령 문장이 인식될 때 (정지어 포함 문장도 여기로 오지 않음)
        on_stop: 정지어가 감지될 때. 말하는 도중에도 호출될 수 있다
        push_to_talk: True면 VAD 대신 begin_utterance() ~ end_utterance() 사이만 인식
        on_speech_start / on_speech_end: 말소리가 들어오기 시작함 / 끝나서 인식에 넘김.
            화면에 "듣는 중" 표시용. 끝 다음에 결과가 없을 수도 있다(잡음으로 판정)
        """
        self.cfg = cfg or VoiceConfig()
        self.on_transcript = on_transcript
        self.on_stop = on_stop
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.transcriber = transcriber or WhisperTranscriber(self.cfg.stt)
        self.stop_words = StopWordDetector(self.cfg.stt.stop_words)
        self.push_to_talk = push_to_talk
        a = self.cfg.audio
        self.segmenter = Segmenter(self.cfg.vad, a.sample_rate, a.frame_samples)
        self._mic = mic
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._muted = False
        self._ptt_active = False
        self._ptt_frames: list[np.ndarray] = []
        self._ptt_start = 0.0
        self._samples = 0
        self._last_partial_ms = 0.0
        self._stop_fired = False   # 한 발화에서 정지 이벤트는 한 번만

    # ---- 수명 주기 -------------------------------------------------------
    def start(self) -> None:
        if hasattr(self.transcriber, "load"):
            self.transcriber.load()   # 첫 명령이 느리지 않도록 미리 로드
        if self._mic is None:
            from .mic import Microphone

            self._mic = Microphone(self.cfg.audio)
        self._worker = threading.Thread(target=self._run, name="voice-listener", daemon=True)
        self._worker.start()
        self._mic.start(self._frames.put)

    def close(self) -> None:
        if self._mic is not None:
            self._mic.stop()
        self._frames.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 제어 ------------------------------------------------------------
    def set_muted(self, muted: bool) -> None:
        """TTS 재생 중 True. 로봇 자기 목소리를 명령으로 받지 않되 정지어는 계속 듣는다."""
        self._muted = muted

    def begin_utterance(self) -> None:
        """말하기 버튼 누름 (push_to_talk 모드)."""
        self._ptt_frames = []
        self._ptt_start = self._samples / self.cfg.audio.sample_rate
        self._last_partial_ms = 0.0
        self._stop_fired = False
        self._ptt_active = True

    def end_utterance(self) -> None:
        """말하기 버튼 뗌. 모은 소리를 인식 큐에 넘긴다."""
        self._ptt_active = False
        self._frames.put(_END_PTT)

    # ---- 처리 ------------------------------------------------------------
    def _run(self) -> None:
        while True:
            frame = self._frames.get()
            if frame is None:
                utt = None if self.push_to_talk else self.segmenter.flush()
                if utt is not None:
                    self._handle_utterance(utt)
                return
            try:
                self.process_frame(frame)
            except Exception:  # noqa: BLE001 - 인식 실패로 리스너가 죽으면 안 됨
                log.exception("음성 처리 중 오류")

    def process_frame(self, frame: np.ndarray) -> None:
        """프레임 하나 처리. 테스트에서는 스레드 없이 직접 호출한다."""
        if frame is _END_PTT:
            self._finish_ptt()
            return
        self._samples += len(frame)
        if self.push_to_talk:
            if self._ptt_active:
                self._ptt_frames.append(frame)
                self._maybe_partial(np.concatenate(self._ptt_frames), self._ptt_start)
            return

        was_speaking = self.segmenter.in_speech
        utt = self.segmenter.feed(frame)
        if self.segmenter.in_speech and not was_speaking:
            self._last_partial_ms = 0.0
            self._stop_fired = False
            self._notify(self.on_speech_start)
        elif was_speaking and not self.segmenter.in_speech:
            self._notify(self.on_speech_end)
        if utt is not None:
            self._handle_utterance(utt)
        elif self.segmenter.in_speech:
            start_s = self._samples / self.cfg.audio.sample_rate - self.segmenter.current_duration_ms / 1000
            self._maybe_partial(self.segmenter.current_audio(), start_s)

    def _maybe_partial(self, audio: np.ndarray, start_s: float) -> None:
        """말하는 도중 최근 구간을 미리 인식해 정지어를 일찍 잡는다."""
        stt = self.cfg.stt
        if self.on_stop is None or self._stop_fired or stt.partial_interval_ms <= 0:
            return
        sr = self.cfg.audio.sample_rate
        dur_ms = audio.size * 1000 / sr
        if dur_ms - self._last_partial_ms < stt.partial_interval_ms:
            return
        self._last_partial_ms = dur_ms
        window = int(sr * stt.partial_window_ms / 1000)
        if audio.size > window:
            start_s += (audio.size - window) / sr
            audio = audio[-window:]
        t = self._transcribe(audio, start_s, start_s + audio.size / sr, partial=True)
        if t.is_stop:
            self._stop_fired = True
            self.on_stop(t)

    @staticmethod
    def _notify(cb: Callable[[], None] | None) -> None:
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001 - 화면 표시 오류로 인식이 멈추면 안 됨
                log.exception("음성 상태 콜백 오류")

    def _finish_ptt(self) -> None:
        if not self._ptt_frames:
            return
        audio = np.concatenate(self._ptt_frames)
        self._ptt_frames = []
        start = self._ptt_start
        self._handle_utterance(Utterance(audio, start, start + audio.size / self.cfg.audio.sample_rate))

    def _handle_utterance(self, utt: Utterance) -> None:
        t = self._transcribe(utt.audio, utt.start_s, utt.end_s)
        if t.is_stop:
            if not self._stop_fired and self.on_stop is not None:
                self.on_stop(t)
            self._stop_fired = False
            return
        self._stop_fired = False
        if not t.text or self._muted:
            return
        self.on_transcript(t)

    def _transcribe(self, audio: np.ndarray, start_s: float, end_s: float, partial: bool = False) -> Transcript:
        t0 = time.perf_counter()
        text, conf, lang = self.transcriber.transcribe(audio, partial=partial)
        return Transcript(
            text=text,
            confidence=conf,
            language=lang,
            start_s=round(start_s, 3),
            end_s=round(end_s, 3),
            latency_s=round(time.perf_counter() - t0, 3),
            stop_word=self.stop_words.match(text) if text else None,
            partial=partial,
        )


# push-to-talk 종료 신호 (큐 안에서 프레임과 구분하기 위한 표식)
_END_PTT = np.zeros(0, dtype=np.float32)
