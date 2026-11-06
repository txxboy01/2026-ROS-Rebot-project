"""faster-whisper 기반 음성 → 텍스트 변환."""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from .config import STTConfig
from .stopword import normalize

log = logging.getLogger(__name__)


def _add_cuda_dll_dirs() -> None:
    """Windows: pip로 설치한 nvidia-cublas-cu12, nvidia-cudnn-cu12의 DLL 경로를 등록.

    ctranslate2는 PATH에서 cublas64_12.dll을 찾는데, pip 패키지는 PATH에 올라가지 않는다.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            bin_dir = os.path.join(root, sub, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


@dataclass
class Transcript:
    text: str
    confidence: float     # 0~1. 0.5 미만이면 다시 묻는 것을 권장 (설계서 5.1)
    language: str
    start_s: float        # 스트림 시작 기준 발화 시작/끝
    end_s: float
    latency_s: float      # 발화 끝 → 텍스트 나오기까지 걸린 시간
    stop_word: str | None = None
    partial: bool = False  # 말하는 도중 미리 인식한 결과인지

    @property
    def is_stop(self) -> bool:
        return self.stop_word is not None

    def to_event(self) -> dict:
        """이벤트 버스 / 로그(events.jsonl)용 딕셔너리."""
        return {"type": "stt", **asdict(self)}


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, partial: bool = False) -> tuple[str, float, str]:
        """(텍스트, 신뢰도, 언어) 반환. audio는 16 kHz float32 모노.

        partial=True는 말하는 도중의 앞부분 인식(정지어 사전 검사)이다.
        """
        ...


class WhisperTranscriber:
    def __init__(self, cfg: STTConfig | None = None):
        self.cfg = cfg or STTConfig()
        self._model = None
        self.device: str | None = None
        self._hallucinations = tuple(normalize(h) for h in self.cfg.hallucinations)
        self._hallucinations_exact = frozenset(normalize(h) for h in self.cfg.hallucinations_exact)
        self._prompt_norm = normalize(self.cfg.initial_prompt or "")

    def load(self) -> None:
        """모델을 불러온다. CUDA가 안 되면 CPU로 넘어간다.

        ctranslate2는 CUDA 라이브러리(cuBLAS, cuDNN)가 없으면 모델 로드가 아니라
        첫 추론에서 실패하므로, 짧은 무음으로 한 번 돌려 보고 판단한다.
        """
        if self._model is not None:
            return
        _add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        devices = ["cuda", "cpu"] if self.cfg.device == "auto" else [self.cfg.device]
        last_err: Exception | None = None
        for dev in devices:
            compute = self.cfg.compute_type
            if compute == "auto":
                compute = "float16" if dev == "cuda" else "int8"
            try:
                t0 = time.perf_counter()
                model = WhisperModel(self.cfg.model, device=dev, compute_type=compute)
                list(model.transcribe(np.zeros(8000, dtype=np.float32), language=self.cfg.language)[0])
                self._model, self.device = model, dev
                log.info("STT 모델 %s 로드 (%s, %s, %.1fs)", self.cfg.model, dev, compute, time.perf_counter() - t0)
                return
            except Exception as e:  # noqa: BLE001 - CUDA 관련 예외 종류가 다양함
                last_err = e
                log.warning("STT %s 사용 실패: %s", dev, e)
        raise RuntimeError(f"STT 모델을 불러오지 못했습니다: {last_err}")

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> tuple[str, float, str]:
        self.load()
        segments, info = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            initial_prompt=None if partial else self.cfg.initial_prompt,
            condition_on_previous_text=False,
            vad_filter=False,          # 구간은 이미 Segmenter가 잘랐음
            without_timestamps=True,
        )
        texts, logprobs, weights = [], [], []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                continue
            texts.append(seg.text.strip())
            logprobs.append(seg.avg_logprob)
            weights.append(max(1, len(seg.text)))
        text = " ".join(t for t in texts if t).strip()
        if not text or self._is_hallucination(text) or is_prompt_echo(text, self._prompt_norm):
            return "", 0.0, info.language
        avg = sum(lp * w for lp, w in zip(logprobs, weights)) / sum(weights)
        conf = round(min(1.0, math.exp(avg)), 3)
        if conf < self.cfg.min_confidence:
            return "", 0.0, info.language
        return text, conf, info.language

    def _is_hallucination(self, text: str) -> bool:
        norm = normalize(text)
        return norm in self._hallucinations_exact or any(h in norm for h in self._hallucinations)


def _longest_common_substring(a: str, b: str) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def is_prompt_echo(text: str, prompt_norm: str, min_chars: int = 6) -> bool:
    """단서 문장을 그대로 또는 일부 읊은 결과인지.

    말소리가 아닌 소리(숨, 딸깍, "음")가 들어오면 모델은 어떤 단서든 거의 매번 읊고,
    이때 신뢰도가 0.7을 넘기도 한다(실측). 그래서 신뢰도가 아니라 글자 겹침으로 판단한다.
    실제 명령은 "빨간블록", "드라이버"처럼 4글자 정도만 겹치므로 6글자 이상 연속으로 같으면 버린다.
    """
    if not prompt_norm:
        return False
    return _longest_common_substring(normalize(text), prompt_norm) >= min_chars
