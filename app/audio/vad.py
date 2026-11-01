"""마이크 프레임을 받아 발화 구간 하나씩 잘라내는 에너지 기반 VAD."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import VADConfig


@dataclass
class Utterance:
    audio: np.ndarray   # float32, -1~1, 모노
    start_s: float      # 스트림 시작 기준 초
    end_s: float


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


class Segmenter:
    def __init__(self, cfg: VADConfig, sample_rate: int, frame_samples: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        frame_ms = frame_samples * 1000 / sample_rate
        self._end_frames = max(1, round(cfg.end_silence_ms / frame_ms))
        self._min_frames = max(1, round(cfg.min_speech_ms / frame_ms))
        self._max_frames = max(1, round(cfg.max_speech_ms / frame_ms))
        self._pre = deque(maxlen=max(1, round(cfg.pre_roll_ms / frame_ms)))
        self._calib_frames = round(cfg.calibration_ms / frame_ms)
        self._calib_count = 0
        # 쉬는 구간 소리 크기 기록. 잡음 기준은 평균이 아니라 상위 분위수로 잡는다.
        # 노이즈 게이트가 있는 헤드셋(G733 등)은 평소 0을 내보내 평균이 0으로 끌려가고,
        # 그러면 게이트가 잠깐 열릴 때마다 말소리로 착각해 발화가 끝나지 않는다(실측).
        self._idle = deque(maxlen=max(1, round(cfg.noise_window_ms / frame_ms)))
        self.noise_floor = cfg.min_rms
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self._frames: list[np.ndarray] = []
        self._loud_run = 0
        self._quiet_run = 0
        self._voiced = 0
        self._start_sample = 0
        self._speech_level = 0.0
        self._recent = deque(maxlen=self.cfg.smooth_frames)
        self._pre.clear()
        self._samples_seen = getattr(self, "_samples_seen", 0)

    @property
    def start_threshold(self) -> float:
        return max(self.cfg.min_rms, self.noise_floor * self.cfg.start_ratio)

    @property
    def end_threshold(self) -> float:
        """말하는 중 이 아래로 떨어지면 조용한 것으로 본다.

        방 잡음 기준만 쓰면 숨소리·옷 스침 같은 짧은 튐이 침묵 카운트를 계속 초기화해
        문장 사이를 쉬어도 끝나지 않는다(헤드셋 마이크 시험에서 확인). 그래서 방금 말한
        목소리 크기의 일정 비율보다 작아도 조용한 것으로 본다.
        """
        return max(
            self.cfg.min_rms * 0.75,
            self.noise_floor * self.cfg.end_ratio,
            self._speech_level * self.cfg.end_speech_ratio,
        )

    def current_audio(self) -> np.ndarray:
        """말하는 중인 구간의 지금까지 소리. 정지어 사전 검사에 쓴다."""
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    @property
    def current_duration_ms(self) -> float:
        return len(self._frames) * self.frame_samples * 1000 / self.sample_rate

    def feed(self, frame: np.ndarray) -> Utterance | None:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        level = rms(frame)
        frame_start = self._samples_seen
        self._samples_seen += frame.size

        if self._calib_count < self._calib_frames:
            self._calib_count += 1
            self._learn_noise(level)
            self._pre.append((frame_start, frame))
            return None

        if not self.in_speech:
            if level > self.start_threshold:
                self._loud_run += 1
            else:
                self._loud_run = 0
                self._learn_noise(level)   # 조용할 때만 (말소리가 잡음으로 학습되지 않게)
            self._pre.append((frame_start, frame))
            if self._loud_run >= self.cfg.start_frames:
                self.in_speech = True
                self._frames = [f for _, f in self._pre]
                self._start_sample = self._pre[0][0]
                self._voiced = self._loud_run
                self._quiet_run = 0
                self._speech_level = level
                self._recent.clear()
                self._pre.clear()
            return None

        self._frames.append(frame)
        # 프레임 하나가 튀어도 판단이 뒤집히지 않도록 최근 몇 프레임 평균으로 본다
        self._recent.append(level)
        smoothed = sum(self._recent) / len(self._recent)
        if smoothed < self.end_threshold:
            self._quiet_run += 1
        else:
            self._quiet_run = 0
            self._voiced += 1
            # 목소리 크기는 천천히 따라간다 (한 음절이 크다고 기준이 확 올라가지 않게)
            self._speech_level = max(self._speech_level * 0.98, 0.9 * self._speech_level + 0.1 * level)

        if self._quiet_run >= self._end_frames or len(self._frames) >= self._max_frames:
            return self._finish()
        return None

    def _learn_noise(self, level: float) -> None:
        self._idle.append(level)
        q = float(np.percentile(self._idle, self.cfg.noise_percentile))
        self.noise_floor = max(self.cfg.min_rms * 0.25, q)

    def flush(self) -> Utterance | None:
        """스트림을 닫을 때 남은 발화를 내보낸다."""
        return self._finish() if self.in_speech else None

    def _finish(self) -> Utterance | None:
        # 끝의 침묵 부분은 잘라내되 약간 남겨 마지막 음절 보존
        keep_tail = min(self._quiet_run, max(1, self._end_frames // 3))
        frames = self._frames[: len(self._frames) - self._quiet_run + keep_tail] or self._frames
        voiced = self._voiced
        start = self._start_sample
        self.reset()
        if voiced < self._min_frames:
            return None
        audio = np.concatenate(frames)
        return Utterance(
            audio=audio,
            start_s=start / self.sample_rate,
            end_s=(start + audio.size) / self.sample_rate,
        )
