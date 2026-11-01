"""마이크 입력. sounddevice 콜백으로 받은 프레임을 넘겨준다."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .config import AudioConfig


def list_input_devices() -> list[dict]:
    import sounddevice as sd

    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"], "default_sr": d["default_samplerate"]}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def resolve_device(device: str | int | None) -> int | None:
    """장치 이름 일부로 번호를 찾는다. 못 찾으면 사용 가능한 장치 목록과 함께 오류."""
    if device is None or isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    devices = list_input_devices()
    for d in devices:
        if device.lower() in d["name"].lower():
            return d["index"]
    names = ", ".join(f'{d["index"]}:{d["name"]}' for d in devices)
    raise ValueError(f'"{device}" 이름의 마이크가 없습니다. 사용 가능: {names}')


class Microphone:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._stream = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            on_frame(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.frame_samples,
            channels=1,
            dtype="float32",
            device=resolve_device(self.cfg.device),
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
