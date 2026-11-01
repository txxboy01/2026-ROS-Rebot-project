"""음성 인식 실행기.

    python -m app.audio --list-devices       마이크 목록
    python -m app.audio                      마이크로 계속 듣기 (말이 끝나면 자동 인식)
    python -m app.audio --ptt                Enter 누르고 말하고 Enter
    python -m app.audio --file sample.wav    파일 하나 인식
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import wave

import numpy as np

from .config import VoiceConfig
from .listener import VoiceListener
from .stopword import StopWordDetector
from .stt import Transcript, WhisperTranscriber


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """WAV는 직접 읽는다. faster-whisper의 decode_audio는 PyAV 19와 호환되지 않음."""
    if not path.lower().endswith(".wav"):
        from faster_whisper import decode_audio

        return decode_audio(path, sampling_rate=sample_rate)
    with wave.open(path, "rb") as w:
        width, ch, sr = w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"16비트 PCM WAV만 지원합니다 (이 파일은 {width * 8}비트)")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != sample_rate:
        n = round(audio.size * sample_rate / sr)
        audio = np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size), audio).astype(np.float32)
    return audio


def show_levels(cfg: VoiceConfig) -> int:
    """마이크 소리 크기 막대와 VAD 판단을 0.1초마다 표시. 말소리가 안 잡히거나 안 끝날 때 원인 확인용."""
    import queue

    from .mic import Microphone
    from .vad import Segmenter, rms

    seg = Segmenter(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_samples)
    frames: queue.Queue = queue.Queue()
    mic = Microphone(cfg.audio)
    mic.start(frames.put)
    print("소리 크기(RMS) | 막대 | 시작 문턱 / 끝 문턱 | 상태.  끝내려면 Ctrl+C")
    peak, n = 0.0, 0
    try:
        while True:
            f = frames.get()
            peak = max(peak, rms(f))
            utt = seg.feed(f)
            n += 1
            if utt is not None:
                print(f"\n  → 발화 끝: {utt.end_s - utt.start_s:.2f}초 분량을 인식에 넘김")
            if n % 3:
                continue
            bar = "#" * min(40, int(peak / 0.005))
            state = "말하는 중" if seg.in_speech else "조용"
            print(f"\r{peak:.4f} |{bar:<40}| {seg.start_threshold:.4f} / {seg.end_threshold:.4f} | {state}   ",
                  end="", flush=True)
            peak = 0.0
    except KeyboardInterrupt:
        print()
    finally:
        mic.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.audio", description="한국어 음성 명령 인식")
    p.add_argument("--list-devices", action="store_true", help="마이크 목록 출력")
    p.add_argument("--device", help="마이크 이름 일부 또는 번호")
    p.add_argument("--model", help="Whisper 모델 (기본 large-v3-turbo, 가볍게는 small)")
    p.add_argument("--cpu", action="store_true", help="GPU를 쓰지 않음")
    p.add_argument("--file", help="음성 파일 하나를 인식하고 종료")
    p.add_argument("--ptt", action="store_true", help="Enter로 말하기 시작/끝")
    p.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로 출력")
    p.add_argument("--levels", action="store_true", help="마이크 소리 크기와 말소리 판단을 실시간 표시 (설정 조정용)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔 한글 깨짐 방지

    if args.list_devices:
        from .mic import list_input_devices

        for d in list_input_devices():
            print(f'{d["index"]:>3}  {d["name"]}  ({d["channels"]}ch, {int(d["default_sr"])} Hz)')
        return 0

    cfg = VoiceConfig()
    if args.device:
        cfg.audio.device = args.device
    if args.model:
        cfg.stt.model = args.model
    if args.cpu:
        cfg.stt.device = "cpu"

    if args.levels:
        return show_levels(cfg)

    status_shown = False

    def status(text: str) -> None:
        nonlocal status_shown
        if args.json:
            return
        # 같은 줄을 덮어써서 "듣는 중..." 표시를 띄우고 지운다
        print("\r" + " " * 30 + "\r" + text, end="", flush=True)
        status_shown = bool(text)

    def show(t: Transcript) -> None:
        if status_shown:
            status("")
        if args.json:
            print(json.dumps(t.to_event(), ensure_ascii=False), flush=True)
        else:
            tag = "정지" if t.is_stop else "명령"
            extra = " (말하는 중 감지)" if t.partial else ""
            print(f"[{tag}] {t.text}  신뢰도 {t.confidence:.2f}  처리 {t.latency_s:.2f}s{extra}", flush=True)

    if args.file:
        audio = load_audio(args.file, cfg.audio.sample_rate)
        stt = WhisperTranscriber(cfg.stt)
        t0 = time.perf_counter()
        text, conf, lang = stt.transcribe(audio)
        dur = audio.size / cfg.audio.sample_rate
        show(Transcript(text, conf, lang, 0.0, round(dur, 3), round(time.perf_counter() - t0, 3),
                        StopWordDetector(cfg.stt.stop_words).match(text)))
        return 0

    print("모델 불러오는 중...", flush=True)
    listener = VoiceListener(
        on_transcript=show,
        on_stop=show,
        cfg=cfg,
        push_to_talk=args.ptt,
        on_speech_start=lambda: status("● 듣는 중..."),
        on_speech_end=lambda: status(""),
    )
    listener.start()
    dev = getattr(listener.transcriber, "device", "?")
    try:
        if args.ptt:
            print(f"준비됨 ({dev}). Enter 누르고 말한 뒤 다시 Enter. 끝내려면 q 입력 후 Enter.")
            while True:
                if input().strip().lower() == "q":
                    break
                listener.begin_utterance()
                print("  듣는 중...", flush=True)
                input()
                listener.end_utterance()
        else:
            print(f"준비됨 ({dev}). 말하면 인식합니다. 끝내려면 Ctrl+C.")
            while True:
                time.sleep(0.5)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
