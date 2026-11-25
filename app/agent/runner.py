"""음성 입력(VoiceListener)과 명령 해석(CommandAgent)을 잇는다.

- 인식된 문장은 큐에 넣고 별도 스레드에서 하나씩 처리한다. LLM 응답을 기다리는 동안에도
  음성 입력 스레드는 막히지 않아서 정지어를 계속 들을 수 있다.
- 정지어는 큐를 거치지 않는다. 바로 robot.stop()을 부르고, 진행 중인 명령을 취소하고,
  아직 처리하지 않은 명령은 버린다 (설계서 7.2: 정지 경로는 STT·LLM과 분리).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from .agent import AgentReply, CommandAgent

log = logging.getLogger(__name__)

ReplyCallback = Callable[[AgentReply], None]


class VoiceCommandRunner:
    def __init__(self, agent: CommandAgent, on_reply: ReplyCallback,
                 on_event: Callable[[dict], None] | None = None):
        self.agent = agent
        self.on_reply = on_reply
        self.on_event = on_event or (lambda e: None)
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

    # VoiceListener(on_transcript=runner.on_transcript, on_stop=runner.on_stop)
    def on_transcript(self, t) -> None:
        self.on_event(t.to_event())
        self._queue.put(t)

    def on_stop(self, t) -> None:
        self.on_event(t.to_event())
        self.stop()

    def stop(self) -> None:
        """정지어, 화면 정지 버튼 등에서 부른다."""
        try:
            self.agent.robot.stop()
        finally:
            self.agent.cancel()
            dropped = self._drain()
            self.on_event({"type": "estop", "source": "voice", "dropped_commands": dropped})
            self.on_reply(AgentReply("멈췄습니다.", "cancelled"))

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="command-agent", daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=30)
            self._worker = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def _drain(self) -> int:
        n = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return n
            if item is None:          # 종료 신호는 되돌려 둔다
                self._queue.put(None)
                return n
            n += 1

    def _run(self) -> None:
        while True:
            t = self._queue.get()
            if t is None:
                return
            try:
                reply = self.agent.handle(t.text, t.confidence)
            except Exception:  # noqa: BLE001 - 스레드가 죽지 않게
                log.exception("명령 처리 실패")
                continue
            if reply.status != "cancelled":   # 취소는 stop()에서 이미 알렸다
                self.on_reply(reply)
