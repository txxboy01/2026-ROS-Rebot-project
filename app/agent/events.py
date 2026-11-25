"""이벤트 로그 (설계서 6장 logs/events.jsonl). 한 줄에 이벤트 하나."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

EventSink = Callable[[dict], None]


class EventLog:
    def __init__(self, path: str | Path = "logs/events.jsonl", echo: EventSink | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()

    def __call__(self, event: dict) -> None:
        event = {"t": datetime.now().isoformat(timespec="milliseconds"), **event}
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            self.echo(event)
