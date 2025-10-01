from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Watchdog:
    timeout_s: float
    on_timeout: Callable[[], None]

    def __post_init__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._reset_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            triggered = self._reset_event.wait(timeout=self.timeout_s)
            if triggered:
                self._reset_event.clear()
                continue
            self.on_timeout()
            self._reset_event.clear()

    def kick(self):
        self._reset_event.set()

    def stop(self):
        self._stop_event.set()
        self._reset_event.set()
        self._thread.join(timeout=1)


