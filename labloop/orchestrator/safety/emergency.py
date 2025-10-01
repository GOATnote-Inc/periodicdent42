from __future__ import annotations

import threading
from typing import Callable


class EmergencyStop:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []

    def trigger(self) -> None:
        self._event.set()
        for cb in self._callbacks:
            cb()

    def register(self, callback: Callable[[], None]) -> None:
        self._callbacks.append(callback)

    def token(self) -> threading.Event:
        return self._event

    def reset(self) -> None:
        self._event.clear()
