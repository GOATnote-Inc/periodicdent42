from __future__ import annotations

import contextlib
import logging
import threading
from typing import Iterator

LOGGER = logging.getLogger(__name__)


class SafetyGuard:
    def __init__(self, cancel_token: threading.Event) -> None:
        self.cancel_token = cancel_token

    def check(self) -> None:
        if self.cancel_token.is_set():
            raise RuntimeError("Operation cancelled by safety guard")

    def __call__(self) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def wrapper() -> Iterator[None]:
            try:
                self.check()
                yield
            except Exception as exc:
                LOGGER.error("Safety guard triggered: %s", exc)
                self.cancel_token.set()
                raise
        return wrapper()
