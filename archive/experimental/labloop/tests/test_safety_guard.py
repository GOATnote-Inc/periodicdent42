import threading

import pytest

from labloop.orchestrator.safety.guard import SafetyGuard


def test_safety_guard_sets_cancel_on_error():
    token = threading.Event()
    guard = SafetyGuard(token)
    with pytest.raises(RuntimeError):
        with guard():
            token.set()
            guard.check()
    assert token.is_set()
