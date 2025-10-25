from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from typing import Any


def deterministic_run_id(plan_payload: Any) -> str:
    data = repr(plan_payload).encode("utf-8")
    digest = hashlib.sha1(data).hexdigest()[:8]
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"run-{timestamp}-{digest}"


def random_task_id() -> str:
    return uuid.uuid4().hex[:8]


def feature_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name.upper())
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
