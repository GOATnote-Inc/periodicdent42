"""Utilities for redacting potentially sensitive data before logging or storage."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN_PATTERN = re.compile(r"\b(?:MRN|Medical\s*Record\s*Number)[:#]?\s*\w+\b", re.IGNORECASE)


def generate_request_id() -> str:
    """Return a random identifier suitable for correlating logs without PII."""

    return uuid4().hex


def _redact_text(value: str) -> str:
    """Best-effort removal of common identifiers from free text."""

    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
    redacted = _PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = _SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
    redacted = _MRN_PATTERN.sub("[REDACTED_MRN]", redacted)
    return redacted


def sanitize_payload(value: Any) -> Any:
    """Recursively redact common PHI/PII patterns in the supplied value."""

    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {k: sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_payload(v) for v in value)
    return value

