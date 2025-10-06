from __future__ import annotations

import hashlib
from typing import Iterable


def stable_hash(values: Iterable[str]) -> str:
    """Compute a stable SHA-256 hash for a collection of string values."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


__all__ = ["stable_hash"]
