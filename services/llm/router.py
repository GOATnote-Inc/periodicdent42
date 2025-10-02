from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SupportsRouterContext(Protocol):
    query: str
    arm: str | None


@dataclass
class RouterDecision:
    arm: str
    policy: str


DEFAULT_ARM = "balanced"


def select_arm(request: SupportsRouterContext) -> RouterDecision:
    if request.arm:
        return RouterDecision(arm=request.arm, policy="user_override")
    if len(request.query) > 280:
        return RouterDecision(arm="high-accuracy", policy="length-rule")
    return RouterDecision(arm=DEFAULT_ARM, policy="bandit-placeholder")


__all__ = ["RouterDecision", "select_arm"]
