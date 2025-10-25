from __future__ import annotations

from dataclasses import dataclass

from services.telemetry.store import TelemetryStore


@dataclass
class DemoDashboard:
    total_chats: int
    arms: list[str]


def build_demo_dashboard(store: TelemetryStore) -> DemoDashboard:
    arms = [record.arm for record in store.records]
    return DemoDashboard(total_chats=len(store.records), arms=arms)


__all__ = ["build_demo_dashboard", "DemoDashboard"]
