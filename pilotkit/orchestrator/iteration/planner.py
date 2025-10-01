from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import json

from ..models.schemas import IterationBacklogItem, IterationPlan, IterationPlanRequest

DEFAULT_OWNERS = ["Pilot PM", "Workflow Lead", "Eng Enablement"]


def rank_backlog(request: IterationPlanRequest) -> IterationPlan:
    items: List[IterationBacklogItem] = []
    guardrails = request.guardrails or {}
    for idx, theme in enumerate(request.top_feedback_themes):
        impact = max(abs(request.metrics_delta.get("cycle_time_pct", -20)) / 10, 1)
        confidence = max(0.5, 1 - (idx * 0.1))
        effort = 1 + idx * 0.5
        ice = (impact * confidence) / effort
        owner = DEFAULT_OWNERS[idx % len(DEFAULT_OWNERS)]
        metric = "cycle_time_p50" if "cycle" in theme.lower() else "yield_rate"
        guardrail = guardrails.get(metric, "Yield must not drop below 95%")
        items.append(
            IterationBacklogItem(
                title=f"{theme} improvement",
                description=f"Address theme '{theme}' to unlock metric gains.",
                impact=float(round(impact, 2)),
                confidence=float(round(confidence, 2)),
                effort=float(round(effort, 2)),
                ice_score=float(round(ice, 2)),
                owner=owner,
                eta_days=14 + idx * 7,
                metric=metric,
                guardrail=guardrail,
            )
        )
    items.sort(key=lambda x: x.ice_score, reverse=True)
    summary = (
        "Prioritized actions to sustain pilot gains and mitigate key failure modes. "
        f"Top theme: {items[0].title if items else 'n/a'}."
    )
    return IterationPlan(generated_at=datetime.utcnow(), items=items[:5], summary=summary)


def export_plan(plan: IterationPlan, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = plan.generated_at.strftime("%Y%m%d-%H%M%S")
    markdown_path = output_dir / f"iteration-plan-{timestamp}.md"
    json_path = output_dir / f"iteration-plan-{timestamp}.json"
    lines = ["# Next Iteration Plan", "", f"Generated: {plan.generated_at.isoformat()}", ""]
    for item in plan.items:
        lines.extend(
            [
                f"## {item.title}",
                f"- Owner: {item.owner}",
                f"- ICE: {item.ice_score}",
                f"- Impact: {item.impact}",
                f"- Confidence: {item.confidence}",
                f"- Effort: {item.effort}",
                f"- Target Metric: {item.metric}",
                f"- Guardrail: {item.guardrail}",
                f"- ETA: {item.eta_days} days",
                "",
                item.description,
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines))
    json_path.write_text(plan.json(indent=2))
    return {"markdown": markdown_path, "json": json_path}
