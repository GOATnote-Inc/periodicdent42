from datetime import datetime

from datetime import datetime

from pilotkit.orchestrator.feedback.processor import FeedbackProcessor
from pilotkit.orchestrator.iteration.planner import rank_backlog
from pilotkit.orchestrator.models.schemas import FeedbackItem, IterationPlanRequest


def test_feedback_dedup_and_cluster():
    processor = FeedbackProcessor()
    now = datetime.utcnow()
    items = [
        FeedbackItem(ts=now, severity="P1", step="queue", text="Slow queue assignment", tags=[]),
        FeedbackItem(ts=now, severity="P2", step="queue", text="Slow queue assignment", tags=[]),
        FeedbackItem(ts=now, severity="P1", step="work", text="Instrument error fails run", tags=[]),
    ]
    themes = processor.aggregate_insights(items)
    assert len(themes) == 2
    assert any("latency" in theme["theme"].lower() or "slow" in theme["theme"].lower() for theme in themes)


def test_iteration_plan_top_items():
    request = IterationPlanRequest(
        metrics_delta={"cycle_time_pct": -25},
        top_feedback_themes=["Queue delays", "Defect escapes", "Onboarding friction", "UI polish", "Analytics gap", "Extra"],
        guardrails={"yield_rate": "Maintain > 0.97"},
    )
    plan = rank_backlog(request)
    assert len(plan.items) == 5
    assert all(item.ice_score > 0 for item in plan.items)
    assert any("yield" in item.guardrail.lower() for item in plan.items)
