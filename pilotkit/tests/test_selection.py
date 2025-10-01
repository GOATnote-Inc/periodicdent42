from datetime import datetime, timedelta

import pytest

from pilotkit.orchestrator.models.schemas import Candidate, CandidateScoreRequest, ScoreWeights
from pilotkit.orchestrator.main import score_candidates


@pytest.mark.asyncio
async def test_high_fit_candidate_ranked_first():
    request = CandidateScoreRequest(
        weights=ScoreWeights(),
        candidates=[
            Candidate(
                name="Fast Lab",
                exec_sponsor=5,
                data_access=5,
                workflow_speed=5,
                potential_value=5,
                risk=1,
                champion=5,
            ),
            Candidate(
                name="Slow Lab",
                exec_sponsor=2,
                data_access=2,
                workflow_speed=1,
                potential_value=3,
                risk=4,
                champion=2,
            ),
        ],
    )
    response = await score_candidates(request, True)
    assert response.ranked[0].candidate.name == "Fast Lab"
    assert response.ranked[0].score > response.ranked[1].score
