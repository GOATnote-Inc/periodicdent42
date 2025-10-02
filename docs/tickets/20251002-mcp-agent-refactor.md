# Ticket: Refactor MCP Agent Pipeline

## Context
- `app/src/reasoning/mcp_agent.py` contains orchestration logic for model control program integration.
- File is >300 lines with TODO markers and mixed responsibilities (planning, execution, reward shaping).

## Problem Statement
Difficult to extend or test; hardware safety review flagged lack of interface boundaries.

## Reproduction Steps
1. Open `app/src/reasoning/mcp_agent.py`.
2. Observe nested logic in `run_control_loop` and TODO comments referencing PySCF and instrument APIs.
3. No unit tests cover behaviours; only docstrings describe flow.

## Proposed Plan
- Split file into `planner.py`, `executor.py`, `safety.py` modules with typed interfaces.
- Introduce dependency injection for instrument drivers so tests can provide fakes.
- Document sequence diagram in `docs/ARCHITECTURE_MAP.md` for control loop.
- Add pytest suite verifying planner decision matrix and safety halt scenarios.

## Acceptance Criteria
- Each module <200 lines with docstrings and type hints.
- 80% coverage on new planner/executor tests.
- Safety halt path raises descriptive exception and is logged.

## Test Plan
- `pytest app/tests/reasoning/test_mcp_agent.py` (new).
- Static type check with `pyright` (if available) or `mypy`.
