# Deprecated Module

`pilotkit` is maintained for archival purposes only. The canonical orchestrator lives in `services/agents/orchestrator.py`, which
powers both the API and demo experiences. Shared utilities should be migrated into the `core` package.

## Retirement Timeline

- Q4 FY25: No new features added to `pilotkit`.
- Q1 FY26: Freeze repository state and migrate any remaining notebooks to the services pipeline.
- Q2 FY26: Remove deployment manifests and archive directory.

Report remaining dependencies in `docs/telemetry-and-router-runbook.md` before final removal.
