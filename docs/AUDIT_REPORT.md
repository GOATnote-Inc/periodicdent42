# Repository Audit Report

_Last updated: 2025-10-02_

## Overview

This audit used static import analysis (`make graph`) and heuristic quality checks (`make audit`) to map the Periodic Labs monorepo. The codebase contains multiple partially overlapping vertical slices (`app`, `apps`, `labloop`, `pilotkit`, `midupdate`, `synthloop`) that target similar autonomous lab loops. Several critical subsystems—routing, guardrails, hardware orchestration—are duplicated with diverging implementations. Documentation and automation for the current "master" slice (`apps/api` + `services/*`) is sparse, creating runability and maintenance risk.

## Scorecard

| Area | Score (0-5) | Notes |
| --- | --- | --- |
| Runability | 2 | Multiple stacks, missing single entrypoint; demos require manual wiring of API + web.
| Clarity | 2 | Competing implementations (`labloop`, `pilotkit`, `app`) with overlapping responsibilities and TODO-heavy drivers.
| Cohesion | 2 | Shared utilities scattered across `services`, `src`, `labloop`.
| Coupling | 3 | Core RAG service is relatively isolated, but orchestration code references global settings and in-memory stores.
| Test Coverage | 2 | Only a handful of unit tests (`tests/test_safety_gateway.py`, `app/tests/*`). Most services untested.
| User Demo | 1 | Prior UI (`apps/web/app/demo`) is static; no live API wiring.
| Maintenance Risk | 4 | Large backlog of TODOs, no enforced lint/test entrypoint outside `app/`.

## Findings

| Path | Category | Issue | Evidence | Recommended Action | Effort (1-5) | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| services/rag/pipeline.py | Architecture | `ChatPipeline.default` re-ingests corpus and rebuilds embeddings on each process start, and `run` mutates guardrail payloads inline. No caching or streaming support. | Pipeline builds index via `ingest()` (loads full corpus) and constructs `guardrail_status` dictionaries manually. | Extract ingest bootstrap into singleton, add streaming-friendly interface, and cover with unit tests for guardrail serialization. | 3 | High |
| services/telemetry/store.py | Data | Telemetry persistence is in-memory only; no rotation, no serialization, so demo data disappears per process and tests cannot assert behaviour. | `TelemetryStore.in_memory()` returns dataclass list without flush/save. | Implement filesystem-backed or SQLite telemetry adapter with tests; wire to `.env` toggle. | 2 | High |
| services/llm/router.py | Logic | Rule-based routing lacks logging, thresholds, or evaluation hooks; ambiguous `DEFAULT_ARM`. Missing tests covering override precedence. | Simple length check without instrumentation. | Add structured logging + config thresholds, create unit tests for override/length/default paths. | 2 | Medium |
| services/rag/index.py | Performance | `top_k` keyword scoring ignores embeddings, ANN timings hardcoded to 1.2ms, no evaluation dataset wiring. | Simplified overlap scoring and static ANN latency. | Integrate cosine similarity using embeddings, load eval dataset, and expose latency metrics. | 4 | Medium |
| services/telemetry/dash_data.py | Dead-end | Generates `Dash` payloads but no consumer in repo; not referenced from API. | No inbound references from `rg dash_data`. | Either archive under `archive/` or integrate with frontend dashboard plan. | 1 | Medium |
| app/src/reasoning/mcp_agent.py | Confusing | 300+ line agent with embedded TODOs for hardware integration; mixes planning, execution, validation without interfaces. | Multiple TODO comments, nested `try/except`, no tests. | Break into planner/executor modules, design hardware adapter protocol, add unit tests for planner. | 5 | High |
| src/experiment_os/drivers/* | Confusing | Hardware drivers filled with TODO placeholders and docstrings warning about missing vendor logic. Hard to tell readiness. | `# TODO` markers across modules. | Move unfinished drivers to `archive/<date>/` until vendor implementation provided; leave shim interface. | 2 | High |
| labloop/ (entire) | Duplication | Full autonomous loop duplicate of current stack; unclear which is canonical. README instructs separate Quickstart. | Overlaps with `services/` orchestrator; not referenced in top-level README. | Archive the slice or clearly document difference in README + Quickstart to avoid diverging orchestrators. | 4 | High |
| pilotkit/ (entire) | Duplication | Another orchestrator/Next.js UI pair with tests; not referenced anywhere else. Maintainers risk splitting fixes. | Directory structure replicates labloop. | Decide on canonical orchestrator; archive others or merge shared libs into `/services`. | 4 | Medium |
| apps/web/app/demo/page.tsx | UX | Static placeholders; no live API connection or run instructions, so stakeholders cannot test pipeline. | Hard-coded `DEFAULT_ANSWER`. | Replace with API-backed demo (this PR introduces `/demos/rag-chat`). | 2 | High |
| tests/ | Testing | Only one root-level test; `services/*` and `apps/api` lack coverage. | `tests/test_safety_gateway.py` the only root test. | Introduce pytest suites for router, guardrails, telemetry store. | 3 | High |

## Next Steps

1. Land this PR to establish reproducible tooling, demo UI, and living docs.
2. Plan follow-up PRs per ticket files in `docs/tickets/`.
3. Decide on archival vs merge strategy for `labloop` and `pilotkit` to cut duplication.
4. Prioritize persistence + testing improvements for the production RAG slice (`apps/api`, `services/*`).

