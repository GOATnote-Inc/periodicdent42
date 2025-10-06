# Telemetry and Router Runbook

## Overview

Phase-2 hardening introduces a Cloud SQL-backed telemetry store, persistent RAG index caching, and an observable LLM router. This
runbook captures operational procedures for the new components.

## Architecture

- **Telemetry store**: PostgreSQL tables (`telemetry_runs`, `telemetry_events`, `telemetry_metrics`, `telemetry_errors`,
  `telemetry_artifacts`) managed via Alembic migrations in `infra/db/migrations`.
- **RAG cache**: `services.rag.index_store.RagIndex` persists FAISS-like embeddings on disk in `.cache/rag` by default.
- **Router**: `services.router.llm_router.LLMRouter` hashes inputs, applies configured thresholds, emits structured JSON logs, and
  records Prometheus counters (`router_decisions_total`).

## Security Posture

- **PII/PHI**: The telemetry store only tracks synthetic demo content. No PII/PHI is ingested.
- **Access control**: Cloud SQL IAM governs access in production. For local development, use per-user credentials.
- **Data retention**: Retain production telemetry for 30 days. Review weekly; export anomalies to BigQuery before truncation.
- **Purge dev data**: Run `python -m tools.telemetry tail --last 5` to confirm targets, then execute a SQL `DELETE` with run IDs or
  drop the SQLite file (`telemetry.db`) for local environments.

## Operations

### Database Migrations

```bash
make db-upgrade                       # apply latest migrations
env MESSAGE="add telemetry column" make db-migrate  # create a new migration stub
```

`DATABASE_URL` or the `DB_*` environment variables configure connections. CI uses SQLite for isolation.

### Telemetry API

- `GET /api/telemetry/runs?limit=50&status=completed`
- `GET /api/telemetry/runs/{id}`
- `GET /api/telemetry/runs/{id}/events`

Use these routes to drive dashboards or ad-hoc investigations. Responses are read-only snapshots of the database.

### Router Observability

- Logs: look for `router_decision` entries with a `payload` JSON blob containing input hash, tokens, and reason codes.
- Metrics: scrape `/metrics` and inspect `router_decisions_total{arm="high-accuracy",reason="uncertainty_high"}` for drift.
- Threshold tuning: update `configs/service.yaml` (`router` section) or set environment overrides (`ROUTER_LATENCY_BUDGET_MS`,
  `ROUTER_MAX_CONTEXT_TOKENS`, `ROUTER_UNCERTAINTY_THRESHOLD`).

### RAG Cache Maintenance

- Default cache path: `.cache/rag` (ignored by git).
- Override via `RAG_CACHE_DIR` or `RAG_CACHE_TTL_SECONDS`.
- Purge cache by deleting the directory; it will be rebuilt on next request.

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| Telemetry endpoints empty | Confirm migrations ran (`make db-upgrade`) and that `DATABASE_URL` points to Cloud SQL. |
| Router always chooses `high-accuracy` | Inspect logs for `uncertainty_high` reason; adjust thresholds if necessary. |
| Slow repeat queries | Ensure `.cache/rag/index_meta.json` exists; otherwise rebuild embeddings via `/api/chat`. |
| Coverage gate failing | Run `pytest --cov-report=term-missing` locally and review uncovered branches in new modules. |

## Incident Response

1. Capture the failing request, router log payload, and telemetry run ID.
2. File an incident ticket with the run ID and attach the `telemetry_artifacts` metadata.
3. Use the CLI `python -m tools.telemetry tail --last 50` to inspect recent runs for patterns.
4. Roll back by redeploying the previous container image if telemetry persistence impacts availability.

## Contacts

- **Primary**: Platform Reliability (@oncall-platform)
- **Secondary**: AI Experience Squad (@ai-exp)
- **Escalation**: security@periodicdent42.com for data-related incidents
