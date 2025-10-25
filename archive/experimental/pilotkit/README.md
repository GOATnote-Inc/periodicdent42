# PilotKit

PilotKit packages everything required to run a controlled Intelligence Layer pilot: partner selection, workflow instrumentation, analytics, feedback loops, and iteration planning. The repo ships with a FastAPI orchestrator, a Next.js internal UI, reproducible configs, and deterministic mock data so teams can practice end-to-end before connecting to a real workflow.

## What Ships

- **Partner selection + playbook** — weighted scorecard, YAML-driven playbook generator, and lightweight legal templates.
- **Workflow instrumentation** — mock + HTTP adapters, value-stream mapper, cycle/yield/throughput metrics with JSONL, Parquet, and SQLite persistence.
- **Pilot orchestrator API** — FastAPI service with endpoints for scoring, configs, metrics, feedback, surveys, reports, and iteration plans. Includes SSE stream.
- **Experiment analytics** — bootstrap confidence intervals, statistical tests, Difference-in-Differences helper, and Markdown/PNG impact reports.
- **Feedback system** — auto-tagging, deduplication, theme clustering, severity triage, SUS/NPS summary, and interview kit generator stubs.
- **Next.js UI** — Tailwind + shadcn inspired styling with pages for pilot setup, dashboards, feedback inbox, and impact reports.
- **Iteration planner** — ICE-scored backlog export to Markdown + JSON.
- **Reproducible demos** — Typer CLI + Makefile to simulate a pilot with >20% cycle-time improvement and run reports.

## Getting Started

```bash
make venv
make dev
```

The `make dev` target launches the FastAPI orchestrator on port 8000 and the Next.js UI on port 3000. Default credentials are `pilot/changeme` (see `.env.example`).

### Demo Workflow

1. `make demo` — resets the data directory, simulates baseline + pilot weeks with deterministic seed, writes events/metrics, generates the pilot playbook, and prints KPI shifts.
2. `make report` — regenerates the Markdown + chart impact report in `reports/`.
3. `make iteration` — synthesizes an ICE-ranked backlog from the latest metrics + seeded feedback and drops files in `reports/`.
4. `make demo-feedback` — optional helper to seed 10 feedback submissions and view clustered themes.

Re-running commands is deterministic thanks to fixed seeds and YAML configs.

## Testing

```bash
make test
```

Unit tests cover candidate ranking, metric calculations, bootstrap statistics, feedback clustering, and iteration planning outputs.

## Docker

- `docker-compose up --build` brings up the orchestrator and UI with basic auth enforced.
- Individual Dockerfiles live in `docker/orchestrator.Dockerfile` and `docker/ui.Dockerfile`.

## Security + Data Minimization

- `.env.example` toggles PII redaction and adapter selection (mock vs HTTP).
- Feedback ingestion redacts obvious PII markers and hashes text only for deduplication.
- Stored datasets avoid raw PII; only hashed references remain for deduplication.

## Repo Layout

```
pilotkit/
  orchestrator/
    adapters/
    analytics/
    feedback/
    iteration/
    models/
    playbook/
    report/
    storage/
    main.py
    cli.py
  ui/
    app/
    components/
  configs/
  reports/
  tests/
  docker/
  docker-compose.yml
  Makefile
```

Swap the mock adapter for the HTTP adapter when you are ready for a production integration. Metrics, analytics, and UI components automatically adapt to the live data feed.
