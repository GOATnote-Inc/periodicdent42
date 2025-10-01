# Synthesis Automation v1

This repository contains a runnable vertical slice of the powder synthesis automation stack.

## Components

- **orchestrator** – FastAPI service providing plan submission, run control, QC, provenance, bundles, SSE streaming, and safety interlocks.
- **ui** – Next.js internal dashboard with run tables, QC report views, and provenance JSON.
- **scripts** – CLI utilities and a demo workflow producing positive and negative outcomes.

## Quick start

```bash
cp .env.example .env
make dev
```

The orchestrator will be available on http://localhost:8080 with basic auth (`admin`/`changeme`). The UI runs on http://localhost:3000.

## Demo

With the orchestrator running (locally or via Docker), run:

```bash
python scripts/demo.py
```

Three runs will execute on the simulator, including intentional failures to showcase negative-result capture.

## Tests

```bash
pytest
```

