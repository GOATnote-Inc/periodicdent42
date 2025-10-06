# Autonomous R&D Intelligence Layer

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Coverage](coverage.svg)](coverage.svg)
[![License](https://img.shields.io/badge/license-PROPRIETARY-red.svg)](LICENSE)

## Evidence-Based Claims (Audited 2025-10-06)

![Hermetic Builds](https://img.shields.io/badge/Hermetic_Builds-Config_Ready_(322_lines)-yellow?logo=nixos)
![ML Test Selection](https://img.shields.io/badge/ML_Test_Selection-10.3%25_CI_reduction_(N=100_synthetic)-orange?logo=scikitlearn)
![Chaos Engineering](https://img.shields.io/badge/Chaos_Engineering-93%25_pass_@_10%25_chaos_(N=15)-green?logo=pytest)
![Continuous Profiling](https://img.shields.io/badge/Continuous_Profiling-2134×_speedup_(N=2_validated)-brightgreen?logo=python)

**🔬 Rigorous Evidence**: All claims audited with 95% confidence intervals. See [EVIDENCE.md](./EVIDENCE.md) for full validation data.

**✅ Validated**: Continuous Profiling achieves 2134× speedup (AI: 0.056s vs manual: 120s), exceeding claimed 360× by 6×. See [reports/manual_vs_ai_timing.json](./reports/manual_vs_ai_timing.json).

**⚠️ Honest Findings**: ML achieves 10.3% CI time reduction with synthetic data (not 70% claimed). Real data collection (50+ runs) estimated to yield 40-60% reduction. See [recruiter_brief_periodiclabs.md](./recruiter_brief_periodiclabs.md) for deployment roadmap.

## Table of Contents
- [Hooks Quickstart](#hooks-quickstart)
- [Overview](#overview)
- [Capabilities](#capabilities)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Run the API](#run-the-api)
- [Configuration](#configuration)
- [Testing & Quality](#testing--quality)
- [Deployment](#deployment)
- [Documentation Index](#documentation-index)
- [Support & Licensing](#support--licensing)

## Hooks Quickstart

Cursor 1.7 enforces safety and transparency via repository hooks. The following automations are installed in `.cursor/hooks.json`:
- **beforeShellExecution** → `.cursor/hooks/check_cmd.sh` blocks destructive commands, sandboxes anything outside `.cursor/allowlist.json`, and requires an explicit **ASK** for package installs.
- **beforeReadFile** → `.cursor/hooks/redact.sh` scrubs secrets (API keys, tokens, SSH material) before context is sent to any model and fingerprints every finding.
- **afterFileEdit** → `.cursor/hooks/audit.sh` appends signed JSON lines to `.cursor/audit.log` so we can trace who changed what.
- **stop** → `.cursor/hooks/notify.sh` summarizes edits, blocked commands, and pending approvals at session shutdown.

**Override & debug:**
1. To intentionally bypass a command block, run it with `ASK: ` prefixed so reviewers see the intent.
2. Review `.cursor/allowlist.json` to expand the sandbox allowlist; changes should be code-reviewed.
3. Inspect `.cursor/audit.log` and `.cursor/redaction.log` for troubleshooting. Add `set -x` inside a hook script while debugging, and remove it once resolved.

## Overview

The Autonomous R&D Intelligence Layer is a production-grade platform that accelerates laboratory research workflows for materials science, chemistry, and physics. It combines dual Gemini models, reinforcement learning, and a safety-first execution engine to recommend, validate, and monitor experiments end-to-end.

## Capabilities

- **Dual-model reasoning:** Vertex AI Gemini 2.5 Flash for rapid ideation and Gemini 2.5 Pro for high-confidence verification.
- **Closed-loop experimentation:** Reinforcement learning planners, experiment OS scheduling, and instrument drivers.
- **Safety kernel:** Rust-backed interlocks and policy enforcement with fail-safe behaviour.
- **Scientific memory:** Provenance tracking, domain ontologies, and retrieval-augmented generation for literature context.
- **Cloud-native operations:** Google Cloud Run deployment, Cloud Storage integration, and Secret Manager–backed secrets.
- **Phase 2 hardware integration:** UV-Vis spectrometer driver with safety interlocks, campaign orchestration, and Cloud SQL/Storage logging.

## System Architecture

```
Client (UI / CLI)
   │
   ├── FastAPI service (`app/src/api`) – request routing, SSE streaming
   │      └── Vertex AI services (`app/src/services`) – Gemini models, storage, database
   │
   ├── Reasoning engines (`app/src/reasoning`, `src/reasoning`) – RL agents, RAG, planners
   │
   └── Safety layer (`src/safety`) – Rust safety kernel + Python gateway
            └── Experiment OS & connectors (`src/experiment_os`, `src/connectors`)
```

## Repository Layout

| Path | Description |
| --- | --- |
| `app/` | FastAPI backend with Makefile, Dockerfile, and static UI assets. |
| `src/` | Shared experiment OS, safety gateway, RL planners, and connectors. |
| `configs/` | Data schemas, safety policies, and operational limits. |
| `docs/` | Detailed product briefs, deployment guides, and architecture references. |
| `infra/` | Infrastructure automation and Cloud Run deployment scripts. |
| `scripts/` | Utility scripts for validation, bootstrapping, and CI helpers. |
| `tests/` | Safety gateway and integration tests outside the FastAPI app. |

## Getting Started

### Prerequisites

- Python 3.12+
- Google Cloud SDK (for deployment)
- Docker (for container builds)
- Optional: Rust toolchain (for working on the safety kernel)

### Local Setup

```bash
# Clone and enter the repository
git clone https://github.com/Periodic-Labs/periodicdent42.git
cd periodicdent42

# Create a shared virtualenv and install backend requirements
make setup

# Install app-specific extras (FastAPI service)
cd app && make setup-local
```

### Run the API

```bash
# Copy the environment template and adjust as needed
cp .env.example .env

# Start the FastAPI service on :8000
make run-api

# Smoke test the chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the UV-Vis campaign"}'
```

### 5-minute Hello Demo

```bash
# Terminal 1 – FastAPI backend
make run.api

# Terminal 2 – Next.js demos (http://localhost:3000)
make demo

# Visit the hybrid chat UI
open http://localhost:3000/demos/rag-chat
```

The demo proxies requests through `/api/rag-chat` (Next.js) to the FastAPI `/api/chat` route. If the backend is offline the UI will return a simulated answer and surface an error banner.

Static dashboards located in `app/static/` are served automatically by the FastAPI application.

## Configuration

Environment variables are loaded via `app/src/utils/settings.py` (Pydantic settings). Supply them in `.env` for local work or Secret Manager for production.

| Variable | Description | Default |
| --- | --- | --- |
| `PROJECT_ID` | Google Cloud project identifier. | `periodicdent42` |
| `LOCATION` | Vertex AI region. | `us-central1` |
| `ENVIRONMENT` | Deployment environment label (development/staging/prod). | `development` |
| `GEMINI_FLASH_MODEL` | Fast Gemini model for preliminary reasoning. | `gemini-2.5-flash` |
| `GEMINI_PRO_MODEL` | Accurate Gemini model for verification. | `gemini-2.5-pro` |
| `GCP_SQL_INSTANCE` | Cloud SQL instance (`project:region:instance`). | `None` |
| `DB_USER` / `DB_PASSWORD` / `DB_NAME` | Database credentials and name. | `ard_user` / `None` / `ard_intelligence` |
| `DATABASE_URL` | Complete SQLAlchemy URL (overrides individual DB_* vars). | `sqlite:///./telemetry.db` |
| `GCS_BUCKET` | Cloud Storage bucket for experiment artifacts. | `None` |
| `UV_VIS_DATASET_PATH` | Optional override for the UV-Vis reference dataset. | `configs/uv_vis_reference_library.json` |
| `LOCAL_STORAGE_PATH` | Filesystem fallback when Cloud Storage is unavailable. | `data/local_storage` |
| `API_KEY` | API key for authenticated access (enable via `ENABLE_AUTH`). | `None` |
| `ALLOWED_ORIGINS` | Comma-separated CORS whitelist. | `""` |
| `ENABLE_METRICS` / `ENABLE_TRACING` | Observability feature flags. | `True` |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute per IP when rate limiting is enabled. | `60` |
| `ROUTER_LATENCY_BUDGET_MS` | Override router latency threshold. | `800` |
| `ROUTER_MAX_CONTEXT_TOKENS` | Override router context token ceiling. | `280` |
| `ROUTER_UNCERTAINTY_THRESHOLD` | Override router uncertainty cutoff. | `0.55` |
| `RAG_CACHE_DIR` | Filesystem directory for persisted embeddings. | `.cache/rag` |
| `RAG_CACHE_TTL_SECONDS` | Time-to-live for cached embeddings. | `86400` |

Secrets omitted from `.env` are fetched from Google Secret Manager when running within GCP (see `get_secret_from_manager`).

## Testing & Quality

```bash
# API unit & integration tests
cd app
make test

# Coverage report
make test-coverage

# Telemetry CLI helper
python -m tools.telemetry tail --last 10

# Linting & formatting checks (FastAPI + repo root)
cd ..
make lint

# Safety kernel and shared logic tests
pytest tests/ -v --tb=short

# Static repo audit & dependency graph
make audit
make graph

# Telemetry migrations (Cloud SQL / SQLite)
make db-upgrade

# Phase 2 UV-Vis campaign simulation
make campaign
```

## Deployment

Cloud Run deployment is managed from the `app/` directory.

```bash
# Build container locally
cd app
make build

# Submit build to Cloud Build
make gcloud-build

# Deploy to Cloud Run (uses infra/scripts/deploy_cloudrun.sh)
make deploy

# Tail logs after deployment
make logs
```

Prior to deployment ensure required Google Cloud APIs are enabled and IAM roles provisioned using scripts in `infra/scripts/`.

## Documentation Index

Key documentation lives under `docs/`:

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) – Executive overview and roadmap.
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) – Detailed deployment procedure.
- [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md) – Current validation status and metrics.
- [docs/architecture.md](docs/architecture.md) – Deep dive into system design and data flows.
- [docs/QUICKSTART.md](docs/QUICKSTART.md) – Expanded setup instructions.
- [docs/telemetry-and-router-runbook.md](docs/telemetry-and-router-runbook.md) – Telemetry persistence, router tuning, and RAG cache ops.

## Support & Licensing

This repository contains proprietary and confidential software. Usage is restricted to authorized collaborators.

- Licensing requests: B@thegoatnote.com (see [LICENSE](LICENSE) and [LICENSING_GUIDE.md](LICENSING_GUIDE.md)).
- Authorized users: see [AUTHORIZED_USERS.md](AUTHORIZED_USERS.md).
- Deployment, compliance, or partnership questions: refer to [docs/contact.md](docs/contact.md).

"Honest iteration over perfect demos"—document limitations, iterate quickly, and accelerate science.

## Rust Core Workspace

A production-grade Rust workspace now lives in [`rust/`](rust/), providing deterministic planning logic, network services, Python bindings, and a WASM demo. Highlights:

- `core`: pure domain models and a deterministic planner with tracing spans and property-based tests.
- `service`: Axum + Tonic application exposing HTTP (OpenAPI docs at `/docs`) and gRPC APIs with SQLx-ready repositories and full telemetry wiring.
- `pycore`: PyO3 bindings packaged with maturin for Python interoperability.
- `wasm-demo`: browser planner demo sharing the same core logic compiled to WebAssembly.

### Prerequisites (macOS/Linux)

1. Install Rust 1.80+ via `rustup` and add the `wasm32-unknown-unknown` target.
2. Install supporting tooling: `cargo-watch`, `cargo-audit`, `cargo-deny`, `maturin`, and `wasm-pack`.
3. Install `protoc` for protobuf codegen.
4. (Optional) Install Docker for container builds.

Run `just setup` to automate most of the toolchain installation.

### Local Development

```bash
# launch both HTTP and gRPC servers with telemetry
just run

# run unit + property + integration tests
just test

# build the Python wheel
just pywheel

# build the WASM demo into rust/wasm-demo/pkg
just wasm
```

The service reads configuration from `configs/service.yaml` or environment variables prefixed with `SERVICE__`. Health and readiness probes are exposed at `/healthz` and `/readyz`; Prometheus metrics at `/metrics`.

### Example Calls

#### HTTP

```bash
curl -s http://localhost:8080/v1/plan \
  -H 'Content-Type: application/json' \
  -d '{"objective":{"description":"Quick screen","metrics":[{"name":"yield","target":0.9}]}}'
```

Response excerpt:

```json
{
  "id": "...",
  "objective": {"description": "Quick screen", "metrics": [{"name": "yield", "target": 0.9}]},
  "rationale": [{"option": "Optimize yield", "score": 0.92, "why": "Target 0.900 + jitter 0.020"}]
}
```

Docs are served at http://localhost:8080/docs.

#### gRPC

```bash
grpcurl -plaintext localhost:50051 periodic.ExperimentService.Plan \
  -d '{"objective":{"description":"Quick screen","metrics":[{"name":"yield","target":0.9}]}}'
```

#### Python

```python
from pycore import plan
plan({
    "description": "Lab automation",
    "metrics": [{"name": "yield", "target": 0.9}],
})
```

#### WASM Demo

Serve `rust/wasm-demo/index.html` with any static file server after running `just wasm` and open it in a browser to see live planning results rendered in under 100ms for small objectives.

### Observability & Security

- Structured JSON logs and OpenTelemetry traces are enabled by default; configure `SERVICE__TELEMETRY__OTLP_ENDPOINT` to export traces to OTLP collectors.
- Request IDs and rationale traces are embedded in spans for correlation.
- SQLx repositories provide both in-memory (default) and Postgres feature-gated implementations.

### Design Notes

- **Axum + Tonic**: unified async stack with Tower middleware, matching our tracing story and enabling shared state between HTTP and gRPC.
- **PyO3 + maturin**: produces ABI3-compatible wheels for Python consumers without duplicating planner logic.
- **Feature flags** isolate instrumentation, GPU hooks, WASM optimizations, Python bindings, and Postgres connectivity for minimal builds.

### Mastery Demo Scaffolding

The repository now contains scaffolding for the CODEx "Mastery Demo" build focused on hybrid RAG experimentation. Highlights:

- `apps/api/` exposes a FastAPI entrypoint with a synthetic `/api/chat` route wired into the new service modules.
- `apps/web/` contains a Next.js UI skeleton with `/demo` and `/evals` pages plus reusable components (RagSources, VectorStats, GuardrailChips, EvalRunCard, RouterBadge).
- `services/` introduces modular packages for retrieval, LLM routing, guardrails, agents, telemetry, and evaluations.
- `datasets/synthetic/` holds a deterministic synthetic corpus (60 markdown docs) and evaluation pairs (120 Q/A rows) for offline testing.

Root-level Makefile commands such as `make ingest`, `make run.api`, and `make eval.offline` are stubbed for quick iteration and will be expanded in subsequent vertical slices.
