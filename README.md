# Autonomous R&D Intelligence Layer

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-PROPRIETARY-red.svg)](LICENSE)

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

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
cd app
pip install -r requirements.txt
```

### Run the API

```bash
# Define minimum configuration (see Configuration section for full list)
cat > .env <<'ENV'
PROJECT_ID=periodicdent42
LOCATION=us-central1
LOG_LEVEL=INFO
ENV

# Start the FastAPI service with autoreload
make dev

# Test endpoints
curl http://localhost:8080/healthz
curl -N -X POST http://localhost:8080/api/reasoning/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Suggest perovskite experiments", "context": {"domain": "materials"}}'
```

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
| `GCS_BUCKET` | Cloud Storage bucket for experiment artifacts. | `None` |
| `API_KEY` | API key for authenticated access (enable via `ENABLE_AUTH`). | `None` |
| `ALLOWED_ORIGINS` | Comma-separated CORS whitelist. | `""` |
| `ENABLE_METRICS` / `ENABLE_TRACING` | Observability feature flags. | `True` |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute per IP when rate limiting is enabled. | `60` |

Secrets omitted from `.env` are fetched from Google Secret Manager when running within GCP (see `get_secret_from_manager`).

## Testing & Quality

```bash
# API unit & integration tests
cd app
make test

# Coverage report
make test-coverage

# Linting & formatting checks
make lint

# Safety kernel and shared logic tests
cd ..
pytest tests/ -v --tb=short
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

## Support & Licensing

This repository contains proprietary and confidential software. Usage is restricted to authorized collaborators.

- Licensing requests: B@thegoatnote.com (see [LICENSE](LICENSE) and [LICENSING_GUIDE.md](LICENSING_GUIDE.md)).
- Authorized users: see [AUTHORIZED_USERS.md](AUTHORIZED_USERS.md).
- Deployment, compliance, or partnership questions: refer to [docs/contact.md](docs/contact.md).

"Honest iteration over perfect demos"—document limitations, iterate quickly, and accelerate science.
