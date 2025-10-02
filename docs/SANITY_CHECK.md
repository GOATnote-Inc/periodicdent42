# Sanity Check — Autonomous R&D Intelligence Layer

## Executive Snapshot
- **Product:** Autonomous R&D Intelligence Layer delivering dual-Gemini scientific co-pilot + autonomous lab execution for materials R&D teams.
- **Stack maturity:** FastAPI service with Vertex AI + RL stack, simulated UV-Vis workflows, Cloud Run deployment scripts, and partial CI; core flows are implemented but integrations remain simulation-first.
- **Users:** Applied scientists and lab ops teams needing guided experiment design, execution logging, and safety enforcement.
- **Success criteria:** Reliable dual-model responses, campaign logging to storage/DB, safe hardware orchestration, and deployable Cloud Run service with green CI gate.
- **Top risks:**
  1. Local storage fallback is broken (`LocalStorageBackend.list_experiments` references non-existent GCS client), so core campaign listing fails outside GCP.
  2. CI/CD health probes hit `/healthz` while the API only exposes `/health`, breaking automated deploy verification.
  3. README instructs `cp .env.example .env` but no template exists, blocking first-run setup and contradicting security docs.
- **Largest opportunity:** Unblock automated deploys by fixing the `/healthz` vs `/health` mismatch and hardening post-deploy smoke checks.

## Architecture
```mermaid
graph TD
    UI[Static UI / CLI] --> API[FastAPI Service]
    API -->|SSE| DualAgent[DualModelAgent]
    DualAgent --> Flash[Vertex AI Gemini Flash]
    DualAgent --> Pro[Vertex AI Gemini Pro]
    API --> DB[(Cloud SQL / SQLAlchemy)]
    API --> Storage[Storage Backend (GCS or Local)]
    API --> Campaign[UV-Vis Campaign Runner]
    Campaign --> Driver[Simulated UV-Vis Driver]
    Campaign --> Safety[Safety Gateway]
    API --> SSE[Server-Sent Events]
    Safety --> RustKernel[Rust Safety Kernel]
    API --> SecretManager[(Secret Manager / Env Vars)]
    CI[GitHub Actions CI/CD] --> API
    CI --> CloudRun[Cloud Run]
    Observability[Logging / Monitoring Scripts] --> CloudRun
```

- **FastAPI gateway** (`app/src/api/main.py`) streams dual Gemini responses, orchestrates campaigns, and exposes storage endpoints.
- **Reasoning layer** (`app/src/reasoning/dual_agent.py`) fans out to Gemini Flash & Pro in parallel for preview + verified replies.
- **Safety & execution** (`src/safety/gateway.py`, `app/src/lab/campaign.py`, `app/src/drivers/uv_vis.py`) simulate instrument control with enforced interlocks and persistence hooks.
- **Persistence** via SQLAlchemy models and Cloud/Local storage (`app/src/services/db.py`, `app/src/services/storage.py`).
- **CI/CD & Infra** (`.github/workflows/cicd.yaml`, `infra/scripts/deploy_cloudrun.sh`) build, deploy, and smoke-test Cloud Run.

## Roadmap
### 2-week plan (Do Now)
| Task | Owner | Est. | Acceptance Criteria |
| --- | --- | --- | --- |
| Fix Cloud Run health probe path and add `/health` smoke test in workflow & deploy script. | Backend | 0.5d | GH Actions pipeline completes deploy stage with 200 response from `/health`. |
| Split storage backends: repair `LocalStorageBackend.list_experiments`/`delete_result`, add unit coverage. | Backend | 1d | Local campaign run lists experiments without AttributeError; new tests pass. |
| Ship `.env.example` aligned with settings defaults and document secret sourcing. | DX | 0.5d | New template copied during `make setup`; README instructions accurate. |
| Add GCP service mocks to reasoning SSE test for deterministic 200 path and ensure agent initialization coverage. | Backend | 1d | Smoke test asserts SSE event ordering using async client. |
| Harden README quickstart + scripts/dev.sh for single-command bootstrap. | DX | 0.5d | Fresh clone to running API with documented command; verified in CI job. |
| Enable `ruff`/`pytest` in root CI (`.github/workflows/ci.yml`) to fail fast on regressions. | Platform | 1d | CI runs lint+tests in <5 min, gating PRs. |
| Instrument FastAPI logging + metrics toggle defaults (Prometheus) for campaign + reasoning endpoints. | Platform | 1d | `/metrics` emits request latency and campaign counters locally. |

### 6-week plan (Next)
| Initiative | Definition of Done |
| --- | --- |
| Real hardware adapter integration pilot | Abstract driver interface aligns with Experiment OS; one physical instrument stub validated end-to-end with safety gateway approvals. |
| Reinforcement learning policy eval pipeline | Offline training scripts produce reproducible metrics, tracked in dashboards with CI gating on regression thresholds. |
| Multi-tenant auth & API keys | Enable API key enforcement by default, rotated via Secret Manager, with docs + integration tests. |
| Observability stack | Structured logging to Cloud Logging, metrics to Cloud Monitoring, tracing toggle documented; runbooks in `/docs`. |
| Data persistence hardening | Database migrations managed via Alembic, 95% coverage on persistence helpers, fallback to local SQLite for dev. |
| Frontend UX iteration | Static dashboards replaced by interactive UI with WebSocket/SSE preview + final states, including error surfacing. |
| Safety kernel production readiness | Rust kernel compiled in CI, YAML policies versioned, Python gateway coverage >90%, manual override protocol documented. |

## Largest Opportunity — Deep Dive
- **Problem:** Deploy pipeline cannot verify service health because scripts probe `/healthz`, while the FastAPI app intentionally exposes `/health`. Production pushes will fail or, worse, falsely signal success without a real check.
- **Evidence:** FastAPI defines `/health` and documents avoiding `/healthz` (`app/src/api/main.py` L171-187). GitHub Actions and deploy script hit `/healthz` (`.github/workflows/cicd.yaml` L95-107, `infra/scripts/deploy_cloudrun.sh` L48-54).
- **Solution:** Standardize on `/health` for post-deploy checks, add explicit non-200 failure handling, and extend smoke test to call `/api/reasoning/query` with short timeout to ensure agent readiness.
- **Effort:** S (≤0.5d). **Impact:** High (restores automated deploy confidence). **Confidence:** 85% (low complexity, code clearly scoped).
- **48-hour thin slice:**
  1. Update workflow/script to call `/health` and parse JSON for `status == "ok"`.
  2. Add short `curl` POST to `/api/reasoning/query` expecting 503/200 for Vertex readiness with logging.
  3. Run workflow via branch push to confirm green deploy summary.
- **Rollback:** Revert workflow/script to previous commit; manual deploy instructions remain.
- **Fallback:** If `/api/reasoning/query` check flaky, gate on `/health` only and log TODO to mock Vertex in Cloud Run smoke test.

## Guardrails & Quality
- **Security:** No secrets checked in; however, `.env.example` missing though referenced. Introduce template and reinforce Secret Manager usage in README. Confirm Secret Manager fallback (`app/src/utils/settings.py`) handles missing secrets gracefully.
- **Testing:** FastAPI health & reasoning smoke tests exist (`app/tests/test_health.py`, `app/tests/test_reasoning_smoke.py`), plus UV-Vis campaign coverage (`app/tests/unit/test_uv_vis_driver.py`) and safety gateway tests (`tests/test_safety_gateway.py`). Need deterministic SSE test and storage regression coverage.
- **CI/CD:** Two workflows exist; only `cicd.yaml` runs real tests/deploy but fails health probe. Root `ci.yml` is stub—replace with lint/test suite to gate PRs.
- **Observability:** Logging configured but Prometheus metrics toggles default on without exporters. Add `/metrics` integration or document disable path.
- **DX:** Rich docs but outdated instructions (`README.md` references `.env.example`). Provide dev bootstrap scripts, ensure `make setup` works sans heavy chemistry libs for quick starts.

## Concrete Changes
- Update `.github/workflows/cicd.yaml` & `infra/scripts/deploy_cloudrun.sh` to target `/health` and assert JSON `status == ok`.
- Refactor `app/src/services/storage.py` to separate cloud vs local backends, fixing duplicate method definitions and adding unit tests.
- Create `.env.example` aligned with `app/src/utils/settings.py`; adjust README quickstart accordingly.
- Enhance CI workflow to run `ruff` + `pytest` with caching, failing on regressions.
- Add regression tests for storage listing and SSE streaming path.
- Introduce `/scripts/dev.sh` & `/scripts/ci.sh` wrappers for consistent developer + CI entry points.

## Appendix
- **Assumptions:** Cloud Run remains target runtime; Vertex AI credentials managed via Secret Manager; RL training stays offline for now.
- **Open questions:** Need clarity on production storage bucket naming, auth rollout timeline, and whether real hardware integration is scheduled this quarter.
- **Risks if ignored:**
  - *30 days:* Broken deploy automation causes manual releases and drift from main.
  - *60 days:* Storage fallback bug blocks local testing, reducing confidence in campaign features.
  - *90 days:* Lack of observability + auth makes production pilots risky, delaying customer onboarding.
