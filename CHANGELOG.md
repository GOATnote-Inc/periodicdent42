# Changelog

## [Unreleased]
### Added
- Repository-wide audit tooling (`make graph`, `make audit`) plus generated `docs/ARCHITECTURE_MAP.md`, `docs/AUDIT_REPORT.md`, and machine-readable `docs/audit.json`.
- Frontend component library (`RunPanel`, `ResultPane`, `HowItWorks`, `ErrorBanner`) and hybrid RAG chat demo at `/demos/rag-chat` with FastAPI proxy.
- Developer enablement docs: refreshed README quickstart, new `CONTRIBUTING.md`, demo index, ticket templates, and `.env.example` updates.
- Persistent telemetry store backed by Cloud SQL with Alembic migrations, FastAPI telemetry routes, and a CLI tail helper.
- RAG index caching with on-disk reuse (`RagIndex`) plus TTL/content-hash invalidation hooks.
- Structured LLM router module with Prometheus counters and JSON decision logs.
- Static coverage badge (`coverage.svg`) and CI coverage gate (>=60%).

### Changed
- Root `Makefile` now provides unified lint/test/demo targets.
- FastAPI now routes through the consolidated `services/agents/orchestrator` entry point; labloop/pilotkit marked deprecated.
- README, agents guide, and database setup docs updated with telemetry endpoints, migrations, and configuration knobs.

### Fixed
- Documented duplication hotspots and prioritized remediation tickets for telemetry persistence, MCP agent refactor, and labloop archival decision.
