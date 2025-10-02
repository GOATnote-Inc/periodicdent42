# Changelog

## [Unreleased]
### Added
- Repository-wide audit tooling (`make graph`, `make audit`) plus generated `docs/ARCHITECTURE_MAP.md`, `docs/AUDIT_REPORT.md`, and machine-readable `docs/audit.json`.
- Frontend component library (`RunPanel`, `ResultPane`, `HowItWorks`, `ErrorBanner`) and hybrid RAG chat demo at `/demos/rag-chat` with FastAPI proxy.
- Developer enablement docs: refreshed README quickstart, new `CONTRIBUTING.md`, demo index, ticket templates, and `.env.example` updates.

### Changed
- Root `Makefile` now provides unified lint/test/demo targets.

### Fixed
- Documented duplication hotspots and prioritized remediation tickets for telemetry persistence, MCP agent refactor, and labloop archival decision.
