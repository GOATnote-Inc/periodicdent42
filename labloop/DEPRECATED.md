# Deprecated Module

The `labloop` orchestrator is now superseded by the consolidated `services/agents/orchestrator.py` entry point. New features
and integrations should target the services pipeline. Existing experiments can continue to reference `labloop` for now, but the
module will be retired after the telemetry and router rollout is validated.

## Migration Plan

1. Repoint any CLI or automation scripts to use `services.agents.orchestrator.Orchestrator`.
2. Migrate shared helpers into the `core` package. The hashing utilities and telemetry helpers are already available.
3. Validate workflows against the new `/api/chat` endpoint and telemetry APIs.
4. Archive `labloop` after the Q4 release once dashboards confirm zero usage.

Please avoid adding new code paths in this directory. Future updates should happen in `services/`.
