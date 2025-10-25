# LabLoop Vertical Slice

This repository provides a minimal end-to-end autonomous laboratory loop with a FastAPI orchestrator, scheduler, instrument simulators, provenance logging, and a lightweight Next.js UI. The project is designed to run entirely in simulation by default with hooks for real hardware once safety limits are provided.

## Structure

```
labloop/
  orchestrator/        # FastAPI app, scheduler, adapters
  ui/                  # Next.js provenance dashboard
  examples/            # Plan examples for XRD and transport
  tests/               # Unit tests for scheduler and safety
  docker/              # Container definitions (skeleton)
  Makefile             # Helper workflows
  .env.example         # Feature flags and defaults
```

## Quickstart

```bash
make venv
source .venv/bin/activate
make dev
```

The default configuration uses the simulated instrument backend (`REAL_DEVICE=false`). To start the orchestrator API locally:

```bash
uvicorn labloop.orchestrator.main:app --reload
```

Then, in another shell, submit an example plan via the CLI:

```bash
export LABLOOP_API=http://127.0.0.1:8000
python labloop/cli.py submit-plan labloop/examples/xrd_quick_scan.yaml
```

Use the CLI run loop to execute tasks and monitor the scheduler decisions:

```bash
python labloop/cli.py run-loop <run_id> --max-steps 6
```

Navigate to `labloop/ui`, install dependencies (`npm install`), and run `npm run dev` to open the provenance dashboard.

## Safety

Hardware execution requires setting `REAL_DEVICE=true` and providing a fully populated `safety_limits` section in the plan plus `operator_ack: true`. The orchestrator refuses to operate a real backend unless all limits are known.

## Tests

```bash
pytest labloop/tests -q
```

## Provenance Bundle

A helper script (`labloop/orchestrator/offline_loop.py`) demonstrates simulated runs and prints the monotonic reduction in predictive variance.
