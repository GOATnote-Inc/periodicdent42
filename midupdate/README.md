# Mid-Term Model Update and Glass-Box Planner

This package provides a runnable vertical slice for a mid-term model update pipeline,
structured glass-box planner, orchestrator, and UI needed to run internal "north-star"
experimental campaigns in simulation. The goal is to demonstrate end-to-end provenance
from data ingestion through campaign execution with transparent rationales.

## Features
- Synthetic dataset builder with domain filtering and stratified splits.
- Lightweight training pipeline (Hydra configs) that exports model artifacts and metadata.
- FastAPI orchestrator exposing planning and critique endpoints, constraint checks, and
  model registry reporting.
- Gaussian-process-inspired scheduler with expected information gain proxy and runtime
  estimates.
- Campaign runner capable of simulating superconducting, XRD, and synthesis campaigns.
- Event logging to JSONL and Parquet plus provenance bundle export.
- Next.js UI scaffold with Tailwind + shadcn/ui for live monitoring.

Consult the `Makefile` for the main automation entry points and `docs/` in the root of
this repository for broader program context.
