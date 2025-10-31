# Frontier Kernel Engineer Excellence Plan (October 2025)

## North-Star Outcome (12 Months)
Deliver a provably reliable kernel toolchain that sustains >95% theoretical FLOP utilization across 10k+ Blackwell GPUs while enabling rapid physics-driven experimentation with zero P0 safety incidents.

## Strategic Pillars

### 1. Frontier-Scale Execution Readiness
- **Objectives**: Harden attention/MoE kernels for multi-pod deployment using NVIDIA CUDA 13.0.2 features (cooperative launch clusters, asynchronous execution graphs) and CUTLASS 4.3.0 CuTe DSL specializations for Hopper/Blackwell tensor cores.
- **Execution Plan**:
  1. Build synthetic stress harness that sweeps batch/sequence and expert fan-out; run nightly on 1k-GPU sandbox, expanding to 8k pods by Q2.
  2. Instrument kernels with CUDA 13.0.2 PC sampling + Device Graph Trace to locate warp-scheduler and NVLink stalls; feed into CUTLASS autotuning sweeps.
  3. Develop deterministic collective pipelines using CUDA 13.0.2 stream-ordered memory allocator and CUDA Graph updates to eliminate launch jitter.
  4. Codify multi-node playbooks (failure trees, recovery runbooks) before production launch.
- **Deliverables**: Multi-node performance dossier, reproducible harness repo, tuning parameter registry, runbook wiki.
- **Metrics**: <3% perf regression when scaling from 128 to 8192 GPUs; <0.1% job-abort rate attributable to kernel faults.
- **Inversion (How it fails)**: Kernel regressions appear only at scale → Mitigation: maintain escalating-scale continuous integration tiers with synthetic data + recorded real traces; gate merges on 95th percentile latency parity.

### 2. Tooling, Observability, and Safety Guardrails
- **Objectives**: Create safety-first DevOps fabric that catches kernel bugs pre-production and supports 10k-GPU hotfix rollbacks within 30 minutes.
- **Execution Plan**:
  1. Ship CUDA 13.0.2 Compute Sanitizer profiles and memory checking in CI; require zero new leak/regression budget per release.
  2. Integrate Nsight Systems CLI and CUPTI tracing into data lake with anomaly detection dashboards; align with lab on red/amber alert thresholds.
  3. Build “flight recorder” capturing warp states, NVLink saturation, and CUTLASS DSL tile occupancy; autopublish incident reports.
  4. Implement signed kernel artifact pipeline (repro builds, SBOMs, confidential compute attestation) to meet safety/security SLAs.
- **Deliverables**: CI/CD pipelines, observability dashboards, security attestation docs, automated incident response bots.
- **Metrics**: Mean time to detect <5 minutes; mean time to rollback <30 minutes; zero unsigned binaries in fleet.
- **Inversion**: Alert fatigue or blind spots → Mitigation: quarterly chaos drills injecting synthetic warp divergence, verifying alert routing and human on-call load.

### 3. Scientific Integration & Curiosity Engine
- **Objectives**: Translate GPU breakthroughs into lab discoveries by pairing kernel innovation with experimental feedback loops.
- **Execution Plan**:
  1. Embed with two flagship scientific programs (e.g., fusion simulations, materials inverse design) to co-own success metrics.
  2. Build dual-path prototyping: CUTLASS 4.3.0 CuTe DSL notebooks for rapid iteration plus production C++ kernels; share findings through open lab seminars.
  3. Establish cross-disciplinary “science sprints” every six weeks to validate hypotheses with real experimental datasets.
  4. Publish shared documentation bridging physics models to GPU optimizations (precision trade-offs, tensor core usage constraints).
- **Deliverables**: Joint milestone roadmaps, reproducible notebooks, cross-functional postmortems, knowledge base articles.
- **Metrics**: ≥3 co-authored experiment wins per half; scientist satisfaction score >4.5/5.
- **Inversion**: Kernels optimized for synthetic benchmarks only → Mitigation: require experimental validation stage-gates before declaring performance wins.

### 4. Ownership & Leadership under Ambiguity
- **Objectives**: Demonstrate founder-level initiative by driving roadmap, aligning stakeholders, and removing blockers without prompting.
- **Execution Plan**:
  1. Operate an autonomous frontier-readiness squad (kernels, infra, science) with weekly operating reviews.
  2. Maintain open RFC pipeline; each major kernel change accompanied by risk assessment, safety plan, and fallback path.
  3. Mentor junior engineers on CUTLASS DSL patterns, CUDA 13.0.2 best practices, and debugging rituals.
  4. Conduct after-action reviews on every incident or scaling challenge; convert insights into process improvements.
- **Deliverables**: Squad charters, RFC archive, mentorship logs, continuous improvement tracker.
- **Metrics**: Stakeholder NPS >50; 90% of high-severity issues proactively detected by squad.
- **Inversion**: Leadership vacuum due to context overload → Mitigation: adopt rotational delegation model and maintain single source-of-truth dashboards.

## Capability-Building Roadmap
- **First 30 Days**: Deploy CUTLASS 4.3.0 CuTe DSL environment, replicate baseline kernels, audit existing observability gaps, shadow scientific teams.
- **Days 31-90**: Launch scale-testing harness, implement Compute Sanitizer CI gates, deliver first cross-disciplinary sprint outcome.
- **Days 91-180**: Harden multi-node graph execution, ship flight recorder telemetry, publish two knowledge base entries.
- **Days 181-365**: Achieve full frontier-scale deployment, lead chaos drills, co-author research output with scientific partners.

## References
- CUTLASS 4.3.0 README (Oct 2025) describing CuTe DSL, tensor core coverage, and supported data types. 【1495e2†L1-L39】
