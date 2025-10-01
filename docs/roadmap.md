# Autonomous R&D Intelligence Layer - Project Roadmap

## Vision

Transform the fundamental challenges of autonomous physical experimentation into durable competitive advantages, enabling AI-driven discovery at 10x the velocity of traditional manual workflows while maintaining scientific rigor and safety.

## Strategic Moats

Our platform's value derives from five interlocking moats:

### 1. Execution Moat
**Challenge**: Physical experiments demand millisecond-level control, diverse hardware, and fault tolerance.  
**Our Advantage**: Rust-based safety kernel + robust driver layer + intelligent queueing → 99.9% uptime.

### 2. Data Moat  
**Challenge**: Scientific data is noisy, heterogeneous, and requires domain expertise to interpret.  
**Our Advantage**: Physics-aware schemas + uncertainty quantification + provenance tracking → high SNR dataset.

### 3. Trust Moat
**Challenge**: Autonomous systems must be safe, auditable, and compliance-ready.  
**Our Advantage**: Glass-box AI + policy-as-code + audit trails → regulatory approval + researcher confidence.

### 4. Time Moat
**Challenge**: Lab time is expensive; random exploration wastes months.  
**Our Advantage**: Bayesian EIG optimization + parallel scheduling → 10x faster learning.

### 5. Interpretability Moat
**Challenge**: Black-box AI lacks credibility in scientific contexts.  
**Our Advantage**: Ontologies + explainable plans + natural language reasoning → trusted by domain experts.

## Phase Breakdown

### Phase 0: Foundations (Weeks 1-4) ✅
**Goal**: Build infrastructure before intelligence.

**Deliverables**:
1. **Data Contract** (`configs/data_schema.py`)
   - Pydantic models: `Experiment`, `Measurement`, `Sample`, `Result`
   - Fields: metadata, units (Pint), uncertainty, provenance links
   - JSON-LD serialization for interoperability

2. **Safety Kernel V1** (`src/safety/kernel.rs`)
   - Resource limits: temperature, pressure, reagent volumes
   - Dead-man switch: periodic heartbeat or shutdown
   - Policy DSL: YAML rules → Rust enforcement
   - Manual override with audit logging

3. **Experiment OS** (`src/experiment_os/core.py`)
   - Priority-based queue with preemption
   - Driver registry (adapter pattern)
   - Event loop with async dispatch
   - Dummy drivers: Simulated XRD, NMR

4. **Provenance V1** (`src/memory/store.py`)
   - PostgreSQL + TimescaleDB for time-series
   - CRUD operations with foreign key integrity
   - SHA-256 hashing for data integrity
   - Lineage tracking (experiment → measurement → analysis)

**Milestones**:
- ✅ Data schema validated with 3 instrument types
- ✅ Safety kernel passes 20+ red-team tests
- ✅ Queue handles 100 concurrent simulated experiments
- ✅ All data writes include provenance

**Risks**:
- Schema too rigid → Mitigation: Use JSON fields for extensibility
- Rust safety kernel over-engineered → Mitigation: Start simple, iterate based on failures

---

### Phase 1: Intelligence (Weeks 5-8) 🚀
**Goal**: Add AI-driven planning and close the first autonomous loop.

**Deliverables**:
1. **Simulator Integration** (`src/connectors/simulators/`)
   - DFT connector: PySCF for quantum chemistry
   - MD connector: ASE for molecular dynamics
   - Cheminformatics: RDKit for property prediction
   - Validation: Benchmark against known results (e.g., water bandgap)

2. **EIG-Driven Planning** (`src/reasoning/eig.py`)
   - Bayesian optimization with Gaussian process surrogates
   - EIG calculation: `H(θ) - E[H(θ|y)]` using `scipy.stats`
   - Cost model: EIG/hour metric (account for experiment duration)
   - Batch selection: Greedy or submodular optimization

3. **DoE Primitives** (`src/experiment_os/doe.py`)
   - Controls: Positive, negative, solvent-only
   - Replication: Automatic n=3 for key experiments
   - Randomization: Block designs to avoid temporal bias
   - Power analysis: Sample size calculation

4. **First Closed Loop**
   - **Scenario**: Optimize bandgap of binary alloy (A₁₋ₓBₓ)
   - **Flow**: DFT simulation → EIG planner → run top-3 compositions → update GP → repeat
   - **Success metric**: Find optimum in <10 iterations vs. 20 for grid search

**Milestones**:
- ✅ 2 simulators + 2 dummy instruments operational
- ✅ EIG planner beats random by 2x on benchmark problem
- ✅ First closed-loop completes in <1 hour
- ✅ Provenance graph visualized in UI

**Risks**:
- Simulators too slow → Mitigation: Use surrogate models, parallel execution
- EIG calculation intractable → Mitigation: Monte Carlo approximation, simplified posteriors

---

### Phase 2: Mid-Training (Months 3-4) 🎯
**Goal**: Train RL agents on simulators before deploying to hardware.

**Deliverables**:
1. **RL Environment** (`src/reasoning/rl_env.py`)
   - Gym-compatible interface
   - State: Current beliefs (GP mean/variance), budget remaining
   - Action: Select next experiment (discrete or continuous space)
   - Reward: EIG gained per unit cost

2. **Policy Training** (`src/reasoning/rl_agent.py`)
   - Algorithm: PPO or SAC (PyTorch)
   - Curriculum learning: Easy → hard problems
   - Reward shaping: Penalize unsafe actions, reward replication
   - Glass-box logging: Attention maps, value estimates

3. **Simulator Gym**
   - Suite of test problems: Bandgap, reaction yield, catalyst selectivity
   - Ground truth from high-fidelity sims (DFT, MD)
   - Difficulty levels: Smooth vs. rugged landscapes

4. **Safety in RL**
   - Constrained RL: Lagrangian multipliers for safety constraints
   - Shielding: Policy filtered through safety checker before execution

**Milestones**:
- ✅ RL agent matches or beats Bayesian optimization on 5 benchmarks
- ✅ Zero safety violations during training
- ✅ Interpretable policies (decision tree extraction)

**Risks**:
- Sim-to-real gap → Mitigation: Domain randomization, real-data fine-tuning
- RL sample inefficiency → Mitigation: Offline RL, pre-training on historical data

---

### Phase 3: Real-World (Months 5-6) 🔬
**Goal**: Deploy to real hardware with closed-loop autonomy.

**Deliverables**:
1. **Hardware Integration**
   - XRD (X-ray diffraction): Phase identification
   - NMR (Nuclear magnetic resonance): Structure elucidation
   - Synthesis robot: Automated mixing, heating, stirring
   - Rust drivers for each instrument

2. **Safety V2**
   - Redundant sensors (e.g., dual thermocouples)
   - Shadow simulations: Run sim in parallel, flag anomalies
   - Automatic shutoff: If real diverges from sim by >3σ
   - Emergency stop buttons (physical + software)

3. **Closed-Loop Demos**
   - **Demo 1**: Optimize XRD crystallinity of thin film
   - **Demo 2**: Find NMR-validated ligand for metal complex
   - **Demo 3**: End-to-end synthesis → characterization → replanning

4. **Monitoring Dashboard** (Next.js UI)
   - Real-time telemetry: Instrument status, queue depth
   - Experiment provenance viewer: Graph of all steps
   - Manual override panel: Approve/reject proposed experiments

**Milestones**:
- ✅ 100 hardware experiments executed without human intervention
- ✅ Zero safety incidents
- ✅ 5x speedup over manual workflows (measured wall-clock time)

**Risks**:
- Hardware failures → Mitigation: Graceful degradation, automatic retries
- Researcher resistance → Mitigation: Co-pilot mode, extensive training, pilot studies

---

### Phase 4: Scale (Months 7-9) 🌐
**Goal**: Multi-lab deployment and federated learning.

**Deliverables**:
1. **Multi-Tenancy**
   - RBAC: Researchers can only access their projects
   - Resource quotas: CPU, instrument time, storage
   - Cost tracking: Charge-back per experiment

2. **Federated Learning**
   - Train global model across labs without sharing raw data
   - Differential privacy for sensitive experiments
   - Model aggregation (FedAvg or FedProx)

3. **Ontology V2**
   - Materials ontology: Crystal systems, space groups, compositions
   - Reaction ontology: Reactants, products, mechanisms
   - Property ontology: Bandgap, conductivity, toxicity
   - Semantic queries: "Find all perovskites with bandgap 1.5-2.0 eV"

4. **Advanced Planning**
   - Multi-objective: Pareto frontiers (cost vs. performance)
   - Long-horizon: Multi-step synthesis plans (A → B → C → target)
   - Contingent plans: If X fails, try Y

**Milestones**:
- ✅ 3 labs deployed with shared knowledge base
- ✅ Federated model outperforms single-lab baselines
- ✅ 1000+ experiments in provenance database

**Risks**:
- Network latency → Mitigation: Edge computing, async updates
- Data heterogeneity → Mitigation: Transfer learning, domain adaptation

---

### Phase 5: Autopilot (Months 10-12) 🚀
**Goal**: Full autonomy with strategic human oversight.

**Deliverables**:
1. **Strategic Planning**
   - Goal: "Discover photocatalyst for CO2 → methanol"
   - System decomposes into subgoals, experiments, analyses
   - Monte Carlo tree search over research space

2. **Natural Language Interface**
   - "Optimize bandgap of perovskite for solar cells"
   - System: Proposes plan → user approves → executes → reports results

3. **Automated Reporting**
   - Generate publication-quality figures (matplotlib, seaborn)
   - Draft methods section with full experimental details
   - Suggest follow-up experiments

4. **Continuous Learning**
   - Online RL: Update policy as new data arrives
   - Meta-learning: Fast adaptation to new problem classes
   - Causal inference: Identify mechanisms, not just correlations

**Milestones**:
- ✅ 10 autonomous "discovery sprints" (1 week each, no human input)
- ✅ 1 paper co-authored by system (human edits only)
- ✅ System proposes novel hypothesis validated by expert

**Risks**:
- Runaway experimentation → Mitigation: Budget limits, tiered approval
- Lack of novelty → Mitigation: Curiosity bonuses, exploration incentives

---

## Technical Architecture

### Layers (Bottom-Up)

1. **Experiment OS** (Python)
   - Queue, drivers, scheduling
   - Event loop, resource allocation
   - Telemetry collection

2. **Safety Kernel** (Rust)
   - Interlocks, limits, policy enforcement
   - Runs in separate process for isolation
   - Communicates via IPC (shared memory or sockets)

3. **Connectors** (Python)
   - Adapters for simulators (PySCF, RDKit, ASE)
   - Adapters for instruments (vendor APIs or SCPI)
   - Unified interface: `run_experiment(protocol) → results`

4. **Scientific Memory** (PostgreSQL + Python)
   - Provenance graph: Experiments → Measurements → Analyses
   - Vector embeddings for semantic search (sentence-transformers)
   - Time-series tables (TimescaleDB) for telemetry

5. **Reasoning & Planning** (Python + PyTorch)
   - Bayesian optimization (BoTorch)
   - RL agents (PyTorch + Stable-Baselines3)
   - Symbolic planner (PDDL or custom logic)
   - Glass-box logging (JSON decision logs)

6. **Actuation** (Python)
   - Dispatches experiments to hardware/sims
   - Monitors progress, handles failures
   - Generates reports (PDF via LaTeX, JSON)

7. **Governance** (Python + FastAPI + Next.js)
   - RBAC (roles: admin, researcher, reviewer)
   - Approval workflows (experiments require sign-off)
   - Audit logs (who did what when)
   - UI for provenance, dashboards, overrides

### Data Flow

```
[User/Agent] → Governance → Reasoning → Planning → Actuation
                    ↓             ↓           ↓          ↓
                  RBAC       EIG calc    Queue    Instrument
                    ↓             ↓           ↓          ↓
                 Audit   Scientific Memory ← Provenance ← Results
```

### Key Technologies

| Component | Primary | Rationale |
|-----------|---------|-----------|
| Backend | FastAPI (Python 3.12) | Async, type-safe, OpenAPI docs |
| Safety | Rust | Memory safety, real-time performance |
| Frontend | Next.js (TypeScript) | SSR, modern React, strong typing |
| Database | PostgreSQL + TimescaleDB | ACID + time-series optimization |
| AI/ML | PyTorch | Research-friendly, dynamic graphs |
| Science | PySCF, RDKit, ASE | Mature, well-tested simulators |
| Planning | NetworkX | Graph-based reasoning, visualization |
| Compute | Docker + Kubernetes | Scalable, reproducible deployments |

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Hardware failures | High | Medium | Redundancy, automatic retries, graceful degradation |
| Data corruption | Low | High | Immutable storage, cryptographic hashing, backups |
| AI errors | Medium | High | Dry-run mode, confidence thresholds, human approval |
| Performance bottlenecks | Medium | Medium | Profiling, caching, async execution, parallelism |

### Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| False discoveries | Medium | High | Multiple testing correction, replication requirements |
| Systematic bias | Medium | Medium | Calibration, positive/negative controls, randomization |
| Overfitting | High | Medium | Hold-out test sets, cross-validation, regularization |

### Organizational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Lack of trust | High | High | Glass-box design, pilot studies, co-pilot mode |
| Compliance issues | Low | High | Audit trails, RBAC, policy reviews, external audits |
| Technical debt | Medium | Medium | Modular architecture, comprehensive tests, refactoring sprints |

---

## Key Performance Indicators (KPIs)

### Phase 0 (Weeks 1-4)
- [ ] Data schema validated with 3 instrument types
- [ ] Safety kernel passes 20 red-team tests
- [ ] Queue handles 100 concurrent experiments
- [ ] All data writes include provenance

### Phase 1 (Weeks 5-8)
- [ ] EIG planner beats random by 2x on benchmark
- [ ] 2 simulators + 2 instruments integrated
- [ ] First closed-loop cycle completes in <1 hour
- [ ] Provenance graph visualized in UI

### Phase 2-5 (Months 3-12)
- [ ] 10x improvement in learning velocity (EIG/hour)
- [ ] 99.9% system uptime with hardware
- [ ] Zero safety incidents
- [ ] 100% provenance coverage
- [ ] Multi-lab deployment (>3 sites)
- [ ] 1 publication co-authored by system

---

## First 90 Days (Detailed)

### Weeks 1-4: Phase 0 Foundations

**Week 1: Data Contract**
- Days 1-2: Design Pydantic schemas (`Experiment`, `Measurement`, `Sample`)
- Days 3-4: Add unit validation (Pint), uncertainty fields
- Day 5: Write pytest tests, validate with dummy data

**Week 2: Safety Kernel**
- Days 1-2: Set up Rust project, define policy DSL (YAML schema)
- Days 3-4: Implement resource limits, dead-man switch
- Day 5: Integration tests with Python via FFI (ctypes or PyO3)

**Week 3: Experiment OS**
- Days 1-2: Implement priority queue, driver registry
- Days 3-4: Add async event loop, dummy XRD/NMR drivers
- Day 5: End-to-end test: Submit 10 experiments, verify execution order

**Week 4: Provenance**
- Days 1-2: Set up PostgreSQL + TimescaleDB, design schema
- Days 3-4: Implement `store.py` CRUD operations
- Day 5: Add SHA-256 hashing, visualize lineage graph in notebook

### Weeks 5-8: Phase 1 Intelligence

**Week 5: Simulators**
- Days 1-2: PySCF connector for DFT (H2O bandgap benchmark)
- Days 3-4: RDKit connector for molecular properties
- Day 5: Validate outputs, add error handling

**Week 6: EIG Planning**
- Days 1-3: Implement Bayesian optimization with BoTorch
- Days 4-5: EIG calculation using `scipy.stats.entropy`

**Week 7: DoE**
- Days 1-2: Add controls, replicates to experiment protocol
- Days 3-4: Latin hypercube sampling for initial designs
- Day 5: Power analysis function

**Week 8: First Loop**
- Days 1-3: Integrate: DFT → EIG → queue → update GP
- Days 4-5: Run benchmark (binary alloy), measure KPIs

### Weeks 9-12: Stabilization & Demos
- Refactor based on learnings
- Add UI for provenance viewer
- Prepare demo for stakeholders
- Begin Phase 2 planning

---

## Success Criteria

**Short-Term (3 months)**:
- ✅ System runs first 100 experiments without crashes
- ✅ EIG-driven planning demonstrably better than baselines
- ✅ All safety tests pass

**Medium-Term (6 months)**:
- ✅ Real hardware integrated, closed-loop demos successful
- ✅ 5x speedup over manual workflows
- ✅ Stakeholder buy-in for Phase 4 expansion

**Long-Term (12 months)**:
- ✅ 10x learning velocity achieved
- ✅ Multi-lab deployment operational
- ✅ System proposes novel scientific hypothesis

---

## Resources & Team

**Roles Needed**:
1. **Full-Stack Engineer**: Next.js UI, FastAPI backend
2. **AI/ML Engineer**: RL agents, Bayesian optimization
3. **Systems Engineer**: Rust safety kernel, instrument drivers
4. **Domain Scientist**: Validate experiments, interpret results
5. **DevOps Engineer**: Kubernetes, CI/CD, monitoring

**External Dependencies**:
- Hardware vendors (XRD, NMR manufacturers)
- Simulation software licenses (PySCF, commercial DFT codes)
- Cloud compute (AWS/GCP for training RL agents)

---

## Conclusion

This roadmap transforms the challenges of autonomous experimentation into a defensible competitive moat. By systematically addressing execution, data quality, trust, speed, and interpretability, we build a platform that accelerates R&D while maintaining scientific rigor. Each phase de-risks the next, ensuring steady progress toward full autonomy.

**Next Steps**: Execute Phase 0 (Weeks 1-4) to establish the foundation, then iterate rapidly based on real-world feedback.

