# Development Instructions

## Objective

Build an **Autonomous R&D Intelligence Layer** for materials science, chemistry, and physics research, combining AI-driven experimental design with safety-critical execution.

## Core Moats (Strategic Advantages)

### 1. Execution Moat
**Challenge**: Physical experiments require reliable hardware control, real-time orchestration, and fault tolerance.  
**Solution**: 
- Rust-based safety kernel for instrument control with low-latency interlocks
- Robust driver abstraction layer supporting diverse equipment (XRD, NMR, synthesis reactors)
- Priority-based queue management with preemption and resource allocation
- Automatic retry logic, graceful degradation, and error recovery

### 2. Data Moat
**Challenge**: Scientific data is heterogeneous, noisy, and requires domain-specific validation.  
**Solution**:
- Physics-aware data schemas with unit validation (Pint integration)
- Structured uncertainty quantification (measurement error, systematic bias)
- SNR-aware data quality scoring and outlier detection
- Provenance tracking from raw instrument output to derived insights
- Time-series optimization for fast retrieval (TimescaleDB)

### 3. Trust Moat
**Challenge**: Autonomous systems must be auditable, safe, and compliant with lab regulations.  
**Solution**:
- Glass-box AI with explainable reasoning (every decision logged with rationale)
- Policy-as-code for safety rules (e.g., max temperature, material compatibility)
- Role-based access control (RBAC) for experiment approval workflows
- Audit trails with cryptographic integrity (SHA-256 hashing of events)
- Dry-run simulation before hardware execution
- Dead-man switch and human override mechanisms

### 4. Time Moat
**Challenge**: Lab time is expensive; inefficient experimentation wastes months.  
**Solution**:
- Bayesian experimental design with Expected Information Gain (EIG) per hour
- Active learning to prioritize high-uncertainty regions of parameter space
- Parallel experiment scheduling across multiple instruments
- Real-time adaptive planning (replan based on incoming data)
- Predictive maintenance to minimize downtime

### 5. Interpretability Moat
**Challenge**: Black-box AI systems lack scientific credibility and fail to build researcher trust.  
**Solution**:
- Scientific ontologies (materials, reactions, properties) for semantic reasoning
- Explainable planning graphs (NetworkX) with human-readable rationales
- Visualization of decision boundaries and uncertainty regions
- Natural language summaries of experimental protocols
- Citation of domain knowledge (papers, databases) in reasoning chains

## Development Principles

### Glass-Box Design
- **Every AI decision must be explainable**: Log rationale, confidence scores, and alternatives considered
- **Use symbolic reasoning where possible**: Physics models, domain heuristics, not just neural nets
- **Visualize agent behavior**: Planning graphs, attention maps, reward shaping

### Safety-First Engineering
- **Fail-safe defaults**: Instruments power off on error, not stay in unknown states
- **Redundancy**: Shadow simulations run in parallel to catch anomalies
- **Validation layers**: Schema checks, physics constraints, manual approval gates
- **Graceful degradation**: System stays partially operational even if components fail

### Scientific Rigor
- **Uncertainty everywhere**: Every measurement and prediction includes error bars
- **Replication protocols**: Automated controls, replicates, and statistical power analysis
- **Null hypothesis testing**: Agents test for no-effect before claiming discoveries
- **Data versioning**: Immutable datasets with lineage tracking

### Modularity & Extensibility
- **Plugin architecture**: New instruments, simulators, and planning algorithms drop in easily
- **Clear interfaces**: Connectors define standard APIs for heterogeneous systems
- **Microservices**: Components can be upgraded or replaced independently
- **Open standards**: Use JSON-LD for metadata, OpenAPI for APIs

## Tech Stack Rationale

### Python 3.12 (Backend Core)
- **Why**: Rich scientific ecosystem (NumPy, SciPy, RDKit), rapid prototyping
- **Use for**: Data processing, AI agents, simulator integration, API orchestration
- **Standards**: Type hints (PEP 484), Pydantic for validation, async/await for I/O

### Rust (Safety-Critical Systems)
- **Why**: Memory safety, concurrency without data races, real-time performance
- **Use for**: Instrument drivers, interlocks, resource limits, low-level hardware control
- **Standards**: `#![forbid(unsafe_code)]` where possible, comprehensive error types, integration tests

### Next.js + TypeScript (Frontend)
- **Why**: Modern React framework with SSR, strong typing, excellent UX libraries
- **Use for**: Provenance viewer, experiment dashboards, approval workflows, real-time monitoring
- **Standards**: Component-driven design, Tailwind CSS, React Query for data fetching

### FastAPI (API Layer)
- **Why**: High-performance async Python, automatic OpenAPI docs, WebSocket support
- **Use for**: REST/GraphQL APIs for frontend, experiment submission endpoints, telemetry
- **Standards**: Pydantic models, OAuth2 authentication, rate limiting

### PostgreSQL + TimescaleDB (Data Layer)
- **Why**: ACID compliance, JSON support, time-series optimization, full-text search
- **Use for**: Experiment metadata, provenance graphs, measurement time-series
- **Standards**: Foreign keys for referential integrity, indexes on query hotpaths

### PyTorch (AI/ML)
- **Why**: Research-friendly, dynamic computation graphs, strong ecosystem
- **Use for**: Reinforcement learning agents, uncertainty quantification (ensembles), surrogate models
- **Standards**: Separate training/inference code, ONNX export for deployment

### Scientific Libraries
- **PySCF**: Ab initio quantum chemistry (DFT, MP2)
- **RDKit**: Cheminformatics (SMILES, molecular descriptors)
- **ASE**: Atomistic simulations (MD, geometry optimization)
- **SymPy**: Symbolic math for EIG calculations
- **NetworkX**: Planning graphs, dependency analysis

## Project Structure Guidelines

```
src/
  experiment_os/       # Core orchestration (Python)
    core.py            # Experiment queue, driver registry, event loop
    drivers/           # Instrument drivers (adapter pattern)
    scheduling.py      # Priority queue, resource allocation
  
  safety/              # Safety-critical layer (Rust)
    kernel.rs          # Interlocks, limits enforcement
    policy.rs          # Safety policy DSL parser
    interlock.rs       # Hardware shutdown logic
  
  connectors/          # Simulator & instrument adapters (Python)
    adapters.py        # Base adapter interface
    simulators/        # DFT, MD, FEA connectors
    instruments/       # XRD, NMR, synthesis equipment
  
  memory/              # Scientific memory layer (Python)
    store.py           # Provenance data lake
    query.py           # Physics-aware queries (e.g., "similar structures")
    embeddings.py      # Vector search for experiments
  
  reasoning/           # AI agents & planning (Python + PyTorch)
    agent.py           # RL agent with glass-box logging
    planner.py         # Symbolic planner using domain heuristics
    eig.py             # Bayesian EIG calculations
  
  actuation/           # Execution & monitoring (Python)
    executor.py        # Dispatches experiments to hardware/sims
    monitor.py         # Real-time telemetry collection
    reporting.py       # Generates experiment reports (PDF/JSON)
  
  governance/          # RBAC, audits, UI APIs (Python)
    auth.py            # Role-based access control
    audit.py           # Audit log with cryptographic integrity
    interfaces.py      # FastAPI endpoints for UI

configs/
  data_schema.py       # Pydantic models for experiments, measurements
  safety_policies.yaml # Human-readable safety rules
  instrument_limits.yaml # Per-instrument resource limits

tests/
  unit/                # Fast isolated tests (pytest)
  integration/         # Multi-component tests (pytest)
  safety/              # Red-team adversarial tests
  
docs/
  roadmap.md           # Phases, milestones, KPIs
  architecture.md      # System design deep-dive
  safety.md            # Risk mitigation strategies

.cursor/rules/
  execution_moat.mdc   # Guidelines for drivers & reliability
  data_moat.mdc        # Guidelines for schemas & provenance
  trust_moat.mdc       # Guidelines for safety & audits
  time_moat.mdc        # Guidelines for EIG optimization
  interpretability_moat.mdc # Guidelines for glass-box AI
```

## Development Workflow

### Phase 0 (Weeks 1-4): Foundations
**Goal**: Establish core infrastructure before adding intelligence.

1. **Data Contract** (Week 1)
   - Define Pydantic models in `configs/data_schema.py`
   - Include: `Experiment`, `Measurement`, `Sample`, `Protocol`, `Result`
   - Fields: metadata, timestamps, uncertainty, units, provenance links

2. **Safety V1** (Week 2)
   - Implement Rust safety kernel in `src/safety/kernel.rs`
   - Features: resource limits, dead-man switch, manual override
   - Write policy DSL for safety rules (YAML → Rust structs)

3. **Experiment OS** (Week 3)
   - Build core queue in `src/experiment_os/core.py`
   - Implement dummy drivers (simulated XRD, NMR)
   - Add event loop for dispatching experiments

4. **Provenance V1** (Week 4)
   - Set up PostgreSQL with TimescaleDB extension
   - Implement `src/memory/store.py` with CRUD operations
   - Add SHA-256 hashing for data integrity

### Phase 1 (Weeks 5-8): Intelligence
**Goal**: Add AI-driven planning and simulator integration.

1. **Simulator Integration** (Week 5)
   - Connect PySCF (DFT), RDKit (cheminformatics)
   - Implement adapters in `src/connectors/simulators/`
   - Validate outputs against known benchmarks

2. **EIG-Driven Planning** (Week 6)
   - Implement Bayesian optimization in `src/reasoning/eig.py`
   - Use `scipy.stats` for entropy calculations
   - Prioritize experiments by EIG/hour metric

3. **DoE Primitives** (Week 7)
   - Add controls, replicates, randomization
   - Implement Latin hypercube sampling
   - Power analysis for sample size estimation

4. **First Closed Loop** (Week 8)
   - Run end-to-end: simulator → EIG → experiment → update beliefs
   - Validate on toy problem (e.g., optimize bandgap via DFT)
   - Measure KPI: EIG/hour improvement over random sampling

## Coding Standards

### Python
- **Type hints everywhere**: Use `mypy` in strict mode
- **Pydantic for validation**: All external data passes through schemas
- **Async where I/O-bound**: Use `asyncio` for concurrency
- **Docstrings**: NumPy style with examples
- **Error handling**: Explicit exception types, never bare `except:`
- **Testing**: `pytest` with `pytest-cov` for coverage (aim for 80%+)

### Rust
- **Forbid unsafe**: Use `#![forbid(unsafe_code)]` unless absolutely necessary
- **Error types**: Use `thiserror` for domain errors
- **Logging**: `tracing` crate with structured fields
- **Testing**: Unit tests + integration tests + property-based tests (proptest)

### TypeScript/Next.js
- **Strict mode**: Enable all strict checks in `tsconfig.json`
- **Component isolation**: Each component in own file with tests
- **API types**: Generate from OpenAPI schema (use `openapi-typescript`)
- **State management**: React Query for server state, Context for UI state

## Risk Mitigation

### Technical Risks
1. **Hardware faults**: Shadow simulations, redundant sensors, automatic retries
2. **Data corruption**: Immutable storage, cryptographic hashes, backups
3. **AI errors**: Dry-run mode, confidence thresholds, human approval gates
4. **Performance bottlenecks**: Profiling, caching, async execution

### Scientific Risks
1. **False discoveries**: Multiple testing correction, replication requirements
2. **Systematic bias**: Calibration protocols, positive/negative controls
3. **Overfitting**: Hold-out test sets, cross-validation, regularization

### Organizational Risks
1. **Lack of trust**: Glass-box design, extensive documentation, pilot studies
2. **Compliance issues**: Audit trails, RBAC, safety policy reviews
3. **Technical debt**: Modular architecture, comprehensive tests, refactoring sprints

## Key Performance Indicators (KPIs)

### Phase 0 (Weeks 1-4)
- ✅ Data schema validated with 3 instrument types
- ✅ Safety kernel passes red-team tests
- ✅ Queue handles 100+ concurrent experiments

### Phase 1 (Weeks 5-8)
- ✅ EIG-driven planning beats random by 2x (measured by surrogate accuracy)
- ✅ 2 simulators + 2 instruments integrated
- ✅ First closed-loop cycle < 1 hour

### Phase 2+ (Months 3-12)
- 🎯 10x improvement in learning velocity (EIG/hour)
- 🎯 99.9% system uptime with hardware
- 🎯 Zero safety incidents
- 🎯 100% provenance coverage
- 🎯 Multi-lab deployment (>3 sites)

## Getting Help

### Context Management in Cursor
- **Use @-references**: `@instructions.md`, `@src/reasoning/agent.py`
- **Open only relevant files**: Don't pollute context with unrelated code
- **Leverage .cursor/rules**: Rules automatically inject moat-specific guidance

### Prompting Best Practices
1. **Start with skeletons**: "Generate the structure for the EIG optimizer"
2. **Iterate with specifics**: "Add Bayesian optimization using scipy"
3. **Request explanations**: "Explain why this approach prevents X risk"
4. **Validate against moats**: "Does this preserve glass-box interpretability?"

### Common Pitfalls
- ❌ Coupling components tightly → Use dependency injection, clear interfaces
- ❌ Ignoring edge cases → Write property-based tests, fuzz inputs
- ❌ Black-box AI → Every prediction needs uncertainty + rationale
- ❌ Weak safety → Default to fail-safe, not fail-dangerous
- ❌ Poor docs → Code should be self-explanatory + commented thoroughly

## Next Steps

After completing Phase 0 and 1, focus areas include:
1. **RL mid-training**: Train agents on simulator data before real hardware
2. **Multi-objective optimization**: Pareto frontiers for cost/performance/time
3. **Federated learning**: Share models across labs without raw data
4. **Natural language interface**: "Optimize catalyst for CO2 reduction"
5. **Automated report generation**: Publication-ready figures and text

---

**Remember**: Every line of code should advance one or more moats. If it doesn't improve execution, data quality, trust, speed, or interpretability, question whether it belongs in the system.

