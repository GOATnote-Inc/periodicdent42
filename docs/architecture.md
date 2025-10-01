# System Architecture

## Overview

The Autonomous R&D Intelligence Layer is designed as a multi-layered system that transforms experimental challenges into strategic advantages through five interlocking moats.

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│              Governance Layer (RBAC, Audits, UI)            │
│  - Role-based access control                                │
│  - Approval workflows                                       │
│  - Audit logs (immutable, cryptographically signed)         │
│  - Next.js dashboard for provenance & monitoring            │
├─────────────────────────────────────────────────────────────┤
│       Reasoning & Planning (AI Agents, EIG Optimization)    │
│  - Bayesian experimental design (EIG/hour maximization)     │
│  - RL agents (PPO/SAC) for policy learning                  │
│  - Symbolic planner with domain heuristics                  │
│  - Glass-box decision logging                               │
├─────────────────────────────────────────────────────────────┤
│     Scientific Memory (Provenance, Physics-Aware Queries)   │
│  - PostgreSQL + TimescaleDB for time-series                 │
│  - Vector embeddings for semantic search                    │
│  - Provenance graph (NetworkX) for lineage tracking         │
│  - SHA-256 hashing for data integrity                       │
├─────────────────────────────────────────────────────────────┤
│          Connectors (Simulators ↔ Instruments)              │
│  - Simulator adapters: PySCF (DFT), ASE (MD), RDKit (chem)  │
│  - Instrument drivers: XRD, NMR, synthesis robots           │
│  - Unified Protocol interface                               │
│  - Cost estimation for EIG calculations                     │
├─────────────────────────────────────────────────────────────┤
│   Experiment OS (Queue, Drivers, Scheduling) + Safety       │
│  - Priority queue with resource allocation                  │
│  - Async event loop for concurrent execution                │
│  - Rust safety kernel (interlocks, limits, dead-man switch) │
│  - Automatic retries, timeout enforcement                   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Experiment Submission → Execution → Learning

1. **User/Agent** submits experiment specification
2. **Governance** checks RBAC permissions, logs audit event
3. **Reasoning** evaluates EIG, generates decision log with rationale
4. **Safety Kernel** validates against policies (Rust)
5. **Experiment OS** enqueues with priority, allocates resources
6. **Connectors** dispatch to simulator or instrument
7. **Results** stored in Scientific Memory with provenance hash
8. **Reasoning** updates model (GP/RL) with new data
9. **Governance** notifies user, generates report

```
User Input
    ↓
[Governance] → RBAC Check → Audit Log
    ↓
[Reasoning] → EIG Calculation → Decision Log
    ↓
[Safety] → Policy Check → Interlock Status
    ↓
[Experiment OS] → Queue → Resource Allocation
    ↓
[Connectors] → Simulator/Instrument → Result
    ↓
[Memory] → Store Result → Update Provenance
    ↓
[Reasoning] → Update Model → Generate Plan
    ↓
[Governance] → Report → UI Dashboard
```

## Component Details

### 1. Experiment OS (Python)

**Location**: `src/experiment_os/core.py`

**Key Classes**:
- `ExperimentQueue`: Priority-based queue with FIFO for same priority
- `DriverRegistry`: Manages instrument/simulator drivers
- `ExperimentOS`: Main orchestration system
- `InstrumentDriver` (ABC): Base class for all drivers
- `Resource`: Tracks capacity and availability

**Responsibilities**:
- Accept experiment submissions
- Schedule based on priority and resource availability
- Execute experiments with timeout enforcement
- Handle retries and error recovery
- Emit structured logs for monitoring

**Moat**: EXECUTION - Reliability under chaos, 99.9% uptime

### 2. Safety Kernel (Rust)

**Location**: `src/safety/src/lib.rs`

**Key Components**:
- `SafetyKernel`: Policy enforcement engine
- `SafetyPolicy`: Rule definitions (YAML → Rust)
- `DeadManSwitch`: Automatic shutdown on heartbeat loss
- `ResourceLimit`: Per-instrument constraints

**Responsibilities**:
- Validate experiments against safety policies
- Enforce temperature, pressure, reagent limits
- Provide dead-man switch (5s timeout)
- Expose Python bindings via PyO3

**Moat**: TRUST - Fail-safe by default, memory-safe enforcement

### 3. Data Contract (Pydantic)

**Location**: `configs/data_schema.py`

**Key Models**:
- `Sample`: Material composition
- `Protocol`: Instrument parameters
- `Experiment`: Complete specification
- `Measurement`: Raw data with units & uncertainty
- `Result`: Aggregated measurements + derived properties
- `Decision`: AI decision with rationale

**Responsibilities**:
- Type-safe data validation
- Unit validation (Pint)
- Uncertainty quantification
- Provenance hash computation (SHA-256)

**Moat**: DATA - High-quality, physics-aware schemas

### 4. EIG Optimizer (Python)

**Location**: `src/reasoning/eig_optimizer.py`

**Key Classes**:
- `GaussianProcessSurrogate`: GP model for uncertainty
- `EIGOptimizer`: Bayesian experimental design
- `EIGResult`: Experiment with EIG and cost metrics

**Responsibilities**:
- Calculate Expected Information Gain
- Optimize EIG/cost ratio
- Greedy batch selection for parallel experiments
- Generate glass-box decision logs

**Moat**: TIME - 10x learning velocity via smart planning

### 5. Simulators (Python)

**Location**: `src/connectors/simulators.py`

**Adapters**:
- `DFTSimulator`: PySCF wrapper for quantum chemistry
- `MDSimulator`: ASE wrapper for molecular dynamics
- `ChemInformaticsSimulator`: RDKit for molecular properties

**Responsibilities**:
- Unified interface across diverse simulation tools
- Cost estimation for EIG calculations
- Structured uncertainty in outputs

**Moat**: TIME - Cheap virtual experiments guide expensive real ones

### 6. Provenance & Memory (Future)

**Location**: `src/memory/store.py` (to be implemented)

**Planned Features**:
- PostgreSQL + TimescaleDB for time-series
- Provenance graph using NetworkX
- Vector search for semantic queries (sentence-transformers)
- Immutable storage with cryptographic integrity

**Moat**: DATA - Complete audit trails, physics-aware queries

### 7. Governance (Future)

**Location**: `src/governance/interfaces.py` (to be implemented)

**Planned Features**:
- FastAPI endpoints for UI
- RBAC with role-based permissions
- Approval workflows (for high-risk experiments)
- Audit log viewer

**Moat**: TRUST - Transparent, auditable, compliant

## Technology Stack

| Layer | Primary Technology | Rationale |
|-------|-------------------|-----------|
| Backend | Python 3.12 + FastAPI | Scientific ecosystem, async I/O |
| AI/ML (Primary) | **Gemini 2.5 Pro + Flash** | Latest Google AI models (Oct 2025) |
| AI/ML (Training) | PyTorch, BoTorch | Research-friendly, Bayesian opt |
| Cloud Platform | **Google Cloud (Vertex AI)** | Managed AI deployment, scalability |
| Safety | Rust | Memory safety, real-time performance |
| Frontend | Next.js + TypeScript | Modern React, SSR, strong typing |
| Database | PostgreSQL + TimescaleDB | ACID + time-series optimization |
| Science | PySCF, RDKit, ASE | Mature simulation tools |
| Planning | NetworkX, SymPy | Graph reasoning, symbolic math |
| Compute | **AI Hypercomputer (TPU/GPU)** | Intensive training and simulations |

## Design Principles

### 1. Glass-Box AI
- Every decision logged with rationale
- Explainable plans (NetworkX graphs)
- SHAP values for feature importance
- Natural language summaries

### 2. Safety-First
- Fail-safe defaults (power off, not unknown state)
- Redundant checks (Rust kernel + Python layer)
- Dry-run mode before hardware execution
- Human approval gates for high-risk

### 3. Modularity
- Clear interfaces between layers
- Adapter pattern for drivers
- Dependency injection for testing
- Independent deployment of services

### 4. Observability
- Structured logging (structlog)
- Distributed tracing (trace IDs)
- Metrics (Prometheus format)
- Real-time dashboards (Next.js)

## Deployment Architecture (Google Cloud - October 2025)

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐    ┌──────────────────────────────┐    │
│  │  Next.js   │───▶│  Cloud Run (FastAPI)         │    │
│  │  Frontend  │    │  - Auto-scaling 1-10         │    │
│  └────────────┘    │  - Serverless containers     │    │
│                    └──────────────────────────────┘    │
│                                │                         │
│                    ┌───────────┴────────────┐           │
│                    ↓                        ↓           │
│         ┌──────────────────┐    ┌─────────────────┐    │
│         │  Vertex AI       │    │ Cloud SQL       │    │
│         │  ├─Gemini 2.5 Pro│    │ (PostgreSQL +   │    │
│         │  ├─Gemini 2.5    │    │  TimescaleDB)   │    │
│         │  │  Flash         │    └─────────────────┘    │
│         │  └─Custom Models │                            │
│         └──────────────────┘                            │
│                    │                                     │
│         ┌──────────┴──────────────┐                     │
│         ↓                         ↓                     │
│  ┌─────────────────┐   ┌──────────────────┐            │
│  │ Cloud Storage   │   │ AI Hypercomputer │            │
│  │ (Data Lake)     │   │ (TPU v5e / GPU)  │            │
│  │ - Experiments   │   │ - RL training    │            │
│  │ - Results       │   │ - DFT at scale   │            │
│  └─────────────────┘   └──────────────────┘            │
│                                                          │
└─────────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ↓                         ↓
    ┌───────────────┐        ┌──────────────────┐
    │ Safety Kernel │        │ Lab Instruments  │
    │ (Rust)        │        │ - XRD, NMR       │
    │ - On-prem or  │        │ - Synthesis      │
    │ - GDC         │        │ - Robotics       │
    └───────────────┘        └──────────────────┘
```

### Dual-Model AI Pattern (Gemini 2.5)

```
User Query → FastAPI Endpoint
              │
              ├─────────────────┬──────────────────┐
              ↓ (Parallel)      ↓ (Parallel)       │
         Gemini 2.5 Flash  Gemini 2.5 Pro         │
         - Latency: <2s    - Latency: 10-30s      │
         - Cost: Low       - Accuracy: High        │
              ↓                  ↓                  │
         Quick Preview      Verified Response      │
         (shown instantly)  (replaces when ready)  │
              └─────────────────┴──────────────────┘
                             ↓
                    UI updates in real-time
```

### Production Setup (Current Best Practices)

**Cloud Services (Google Cloud)**:
- **Cloud Run**: Serverless FastAPI with auto-scaling
- **Vertex AI**: Gemini 2.5 Pro/Flash deployment
- **Cloud SQL**: Managed PostgreSQL + TimescaleDB
- **Cloud Storage**: Data lake for experiments
- **Cloud Monitoring**: Metrics, logs, alerts
- **AI Hypercomputer**: TPU/GPU for intensive compute

**On-Premises (Optional)**:
- **Google Distributed Cloud (GDC)**: Gemini models on-premises
- **NVIDIA Blackwell**: Hardware for GDC deployment
- **Air-gapped option**: For high-security environments

**Integration**:
- **Model Context Protocol (MCP)**: Connect Gemini to tools
- **Vertex AI SDK**: Python client for all models
- **Cloud Build**: CI/CD pipelines
- **Secret Manager**: API keys and credentials

## Security

### Authentication
- OAuth2 for user login
- API keys for programmatic access
- JWT tokens for session management

### Authorization
- RBAC with roles: Admin, Researcher, Reviewer, Viewer
- Per-experiment permissions
- Approval workflows for high-risk

### Data Protection
- SHA-256 hashing for integrity
- HMAC signatures for audit logs
- Encryption at rest (database)
- TLS for data in transit

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Queue latency | <50ms | ~20ms |
| Safety check | <10ms | ~5ms |
| EIG calculation | <100ms | ~150ms |
| Experiment execution | 1-10 hours | 0.5s (dummy) |
| Database write | <10ms | N/A |
| UI render | <1s | N/A |

## Future Enhancements

1. **Multi-tenancy**: Separate workspaces for different teams
2. **Federated learning**: Share models across labs without raw data
3. **NLP interface**: "Optimize bandgap of perovskite" (via Gemini 2.5 Pro)
4. **Automated reporting**: Publication-ready figures and text (via Gemini 2.5 Pro)
5. **Causal inference**: Identify mechanisms, not just correlations

---

## Google Cloud Integration (October 2025)

### Key Resources

📘 **[Google Cloud Deployment Guide](google_cloud_deployment.md)**  
Complete guide to deploying on Google Cloud with Gemini 2.5 Pro/Flash, including:
- Dual-model architecture (fast preview + accurate response)
- Vertex AI setup and configuration
- Cloud Run deployment for FastAPI
- Cost optimization strategies
- Security and compliance
- Migration path from local development

📘 **[Gemini Integration Examples](gemini_integration_examples.md)**  
Production-ready code samples for:
- Dual-model EIG planning (Flash + Pro)
- Safety policy validation with AI reasoning
- Literature search and RAG (Retrieval Augmented Generation)
- Natural language experiment submission
- Automated report generation

### Quick Start (Google Cloud)

```bash
# 1. Create GCP project
gcloud projects create periodicdent42

# 2. Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable run.googleapis.com

# 3. Deploy backend
gcloud run deploy ard-backend \
  --source . \
  --region us-central1

# 4. Test Gemini integration
python -c "
from google.cloud import aiplatform
aiplatform.init(project='periodicdent42')
model = aiplatform.GenerativeModel('gemini-2.5-flash')
response = model.generate_content('Hello!')
print(response.text)
"
```

### Current Models (October 2025)

| Model | Context Window | Best For | Cost (per 1M tokens) |
|-------|---------------|----------|---------------------|
| **Gemini 2.5 Flash** | 1M+ tokens | Fast feedback, previews | $0.075 input / $0.30 output |
| **Gemini 2.5 Pro** | 1M+ tokens | Verified analysis, reasoning | $1.25 input / $5.00 output |

**Recommendation**: Use **dual-model pattern** - Flash for instant UI updates, Pro for verified scientific results.

---

For implementation details, see source code and inline comments. Each file includes rationale for design choices tied to strategic moats.

**Related Documentation**:
- [Google Cloud Deployment Guide](google_cloud_deployment.md) - Full GCP setup
- [Gemini Integration Examples](gemini_integration_examples.md) - Code samples
- [QUICKSTART.md](QUICKSTART.md) - Local development setup

