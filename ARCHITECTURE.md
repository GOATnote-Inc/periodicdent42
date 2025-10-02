# System Architecture - Autonomous R&D Intelligence Layer

**Last Updated**: October 1, 2025  
**Version**: 1.0  
**Status**: Production (FastAPI), Research (Adaptive Router)

---

## Executive Summary

The Autonomous R&D Intelligence Layer is a **production-grade AI platform** for optimizing physical experiments in materials science, chemistry, and manufacturing. The system uses dual Gemini models (Flash + Pro) and reinforcement learning to accelerate experimental research.

**Core Value Proposition**: Reduce experiment cycle time by 10-100× through intelligent experiment design and automated execution.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   Web UI    │  │    CLI      │  │   Jupyter   │  │    API    │ │
│  │  (Tailwind) │  │  (Typer)    │  │  Notebooks  │  │  Clients  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
└─────────┼─────────────────┼─────────────────┼───────────────┼───────┘
          │                 │                 │               │
┌─────────┼─────────────────┼─────────────────┼───────────────┼───────┐
│         ▼                 ▼                 ▼               ▼       │
│                   FastAPI Application Layer                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Authentication │  Rate Limiting  │  Security Headers         │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │  /api/reasoning/query       - AI reasoning endpoint          │  │
│  │  /api/storage/experiment    - Experiment result storage      │  │
│  │  /api/health                - Health checks                  │  │
│  │  /                          - Static UI serving              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ▲                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                              ▼                                      │
│                     Reasoning & AI Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Dual Agent   │  │ MCP Agent    │  │  Adaptive Router         │ │
│  │ (Gemini)     │  │              │  │  (Experimental)          │ │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────────┤ │
│  │ Flash: Fast  │  │ Model Context│  │ Noise Estimator          │ │
│  │ Pro: Accurate│  │ Protocol     │  │ BO vs RL Routing         │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              Optimization Algorithms                          │ │
│  ├──────────────┬──────────────┬──────────────┬─────────────────┤ │
│  │ PPO+ICM (RL) │ Bayesian Opt │ Hybrid Opt   │ EIG Optimizer   │ │
│  │              │ (GP-UCB)     │              │                 │ │
│  └──────────────┴──────────────┴──────────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                              ▼                                      │
│                   Services & Integration Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Vertex AI    │  │ Cloud Storage│  │  Cloud SQL (PostgreSQL)  │ │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────────┤ │
│  │ Gemini Flash │  │ GCS Buckets  │  │ Experiment Results       │ │
│  │ Gemini Pro   │  │ Artifacts    │  │ Metadata                 │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Secret Mgr   │  │ Monitoring   │  │  Logging                 │ │
│  ├──────────────┤  ├──────────────┤  ├──────────────────────────┤ │
│  │ API Keys     │  │ Cloud Monitor│  │ Structured Logging       │ │
│  │ Credentials  │  │ Metrics      │  │ Audit Trails             │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                              ▼                                      │
│                   Experiment Execution Layer                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Experiment OS (Python)                      │  │
│  ├──────────────┬──────────────┬──────────────┬─────────────────┤  │
│  │ Connectors   │ Queue Mgmt   │ Drivers      │ Simulators      │  │
│  └──────────────┴──────────────┴──────────────┴─────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Safety Gateway (Rust)                            │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ Hardware interlocks │ Resource limits │ Emergency stop       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Hardware Drivers                           │  │
│  ├──────────────┬──────────────┬──────────────┬─────────────────┤  │
│  │ XRD          │ NMR          │ UV-Vis       │ Custom          │  │
│  └──────────────┴──────────────┴──────────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

### Production Application (`app/`)

```
app/
├── src/                        # FastAPI application
│   ├── api/                    # API endpoints
│   │   ├── main.py            # FastAPI app, routes, middleware
│   │   └── security.py        # Auth, rate limiting, security headers
│   │
│   ├── reasoning/             # AI reasoning modules
│   │   ├── dual_agent.py      # Dual Gemini model orchestration
│   │   └── mcp_agent.py       # Model Context Protocol agent
│   │
│   ├── services/              # Google Cloud integrations
│   │   ├── vertex.py          # Vertex AI (Gemini)
│   │   ├── storage.py         # Cloud Storage (GCS)
│   │   └── db.py              # Cloud SQL (PostgreSQL)
│   │
│   ├── monitoring/            # Observability
│   │   └── metrics.py         # Cloud Monitoring metrics
│   │
│   └── utils/                 # Utilities
│       ├── settings.py        # Configuration management
│       └── sse.py             # Server-Sent Events (streaming)
│
├── static/                    # Frontend (HTML/CSS/JS)
│   ├── index.html             # Main UI
│   ├── benchmark.html         # Validation results viewer
│   ├── breakthrough.html      # Research findings
│   └── rl-training.html       # RL training dashboard
│
├── tests/                     # Test suite
│   ├── test_health.py         # Health endpoint tests
│   ├── test_reasoning_smoke.py # Reasoning smoke tests
│   ├── test_security.py       # Security middleware tests
│   └── unit/                  # Unit tests
│       ├── test_core.py.old   # Legacy core tests
│       └── test_adaptive_router.py # Adaptive router tests
│
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image
└── README.md                  # App-specific documentation
```

### Research Modules (`src/`)

```
src/
├── reasoning/                 # Optimization algorithms (research)
│   ├── ppo_agent.py          # PPO with ICM (RL)
│   ├── rl_agent.py           # Base RL agent
│   ├── rl_env.py             # RL environment
│   ├── eig_optimizer.py      # Expected Information Gain
│   ├── hybrid_optimizer.py   # BO + RL hybrid
│   ├── agentic_optimizer.py  # Agent-based optimization
│   ├── curiosity_module.py   # Intrinsic Curiosity Module
│   ├── rag_system.py         # RAG for scientific knowledge
│   │
│   └── adaptive/             # Adaptive routing (experimental)
│       ├── __init__.py       # Experimental warnings
│       ├── noise_estimator.py # Noise estimation from pilots
│       └── router.py         # BO vs RL routing logic
│
├── experiment_os/            # Experiment execution
│   ├── core.py               # Experiment OS core
│   └── drivers/              # Hardware drivers
│       ├── xrd_driver.py     # X-Ray Diffraction
│       ├── nmr_driver.py     # Nuclear Magnetic Resonance
│       └── uvvis_driver.py   # UV-Visible Spectroscopy
│
├── connectors/               # Simulators and connectors
│   └── simulators.py         # Simulation connectors
│
└── safety/                   # Safety-critical systems
    ├── gateway.py            # Python interface
    ├── Cargo.toml            # Rust project config
    └── src/
        └── lib.rs            # Rust safety kernel
```

### Infrastructure (`infra/`)

```
infra/
├── scripts/
│   ├── enable_apis.sh        # Enable required Google Cloud APIs
│   ├── setup_iam.sh          # Configure IAM roles
│   ├── create_secrets.sh     # Create secrets in Secret Manager
│   └── deploy_cloudrun.sh    # Deploy to Cloud Run
│
└── monitoring/
    ├── dashboard.json        # Cloud Monitoring dashboard
    └── setup_dashboard.sh    # Dashboard setup script
```

### Configuration (`configs/`)

```
configs/
├── data_schema.py            # Physics-aware data schemas
└── safety_policies.yaml      # Safety policy definitions
```

### Scripts (`scripts/`)

```
scripts/
├── bootstrap.py              # Project initialization
├── train_ppo.py              # Train PPO agent
├── train_ppo_expert.py       # Train expert PPO agent
├── train_rl_agent.py         # Train RL agents
├── validate_rl_system.py     # RL system validation
├── validate_stochastic.py    # Stochastic validation
├── get_secrets.sh            # Retrieve secrets from Secret Manager
├── setup_local_dev.sh        # Local development setup
├── init_secrets_and_env.sh   # Initialize secrets and environment
├── rotate_api_key.sh         # Rotate API keys
└── check_security.sh         # Security integrity checks
```

### Documentation (`docs/`)

```
docs/
├── architecture.md           # Original architecture doc
├── instructions.md           # Development instructions
├── roadmap.md                # Product roadmap
├── QUICKSTART.md             # Quick start guide
├── README_CLOUD.md           # Cloud deployment guide
├── google_cloud_deployment.md # Detailed GCP deployment
├── gemini_integration_examples.md # Gemini examples
├── safety_gateway.md         # Safety system documentation
├── contact.md                # Contact information
└── SECURITY.md               # Security documentation
```

---

## Component Details

### 1. FastAPI Application Layer

**Purpose**: Production HTTP API for AI reasoning and experiment orchestration

**Key Files**:
- `app/src/api/main.py` - Main application, routes, CORS, middleware
- `app/src/api/security.py` - Authentication, rate limiting, security headers

**Security Features**:
- API key authentication (Secret Manager)
- IP-based rate limiting (sliding window)
- CORS with strict origin validation
- Security headers (HSTS, X-Content-Type-Options, etc.)
- Error sanitization (no stack traces exposed)

**Endpoints**:
- `POST /api/reasoning/query` - AI reasoning with streaming (SSE)
- `POST /api/storage/experiment` - Store experiment results
- `GET /api/storage/experiments` - List experiments
- `GET /health` - Health check (unauthenticated for Cloud Run)
- `GET /` - Serve static UI

**Deployment**:
- **Platform**: Google Cloud Run (serverless containers)
- **Scaling**: 1-10 instances, 2 GB RAM, 2 vCPU per instance
- **Timeout**: 300s
- **Authentication**: IAM + API key

### 2. AI Reasoning Layer

#### Dual-Model Pattern
**File**: `app/src/reasoning/dual_agent.py`

**Strategy**: Use two Gemini models for cost/quality optimization
- **Gemini 2.5 Flash**: Fast, cheap (routine queries)
- **Gemini 2.5 Pro**: Slow, accurate (complex reasoning)

**Routing Logic**:
1. Simple queries → Flash
2. Complex queries → Pro
3. Confidence threshold: If Flash < 0.8 confidence → escalate to Pro

#### Optimization Algorithms

**Bayesian Optimization** (Standard)
- **Implementation**: GPyOpt or BoTorch
- **Use case**: Clean data, known noise model
- **Strengths**: Sample-efficient, uncertainty quantification
- **Limitations**: Struggles with high noise (σ > 1.5)

**Reinforcement Learning** (PPO + ICM)
- **Files**: `src/reasoning/ppo_agent.py`, `src/reasoning/rl_agent.py`
- **Use case**: High-noise environments, exploration-heavy problems
- **Strengths**: Noise-robust, adaptive exploration
- **Limitations**: Requires training, sample-inefficient initially
- **Status**: Validated on Branin function (preliminary)

**Adaptive Router** (Experimental)
- **Files**: `src/reasoning/adaptive/router.py`, `noise_estimator.py`
- **Purpose**: Automatically select BO vs RL based on estimated noise
- **Status**: Research prototype, needs validation (see Phase 1 plan)
- **Test Coverage**: 96% router, 74% noise estimator, 21 tests passing

#### Expected Information Gain (EIG)
**File**: `src/reasoning/eig_optimizer.py`

**Purpose**: Optimize for learning velocity (information gain per experiment)

**Method**:
1. Estimate information gain for each candidate experiment
2. Select experiment that maximizes EIG
3. Balances exploration vs exploitation

**Use case**: When experiments are very expensive and you want to learn maximally from each one

### 3. Google Cloud Services Integration

#### Vertex AI
**File**: `app/src/services/vertex.py`

**Purpose**: Interface to Gemini models

**Features**:
- Model selection (Flash vs Pro)
- Streaming responses (SSE)
- Error handling and retries
- Token usage tracking

#### Cloud Storage (GCS)
**File**: `app/src/services/storage.py`

**Purpose**: Store experiment results, artifacts, validation data

**Buckets**:
- `periodicdent42-experiments` - Experiment results
- `periodicdent42-validation` - Validation results
- `periodicdent42-artifacts` - Model artifacts, plots

#### Cloud SQL (PostgreSQL)
**File**: `app/src/services/db.py`

**Purpose**: Structured data storage (metadata, provenance)

**Schema** (planned):
- `experiments` - Experiment metadata
- `results` - Experimental results
- `runs` - Optimization run tracking

#### Secret Manager
**Integration**: `app/src/utils/settings.py`

**Secrets Stored**:
- `api-key` - API authentication key
- Database credentials (if using Cloud SQL)

**Access Pattern**:
1. Application pulls secrets at startup
2. Secrets cached in memory
3. Service account has `secretAccessor` role

### 4. Experiment Execution Layer

#### Experiment OS
**File**: `src/experiment_os/core.py`

**Purpose**: Queue management, experiment scheduling, driver coordination

**Features**:
- Priority queue for experiments
- Experiment state machine
- Rollback and retry logic
- Provenance tracking

#### Hardware Drivers
**Directory**: `src/experiment_os/drivers/`

**Drivers**:
1. **XRD** (X-Ray Diffraction) - `xrd_driver.py`
   - Crystal structure analysis
   - Lattice parameter measurement

2. **NMR** (Nuclear Magnetic Resonance) - `nmr_driver.py`
   - Molecular structure determination
   - Chemical composition analysis

3. **UV-Vis** (UV-Visible Spectroscopy) - `uvvis_driver.py`
   - Absorbance spectra
   - Band gap measurement

**Common Interface**:
```python
class Driver:
    def connect() -> None
    def configure(params: Dict) -> None
    def run_measurement(sample: Sample) -> Result
    def disconnect() -> None
```

#### Safety Gateway (Rust)
**Files**: `src/safety/src/lib.rs`, `src/safety/gateway.py`

**Purpose**: Hardware interlocks, resource limits, emergency stop

**Why Rust**: Safety-critical code benefits from Rust's memory safety guarantees

**Features**:
- Parameter validation before hardware execution
- Timeout enforcement
- Emergency stop capability
- Audit logging (immutable, tamper-evident)

**Python Interface**:
```python
from src.safety.gateway import SafetyGateway

gateway = SafetyGateway()
if gateway.validate_action(action):
    execute_hardware_action(action)
else:
    log_safety_violation(action)
```

### 5. Frontend (Static UI)

**Technology**: HTML5, Tailwind CSS, vanilla JavaScript

**Why Static**: 
- No build step required
- Fast initial load
- Served directly by FastAPI
- Easy deployment

**Pages**:
- `index.html` - Main dashboard, AI query interface
- `benchmark.html` - Validation results viewer
- `breakthrough.html` - Research findings display
- `rl-training.html` - RL training metrics and curves

**Features**:
- Server-Sent Events (SSE) for streaming AI responses
- Responsive design (mobile-friendly)
- Dark mode support (planned)

---

## Data Flow

### AI Reasoning Request Flow

```
User (Web UI or API) 
  ↓
  POST /api/reasoning/query {"query": "Design next experiment"}
  ↓
FastAPI Main (security middleware)
  ↓ Authenticate (API key)
  ↓ Rate limit check
  ↓ Security headers
  ↓
Dual Agent (reasoning/dual_agent.py)
  ↓
  ├─→ Gemini Flash (fast path)
  │     ↓ confidence < 0.8?
  │     ↓
  └─→ Gemini Pro (accurate path)
        ↓
  Response (streamed via SSE)
  ↓
FastAPI Main (format SSE events)
  ↓
User (real-time updates)
```

### Experiment Execution Flow

```
AI Agent decides next experiment
  ↓
Experiment OS (queue experiment)
  ↓
Safety Gateway (validate parameters)
  ↓ ✓ pass safety checks
  ↓
Hardware Driver (execute measurement)
  ↓ ✓ measurement complete
  ↓
Result Processing
  ↓
  ├─→ Cloud Storage (raw data)
  ├─→ Cloud SQL (metadata)
  └─→ AI Agent (feedback for next decision)
```

### Adaptive Routing Flow (Experimental)

```
User provides pilot data (3-5 experiments)
  ↓
Noise Estimator (estimate σ from pilot)
  ↓ σ = 0.8 (medium noise)
  ↓
Adaptive Router (apply routing logic)
  ↓
  ├─→ σ < 0.5? → Bayesian Optimization
  ├─→ 0.5 ≤ σ < 1.5? → BO (conservative)
  └─→ σ ≥ 1.5? → RL (PPO+ICM)
        ↓
  Decision (method, confidence, reasoning)
  ↓
Execute optimization with selected method
```

---

## Deployment Architecture

### Development Environment

```
Local Machine
├── Docker Compose (optional)
│   ├── PostgreSQL container
│   └── Redis container (future)
│
├── Python 3.12 venv
│   ├── FastAPI (uvicorn --reload)
│   └── All dependencies
│
└── Environment Variables
    ├── .env file (local secrets)
    └── PROJECT_ID=periodicdent42
```

**Local Development Setup**:
```bash
# One command setup
bash scripts/init_secrets_and_env.sh

# Manual setup
cd app && python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=".:${PYTHONPATH}"
uvicorn src.api.main:app --reload --port 8080
```

### Production Environment (Google Cloud)

```
Internet
  ↓ HTTPS
Cloud Load Balancer
  ↓
Cloud Run (FastAPI container)
  ├── Auto-scaling: 1-10 instances
  ├── 2 GB RAM, 2 vCPU per instance
  ├── Timeout: 300s
  └── Service Account: periodicdent42@*.iam.gserviceaccount.com
      ↓
  ┌───┴────────────────────────────────────┐
  ↓                   ↓                    ↓
Vertex AI        Cloud Storage      Secret Manager
(Gemini)         (GCS Buckets)      (API Keys)
```

**Deployment Process**:
```bash
# Automated deployment
cd infra/scripts
bash enable_apis.sh          # Enable GCP APIs
bash setup_iam.sh            # Configure IAM
bash create_secrets.sh       # Create secrets
bash deploy_cloudrun.sh      # Deploy app
```

**Configuration**:
- **Region**: us-central1
- **Authentication**: IAM + API key (X-API-Key header)
- **Monitoring**: Cloud Monitoring + Logs Explorer
- **Secrets**: Secret Manager (automatic rotation supported)

---

## Security Architecture

### Defense in Depth

```
Layer 1: Network (Cloud Run, HTTPS only)
  ↓
Layer 2: Application (CORS, rate limiting)
  ↓
Layer 3: Authentication (API key from Secret Manager)
  ↓
Layer 4: Authorization (exempt paths, role-based future)
  ↓
Layer 5: Data (error sanitization, no stack traces)
  ↓
Layer 6: Audit (structured logging, Cloud Logging)
```

### Security Features

1. **Authentication**
   - API key (32-byte random, SHA-256 hashed)
   - Stored in Google Secret Manager
   - Rotated quarterly (or on-demand)

2. **Rate Limiting**
   - IP-based sliding window
   - Default: 60 requests/minute per IP
   - Configurable via `RATE_LIMIT_PER_MINUTE`

3. **CORS**
   - Strict origin validation
   - No wildcards (`allow_origins="*"` disabled)
   - Credentials disabled for safety

4. **Security Headers**
   - HSTS: `max-age=31536000`
   - X-Content-Type-Options: `nosniff`
   - X-Frame-Options: `DENY`
   - Referrer-Policy: `strict-origin-when-cross-origin`
   - Permissions-Policy: restrictive defaults

5. **Error Handling**
   - Generic error messages to users
   - Detailed logs server-side only
   - Structured error codes (no stack traces)

6. **Secret Management**
   - No secrets in code or environment variables (except local dev)
   - All secrets in Google Secret Manager
   - Service account with minimal permissions
   - Pre-commit hook prevents secret commits

### Security Monitoring

- **Logs**: Cloud Logging (structured JSON)
- **Metrics**: Cloud Monitoring (request rates, errors, latencies)
- **Alerts**: On rate limit violations, auth failures, 5xx errors
- **Audit**: All API requests logged with IP, user, timestamp

---

## Testing Strategy

### Test Pyramid

```
        ╱╲
       ╱  ╲        E2E Tests (manual, smoke tests)
      ╱────╲       
     ╱      ╲      Integration Tests (API endpoints, database)
    ╱────────╲     
   ╱          ╲    Unit Tests (algorithms, logic, utilities)
  ╱────────────╲   
 ╱______________╲  
```

### Test Suite

**Unit Tests** (96% target coverage):
- `app/tests/unit/test_adaptive_router.py` - Adaptive router (21 tests)
- Noise estimator methods
- Routing logic
- Edge cases

**Integration Tests**:
- `app/tests/test_health.py` - Health endpoint
- `app/tests/test_reasoning_smoke.py` - Reasoning smoke tests
- `app/tests/test_security.py` - Security middleware

**Validation Tests**:
- `scripts/validate_rl_system.py` - RL system validation
- `scripts/validate_stochastic.py` - Stochastic performance validation

**Run Tests**:
```bash
cd app
export PYTHONPATH=".:${PYTHONPATH}"
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Current Coverage
- `app/src/api/main.py`: ~70%
- `src/reasoning/adaptive/router.py`: 96%
- `src/reasoning/adaptive/noise_estimator.py`: 74%
- Overall: ~50% (target: >80% for critical paths)

---

## Performance Characteristics

### Latency

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| `/api/reasoning/query` (Flash) | 1.5s | 3s | 5s |
| `/api/reasoning/query` (Pro) | 5s | 10s | 15s |
| `/api/storage/experiment` | 100ms | 200ms | 500ms |
| `/health` | 50ms | 100ms | 200ms |

### Throughput

- **Cloud Run**: 1-10 instances (auto-scaling)
- **Max concurrent**: 80 requests/instance
- **Theoretical max**: 800 concurrent requests
- **Typical load**: 10-50 req/min

### Cost Optimization

**Gemini Model Costs** (approximate):
- Flash: $0.10 per 1M tokens
- Pro: $1.25 per 1M tokens
- **Strategy**: Route simple queries to Flash (10× cheaper)

**Cloud Run Costs**:
- $0.00002400 per vCPU-second
- $0.00000250 per GiB-second
- Idle instances scaled to zero (cost = $0)

**Estimated Monthly Cost** (1000 queries/day):
- Gemini: ~$50
- Cloud Run: ~$20
- Storage: ~$10
- **Total**: ~$80/month

---

## Development Workflow

### Git Workflow

```
main (protected)
  ↓
feat-* (feature branches)
  ↓ PR + review
  ↓ CI tests pass
  ↓ merge
main
```

**Branching Strategy**:
- `main` - Production-ready code
- `feat-*` - Feature branches (e.g., `feat-api-security-d53b7`)
- `fix-*` - Bug fixes
- `docs-*` - Documentation updates

**Pre-commit Hook**:
- Security checks (no secrets in code)
- Linter (ruff)
- Formatter (black)
- Tests (optional, can be run in CI)

### CI/CD Pipeline (Planned)

```
Push to GitHub
  ↓
GitHub Actions
  ├─→ Run tests (pytest)
  ├─→ Lint (ruff)
  ├─→ Security scan (TruffleHog)
  ├─→ Build Docker image
  └─→ Deploy to Cloud Run (on main only)
```

**Current Status**: Manual deployment via `infra/scripts/deploy_cloudrun.sh`

---

## Monitoring & Observability

### Logging

**Format**: Structured JSON (Cloud Logging compatible)

```json
{
  "severity": "INFO",
  "message": "Reasoning request completed",
  "timestamp": "2025-10-01T18:00:00Z",
  "labels": {
    "endpoint": "/api/reasoning/query",
    "model": "gemini-2.5-flash",
    "latency_ms": 1523,
    "request_id": "abc123"
  }
}
```

**Log Levels**:
- `DEBUG` - Detailed debugging info
- `INFO` - Normal operations
- `WARNING` - Potential issues (e.g., slow queries)
- `ERROR` - Errors (caught exceptions)
- `CRITICAL` - System failures

### Metrics

**Key Metrics** (Cloud Monitoring):
- Request rate (req/s)
- Error rate (%)
- Latency (P50, P95, P99)
- Rate limit violations
- Authentication failures
- Gemini token usage
- Cost per query

### Alerts (Recommended)

- Error rate > 5% (5 min)
- Latency P95 > 10s (5 min)
- Rate limit violations > 100/min
- 5xx errors > 10 (1 min)
- Authentication failures > 50/min

---

## Roadmap & Future Work

### Production Hardening (Phase 2A)

1. **Cost Control**
   - Token budget limits per user
   - Query complexity scoring
   - Automatic Flash → Pro routing

2. **Safety System**
   - Hardware interlock validation
   - Emergency stop testing
   - Audit trail verification

3. **Monitoring**
   - Automated alerts
   - Performance dashboards
   - Cost tracking

4. **Database Integration**
   - Cloud SQL setup
   - Experiment metadata storage
   - Provenance tracking

### Hardware Integration (Phase 2B)

1. **Driver Development**
   - Real XRD integration
   - Real NMR integration
   - Real UV-Vis integration

2. **Calibration**
   - Measure actual noise levels
   - Validate adaptive router on real data
   - Safety system stress testing

3. **Queue Management**
   - Priority scheduling
   - Multi-user support
   - Concurrent experiments

### Adaptive Router Validation (Phase 1)

**Timeline**: 4-6 weeks

**Steps**:
1. Pre-register experiments on OSF
2. Implement benchmark functions (Ackley, Rastrigin, Rosenbrock, Hartmann6)
3. Run 4,500 experiments (5 functions × 6 noise levels × 5 methods × 30 trials)
4. Statistical analysis (effect sizes, confidence intervals)
5. Decision: Continue to Phase 2 or pivot

**Success Criteria**:
- RL shows advantage on ≥3/5 functions at σ≥2.0
- p < 0.01 (Bonferroni corrected)
- Cohen's d > 0.5 (medium effect)

**See**: `PHASE1_PREREGISTRATION.md`, `PHASE1_CHECKLIST.md`, `RESEARCH_LOG.md`

---

## Key Design Decisions

### 1. Dual Gemini Model Pattern

**Rationale**: Cost optimization without sacrificing quality
- Flash for 80% of queries (routine)
- Pro for 20% (complex reasoning)
- 10× cost reduction

**Alternative Considered**: Single model (Pro only)
- Rejected: Too expensive at scale

### 2. Static Frontend

**Rationale**: Simplicity, fast deployment, no build step
- HTML/CSS/JS served by FastAPI
- No React/Next.js complexity for MVP
- Easy to iterate

**Future**: May migrate to Next.js if UI complexity grows

### 3. Serverless (Cloud Run) over VMs

**Rationale**: Cost, auto-scaling, zero-downtime deployments
- Pay only for requests (idle = $0)
- Automatic scaling (1-10 instances)
- No infrastructure management

**Trade-off**: Cold start latency (~1-2s)
- Mitigated by min_instances=1 for production

### 4. API Key Auth (not OAuth)

**Rationale**: Simplicity for MVP, API client-friendly
- Easy for programmatic access
- No token refresh complexity
- Sufficient for B2B API

**Future**: Add OAuth for multi-user web app

### 5. Rust Safety Kernel

**Rationale**: Memory safety for hardware control
- Prevents crashes that could damage equipment
- Compile-time guarantees
- Fast execution

**Trade-off**: Development velocity
- Rust is harder than Python
- Only used for critical path (hardware control)

---

## Known Issues & Limitations

### Current Limitations

1. **No Multi-User Support**
   - Single API key shared by all users
   - No user-level permissions or quotas
   - Future: Add OAuth + RBAC

2. **No Database Persistence**
   - Experiment results stored in GCS only
   - No queryable metadata database
   - Future: Add Cloud SQL integration

3. **Limited Hardware Drivers**
   - XRD, NMR, UV-Vis are stubs (not connected to real hardware)
   - Future: Real hardware integration in Phase 2B

4. **Manual Deployment**
   - No CI/CD pipeline yet
   - Deploy via shell script
   - Future: GitHub Actions

5. **Adaptive Router Unvalidated**
   - Only tested on Branin function (n=10)
   - Needs Phase 1 validation (4,500 experiments)
   - May not replicate

### Security Considerations

1. **API Key Rotation**
   - Manual rotation (no automatic expiry)
   - Future: Implement 90-day auto-rotation

2. **Rate Limiting**
   - IP-based only (can be bypassed with proxies)
   - Future: Add user-level rate limiting

3. **No DDoS Protection**
   - Cloud Run has some built-in protection
   - Future: Add Cloud Armor

---

## References

### External Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [Vertex AI Gemini](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)
- [Open Science Framework](https://osf.io) - For pre-registration

### Internal Documentation
- `README.md` - Project overview and quick start
- `docs/architecture.md` - Original architecture document
- `docs/google_cloud_deployment.md` - Detailed GCP deployment
- `PRODUCTION_DEPLOYMENT_GUIDE.md` - Production deployment
- `LOCAL_DEV_SETUP.md` - Local development setup
- `SECURITY_QUICKREF.md` - Security quick reference
- `RESEARCH_LOG.md` - Research activity log

### Research Documentation
- `BREAKTHROUGH_FINDING.md` - Preliminary RL vs BO finding (de-hyped)
- `ADAPTIVE_ROUTER_PROTOTYPE.md` - Adaptive router technical docs
- `PHASE1_PREREGISTRATION.md` - Validation experiment design
- `PHASE1_CHECKLIST.md` - Validation implementation checklist

---

## Contact & Support

**Project**: Autonomous R&D Intelligence Layer (Periodic Labs)  
**Repository**: https://github.com/GOATnote-Inc/periodicdent42  
**Documentation**: See `docs/` directory  

**For Issues**:
- Security issues: See `SECURITY.md`
- Bug reports: GitHub Issues
- Feature requests: GitHub Discussions

---

**Last Updated**: October 1, 2025  
**Version**: 1.0  
**Next Review**: After Phase 1 validation complete

---

*"Architecture is a hypothesis. Code is the experiment. Production is the result."*

