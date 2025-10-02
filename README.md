# Autonomous R&D Intelligence Layer

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-PROPRIETARY-red.svg)](LICENSE)

## Overview

An autonomous experimentation platform designed for materials science, chemistry, and physics research. This system combines AI reasoning, safety-critical execution, and interpretable decision-making to support laboratory workflows.

## Key Moats

1. **Execution Moat**: Reliable instrument drivers, queue management, and hardware control
2. **Data Moat**: Physics-aware schemas, uncertainty quantification, and provenance tracking
3. **Trust Moat**: Safety-first design, audit trails, and compliance-ready SOPs
4. **Time Moat**: Expected Information Gain (EIG) optimization for maximum learning velocity
5. **Interpretability Moat**: Glass-box agents with explainable reasoning and scientific ontologies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Governance Layer (UI/RBAC)               │
├─────────────────────────────────────────────────────────────┤
│  Reasoning & Planning (AI Agents, EIG Optimization)         │
├─────────────────────────────────────────────────────────────┤
│  Scientific Memory (Provenance, Physics-Aware Queries)      │
├─────────────────────────────────────────────────────────────┤
│  Connectors (Simulators ↔ Instruments)                      │
├─────────────────────────────────────────────────────────────┤
│  Experiment OS (Queue, Drivers) + Safety Kernel (Rust)      │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Frontend**: Next.js (TypeScript) - Provenance viewer, dashboards
- **Backend**: FastAPI (Python 3.12) - Orchestration, AI reasoning
- **Safety-Critical**: Rust - Instrument control, interlocks, resource limits
- **AI/ML**: PyTorch, NumPy, SciPy, SymPy
- **Science**: PySCF, RDKit, ASE (chemistry/materials simulations)
- **Data**: PostgreSQL + TimescaleDB, NetworkX for planning graphs

## Quick Start

### Local Development (2 Minutes) 🚀

```bash
# One command does everything!
bash scripts/init_secrets_and_env.sh

# Then start the server:
cd app && source venv/bin/activate
uvicorn src.api.main:app --reload --port 8080
```

**See**: [QUICK_START.md](QUICK_START.md) for detailed walkthrough  
**Or**: [LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md) for manual setup

### Production Deployment (10 Minutes)
```bash
# Automated deployment to Google Cloud Run with security enabled
bash infra/scripts/enable_apis.sh      # Enable required APIs
bash infra/scripts/setup_iam.sh        # Configure service accounts
bash infra/scripts/create_secrets.sh   # Generate API keys
bash infra/scripts/deploy_cloudrun.sh  # Deploy secure service
```

**See**: [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) for step-by-step instructions

## Project Structure

```
periodicdent42/
├── src/
│   ├── experiment_os/      # Core queue, drivers, scheduling
│   ├── safety/             # Rust safety kernel
│   ├── connectors/         # Simulator & instrument adapters
│   ├── memory/             # Scientific memory & provenance
│   ├── reasoning/          # AI agents, EIG planning
│   ├── actuation/          # Execution & monitoring
│   ├── governance/         # RBAC, audits, policies
│   └── simulators/         # DFT, MD, and other sim integrations
├── configs/                # Data schemas, policies, limits
├── tests/                  # Unit, integration, red-team tests
├── docs/                   # Roadmap, instructions, architecture
├── ui/                     # Next.js frontend
├── scripts/                # Bootstrap, deployment scripts
└── .cursor/rules/          # AI coding assistant rules (moats)
```

## Development Phases

- **Phase 0** (Weeks 1-4): Foundations - Experiment OS, Data Contracts, Safety V1
- **Phase 1** (Weeks 5-8): Intelligence - Simulators, EIG planning, DoE primitives
- **Phase 2** (Months 3-4): Mid-Training - RL agents, policy optimization
- **Phase 3** (Months 5-6): Real-World - Hardware integration, closed-loop automation
- **Phase 4** (Months 7-9): Scale - Multi-lab, federated learning
- **Phase 5** (Months 10-12): Autopilot - Full autonomy with human oversight

## Key Features

✅ **Safety-First Design**: Rust-based interlocks, dry-run compilation, policy enforcement  
✅ **Interpretable AI**: Decision logging with rationale and explainable plans  
✅ **EIG Optimization**: Bayesian experimental design approach  
✅ **Physics-Aware**: Domain ontologies, unit validation, uncertainty propagation  
✅ **Provenance Tracking**: Audit trails from raw data to insights  
✅ **Security Hardened**: API key authentication, rate limiting, CORS restrictions  
✅ **Tiered Autonomy**: Human-in-loop → co-pilot → autopilot progression  

## Documentation

> **📚 [Complete Documentation Index](DOCUMENTATION_INDEX.md)** - Comprehensive guide to all 60+ documentation files

### 🎯 Start Here

| If you want to... | Read this |
|-------------------|-----------|
| **Get started quickly** | [QUICK_START.md](QUICK_START.md) (2 minutes) |
| **Understand the system** | [ARCHITECTURE.md](ARCHITECTURE.md) ⭐ **Start here for deep dive** |
| **Deploy to production** | [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) |
| **Set up local dev** | [LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md) |
| **Manage secrets securely** | [SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md) 🔐 **Critical** |

### 🏗️ Architecture & Technical

- **[ARCHITECTURE.md](ARCHITECTURE.md)** ⭐ - **Comprehensive system architecture** (400+ lines, start here!)
  - System components and data flow
  - Directory structure explained
  - Deployment architecture
  - Security architecture
  - Performance characteristics
  - Design decisions and trade-offs

### 🔐 Security & Deployment

- **[SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md)** 🔐 - Secrets management guide (critical!)
- **[PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)** - Production deployment
- **[SECURITY_VERIFICATION_OCT2025.md](SECURITY_VERIFICATION_OCT2025.md)** - Latest security audit
- [SECURITY_QUICKREF.md](SECURITY_QUICKREF.md) - Quick security reference
- [docs/google_cloud_deployment.md](docs/google_cloud_deployment.md) - Detailed GCP deployment

### 🔬 Research & Validation

- **[RESEARCH_LOG.md](RESEARCH_LOG.md)** 🔥 - Active research activity log
- **[BREAKTHROUGH_FINDING.md](BREAKTHROUGH_FINDING.md)** - RL vs BO preliminary findings (honest framing)
- **[ADAPTIVE_ROUTER_PROTOTYPE.md](ADAPTIVE_ROUTER_PROTOTYPE.md)** - Adaptive router documentation
- [PHASE1_PREREGISTRATION.md](PHASE1_PREREGISTRATION.md) - Validation experiment design
- [PHASE1_CHECKLIST.md](PHASE1_CHECKLIST.md) - Validation implementation plan

### 💼 Business & Strategy

- [BUSINESS_VALUE_ANALYSIS.md](BUSINESS_VALUE_ANALYSIS.md) - Business value and customer pain points
- [MARKET_ANALYSIS_OCT2025.md](MARKET_ANALYSIS_OCT2025.md) - Market analysis
- [CUSTOMER_PROTOTYPE_STRATEGY.md](CUSTOMER_PROTOTYPE_STRATEGY.md) - Prototype strategy
- [docs/roadmap.md](docs/roadmap.md) - Product roadmap

### ☁️ Google Cloud Integration

**Key Features**:
- ⚡ **Dual-Model AI**: Gemini 2.5 Flash + Pro for cost/quality optimization
- 🚀 **Serverless**: Cloud Run with auto-scaling (1-10 instances)
- 🔒 **Security**: API key auth, rate limiting, CORS, security headers
- 💰 **Cost**: ~$80/month for 1000 queries/day (estimate)

**Documentation**:
- [docs/gemini_integration_examples.md](docs/gemini_integration_examples.md) - Code samples
- [CLOUD_INTEGRATION_SUMMARY.md](CLOUD_INTEGRATION_SUMMARY.md) - Integration overview
- [docs/README_CLOUD.md](docs/README_CLOUD.md) - Cloud documentation index

## Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Safety red-team tests
pytest tests/safety/ --strict

# Rust safety kernel
cargo test --package safety-kernel --all-features
```

## Collaboration

This is proprietary software. For collaboration inquiries, see [docs/contact.md](docs/contact.md) or contact B@thegoatnote.com.

## 📄 License

**⚠️ PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED**

This software is proprietary and confidential. Unauthorized use, reproduction, or distribution is strictly prohibited.

**For licensing inquiries**: B@thegoatnote.com

See [LICENSE](LICENSE) for full terms.  
See [AUTHORIZED_USERS.md](AUTHORIZED_USERS.md) for current authorized users.  
See [LICENSING_GUIDE.md](LICENSING_GUIDE.md) for how to request authorization.

## Contact

For questions about deployment, compliance, or research partnerships, see [docs/contact.md](docs/contact.md).

