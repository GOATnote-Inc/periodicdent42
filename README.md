# Autonomous R&D Intelligence Layer

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-PROPRIETARY-red.svg)](LICENSE)

## Overview

An autonomous experimentation platform that transforms physical R&D challenges into strategic advantages. This system acts as a co-pilot (evolving to autopilot) for materials science, chemistry, and physics research by combining AI reasoning, safety-critical execution, and glass-box interpretability.

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

```bash
# Install dependencies
pip install -r requirements.txt
cargo build --release

# Run bootstrap setup (Phase 0 + 1)
python scripts/bootstrap.py

# Start services
docker-compose up -d

# Run safety checks
cargo test --package safety-kernel

# Launch UI
cd ui && npm run dev
```

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
✅ **Glass-Box AI**: Every decision logged with rationale, explainable plans  
✅ **EIG Optimization**: Bayesian experimental design for maximum learning/hour  
✅ **Physics-Aware**: Domain ontologies, unit validation, uncertainty propagation  
✅ **Provenance Tracking**: Complete audit trails from raw data to insights  
✅ **Tiered Autonomy**: Human-in-loop → co-pilot → autopilot progression  

## Documentation

### Core Documentation
- [Project Roadmap](docs/roadmap.md) - Phases, milestones, KPIs
- [Instructions](docs/instructions.md) - Development guidelines, best practices
- [Architecture](docs/architecture.md) - System design, data flows
- [Quick Start](docs/QUICKSTART.md) - Installation and usage guide

### Google Cloud Integration (October 2025) ☁️
- **[Google Cloud Deployment Guide](docs/google_cloud_deployment.md)** ⭐ - Complete GCP deployment with Gemini 2.5 Pro/Flash
- **[Gemini Integration Examples](docs/gemini_integration_examples.md)** 💻 - Production-ready code samples
- [Cloud Documentation Index](docs/README_CLOUD.md) - Navigation guide for all cloud docs

### Key Features (Cloud)
- ⚡ **Dual-Model AI**: Gemini 2.5 Flash (<2s) + Pro (10-30s) for instant feedback + verified accuracy
- 🚀 **Serverless Deployment**: Cloud Run auto-scaling (1-10 instances)
- 🔒 **Enterprise Security**: GDC for on-premises, HIPAA-ready
- 💰 **Cost-Optimized**: ~$321/month for development, $3.7K/month for production

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, code standards, and PR guidelines.

## 📄 License

**⚠️ PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED**

This software is proprietary and confidential. Unauthorized use, reproduction, or distribution is strictly prohibited.

**For licensing inquiries**: B@thegoatnote.com

See [LICENSE](LICENSE) for full terms.  
See [AUTHORIZED_USERS.md](AUTHORIZED_USERS.md) for current authorized users.  
See [LICENSING_GUIDE.md](LICENSING_GUIDE.md) for how to request authorization.

## Contact

For questions about deployment, compliance, or research partnerships, see [docs/contact.md](docs/contact.md).

