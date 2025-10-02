# Documentation Index

**Last Updated**: October 1, 2025  
**Repository**: periodicdent42 (Autonomous R&D Intelligence Layer)

This document provides a comprehensive index of all documentation in this repository, organized by category.

---

## 🏗️ **Architecture & Overview**

| Document | Description | Status |
|----------|-------------|--------|
| [README.md](README.md) | Project overview, quick start, key features | ✅ Current |
| [ARCHITECTURE.md](ARCHITECTURE.md) | **Comprehensive system architecture** (start here) | ✅ Current |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Project summary and business context | ✅ Current |
| [docs/architecture.md](docs/architecture.md) | Original architecture doc (legacy) | ⚠️ Legacy |

---

## 🚀 **Getting Started**

| Document | Description | Audience |
|----------|-------------|----------|
| [QUICK_START.md](QUICK_START.md) | 2-minute quick start guide | Developers (new) |
| [LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md) | Detailed local development setup | Developers |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Alternative quick start | Developers |
| [docs/instructions.md](docs/instructions.md) | Development instructions | Developers |

**Recommended Order**:
1. Start with `QUICK_START.md` (2 minutes)
2. If setting up locally: `LOCAL_DEV_SETUP.md`
3. For production: `PRODUCTION_DEPLOYMENT_GUIDE.md`

---

## 🔐 **Security**

| Document | Description | Status |
|----------|-------------|--------|
| [SECURITY.md](SECURITY.md) | Security policy and reporting | ✅ Current |
| [SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md) | **Secrets management guide** (critical) | ✅ Current |
| [SECURITY_QUICKREF.md](SECURITY_QUICKREF.md) | Security quick reference | ✅ Current |
| [SECURITY_VERIFICATION_OCT2025.md](SECURITY_VERIFICATION_OCT2025.md) | Security audit report (Oct 2025) | ✅ Current |
| [SECURITY_INCIDENT_REPORT_20251001.md](SECURITY_INCIDENT_REPORT_20251001.md) | API key leak incident (resolved) | ✅ Archived |
| [COMPREHENSIVE_SECURITY_SCAN_RESULTS.md](COMPREHENSIVE_SECURITY_SCAN_RESULTS.md) | 15-point security scan results | ✅ Current |
| [docs/SECURITY.md](docs/SECURITY.md) | Additional security documentation | ✅ Current |
| [docs/safety_gateway.md](docs/safety_gateway.md) | Safety system (hardware) | ✅ Current |

**Key Scripts**:
- `scripts/check_security.sh` - Run security integrity checks
- `scripts/rotate_api_key.sh` - Rotate API keys
- `scripts/get_secrets.sh` - Retrieve secrets from Secret Manager

---

## ☁️ **Deployment**

| Document | Description | Environment |
|----------|-------------|-------------|
| [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) | **Production deployment** (Google Cloud Run) | Production |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Alternative deployment guide | Production |
| [CLOUD_INTEGRATION_SUMMARY.md](CLOUD_INTEGRATION_SUMMARY.md) | Cloud integration overview | Production |
| [docs/google_cloud_deployment.md](docs/google_cloud_deployment.md) | Detailed GCP deployment | Production |
| [docs/README_CLOUD.md](docs/README_CLOUD.md) | Cloud-specific README | Production |
| [MANUAL_STEPS_CHECKLIST.md](MANUAL_STEPS_CHECKLIST.md) | Manual deployment checklist | Production |

**Deployment Scripts** (`infra/scripts/`):
1. `enable_apis.sh` - Enable required Google Cloud APIs
2. `setup_iam.sh` - Configure IAM roles
3. `create_secrets.sh` - Create secrets in Secret Manager
4. `deploy_cloudrun.sh` - Deploy to Cloud Run

**Quick Deploy**:
```bash
cd infra/scripts
bash enable_apis.sh && bash setup_iam.sh && bash create_secrets.sh && bash deploy_cloudrun.sh
```

---

## 🔬 **Research & Validation**

### Active Research

| Document | Description | Status |
|----------|-------------|--------|
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | **Transparent research activity log** | 🔥 Active |
| [BREAKTHROUGH_FINDING.md](BREAKTHROUGH_FINDING.md) | Preliminary RL vs BO finding (honest framing) | ✅ Current |
| [ADAPTIVE_ROUTER_PROTOTYPE.md](ADAPTIVE_ROUTER_PROTOTYPE.md) | Adaptive router technical documentation | ✅ Current |
| [ADAPTIVE_ROUTER_BUILD_SUMMARY.md](ADAPTIVE_ROUTER_BUILD_SUMMARY.md) | Build summary (Oct 1, 2025) | ✅ Current |

### Phase 1 Validation (Planned)

| Document | Description | Status |
|----------|-------------|--------|
| [PHASE1_PREREGISTRATION.md](PHASE1_PREREGISTRATION.md) | **Pre-registration template** (OSF) | 📋 To be completed |
| [PHASE1_CHECKLIST.md](PHASE1_CHECKLIST.md) | Implementation checklist (19 items) | 📋 Not started |
| [VALIDATION_BEST_PRACTICES.md](VALIDATION_BEST_PRACTICES.md) | Validation methodology | ✅ Current |

### Historical Validation

| Document | Description | Status |
|----------|-------------|--------|
| [VALIDATION_RESULTS_ANALYSIS.md](VALIDATION_RESULTS_ANALYSIS.md) | Branin function validation (n=10) | ✅ Archived |
| [VALIDATION_STATUS.md](VALIDATION_STATUS.md) | Validation status summary | ⚠️ Outdated |
| [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md) | Validation summary | ⚠️ Outdated |
| [PROOF_STRATEGY_OCT2025.md](PROOF_STRATEGY_OCT2025.md) | Original proof strategy | ⚠️ Superseded |

**Next Steps**: See `PHASE1_CHECKLIST.md` for actionable items

---

## 💼 **Business & Strategy**

| Document | Description | Audience |
|----------|-------------|----------|
| [BUSINESS_VALUE_ANALYSIS.md](BUSINESS_VALUE_ANALYSIS.md) | **Business value and customer pain points** | Leadership |
| [MARKET_ANALYSIS_OCT2025.md](MARKET_ANALYSIS_OCT2025.md) | Market analysis (Oct 2025) | Leadership |
| [CUSTOMER_PROTOTYPE_STRATEGY.md](CUSTOMER_PROTOTYPE_STRATEGY.md) | Prototype development strategy | Product |
| [NEXT_DEVELOPMENT_PHASE.md](NEXT_DEVELOPMENT_PHASE.md) | Development roadmap | Product |
| [PRODUCTION_HARDENING_TASKS.md](PRODUCTION_HARDENING_TASKS.md) | Production hardening tasks | Engineering |
| [docs/roadmap.md](docs/roadmap.md) | Product roadmap | Product |

---

## 🛠️ **Technical Guides**

### API & Integration

| Document | Description | Audience |
|----------|-------------|----------|
| [docs/gemini_integration_examples.md](docs/gemini_integration_examples.md) | Gemini API examples | Developers |
| `app/src/api/main.py` | FastAPI application source | Developers |

### Hardware Drivers

| Document | Description | Status |
|----------|-------------|--------|
| [DRIVERS_README.md](DRIVERS_README.md) | Hardware drivers overview | ✅ Current |
| `src/experiment_os/drivers/` | Driver implementations | 🚧 Stubs |

### AI Agents

| Document | Description | Status |
|----------|-------------|--------|
| [agents.md](agents.md) | AI agents overview | ✅ Current |
| [EXPERT_RL_DEMO.md](EXPERT_RL_DEMO.md) | Expert RL demonstration | ✅ Current |

---

## 📜 **Legal & Compliance**

| Document | Description | Status |
|----------|-------------|--------|
| [LICENSE](LICENSE) | Proprietary license | ✅ Current |
| [LICENSING_GUIDE.md](LICENSING_GUIDE.md) | Licensing guide | ✅ Current |
| [LEGAL_REVIEW_SUMMARY.md](LEGAL_REVIEW_SUMMARY.md) | Legal review summary | ✅ Current |
| [AUTHORIZED_USERS.md](AUTHORIZED_USERS.md) | Authorized users list | ✅ Current |

---

## 📚 **Historical Documents** (Archive)

These documents capture past work and decisions but are no longer actively maintained.

### Completion Reports

| Document | Description | Date |
|----------|-------------|------|
| [SESSION_COMPLETE.md](SESSION_COMPLETE.md) | Session completion summary | Various |
| [SETUP_COMPLETE.md](SETUP_COMPLETE.md) | Setup completion | Historical |
| [TESTING_COMPLETE.md](TESTING_COMPLETE.md) | Testing completion | Historical |
| [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) | Implementation completion | Historical |
| [EXPERT_VALIDATION_COMPLETE.md](EXPERT_VALIDATION_COMPLETE.md) | Validation completion | Historical |
| [QUICK_WINS_COMPLETE.md](QUICK_WINS_COMPLETE.md) | Quick wins completion | Historical |
| [SECURITY_IMPLEMENTATION_COMPLETE.md](SECURITY_IMPLEMENTATION_COMPLETE.md) | Security implementation | Historical |
| [SECURITY_HARDENING_COMPLETE.md](SECURITY_HARDENING_COMPLETE.md) | Security hardening | Historical |
| [SECURITY_AUDIT_COMPLETE.md](SECURITY_AUDIT_COMPLETE.md) | Security audit | Historical |

### Deployment Reports

| Document | Description | Date |
|----------|-------------|------|
| [DEPLOYMENT_SUCCESS.md](DEPLOYMENT_SUCCESS.md) | Deployment success | Historical |
| [DEPLOYMENT_SUCCESS_VALIDATED.md](DEPLOYMENT_SUCCESS_VALIDATED.md) | Validated deployment | Historical |
| [DEPLOYMENT_VALIDATED.md](DEPLOYMENT_VALIDATED.md) | Deployment validation | Historical |

### Other Historical

| Document | Description | Status |
|----------|-------------|--------|
| [FINAL_SUMMARY.md](FINAL_SUMMARY.md) | Final summary | Historical |
| [HARDENING_SUMMARY.md](HARDENING_SUMMARY.md) | Hardening summary | Historical |
| [SECURITY_AUDIT_REPORT.md](SECURITY_AUDIT_REPORT.md) | Security audit report | Historical |
| [PROBLEMS_SOLVED.md](PROBLEMS_SOLVED.md) | Problems solved log | Historical |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Next steps (outdated) | Superseded by RESEARCH_LOG |
| [STRATEGIC_DECISION.md](STRATEGIC_DECISION.md) | Strategic decision | Historical |
| [TECHNICAL_BUILD_PLAN.md](TECHNICAL_BUILD_PLAN.md) | Technical build plan | Historical |
| [PHASE3_IMPLEMENTATION_PLAN.md](PHASE3_IMPLEMENTATION_PLAN.md) | Phase 3 plan | Historical |
| [UI_IMPROVEMENTS.md](UI_IMPROVEMENTS.md) | UI improvements | Historical |
| [FIXED_HEALTHZ.md](FIXED_HEALTHZ.md) | Health endpoint fix | Historical |
| [COMMANDS_TO_RUN.md](COMMANDS_TO_RUN.md) | Command reference | Superseded |

**Note**: These documents are kept for historical reference but may contain outdated information.

---

## 🗂️ **By Directory**

### Root Directory
- Core documentation (README, ARCHITECTURE, etc.)
- Research logs (RESEARCH_LOG, BREAKTHROUGH_FINDING, etc.)
- Security guides (SECRETS_MANAGEMENT, SECURITY_QUICKREF, etc.)

### `docs/`
- Original documentation (some legacy)
- Detailed guides (deployment, security, architecture)
- Contact and roadmap

### `app/`
- FastAPI application source code
- Tests
- Static frontend (HTML/CSS/JS)
- `app/README.md` - App-specific documentation

### `src/`
- Research modules (RL, adaptive router, experiment OS)
- Core algorithms and drivers

### `infra/`
- Infrastructure scripts (deployment, IAM, secrets)
- Monitoring dashboards

### `scripts/`
- Training scripts (PPO, RL agents)
- Validation scripts
- Utility scripts (secrets, security checks)

### `configs/`
- Data schemas
- Safety policies

---

## 🔍 **Quick Search Guide**

**Looking for...**

| I want to... | Document |
|--------------|----------|
| Get started quickly | [QUICK_START.md](QUICK_START.md) |
| Understand the system | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Deploy to production | [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) |
| Set up local dev | [LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md) |
| Manage secrets securely | [SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md) |
| Understand security | [SECURITY_VERIFICATION_OCT2025.md](SECURITY_VERIFICATION_OCT2025.md) |
| Review research findings | [BREAKTHROUGH_FINDING.md](BREAKTHROUGH_FINDING.md) |
| Plan validation experiments | [PHASE1_PREREGISTRATION.md](PHASE1_PREREGISTRATION.md) |
| Understand business value | [BUSINESS_VALUE_ANALYSIS.md](BUSINESS_VALUE_ANALYSIS.md) |
| See what's been done | [RESEARCH_LOG.md](RESEARCH_LOG.md) |
| Report a security issue | [SECURITY.md](SECURITY.md) |

---

## 📊 **Documentation Statistics**

**Total Documents**: ~60 markdown files  
**Active Documentation**: ~25 files  
**Historical/Archived**: ~35 files  

**By Category**:
- Architecture & Overview: 4
- Getting Started: 4
- Security: 8
- Deployment: 6
- Research: 11
- Business: 6
- Technical Guides: 5
- Legal: 4
- Historical: ~35

**Last Major Update**: October 1, 2025  
**Next Review**: After Phase 1 validation

---

## 🔄 **Documentation Maintenance**

### Active Documents (Update Regularly)
- `RESEARCH_LOG.md` - After each research session
- `ARCHITECTURE.md` - When architecture changes
- `SECRETS_MANAGEMENT.md` - When security practices change
- `PHASE1_CHECKLIST.md` - As validation progresses

### Review Schedule
- **Weekly**: RESEARCH_LOG, PHASE1_CHECKLIST
- **Monthly**: ARCHITECTURE, Security docs
- **Quarterly**: Business docs, roadmap
- **Annually**: LICENSE, legal docs

### Archive Policy
- Move "COMPLETE" and "STATUS" files to `docs/archive/` after 3 months
- Keep README, ARCHITECTURE, and active research docs in root
- Maintain this index as single source of truth

---

## 🤝 **Contributing to Documentation**

**Guidelines**:
1. Keep documentation honest and accurate (no hype)
2. Update RESEARCH_LOG.md for research activities
3. Use clear, structured formatting
4. Include "Last Updated" dates
5. Cross-reference related documents
6. Archive outdated docs (don't delete)

**Before Committing**:
- [ ] Update "Last Updated" date
- [ ] Check links work
- [ ] Run `scripts/check_security.sh` (if adding code examples)
- [ ] Update `DOCUMENTATION_INDEX.md` if adding new docs

---

## 📞 **Support**

**Questions about documentation?**
- Check this index first
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical questions
- See [docs/contact.md](docs/contact.md) for contact information

**Found a broken link or outdated info?**
- Open a GitHub issue
- Tag with `documentation` label

---

**Last Updated**: October 1, 2025  
**Maintainer**: Engineering Team  
**Next Review**: October 8, 2025

---

*"Good documentation is a map, not a maze."*

