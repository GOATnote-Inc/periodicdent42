# Cloud Deployment Documentation

## Overview

This directory contains comprehensive documentation for deploying the Autonomous R&D Intelligence Layer on Google Cloud Platform with Gemini 2.5 models (October 2025).

---

## 📚 Documentation Index

### 1. [google_cloud_deployment.md](google_cloud_deployment.md) ⭐
**Complete deployment guide** with:
- ✅ Dual-model architecture (Gemini 2.5 Flash + Pro)
- ✅ Vertex AI configuration
- ✅ Cloud Run serverless deployment
- ✅ Cloud SQL, Storage, and AI Hypercomputer setup
- ✅ Google Distributed Cloud (on-premises option)
- ✅ Security, monitoring, and cost optimization
- ✅ Step-by-step migration from local to cloud

**Use this for**: Production deployment planning and execution

---

### 2. [gemini_integration_examples.md](gemini_integration_examples.md) 💻
**Production-ready code samples** including:
- ✅ Dual-model EIG planning (fast preview + verified response)
- ✅ Safety policy validation with AI reasoning
- ✅ Literature search with RAG (Retrieval Augmented Generation)
- ✅ Natural language experiment submission
- ✅ Automated scientific report generation

**Use this for**: Implementation reference and code templates

---

### 3. [architecture.md](architecture.md) 🏗️
**System architecture overview** covering:
- ✅ Multi-layer design (Governance → Reasoning → Memory → Connectors → OS)
- ✅ Data flow diagrams
- ✅ Component responsibilities
- ✅ Technology stack rationale
- ✅ Updated with Google Cloud integration (October 2025)

**Use this for**: Understanding overall system design

---

### 4. [roadmap.md](roadmap.md) 🗺️
**Project roadmap** with:
- ✅ Phases 0-5 (Foundations → Autopilot)
- ✅ Milestones and KPIs
- ✅ Risk mitigation strategies
- ✅ First 90 days detailed plan

**Use this for**: Project planning and progress tracking

---

### 5. [instructions.md](instructions.md) 📖
**Development guidelines** including:
- ✅ Core moats (Execution, Data, Trust, Time, Interpretability)
- ✅ Tech stack rationale
- ✅ Coding standards (Python, Rust, TypeScript)
- ✅ Testing requirements
- ✅ Common pitfalls to avoid

**Use this for**: Development best practices

---

### 6. [QUICKSTART.md](QUICKSTART.md) 🚀
**Quick start guide** for:
- ✅ Local installation
- ✅ Running bootstrap script
- ✅ Example code usage
- ✅ Troubleshooting

**Use this for**: Getting started quickly

---

## 🎯 Quick Navigation

### For Cloud Deployment
1. Read [google_cloud_deployment.md](google_cloud_deployment.md) (sections 1-4)
2. Review [gemini_integration_examples.md](gemini_integration_examples.md) (examples 1-2)
3. Follow deployment checklist in [google_cloud_deployment.md](google_cloud_deployment.md#8-deployment-checklist)

### For Local Development
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Review [architecture.md](architecture.md) (component details)
3. Check [instructions.md](instructions.md) (coding standards)

### For Project Planning
1. Read [roadmap.md](roadmap.md) (phases and milestones)
2. Review [architecture.md](architecture.md) (technology stack)
3. Check [google_cloud_deployment.md](google_cloud_deployment.md#10-cost-estimate)

---

## 🔑 Key Concepts

### Dual-Model AI Pattern

The platform uses **Gemini 2.5 Flash** and **Gemini 2.5 Pro** in parallel:

```
User Query
    ↓
    ├─────────────────┬──────────────────┐
    ↓ Fast (<2s)      ↓ Accurate (10-30s)│
Flash Preview    Pro Verified Result    │
    └─────────────────┴──────────────────┘
              ↓
    UI shows both (Flash first, Pro when ready)
```

**Benefits**:
- ⚡ **Instant feedback** for better UX
- 🎯 **High accuracy** for scientific integrity
- 💰 **Cost-effective** (use Flash for 90% of queries)

---

### Google Cloud Services Used

| Service | Purpose | Phase |
|---------|---------|-------|
| **Vertex AI** | Gemini model deployment | 0-1 ✅ |
| **Cloud Run** | Serverless FastAPI | 0-1 ✅ |
| **Cloud SQL** | PostgreSQL + TimescaleDB | 0-1 ✅ |
| **Cloud Storage** | Data lake | 0-1 ✅ |
| **AI Hypercomputer** | RL training, DFT at scale | 2-3 |
| **GDC (optional)** | On-premises deployment | 3-4 |

---

### Cost Estimates

#### Phase 0-1 (Development)
**~$321/month**
- Cloud Run: $50
- Gemini API (Flash + Pro): $36
- Cloud SQL: $200
- Cloud Storage: $25
- Monitoring: $10

#### Phase 3-5 (Production)
**~$3,700/month**
- Cloud Run (scaled): $500
- Gemini API (1B tokens): $200
- AI Hypercomputer: $2,000
- Cloud SQL (HA): $800
- Cloud Storage (10TB): $200

See [google_cloud_deployment.md#10-cost-estimate](google_cloud_deployment.md#10-cost-estimate) for details.

---

## 🛠️ Tools & Technologies

### Current (October 2025)

✅ **Gemini 2.5 Flash** - Fast, cost-effective AI (latest model)  
✅ **Gemini 2.5 Pro** - High-accuracy reasoning (latest model)  
✅ **Vertex AI** - Unified ML platform  
✅ **Model Context Protocol (MCP)** - Tool integration  
✅ **Cloud Run** - Serverless containers  
✅ **AI Hypercomputer** - TPU v5e / GPU compute  

### Not Used (Outdated)

❌ PaLM 2 (superseded by Gemini)  
❌ Codey (superseded by Gemini Code Assist)  
❌ Chirp (superseded by Gemini)  

**Policy**: Always use latest models as of deployment date.

---

## 📞 Support & Resources

### Documentation
- [Google Cloud Vertex AI Docs](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Reference](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro)
- [Model Context Protocol](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling)

### Training
- [Google Cloud Skills Boost](https://www.cloudskillsboost.google/)
- [Vertex AI Tutorials](https://cloud.google.com/vertex-ai/docs/tutorials)

### Community
- [Stack Overflow: google-cloud-aiplatform](https://stackoverflow.com/questions/tagged/google-cloud-aiplatform)
- [Google Cloud Community](https://www.googlecloudcommunity.com/)

---

## 🎓 Learning Path

### Week 1: Foundations
- [ ] Read [architecture.md](architecture.md)
- [ ] Read [QUICKSTART.md](QUICKSTART.md)
- [ ] Run bootstrap script locally
- [ ] Review [gemini_integration_examples.md](gemini_integration_examples.md) (example 1)

### Week 2: Cloud Basics
- [ ] Read [google_cloud_deployment.md](google_cloud_deployment.md) (sections 1-4)
- [ ] Create GCP project and enable APIs
- [ ] Deploy FastAPI to Cloud Run
- [ ] Test Gemini 2.5 Flash integration

### Week 3: Advanced Features
- [ ] Implement dual-model pattern (Flash + Pro)
- [ ] Set up Cloud SQL and Cloud Storage
- [ ] Configure monitoring and alerts
- [ ] Review [gemini_integration_examples.md](gemini_integration_examples.md) (all examples)

### Week 4: Production Ready
- [ ] Complete deployment checklist
- [ ] Run load tests
- [ ] Set up cost budgets
- [ ] Deploy to production

---

## 🚨 Important Notes

### Security
⚠️ **Never commit API keys** to git  
⚠️ Use **Secret Manager** for credentials  
⚠️ Enable **VPC Service Controls** for production  
⚠️ Review **IAM policies** regularly  

### Cost Management
💰 Set **budget alerts** at $100, $500, $1000  
💰 Use **quotas** to prevent runaway costs  
💰 Enable **sustained use discounts**  
💰 Review **cost allocation** monthly  

### Compliance
📋 **HIPAA**: Use GDC for sensitive health data  
📋 **Export Control**: Air-gapped GDC for restricted research  
📋 **GDPR**: Configure data residency in EU regions  

---

## ✅ Deployment Checklist

Quick reference (full checklist in [google_cloud_deployment.md#8-deployment-checklist](google_cloud_deployment.md#8-deployment-checklist)):

- [ ] GCP project created
- [ ] Vertex AI enabled
- [ ] FastAPI deployed to Cloud Run
- [ ] Gemini 2.5 Flash + Pro tested
- [ ] Cloud SQL provisioned
- [ ] Monitoring configured
- [ ] Security audit completed
- [ ] Cost budgets set

---

**Last Updated**: October 1, 2025  
**Next Review**: January 1, 2026

For questions or issues, see [QUICKSTART.md#troubleshooting](QUICKSTART.md#troubleshooting).

