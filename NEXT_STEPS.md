# 🎯 Next Steps: What to Build Next

**Current Status**: ✅ Phase 0 & 1 MVP deployed to production on Cloud Run

---

## 🏆 What We've Accomplished

### Phase 0: Foundations ✅
- ✅ **Data Contract**: Pydantic schemas in `configs/data_schema.py`
- ✅ **Safety Kernel V1**: Rust implementation in `src/safety/`
- ✅ **Experiment OS**: Core queue + driver registry in `src/experiment_os/core.py`
- ✅ **Cloud Infrastructure**: FastAPI on Cloud Run + Vertex AI
- ✅ **Dual-Model AI**: Gemini 2.5 Flash (instant) + Pro (verified)
- ✅ **CI/CD Pipeline**: GitHub Actions → automatic deployment
- ✅ **Health Checks**: Production monitoring working

### Phase 1: Intelligence ✅ (Foundation)
- ✅ **EIG Optimizer**: Bayesian planning in `src/reasoning/eig_optimizer.py`
- ✅ **Simulator Integration**: Connector layer in `src/connectors/simulators.py`
- ✅ **Dual-Agent Pattern**: SSE streaming for real-time feedback

---

## 🚀 Recommended Next Steps (Priority Order)

### Option 1: **Phase 2 - Training (High Impact)** 🎓
**Why**: Get the AI ready for real experiments with domain knowledge.

**Tasks**:
1. **Build RL Environment** (`src/training/gym_env.py`)
   - Wrap Experiment OS as OpenAI Gym environment
   - Reward function: EIG per experiment * safety compliance
   - State space: current knowledge graph + available actions
   - Action space: discrete (select experiment type + parameters)

2. **Fine-tune Gemini Pro** for your domain
   - Collect ~100 examples of good/bad experiment plans
   - Use Vertex AI Model Garden for fine-tuning
   - Compare baseline vs. fine-tuned on held-out test set

3. **Policy Training**
   - Use Stable-Baselines3 PPO algorithm
   - Train on simulated experiments first
   - Gradually transition to real hardware

**Deliverables**:
- Working RL agent that can select experiments
- Fine-tuned Gemini Pro model for materials science
- Curriculum learning pipeline (sim → real transfer)

**Estimated Time**: 2-3 weeks

---

### Option 2: **Phase 2 - Hardware Integration (Real Experiments)** 🔬
**Why**: Start running actual experiments to collect data.

**Tasks**:
1. **Connect First Real Instrument**
   - Pick one: XRD, NMR, or custom setup
   - Write driver: `src/experiment_os/drivers/xrd_bruker.py`
   - Test connection: read status, run calibration
   - Implement safety interlocks (emergency stop)

2. **First Autonomous Loop**
   - Start with simple parameter sweep (e.g., temperature scan)
   - AI suggests next temperature → instrument measures → log result
   - Run for 24 hours, collect 50+ data points
   - Verify data quality and safety compliance

3. **Database Enhancement**
   - Connect Cloud SQL (PostgreSQL + TimescaleDB)
   - Store all experiment results with timestamps
   - Add Cloud Storage for raw instrument files
   - Implement data backup and versioning

**Deliverables**:
- One working instrument driver
- First 50+ autonomous experiments logged
- Production database with real data

**Estimated Time**: 1-2 weeks (depends on hardware availability)

---

### Option 3: **Phase 1.5 - Enhanced RAG & Knowledge Graph** 🧠
**Why**: Give the AI access to scientific literature and prior knowledge.

**Tasks**:
1. **Scientific RAG System**
   - Index papers from arXiv, PubMed, or domain-specific sources
   - Use Vertex AI Vector Search for embeddings
   - Integrate with Gemini Pro for citation-backed answers
   - Example: "What's the optimal bandgap for perovskite solar cells?"

2. **Knowledge Graph V1**
   - Use Neo4j or TigerGraph
   - Nodes: Materials, Properties, Experiments, Papers
   - Edges: "has_property", "cited_by", "derived_from"
   - Ingest Materials Project or ICSD database

3. **MCP Tool Integration**
   - Implement tools in `src/reasoning/mcp_agent.py`:
     - `search_literature(query)` → returns papers
     - `query_knowledge_graph(material)` → returns properties
     - `run_simulation(params)` → DFT calculation
   - Give Gemini Pro access to these tools via function calling

**Deliverables**:
- RAG system with 10,000+ papers indexed
- Knowledge graph with 1,000+ materials
- Gemini Pro with tool-calling capabilities

**Estimated Time**: 2-3 weeks

---

### Option 4: **Polish & Scale Current MVP** 🏗️
**Why**: Make what you have production-ready and robust.

**Tasks**:
1. **Enhanced Observability**
   - Implement custom Cloud Monitoring metrics (EIG per hour)
   - Add structured logging with Cloud Logging
   - Create Grafana dashboard for real-time monitoring
   - Set up alerting (PagerDuty/Slack) for failures

2. **Security Hardening**
   - Implement authentication (OAuth 2.0 via Identity Platform)
   - Add rate limiting and request validation
   - Secrets rotation for Cloud SQL passwords
   - VPC Service Controls for data isolation

3. **Performance Optimization**
   - Add Redis caching for frequent queries
   - Optimize Gemini Pro prompts (reduce tokens)
   - Implement request batching for Vector Search
   - Load testing (can it handle 100 concurrent users?)

4. **Frontend Dashboard**
   - Next.js app showing live experiment status
   - Real-time charts (EIG over time, safety violations)
   - Manual experiment submission form
   - Audit log viewer for transparency

**Deliverables**:
- Production-grade observability
- Secure multi-user system
- Web dashboard for non-technical users

**Estimated Time**: 2-3 weeks

---

## 📊 Decision Matrix

| Option | Impact | Effort | Risk | Time-to-Value |
|--------|--------|--------|------|---------------|
| **Phase 2 - Training** | 🔥🔥🔥 High | Medium | Medium | 2-3 weeks |
| **Phase 2 - Hardware** | 🔥🔥🔥 High | High | High | 1-2 weeks |
| **Phase 1.5 - RAG/KG** | 🔥🔥 Medium | Medium | Low | 2-3 weeks |
| **Polish & Scale** | 🔥 Low-Medium | Medium | Low | 2-3 weeks |

---

## 🎯 My Recommendation

### **Start with Phase 2 - Hardware Integration** (if you have equipment)
**Why**:
1. **Real data beats simulated data** - you need actual experiments to validate the AI
2. **Early feedback loop** - see what works/breaks in practice
3. **Moat building** - start accumulating proprietary data NOW

**Alternative if no hardware yet**:
→ **Phase 1.5 - RAG & Knowledge Graph** to make the AI smarter while waiting for equipment

---

## 📝 Quick Wins (< 1 day each)

Before diving into a big phase, consider these quick improvements:

1. **Add Reasoning Endpoint Tests**
   - Write test for `/api/reasoning/query` with SSE streaming
   - Mock Gemini responses, verify Flash → Pro order
   - Add to CI/CD pipeline

2. **Create `.env.example` for Local Development**
   - Document all required environment variables
   - Add setup instructions for new developers

3. **Implement Simple Web UI**
   - Single HTML page with WebSocket for SSE
   - Text box → query → shows Flash (gray) → Pro (green)
   - Deploy to Cloud Run as separate service

4. **Add Experiment Result Storage**
   - Create Cloud Storage bucket for results
   - Implement `storage.store_result(experiment_id, data)`
   - Test upload/download with real files

5. **Set Up Monitoring Dashboard**
   - Create Cloud Monitoring dashboard
   - Add charts: request latency, error rate, Gemini API costs
   - Set up alert if health check fails

---

## 🤔 Questions to Consider

1. **Do you have physical lab equipment ready to connect?**
   - Yes → Go for Phase 2 - Hardware Integration
   - No → Focus on Phase 1.5 - RAG/KG or training

2. **What's your team size?**
   - Solo → Pick one focused task
   - 2-3 people → Parallel tracks (hardware + AI training)

3. **What's the most painful bottleneck right now?**
   - Lack of data → Hardware integration
   - AI not smart enough → Training & fine-tuning
   - Need domain knowledge → RAG & knowledge graph
   - System not reliable → Polish & scale

4. **What's your timeline to first real results?**
   - < 2 weeks → Quick wins + hardware
   - 1 month → Phase 2 (training or hardware)
   - 3 months → Full Phase 2 + Phase 3

---

## 📚 Resources

- **Roadmap**: `docs/roadmap.md` - Full 18-month plan
- **Architecture**: `docs/architecture.md` - System design
- **Quick Start**: `docs/QUICKSTART.md` - Getting started
- **Deployment**: `DEPLOYMENT_GUIDE.md` - Production setup

---

**What resonates with you? What's the biggest pain point you want to solve next?**

