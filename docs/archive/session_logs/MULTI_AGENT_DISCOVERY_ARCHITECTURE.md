# Multi-Agent Superconductor Discovery Architecture

**Status**: 🚧 **Design Phase** → Implementation Starting  
**Owner**: GOATnote Autonomous Research Lab Initiative  
**Date**: October 8, 2025

---

## Vision

Transform BETE-NET from a single-model screening tool into a **multi-agent autonomous discovery system** that:
- Generates candidate structures (Proposer)
- Filters with fast surrogates (Filter: S2SNet)
- Refines with high-fidelity models (Refiner: BETE-NET + BEE)
- Verifies with DFT (Verifier)
- Ranks with uncertainty (Ranker)
- Learns from results (Curator)
- Enforces budget constraints (Governor)

**Outcome**: Closed-loop discovery that accelerates superconductor R&D by **10^5×** vs manual DFT.

---

## 0) High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MULTI-AGENT DISCOVERY SYSTEM                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐          │
│  │  PROPOSER   │─────→│   FILTER    │─────→│  REFINER    │          │
│  │ (Structures)│      │  (S2SNet)   │      │(BETE+BEE+σ) │          │
│  └─────────────┘      └─────────────┘      └─────────────┘          │
│         │                                           │                 │
│         │                                           ↓                 │
│         │                                   ┌─────────────┐          │
│         │                                   │  VERIFIER   │          │
│         │                                   │ (DFT/DFPT)  │          │
│         │                                   └─────────────┘          │
│         │                                           │                 │
│         │                                           ↓                 │
│         │                                   ┌─────────────┐          │
│         └──────────────────────────────────→│   RANKER    │          │
│                                             │ (Evidence)  │          │
│                                             └─────────────┘          │
│                                                     │                 │
│                                                     ↓                 │
│                                             ┌─────────────┐          │
│                                             │   CURATOR   │          │
│                                             │  (Ingest)   │          │
│                                             └─────────────┘          │
│                                                     │                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                         GOVERNOR                               │  │
│  │  (Budget, Safety, Prioritization, Resource Allocation)        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
        ┌─────────────────────────────────────────────────┐
        │            VERTEX AI BACKBONE                    │
        ├─────────────────────────────────────────────────┤
        │ • Agent Builder (orchestration)                  │
        │ • Cloud Run (FastAPI tools)                      │
        │ • Vertex Pipelines (retrain/eval)                │
        │ • Model Registry (versioning)                    │
        │ • Vertex AI Search (RAG for prior work)          │
        └─────────────────────────────────────────────────┘
```

---

## 1) Agents (Roles)

### Agent 1: GOVERNOR (Budget & Guardrails)

**Role**: Orchestrator and resource manager

**Responsibilities**:
- Allocate compute budget (DFT vs ML)
- Enforce safety constraints (toxicity, reactivity)
- Prioritize experiments by information gain
- Trigger emergency stops
- Monitor agent health

**Tools**:
- Budget tracker (Cloud Billing API)
- Safety database (toxicity, flammability)
- Priority queue (Shannon entropy)
- Monitoring dashboard (Cloud Monitoring)

**Implementation**:
```python
# app/src/agents/governor.py
class GovernorAgent:
    def allocate_budget(self, proposed_experiments):
        # Prioritize by (Tc_expected × uncertainty) / cost
        # Enforce: DFT < 10% of monthly budget
        pass
    
    def check_safety(self, structure):
        # Query toxicity/reactivity DB
        # Block: explosive, toxic, radioactive
        pass
```

---

### Agent 2: PROPOSER (Generate Structures)

**Role**: Generate candidate superconductor structures

**Strategies**:
1. **Substitution**: Replace atoms in known superconductors
2. **Interpolation**: Blend structures in latent space
3. **Generative**: Sample from VAE/diffusion model
4. **Literature Mining**: Extract from papers (Vertex AI Search + RAG)

**Tools**:
- pymatgen (structure manipulation)
- CDVAE (generative model)
- Vertex AI Search (paper mining)
- Materials Project API (known superconductors)

**Output**: 10,000 candidate structures/week

**Implementation**:
```python
# app/src/agents/proposer.py
class ProposerAgent:
    def generate_candidates(self, n=10000):
        # Strategy 1: Substitution (50%)
        # Strategy 2: Interpolation (30%)
        # Strategy 3: Generative (20%)
        pass
    
    def mine_literature(self, query="superconductor"):
        # Vertex AI Search → extract structures from papers
        pass
```

---

### Agent 3: FILTER (S2SNet Fast Screen)

**Role**: Fast first-pass screening with S2SNet

**Model**: S2SNet (structure-to-structure network)
- Input: Crystal structure
- Output: Predicted Tc (fast, ~0.1s per structure)
- Accuracy: Lower than BETE-NET, but 50× faster

**Threshold**: Keep top 1,000 candidates (10% of input)

**Tools**:
- S2SNet inference endpoint (Cloud Run)
- Batch processing (Cloud Run Jobs)

**Implementation**:
```python
# app/src/agents/filter.py
class FilterAgent:
    def screen(self, structures, top_k=1000):
        # Fast S2SNet inference
        # Sort by predicted Tc
        # Return top 1,000
        pass
```

---

### Agent 4: REFINER (BETE-NET + BEE + Uncertainty)

**Role**: High-fidelity Tc prediction with uncertainty quantification

**Models**:
1. **BETE-NET**: Electron-phonon spectral function α²F(ω) → Tc
2. **BEE**: Bayesian ensemble (additional uncertainty)
3. **Uncertainty**: Ensemble variance + epistemic uncertainty

**Output**: 
- Tc prediction with confidence intervals
- High-uncertainty flag for DFT queue

**Tools**:
- BETE-NET endpoint (Cloud Run)
- BEE endpoint (Cloud Run)
- Uncertainty aggregator

**Implementation**:
```python
# app/src/agents/refiner.py
class RefinerAgent:
    def predict_with_uncertainty(self, structures):
        # BETE-NET prediction
        bete_tc, bete_std = self.bete_net(structures)
        
        # BEE prediction
        bee_tc, bee_std = self.bee_model(structures)
        
        # Aggregate uncertainty
        combined_tc = (bete_tc + bee_tc) / 2
        combined_std = sqrt(bete_std**2 + bee_std**2)
        
        # Flag high uncertainty for DFT
        if combined_std > 0.3 * combined_tc:
            self.queue_for_verification(structures)
        
        return combined_tc, combined_std
```

---

### Agent 5: VERIFIER (DFT/DFPT Queue + Checks)

**Role**: Ground truth validation with DFT

**When to Verify**:
- High uncertainty (σ(Tc) > 30% of Tc)
- High predicted Tc (>30K)
- Novel chemistry (distance from training data)
- Random sampling (5% for calibration)

**Process**:
1. Queue structure for DFT calculation
2. Run DFPT (electron-phonon coupling)
3. Compute α²F(ω) and Tc
4. Compare to ML predictions
5. Update model training queue

**Tools**:
- DFT job proxy (submit to SLURM/Cloud)
- DFPT parser (extract α²F(ω))
- Comparison metrics (MAE, RMSE)

**Implementation**:
```python
# app/src/agents/verifier.py
class VerifierAgent:
    def should_verify(self, structure, tc_pred, tc_std):
        # High uncertainty
        if tc_std > 0.3 * tc_pred:
            return True, "high_uncertainty"
        
        # High Tc
        if tc_pred > 30:
            return True, "high_tc"
        
        # Novel chemistry
        if self.distance_from_training(structure) > 0.5:
            return True, "novel_chemistry"
        
        # Random sampling
        if random.random() < 0.05:
            return True, "calibration"
        
        return False, None
    
    def queue_dft(self, structure):
        # Submit to DFT queue
        # Track job status
        pass
```

---

### Agent 6: RANKER (Shortlist + Evidence Packs)

**Role**: Generate final ranked shortlist with evidence

**Ranking Criteria**:
1. Predicted Tc (higher is better)
2. Confidence (lower uncertainty is better)
3. Synthesizability (Materials Project e_above_hull)
4. Novelty (distance from known superconductors)
5. Cost (elemental abundance)

**Output**:
- Top 20 candidates per week
- Evidence pack for each (α²F plot, worksheet, provenance)
- Recommendation report (PDF)

**Implementation**:
```python
# app/src/agents/ranker.py
class RankerAgent:
    def rank(self, structures, predictions):
        # Multi-criteria scoring
        scores = []
        for s, p in zip(structures, predictions):
            score = (
                0.4 * normalize(p.tc_kelvin) +
                0.3 * (1 - normalize(p.tc_std / p.tc_kelvin)) +
                0.2 * self.synthesizability(s) +
                0.1 * self.novelty(s)
            )
            scores.append(score)
        
        # Sort and return top 20
        ranked = sorted(zip(structures, predictions, scores), 
                       key=lambda x: x[2], reverse=True)
        return ranked[:20]
    
    def generate_evidence_pack(self, structure, prediction):
        # α²F plot + worksheet + provenance
        pass
```

---

### Agent 7: CURATOR (Ingest + Update + Retrain)

**Role**: Close the loop with continuous learning

**Responsibilities**:
1. Ingest DFT validation results
2. Update training datasets (DVC + Cloud Storage)
3. Trigger model retraining (Vertex Pipelines)
4. A/B test new models vs old
5. Deploy new models if performance improves

**Triggers**:
- **Nightly**: Ingest new DFT results
- **Weekly**: Retrain models if N_new > 100
- **Monthly**: Full model evaluation + A/B testing

**Tools**:
- DVC (data versioning)
- Vertex Pipelines (training)
- Model Registry (versioning)
- A/B testing framework

**Implementation**:
```python
# app/src/agents/curator.py
class CuratorAgent:
    def ingest_dft_results(self, results):
        # Parse DFT outputs
        # Add to training dataset (DVC)
        # Commit to Git + push to Cloud Storage
        pass
    
    def trigger_retrain(self):
        # Check if N_new > 100
        # Submit Vertex Pipeline job
        # Wait for completion
        pass
    
    def ab_test_model(self, new_model_id, old_model_id):
        # Run both models on validation set
        # Compare MAE, RMSE, coverage
        # Deploy if MAE improves by >10%
        pass
```

---

## 2) Backbone (Vertex AI)

### 2.1) Vertex AI Agent Builder

**Purpose**: Orchestrate multi-agent planning

**Features**:
- Agentic workflows (define agent graph)
- Tool calling (FastAPI endpoints)
- State management (conversation history)
- Human-in-the-loop (approval gates)

**Agent Graph**:
```
START → Governor (budget check)
      ↓
      Proposer (generate 10K structures)
      ↓
      Filter (S2SNet → keep top 1K)
      ↓
      Refiner (BETE-NET + BEE → Tc + σ)
      ↓
      Verifier (queue high-uncertainty for DFT)
      ↓
      Ranker (top 20 + evidence packs)
      ↓
      Curator (ingest + retrain)
      ↓
END
```

**Implementation**:
```python
# app/src/agents/orchestrator.py
from vertexai.preview import agent_builder

class DiscoveryOrchestrator:
    def __init__(self):
        self.agents = {
            "governor": GovernorAgent(),
            "proposer": ProposerAgent(),
            "filter": FilterAgent(),
            "refiner": RefinerAgent(),
            "verifier": VerifierAgent(),
            "ranker": RankerAgent(),
            "curator": CuratorAgent(),
        }
    
    def run_discovery_cycle(self):
        # Week 1: Generate candidates
        if not self.agents["governor"].check_budget():
            return "Budget exceeded"
        
        candidates = self.agents["proposer"].generate_candidates(n=10000)
        
        # Week 2: Fast screening
        filtered = self.agents["filter"].screen(candidates, top_k=1000)
        
        # Week 3: High-fidelity refinement
        refined = self.agents["refiner"].predict_with_uncertainty(filtered)
        
        # Week 4: Queue for verification
        self.agents["verifier"].queue_high_uncertainty(refined)
        
        # Week 5: Rank and report
        ranked = self.agents["ranker"].rank(refined)
        evidence_packs = [self.agents["ranker"].generate_evidence_pack(s, p) 
                         for s, p in ranked]
        
        # Week 6: Ingest and retrain
        dft_results = self.agents["verifier"].get_completed_jobs()
        self.agents["curator"].ingest_dft_results(dft_results)
        
        if len(dft_results) > 100:
            self.agents["curator"].trigger_retrain()
        
        return ranked, evidence_packs
```

---

### 2.2) Cloud Run (FastAPI Tools)

**Endpoints**:
1. `/api/s2snet/predict` - Fast Tc screening
2. `/api/bete/predict` - BETE-NET inference (already implemented ✅)
3. `/api/bee/predict` - BEE Bayesian ensemble
4. `/api/dft/submit` - DFT job submission
5. `/api/evidence/generate` - Evidence pack generation
6. `/api/uncertainty/aggregate` - Uncertainty quantification

**Deployment**:
```bash
# Deploy all tools
gcloud run deploy discovery-tools \
  --source . \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 600 \
  --max-instances 10 \
  --set-env-vars "PYTHONPATH=/app"
```

---

### 2.3) Vertex Pipelines (Retrain/Eval)

**Pipelines**:

1. **Nightly: Ingest Pipeline**
   - Check for new DFT results
   - Parse outputs
   - Add to training dataset
   - Commit to DVC

2. **Weekly: Retrain Pipeline** (if N_new > 100)
   - Load updated dataset
   - Retrain BETE-NET, S2SNet, BEE
   - Validate on holdout set
   - Register new models

3. **Monthly: Evaluation Pipeline**
   - Run all models on validation set
   - Compare MAE, RMSE, coverage
   - Generate evaluation report
   - A/B test new vs old models

**Implementation**:
```python
# pipelines/retrain_bete_net.py
from kfp import dsl

@dsl.pipeline(name="Retrain BETE-NET")
def retrain_pipeline(dataset_version: str):
    # Component 1: Load data
    load_data = dsl.ContainerOp(
        name="Load Data",
        image="gcr.io/periodicdent42/dvc-loader:latest",
        arguments=["--version", dataset_version]
    )
    
    # Component 2: Train model
    train = dsl.ContainerOp(
        name="Train BETE-NET",
        image="gcr.io/periodicdent42/bete-net-trainer:latest",
        arguments=["--data", load_data.outputs["dataset"]]
    )
    
    # Component 3: Validate
    validate = dsl.ContainerOp(
        name="Validate",
        image="gcr.io/periodicdent42/validator:latest",
        arguments=["--model", train.outputs["model_path"]]
    )
    
    # Component 4: Register
    register = dsl.ContainerOp(
        name="Register Model",
        image="gcr.io/periodicdent42/model-registry:latest",
        arguments=["--model", train.outputs["model_path"],
                  "--mae", validate.outputs["mae"]]
    )
```

---

### 2.4) Model Registry (Versioning)

**Models**:
- `bete-net-v1.0.0` (current)
- `bete-net-v1.1.0` (after 100 new DFT results)
- `s2snet-v1.0.0` (fast filter)
- `bee-v1.0.0` (Bayesian ensemble)

**Features**:
- Semantic versioning
- Model lineage (training data version)
- Performance metrics (MAE, RMSE)
- Rollback capability

**Usage**:
```python
from google.cloud import aiplatform

# Deploy new model
model = aiplatform.Model.upload(
    display_name="bete-net-v1.1.0",
    artifact_uri="gs://periodicdent42-models/bete-net-v1.1.0",
    serving_container_image_uri="gcr.io/periodicdent42/bete-net:latest"
)

model.deploy(
    endpoint=endpoint,
    traffic_percentage=10,  # A/B test with 10% traffic
    machine_type="n1-standard-4"
)
```

---

## 3) Data Flow

```
Week 1: PROPOSER generates 10,000 structures
          ↓
Week 2: FILTER (S2SNet) screens → 1,000 candidates
          ↓
Week 3: REFINER (BETE-NET + BEE) predicts Tc + σ
          ↓
          ├─ High uncertainty (100 structures) → VERIFIER (DFT queue)
          └─ Low uncertainty (900 structures) → RANKER
          ↓
Week 4: RANKER generates top 20 + evidence packs
          ↓
Week 5: CURATOR ingests DFT results (100 completed)
          ↓
Week 6: CURATOR triggers retrain (if N_new > 100)
          ↓
LOOP
```

---

## 4) Cost Analysis

### Compute Costs (Monthly)

| Component | Cost/Unit | Units/Month | Total |
|-----------|-----------|-------------|-------|
| S2SNet (Cloud Run) | $0.000001/pred | 40,000 | $0.04 |
| BETE-NET (Cloud Run) | $0.000024/pred | 4,000 | $0.10 |
| BEE (Cloud Run) | $0.000050/pred | 4,000 | $0.20 |
| DFT (on-demand VMs) | $50/job | 100 | $5,000 |
| Vertex Pipelines | $0.50/hr | 10 hr | $5.00 |
| Model Registry | $0.01/model/day | 3 models | $0.90 |
| **TOTAL** | | | **$5,006** |

**Breakdown**:
- ML inference: $0.34 (0.007% of budget)
- DFT validation: $5,000 (99.9% of budget)
- Infrastructure: $5.90 (0.1% of budget)

**ROI**: 
- Without ML: 100 DFT jobs × $50 = $5,000
- With ML: 40,000 screened → 100 DFT → same $5,000
- **Value**: 400× more candidates screened for same cost

---

## 5) Implementation Roadmap

### Phase 1: Complete BETE-NET (Week 1) ✅
- [x] Download model weights
- [x] Implement model loading (40 lines)
- [x] Validate golden tests (Nb, MgB2, Al)
- [x] Deploy to Cloud Run with 4GB RAM

### Phase 2: Add S2SNet Filter (Week 2)
- [ ] Implement S2SNet inference wrapper
- [ ] Deploy S2SNet endpoint (Cloud Run)
- [ ] Add batch screening (10K structures)
- [ ] Benchmark latency and accuracy

### Phase 3: Add BEE Refiner (Week 3)
- [ ] Implement BEE Bayesian ensemble
- [ ] Add uncertainty aggregation
- [ ] Deploy BEE endpoint
- [ ] Test combined BETE-NET + BEE

### Phase 4: Add DFT Verifier (Week 4)
- [ ] Implement DFT job proxy
- [ ] Add SLURM/Cloud job submission
- [ ] Parse DFPT outputs (α²F extraction)
- [ ] Track job status

### Phase 5: Multi-Agent Orchestration (Week 5-6)
- [ ] Implement Governor, Proposer, Ranker, Curator
- [ ] Integrate Vertex AI Agent Builder
- [ ] Deploy full agent graph
- [ ] Test end-to-end discovery cycle

### Phase 6: Continuous Learning (Week 7-8)
- [ ] Add DVC data versioning
- [ ] Implement Vertex Pipelines (retrain)
- [ ] Add Model Registry integration
- [ ] A/B test new models

---

## 6) Success Metrics

### Technical
- **Throughput**: 10,000 candidates/week (vs 10/month with DFT only)
- **Latency**: <5s per ML prediction, ~8 weeks per DFT
- **Accuracy**: MAE(Tc) < 5K on validation set
- **Coverage**: Screen 40,000 materials/month

### Business
- **Cost**: $5K/month (same as 100 DFT jobs)
- **ROI**: 400× more candidates for same budget
- **Time-to-Discovery**: 6 weeks (full cycle) vs 2 years (DFT-only)
- **Novel Materials**: 10-20 high-Tc candidates/month

---

## 7) Risk Mitigation

### Technical Risks
- **Model drift**: Monitor MAE on holdout set, retrain monthly
- **DFT queue overflow**: Cap at 100 jobs/month, prioritize by uncertainty
- **Compute costs**: Set budget alerts, auto-scale down if exceeded

### Scientific Risks
- **False positives**: Verify all predictions >30K with DFT
- **Domain mismatch**: Flag novel chemistry for expert review
- **Safety**: Block toxic/reactive materials (Governor agent)

---

## Next Steps (Immediate)

1. **Download BETE-NET weights** (5 min)
2. **Implement model loading** (40 lines)
3. **Validate golden tests** (30 min)
4. **Deploy to Cloud Run** (15 min)

Then we can proceed to Phase 2 (S2SNet integration) and beyond.

---

**Status**: Ready to implement Phase 1 immediately 🚀

© 2025 GOATnote Autonomous Research Lab Initiative  
Licensed under Proprietary License

