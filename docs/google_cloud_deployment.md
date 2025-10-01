# Google Cloud Deployment Guide
## Autonomous R&D Intelligence Layer - October 2025

**Last Updated**: October 1, 2025  
**Gemini Models**: 2.5 Pro & 2.5 Flash (Latest)

---

## Executive Summary

This guide provides production-ready architecture for deploying the Autonomous R&D Intelligence Layer on Google Cloud Platform, leveraging **dual-model AI reasoning** with Gemini 2.5 Flash (fast preliminary feedback) and Gemini 2.5 Pro (accurate, high-integrity analysis).

### Key Architecture Decisions

- **Dual-Model Pattern**: Flash for instant user feedback (<2s), Pro for verified scientific reasoning (10-30s)
- **Vertex AI**: Unified ML platform for model deployment, monitoring, and scaling
- **Google Distributed Cloud (GDC)**: On-premises option for sensitive data compliance
- **Model Context Protocol (MCP)**: Seamless integration with open-source tools
- **AI Hypercomputer**: TPU/GPU infrastructure for intensive compute

---

## 1. Dual-Model AI Architecture

### Pattern: Parallel Fast + Accurate Reasoning

```
User Query
    ↓
    ├──────────────────────────────────────────┐
    ↓ (Parallel)                               ↓ (Parallel)
┌─────────────────┐                  ┌──────────────────┐
│ Gemini 2.5 Flash│                  │ Gemini 2.5 Pro   │
│ - Speed: <2s    │                  │ - Speed: 10-30s  │
│ - Cost: Low     │                  │ - Accuracy: High │
│ - Use: Preview  │                  │ - Use: Final     │
└─────────────────┘                  └──────────────────┘
    ↓                                         ↓
    ↓                                         ↓
User sees instant                   Pro result replaces
preliminary result                  Flash when ready
(with "Computing..." badge)         (with verification)
```

### Implementation Strategy

#### Phase 1: Instant Feedback (Gemini 2.5 Flash)
- **Purpose**: Provide immediate UI response, preliminary experiment suggestions
- **Latency**: <2 seconds
- **Use Cases**:
  - Initial experiment parameter suggestions
  - Quick literature search summaries
  - Preliminary EIG estimates
  - Natural language query understanding
  - Real-time dashboard updates

#### Phase 2: Verified Response (Gemini 2.5 Pro)
- **Purpose**: High-accuracy scientific reasoning with audit trail
- **Latency**: 10-30 seconds
- **Use Cases**:
  - Final EIG calculations with uncertainty quantification
  - Safety policy validation and explanation
  - Complex experimental design optimization
  - Multi-step reasoning for research hypotheses
  - Publication-quality report generation

### Code Example: Dual-Model Pattern

```python
# src/reasoning/gemini_dual_agent.py
import asyncio
from google.cloud import aiplatform
from typing import Dict, Any, Tuple

class DualModelAgent:
    """
    Parallel Fast + Accurate AI reasoning with Gemini 2.5 Flash and Pro.
    
    Moat: INTERPRETABILITY + TIME - Instant feedback with verified accuracy.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Gemini 2.5 Flash: Fast, cost-effective
        self.flash_model = aiplatform.GenerativeModel("gemini-2.5-flash")
        
        # Gemini 2.5 Pro: Accurate, high-reasoning
        self.pro_model = aiplatform.GenerativeModel("gemini-2.5-pro")
    
    async def query_parallel(
        self, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute both models in parallel.
        
        Returns:
            (flash_response, pro_response) - Flash completes first, Pro follows
        """
        # Launch both models simultaneously
        flash_task = asyncio.create_task(self._query_flash(prompt, context))
        pro_task = asyncio.create_task(self._query_pro(prompt, context))
        
        # Return Flash immediately for UI update
        flash_response = await flash_task
        
        # Pro completes in background, UI updates when ready
        pro_response = await pro_task
        
        return flash_response, pro_response
    
    async def _query_flash(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fast preliminary response."""
        
        # Enhanced prompt with context
        enhanced_prompt = f"""
        Context: {context}
        
        Task: {prompt}
        
        Provide a QUICK preliminary analysis suitable for immediate user feedback.
        Note: This is a fast preview; a more detailed analysis will follow.
        """
        
        response = await self.flash_model.generate_content_async(
            enhanced_prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "top_p": 0.9,
            }
        )
        
        return {
            "model": "gemini-2.5-flash",
            "content": response.text,
            "latency_ms": response.metadata.get("latency_ms", 0),
            "is_preliminary": True,
            "confidence": "medium"
        }
    
    async def _query_pro(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Accurate, verified response with reasoning."""
        
        # Enhanced prompt with scientific rigor
        enhanced_prompt = f"""
        Context: {context}
        
        Task: {prompt}
        
        Provide a COMPREHENSIVE, scientifically rigorous analysis with:
        1. Step-by-step reasoning
        2. Confidence intervals and uncertainty quantification
        3. Citations to domain knowledge (if applicable)
        4. Alternative approaches considered
        5. Safety considerations
        
        This is the FINAL verified response for scientific use.
        """
        
        response = await self.pro_model.generate_content_async(
            enhanced_prompt,
            generation_config={
                "temperature": 0.2,  # Lower temp for consistency
                "max_output_tokens": 8192,  # More detailed
                "top_p": 0.95,
            }
        )
        
        return {
            "model": "gemini-2.5-pro",
            "content": response.text,
            "latency_ms": response.metadata.get("latency_ms", 0),
            "is_preliminary": False,
            "confidence": "high",
            "reasoning_steps": self._extract_reasoning(response.text)
        }
    
    def _extract_reasoning(self, text: str) -> list:
        """Extract reasoning steps for audit trail."""
        # Parse structured reasoning from Pro output
        # In production, use structured output format
        return text.split("\n\n")


# Example usage in FastAPI endpoint
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse

app = FastAPI()
agent = DualModelAgent(project_id="periodicdent42")

@app.post("/api/reasoning/query")
async def query_with_feedback(query: str, context: dict, background_tasks: BackgroundTasks):
    """
    Endpoint that returns Flash immediately, Pro via SSE.
    """
    # Get Flash response (fast)
    flash_response, _ = await agent.query_parallel(query, context)
    
    # Return Flash immediately
    yield {
        "status": "preliminary",
        "response": flash_response,
        "message": "Computing verified response..."
    }
    
    # Pro response streams in when ready
    _, pro_response = await agent.query_parallel(query, context)
    
    yield {
        "status": "final",
        "response": pro_response,
        "message": "Verified response ready"
    }
```

---

## 2. Google Cloud Infrastructure

### 2.1 Vertex AI Deployment

**Service**: Vertex AI  
**Purpose**: Managed ML platform for Gemini models, custom models, and MLOps

#### Setup Commands

```bash
# Install Google Cloud SDK and Vertex AI
pip install google-cloud-aiplatform

# Initialize project
gcloud init
gcloud config set project periodicdent42

# Enable APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com

# Authenticate
gcloud auth application-default login
```

#### Vertex AI Configuration

```python
# configs/vertex_ai_config.py
from google.cloud import aiplatform

VERTEX_AI_CONFIG = {
    "project_id": "periodicdent42",
    "location": "us-central1",  # Choose based on data residency
    
    # Gemini Models
    "models": {
        "flash": "gemini-2.5-flash",      # Fast, cost-effective
        "pro": "gemini-2.5-pro",          # Accurate, high-reasoning
    },
    
    # Prediction settings
    "prediction_config": {
        "machine_type": "n1-standard-4",
        "accelerator_type": None,  # Gemini doesn't need custom accelerators
        "min_replica_count": 1,
        "max_replica_count": 10,
    },
    
    # Context window
    "max_input_tokens": 1_048_576,  # Gemini 2.5 Pro supports 1M+ tokens
    
    # Cost optimization
    "enable_caching": True,  # Cache frequent prompts
    "use_flash_for_preview": True,
}

def initialize_vertex_ai():
    """Initialize Vertex AI with project settings."""
    aiplatform.init(
        project=VERTEX_AI_CONFIG["project_id"],
        location=VERTEX_AI_CONFIG["location"]
    )
```

---

### 2.2 Cloud Run Deployment (Serverless FastAPI)

**Service**: Cloud Run  
**Purpose**: Serverless container platform for FastAPI backend

#### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ /app/src/
COPY configs/ /app/configs/
WORKDIR /app

# Expose port
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Deploy to Cloud Run

```bash
# Build and push container
gcloud builds submit --tag gcr.io/periodicdent42/ard-backend

# Deploy to Cloud Run
gcloud run deploy ard-backend \
  --image gcr.io/periodicdent42/ard-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars "PROJECT_ID=periodicdent42,LOCATION=us-central1"
```

---

### 2.3 Cloud Storage & Database

#### Cloud Storage (Data Lake)

```python
# src/memory/cloud_storage.py
from google.cloud import storage

class CloudStorageBackend:
    """Store experiment results in Cloud Storage."""
    
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def store_result(self, experiment_id: str, result: dict):
        """Store experiment result with provenance."""
        blob_name = f"experiments/{experiment_id}/result.json"
        blob = self.bucket.blob(blob_name)
        
        blob.upload_from_string(
            json.dumps(result),
            content_type="application/json"
        )
        
        # Add metadata
        blob.metadata = {
            "experiment_id": experiment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "provenance_hash": result["provenance_hash"]
        }
        blob.patch()
```

#### Cloud SQL (PostgreSQL + TimescaleDB)

```bash
# Create Cloud SQL instance with TimescaleDB
gcloud sql instances create ard-postgres \
  --database-version=POSTGRES_15 \
  --tier=db-n1-standard-4 \
  --region=us-central1 \
  --database-flags=shared_preload_libraries=timescaledb

# Enable TimescaleDB extension
gcloud sql connect ard-postgres --user=postgres
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

---

### 2.4 AI Hypercomputer (Intensive Compute)

**Service**: AI Hypercomputer (TPUs + GPUs)  
**Purpose**: Train RL agents, run DFT/MD simulations at scale

#### Use Cases for AI Hypercomputer

1. **RL Agent Training** (Phase 2): Train PPO/SAC agents on TPU v5e
2. **DFT Batch Processing**: Parallel PySCF calculations on GPUs
3. **Surrogate Model Training**: GP/Neural network training at scale

#### Setup TPU Training

```python
# scripts/train_rl_agent_tpu.py
import jax
from google.cloud import aiplatform

def train_on_tpu():
    """Train RL agent on TPU v5e."""
    
    # TPU configuration
    tpu_config = {
        "accelerator_type": "TPU_V5_LITEPOD",
        "topology": "2x2x1",  # 4 TPU chips
    }
    
    # Launch training job
    job = aiplatform.CustomTrainingJob(
        display_name="rl-agent-training",
        container_uri="gcr.io/periodicdent42/rl-trainer",
        model_serving_container_image_uri="gcr.io/periodicdent42/rl-server",
    )
    
    job.run(
        replica_count=1,
        machine_type="ct5lp-hightpu-4t",
        accelerator_type="TPU_V5_LITEPOD",
        accelerator_count=4,
        args=["--epochs", "1000", "--batch_size", "256"]
    )
```

---

## 3. Google Distributed Cloud (On-Premises)

**Service**: Google Distributed Cloud (GDC)  
**Purpose**: Run Gemini models on-premises for data residency, compliance, and security

### When to Use GDC

✅ **Regulatory compliance** (HIPAA, GDPR, export control)  
✅ **Sensitive data** (proprietary formulations, patents)  
✅ **Air-gapped environments** (defense, high-security labs)  
✅ **Low-latency requirements** (real-time instrument control)

### GDC Architecture

```
┌──────────────────────────────────────────────┐
│        On-Premises Lab Infrastructure        │
├──────────────────────────────────────────────┤
│  Google Distributed Cloud (GDC)              │
│  ├─ Gemini 2.5 Pro/Flash (local)             │
│  ├─ Vertex AI SDK (local)                    │
│  ├─ Cloud Storage (local cache)              │
│  └─ NVIDIA Blackwell GPUs                    │
├──────────────────────────────────────────────┤
│  Instruments (XRD, NMR, Synthesis Robots)    │
│  - Data never leaves premises                │
│  - Sub-millisecond latency to instruments    │
└──────────────────────────────────────────────┘
         ↕ (Optional sync to Google Cloud)
┌──────────────────────────────────────────────┐
│  Google Cloud (Public)                       │
│  - Federated learning aggregation            │
│  - Non-sensitive data analytics              │
│  - Model updates and monitoring              │
└──────────────────────────────────────────────┘
```

### GDC Setup

1. **Hardware**: NVIDIA Blackwell systems with GDC software stack
2. **Installation**: Google professional services deploy GDC rack
3. **Configuration**: Air-gapped or hybrid cloud connectivity
4. **Models**: Gemini 2.5 Pro/Flash run locally with full capabilities

---

## 4. Model Context Protocol (MCP) Integration

**Purpose**: Seamless integration with open-source tools, databases, and APIs

### MCP Benefits

- Connect Gemini models to external data sources (PySCF, RDKit, lab instruments)
- Function calling with automatic tool selection
- Streaming responses for real-time updates
- Context caching for cost efficiency

### Example: MCP with Lab Instruments

```python
# src/reasoning/mcp_integration.py
from google.cloud import aiplatform
from typing import Dict, Any

class MCPAgent:
    """
    Model Context Protocol agent for connecting Gemini to lab tools.
    """
    
    def __init__(self):
        self.model = aiplatform.GenerativeModel(
            "gemini-2.5-pro",
            tools=[
                self._define_xrd_tool(),
                self._define_dft_tool(),
                self._define_eig_tool(),
            ]
        )
    
    def _define_xrd_tool(self):
        """Define XRD instrument as a tool."""
        return {
            "function_declarations": [{
                "name": "run_xrd_experiment",
                "description": "Run XRD diffraction scan on sample",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sample_id": {"type": "string"},
                        "scan_range": {"type": "string"},
                        "step_size": {"type": "number"}
                    },
                    "required": ["sample_id", "scan_range"]
                }
            }]
        }
    
    async def query_with_tools(self, prompt: str) -> Dict[str, Any]:
        """
        Query Gemini with automatic tool calling.
        
        Gemini decides which tools to call based on context.
        """
        response = await self.model.generate_content_async(prompt)
        
        # Handle function calls
        if response.candidates[0].content.parts[0].function_call:
            func_call = response.candidates[0].content.parts[0].function_call
            
            # Execute tool (e.g., run XRD experiment)
            result = await self._execute_tool(func_call)
            
            # Return result to Gemini for interpretation
            final_response = await self.model.generate_content_async([
                prompt,
                response,
                {"function_response": result}
            ])
            
            return final_response.text
        
        return response.text
```

---

## 5. Cost Optimization

### Pricing (Approximate as of Oct 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Use Case |
|-------|----------------------|------------------------|----------|
| Gemini 2.5 Flash | $0.075 | $0.30 | Preliminary feedback |
| Gemini 2.5 Pro | $1.25 | $5.00 | Final analysis |

### Optimization Strategies

1. **Context Caching**: Cache frequent prompts (e.g., safety policies, domain knowledge)
2. **Flash First**: Use Flash for 90% of queries, Pro for 10% (critical decisions)
3. **Batch Processing**: Group multiple experiments into single API call
4. **Prompt Engineering**: Shorter, more focused prompts reduce token usage

### Example: Context Caching

```python
# Cache domain knowledge for reuse
from google.cloud import aiplatform

# Create cached context (persists for 1 hour)
cached_context = aiplatform.CachedContent.create(
    model_name="gemini-2.5-pro",
    system_instruction="You are an expert in materials science...",
    contents=[
        "# Domain Knowledge\n",
        "Perovskites: ABX3 structure...",
        "DFT calculations: Kohn-Sham equations...",
        # ... 500K tokens of domain knowledge
    ],
    ttl=3600,  # 1 hour
)

# Subsequent queries use cached context (90% cost reduction)
model = aiplatform.GenerativeModel("gemini-2.5-pro", cached_content=cached_context)
response = model.generate_content("What is the bandgap of BaTiO3?")
```

---

## 6. Security & Compliance

### Identity and Access Management (IAM)

```bash
# Grant Vertex AI access to service account
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="serviceAccount:ard-backend@periodicdent42.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Least privilege principle
# - Researchers: aiplatform.user (query models)
# - Admins: aiplatform.admin (deploy models)
# - Auditors: logging.viewer (read audit logs)
```

### Data Encryption

- **At rest**: Cloud Storage and Cloud SQL use AES-256 encryption by default
- **In transit**: TLS 1.3 for all API calls
- **Customer-managed keys**: Use Cloud KMS for additional control

### Audit Logging

```python
# Enable Cloud Audit Logs
from google.cloud import logging

client = logging.Client()
logger = client.logger("ard-audit-log")

# Log all Gemini API calls
logger.log_struct({
    "event": "gemini_api_call",
    "model": "gemini-2.5-pro",
    "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
    "user_id": "alice",
    "timestamp": datetime.utcnow().isoformat(),
    "experiment_id": "exp-12345"
})
```

---

## 7. Monitoring & Observability

### Cloud Monitoring (formerly Stackdriver)

```python
# src/monitoring/metrics.py
from google.cloud import monitoring_v3

def create_custom_metric(project_id: str):
    """Create custom metric for EIG/hour."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    descriptor = monitoring_v3.MetricDescriptor(
        type="custom.googleapis.com/ard/eig_per_hour",
        metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
        value_type=monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
        description="Expected Information Gain per hour",
    )
    
    client.create_metric_descriptor(name=project_name, metric_descriptor=descriptor)
```

### Vertex AI Model Monitoring

```python
# Enable automatic monitoring
from google.cloud import aiplatform

monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="gemini-monitoring",
    endpoint=endpoint,
    logging_sampling_strategy={"random_sample_config": {"sample_rate": 0.1}},
    schedule_config={"monitor_interval": 3600},  # Every hour
    alert_config={"email_alert_config": {"user_emails": ["team@example.com"]}}
)
```

---

## 8. Deployment Checklist

### Pre-Deployment

- [ ] GCP project created and billing enabled
- [ ] Vertex AI API enabled
- [ ] Service accounts configured with least privilege
- [ ] VPC network configured (if using private endpoints)
- [ ] Cloud Storage buckets created for data lake
- [ ] Cloud SQL instance provisioned with TimescaleDB

### Phase 0-1 Deployment (Week 1-8)

- [ ] Deploy FastAPI backend to Cloud Run
- [ ] Configure Gemini 2.5 Flash + Pro dual-model pattern
- [ ] Set up Cloud Storage for experiment results
- [ ] Enable Cloud Monitoring with custom metrics
- [ ] Test end-to-end: Query → Flash → Pro → Result

### Phase 2-3 Deployment (Month 3-6)

- [ ] Deploy RL training pipeline to AI Hypercomputer
- [ ] Integrate real instruments via Cloud IoT Core
- [ ] Set up GDC for on-premises sensitive data (if needed)
- [ ] Deploy Next.js UI to Cloud Run
- [ ] Enable federated learning across multiple labs

### Production Readiness

- [ ] Load testing (1000+ concurrent users)
- [ ] Disaster recovery plan (backup/restore)
- [ ] Security audit (penetration testing)
- [ ] Cost budget alerts configured
- [ ] On-call rotation established

---

## 9. Migration Path from Current Setup

### Current State (Local Development)
```
Local Machine
├── Python 3.12 venv
├── PostgreSQL (local)
├── Dummy instruments
└── pytest (local tests)
```

### Target State (Google Cloud)
```
Google Cloud
├── Cloud Run (FastAPI backend)
├── Vertex AI (Gemini 2.5 Flash + Pro)
├── Cloud SQL (PostgreSQL + TimescaleDB)
├── Cloud Storage (Data lake)
├── AI Hypercomputer (RL training)
└── Cloud Monitoring (Observability)
```

### Migration Steps

1. **Week 1**: Deploy FastAPI to Cloud Run (no Gemini yet)
2. **Week 2**: Integrate Gemini 2.5 Flash for preliminary feedback
3. **Week 3**: Add Gemini 2.5 Pro for verified responses
4. **Week 4**: Migrate PostgreSQL to Cloud SQL
5. **Week 5**: Set up Cloud Storage data lake
6. **Week 6-8**: Full integration testing and optimization

---

## 10. Cost Estimate

### Monthly Cost Breakdown (Phase 0-1)

| Service | Configuration | Monthly Cost |
|---------|--------------|-------------|
| Cloud Run (Backend) | 2 vCPU, 2GB RAM, 10M requests | $50 |
| Gemini 2.5 Flash | 100M input tokens, 20M output | $13.50 |
| Gemini 2.5 Pro | 10M input tokens, 2M output | $22.50 |
| Cloud SQL | db-n1-standard-4, 100GB SSD | $200 |
| Cloud Storage | 1TB data, 10K operations | $25 |
| Cloud Monitoring | Custom metrics, logs | $10 |
| **Total** | | **~$321/month** |

### Scale-Up (Phase 3-5 Production)

| Service | Configuration | Monthly Cost |
|---------|--------------|-------------|
| Cloud Run (Backend) | Auto-scale 1-10 instances | $500 |
| Gemini API (Flash + Pro) | 1B tokens/month | $200 |
| AI Hypercomputer | 4x TPU v5e, 100 hours/month | $2,000 |
| Cloud SQL | High availability, 1TB | $800 |
| Cloud Storage | 10TB data, 100K operations | $200 |
| **Total** | | **~$3,700/month** |

---

## 11. Next Steps

### Immediate (This Week)
1. Create GCP project: `gcloud projects create periodicdent42`
2. Enable billing and APIs
3. Deploy FastAPI backend to Cloud Run
4. Test Gemini 2.5 Flash integration

### Short-Term (Weeks 2-4)
1. Implement dual-model pattern (Flash + Pro)
2. Migrate data to Cloud SQL and Cloud Storage
3. Set up monitoring and alerts
4. Load test with 100 concurrent experiments

### Long-Term (Months 3-12)
1. Deploy RL training to AI Hypercomputer
2. Integrate GDC for sensitive data (if needed)
3. Multi-lab federated learning deployment
4. Scale to 1000+ experiments/day

---

## Resources

### Documentation
- [Vertex AI Gemini 2.5 Pro](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro)
- [Model Context Protocol](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling)
- [Google Distributed Cloud](https://cloud.google.com/distributed-cloud)
- [AI Hypercomputer](https://cloud.google.com/ai-hypercomputer)

### Training
- [Google Cloud Skills Boost](https://www.cloudskillsboost.google/)
- [Vertex AI Tutorials](https://cloud.google.com/vertex-ai/docs/tutorials)

### Support
- Google Cloud Support (Premium tier recommended for production)
- Vertex AI Community Forums
- Stack Overflow: `google-cloud-aiplatform`

---

**Document Version**: 1.0  
**Last Updated**: October 1, 2025  
**Next Review**: January 1, 2026

