# 🚀 READY TO RUN - Complete Command Guide

**All files created! Here's what to run next.**

---

## ✅ Files Created (30+ files)

### Application Code
- ✅ `app/src/api/main.py` - FastAPI app with SSE streaming
- ✅ `app/src/reasoning/dual_agent.py` - Gemini 2.5 Flash + Pro parallel execution
- ✅ `app/src/reasoning/mcp_agent.py` - Model Context Protocol integration
- ✅ `app/src/services/vertex.py` - Vertex AI wrapper
- ✅ `app/src/services/storage.py` - Cloud Storage backend
- ✅ `app/src/services/db.py` - PostgreSQL/Cloud SQL
- ✅ `app/src/monitoring/metrics.py` - Cloud Monitoring
- ✅ `app/src/utils/settings.py` - Configuration management
- ✅ `app/src/utils/sse.py` - Server-Sent Events helpers

### Infrastructure
- ✅ `app/Dockerfile` - Multi-stage production container
- ✅ `app/Makefile` - Dev, test, build, deploy targets
- ✅ `app/requirements.txt` - Python dependencies
- ✅ `infra/scripts/enable_apis.sh` - Enable GCP APIs
- ✅ `infra/scripts/setup_iam.sh` - IAM configuration
- ✅ `infra/scripts/deploy_cloudrun.sh` - Cloud Run deployment

### Tests
- ✅ `tests/test_health.py` - Health check tests
- ✅ `tests/test_reasoning_smoke.py` - SSE streaming tests

### CI/CD
- ✅ `.github/workflows/cicd.yaml` - GitHub Actions pipeline

### Documentation
- ✅ `DEPLOYMENT_GUIDE.md` - Step-by-step deployment
- ✅ `COMMANDS_TO_RUN.md` - This file!
- ✅ `quickdeploy.sh` - One-command deployment

---

## 🎯 Quick Start (Choose One)

### Option A: One-Command Deployment

```bash
cd /Users/kiteboard/periodicdent42
bash quickdeploy.sh
```

**This will**:
1. Enable all GCP APIs
2. Setup IAM permissions
3. Run tests
4. Build Docker image
5. Deploy to Cloud Run
6. Print service URL

⏱️ **Time**: ~10 minutes

---

### Option B: Step-by-Step (Recommended for first time)

#### 1. Local Development & Testing

```bash
# Navigate to app directory
cd /Users/kiteboard/periodicdent42/app

# Install dependencies
python3 -m pip install -r requirements.txt

# Create .env file
cat > .env << 'EOF'
PROJECT_ID=periodicdent42
LOCATION=us-central1
ENVIRONMENT=development
GEMINI_FLASH_MODEL=gemini-2.5-flash
GEMINI_PRO_MODEL=gemini-2.5-pro
PORT=8080
LOG_LEVEL=INFO
EOF

# Run tests (should work without GCP credentials)
python3 -m pytest -v

# Start development server (requires GCP auth)
make dev

# In another terminal, test endpoints
curl http://localhost:8080/healthz
curl -N -X POST http://localhost:8080/api/reasoning/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Test query","context":{}}'
```

#### 2. GCP Setup

```bash
# Set your GCP project
gcloud config set project periodicdent42

# Authenticate
gcloud auth login
gcloud auth application-default login

# Enable APIs (takes ~2 minutes)
bash infra/scripts/enable_apis.sh

# Setup IAM (creates service account + permissions)
bash infra/scripts/setup_iam.sh
```

#### 3. Build & Deploy

```bash
cd /Users/kiteboard/periodicdent42/app

# Build image with Cloud Build (~3 minutes)
gcloud builds submit --tag gcr.io/periodicdent42/ard-backend

# Deploy to Cloud Run (~2 minutes)
export PROJECT_ID=periodicdent42
export REGION=us-central1
bash ../infra/scripts/deploy_cloudrun.sh

# Get service URL
SERVICE_URL=$(gcloud run services describe ard-backend \
  --region us-central1 \
  --format 'value(status.url)')

echo "Service URL: $SERVICE_URL"
```

#### 4. Test Deployed Service

```bash
# Health check (requires auth token)
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $SERVICE_URL/healthz

# Expected output:
# {"status":"ok","vertex_initialized":true,"project_id":"periodicdent42"}

# Test reasoning endpoint with SSE streaming
curl -N -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -X POST $SERVICE_URL/api/reasoning/query \
  -d '{
    "query": "Suggest experiments for optimizing BaTiO3 bandgap",
    "context": {"domain": "materials_science", "budget": 1000}
  }'

# You should see two SSE events:
# event: preliminary  (Flash response <2s)
# event: final        (Pro response 10-30s)
```

---

## 🧪 Test Commands

### Local Tests

```bash
cd /Users/kiteboard/periodicdent42/app

# Run all tests
python3 -m pytest -v

# Run specific test
python3 -m pytest tests/test_health.py -v

# Run with coverage
python3 -m pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
```

### Deployed Service Tests

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe ard-backend \
  --region us-central1 \
  --format 'value(status.url)')

# Test health
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $SERVICE_URL/healthz

# Test API docs
open "$SERVICE_URL/docs"  # Opens Swagger UI

# View logs
gcloud run services logs tail ard-backend --region=us-central1

# Watch logs in real-time
gcloud run services logs tail ard-backend --region=us-central1 --follow
```

---

## 🔍 Verification Checklist

After deployment, verify:

```bash
# ✅ Service is running
gcloud run services describe ard-backend --region=us-central1

# ✅ Health check passes
curl -s -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $(gcloud run services describe ard-backend --region=us-central1 --format='value(status.url)')/healthz \
  | jq .

# ✅ Vertex AI initialized
# Should see: "vertex_initialized": true

# ✅ Logs are flowing
gcloud run services logs read ard-backend --region=us-central1 --limit=10

# ✅ Service account has permissions
gcloud projects get-iam-policy periodicdent42 \
  --flatten="bindings[].members" \
  --filter="bindings.members:ard-backend@periodicdent42.iam.gserviceaccount.com"

# ✅ APIs are enabled
gcloud services list --enabled | grep -E "aiplatform|run|storage|monitoring"
```

---

## 🐛 Troubleshooting Commands

### If tests fail locally

```bash
# Install test dependencies
cd /Users/kiteboard/periodicdent42/app
pip install pytest pytest-asyncio httpx

# Set PYTHONPATH
export PYTHONPATH=/Users/kiteboard/periodicdent42/app:$PYTHONPATH

# Run with verbose output
pytest -vv --tb=long
```

### If deployment fails

```bash
# Check Cloud Build logs
gcloud builds list --limit=5

# Get specific build ID
BUILD_ID=$(gcloud builds list --limit=1 --format='value(id)')
gcloud builds log $BUILD_ID

# Check IAM permissions
gcloud projects get-iam-policy periodicdent42 | grep ard-backend

# Re-run IAM setup
bash infra/scripts/setup_iam.sh

# Check quotas
gcloud compute project-info describe --project=periodicdent42
```

### If service won't start

```bash
# View Cloud Run logs
gcloud run services logs tail ard-backend --region=us-central1 --limit=100

# Check service configuration
gcloud run services describe ard-backend --region=us-central1

# Check for common issues:
# 1. Environment variables set?
gcloud run services describe ard-backend --region=us-central1 \
  --format='value(spec.template.spec.containers[0].env)'

# 2. Service account attached?
gcloud run services describe ard-backend --region=us-central1 \
  --format='value(spec.template.spec.serviceAccountName)'

# 3. Memory/CPU sufficient?
gcloud run services describe ard-backend --region=us-central1 \
  --format='value(spec.template.spec.containers[0].resources)'
```

### If Vertex AI fails

```bash
# Check API is enabled
gcloud services list --enabled | grep aiplatform

# Enable if needed
gcloud services enable aiplatform.googleapis.com

# Check service account permissions
gcloud projects get-iam-policy periodicdent42 \
  --flatten="bindings[].members" \
  --filter="bindings.members:ard-backend@periodicdent42.iam.gserviceaccount.com AND bindings.role:roles/aiplatform.user"

# Test Vertex AI access locally
python3 << 'EOF'
from google.cloud import aiplatform
aiplatform.init(project="periodicdent42", location="us-central1")
model = aiplatform.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Hello!")
print(response.text)
EOF
```

---

## 📊 Monitoring Commands

```bash
# View real-time logs
gcloud run services logs tail ard-backend \
  --region=us-central1 \
  --follow

# Filter for errors
gcloud run services logs tail ard-backend \
  --region=us-central1 \
  | grep ERROR

# View metrics in console
open "https://console.cloud.google.com/run/detail/us-central1/ard-backend/metrics?project=periodicdent42"

# Check custom metrics
gcloud monitoring time-series list \
  --filter='metric.type="custom.googleapis.com/ard/eig_per_hour"' \
  --format=json
```

---

## 🔄 Update & Redeploy

```bash
# After making code changes
cd /Users/kiteboard/periodicdent42/app

# Rebuild and redeploy
gcloud builds submit --tag gcr.io/periodicdent42/ard-backend
bash ../infra/scripts/deploy_cloudrun.sh

# Or use Makefile
make gcloud-build
make deploy

# Check new revision is serving
gcloud run services describe ard-backend --region=us-central1 \
  --format='value(status.latestCreatedRevisionName)'
```

---

## 🎉 Success Indicators

You'll know it's working when:

1. ✅ **Health check returns 200**:
   ```json
   {
     "status": "ok",
     "vertex_initialized": true,
     "project_id": "periodicdent42"
   }
   ```

2. ✅ **SSE streams work**:
   ```
   event: preliminary
   data: {"response": {...}, "message": "Quick preview..."}
   
   event: final
   data: {"response": {...}, "message": "Verified response ready"}
   ```

3. ✅ **Logs show both models**:
   ```
   Flash response sent: 1234ms
   Pro response sent: 15678ms
   ```

4. ✅ **No errors in logs**:
   ```bash
   gcloud run services logs tail ard-backend --region=us-central1 | grep ERROR
   # (should return nothing)
   ```

---

## 💰 Cost Tracking

```bash
# View current spend
gcloud billing accounts list

# Set budget alert
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="ARD Development Budget" \
  --budget-amount=100.00 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90

# View current usage
open "https://console.cloud.google.com/billing?project=periodicdent42"
```

---

## 📚 Next Steps

1. **Test the dual-model pattern**:
   ```bash
   # Send a query and observe Flash (fast) then Pro (accurate)
   curl -N -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     -X POST $SERVICE_URL/api/reasoning/query \
     -H "Content-Type: application/json" \
     -d '{"query":"Explain Bayesian experimental design","context":{}}'
   ```

2. **Set up Cloud SQL** (for production):
   ```bash
   # Create PostgreSQL instance
   gcloud sql instances create ard-postgres \
     --database-version=POSTGRES_15 \
     --tier=db-n1-standard-2 \
     --region=us-central1
   
   # Create database
   gcloud sql databases create ard_intelligence \
     --instance=ard-postgres
   ```

3. **Enable unauthenticated access** (for demos only):
   ```bash
   gcloud run services add-iam-policy-binding ard-backend \
     --region=us-central1 \
     --member="allUsers" \
     --role="roles/run.invoker"
   ```

4. **View API documentation**:
   ```bash
   # Get service URL and open docs
   SERVICE_URL=$(gcloud run services describe ard-backend \
     --region=us-central1 \
     --format='value(status.url)')
   open "$SERVICE_URL/docs"
   ```

---

## 🆘 Getting Help

### Documentation
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Full deployment steps
- [app/README.md](app/README.md) - Application documentation
- [docs/google_cloud_deployment.md](docs/google_cloud_deployment.md) - Complete GCP guide
- [docs/gemini_integration_examples.md](docs/gemini_integration_examples.md) - Code examples

### Useful Links
- [Cloud Run Console](https://console.cloud.google.com/run?project=periodicdent42)
- [Cloud Build Console](https://console.cloud.google.com/cloud-build/builds?project=periodicdent42)
- [Vertex AI Console](https://console.cloud.google.com/vertex-ai?project=periodicdent42)
- [Logs Explorer](https://console.cloud.google.com/logs/query?project=periodicdent42)

---

## ✅ Final Checklist

Before you start:
- [ ] GCP project `periodicdent42` exists
- [ ] Billing is enabled on the project
- [ ] `gcloud` CLI is installed and authenticated
- [ ] Python 3.12+ is installed

After deployment:
- [ ] Health check returns 200
- [ ] SSE streaming works
- [ ] Logs are visible in Cloud Logging
- [ ] No errors in Cloud Run logs
- [ ] Budget alerts are configured

---

**Ready to deploy?** Run:

```bash
cd /Users/kiteboard/periodicdent42
bash quickdeploy.sh
```

**Or follow step-by-step** in Option B above.

🚀 **Let's ship it!**

