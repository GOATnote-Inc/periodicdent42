# Deployment Guide - Autonomous R&D Intelligence Layer

**Complete deployment commands for Google Cloud Platform**

---

## Prerequisites

- Google Cloud SDK installed (`gcloud`)
- Python 3.12+
- Docker (for local testing)
- GCP Project: `periodicdent42`

---

## 🚀 Quick Deploy (One Command Setup)

```bash
# Full deployment script
cd /Users/kiteboard/periodicdent42
bash quickdeploy.sh
```

---

## 📋 Step-by-Step Deployment

### 1. Local Setup & Testing

```bash
# Navigate to project
cd /Users/kiteboard/periodicdent42/app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

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

# Run development server
make dev

# In another terminal, test
curl http://localhost:8080/healthz
```

### 2. GCP Project Setup

```bash
# Set project
gcloud config set project periodicdent42

# Enable all required APIs
bash infra/scripts/enable_apis.sh

# Setup IAM (service accounts and permissions)
bash infra/scripts/setup_iam.sh
```

### 3. Test Deployment

```bash
# Run tests locally
cd app
python3 -m pytest -v

# Expected output:
# ✅ test_health_check_returns_200 PASSED
# ✅ test_reasoning_endpoint_streams_preliminary_and_final PASSED
```

### 4. Build & Deploy to Cloud Run

```bash
# From app directory
cd /Users/kiteboard/periodicdent42/app

# Build image with Cloud Build
make gcloud-build

# Deploy to Cloud Run
export PROJECT_ID=periodicdent42
export REGION=us-central1
bash ../infra/scripts/deploy_cloudrun.sh

# Service will be deployed at:
# https://ard-backend-<hash>-uc.a.run.app
```

### 5. Test Deployed Service

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe ard-backend \
  --region us-central1 \
  --format 'value(status.url)')

# Test health check (requires auth)
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $SERVICE_URL/healthz

# Expected: {"status":"ok","vertex_initialized":true,"project_id":"periodicdent42"}

# Test reasoning endpoint
curl -N -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -X POST $SERVICE_URL/api/reasoning/query \
  -d '{"query":"Suggest experiments for BaTiO3","context":{"domain":"materials"}}'

# You should see SSE streams:
# event: preliminary
# data: {...}
# 
# event: final
# data: {...}
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` in `app/` directory:

```bash
# Core
PROJECT_ID=periodicdent42
LOCATION=us-central1

# Models (October 2025 latest)
GEMINI_FLASH_MODEL=gemini-2.5-flash
GEMINI_PRO_MODEL=gemini-2.5-pro

# Database (set after Cloud SQL setup)
GCP_SQL_INSTANCE=  # Format: project:region:instance
DB_USER=ard_user
DB_PASSWORD=<from Secret Manager>
DB_NAME=ard_intelligence

# Storage
GCS_BUCKET=<from terraform or manual creation>
```

### Secret Manager Setup

```bash
# Store sensitive values
echo -n "your-db-password" | gcloud secrets create db-password \
  --data-file=- \
  --project=periodicdent42

# Grant service account access
gcloud secrets add-iam-policy-binding db-password \
  --member="serviceAccount:ard-backend@periodicdent42.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 🎯 Verification Checklist

After deployment, verify:

- [ ] Health check returns 200: `curl $SERVICE_URL/healthz`
- [ ] Vertex AI initialized: Check `vertex_initialized: true` in health response
- [ ] SSE streaming works: Test `/api/reasoning/query` endpoint
- [ ] Logs visible: `gcloud run services logs tail ard-backend --region=us-central1`
- [ ] Metrics in Cloud Monitoring: Check custom metric `ard/eig_per_hour`

---

## 🐛 Troubleshooting

### "Permission denied" errors

```bash
# Re-run IAM setup
bash infra/scripts/setup_iam.sh

# Verify service account
gcloud iam service-accounts describe \
  ard-backend@periodicdent42.iam.gserviceaccount.com
```

### "Vertex AI not initialized"

```bash
# Check API is enabled
gcloud services list --enabled | grep aiplatform

# Enable if needed
gcloud services enable aiplatform.googleapis.com
```

### "Container failed to start"

```bash
# Check logs
gcloud run services logs tail ard-backend \
  --region=us-central1 \
  --limit=50

# Common issues:
# - Missing environment variables
# - Service account lacks permissions
# - Import errors in Python code
```

### Local development issues

```bash
# If Python imports fail
export PYTHONPATH=/Users/kiteboard/periodicdent42/app:$PYTHONPATH

# If Vertex AI fails locally
# Set up Application Default Credentials:
gcloud auth application-default login
```

---

## 📊 Monitoring

### View Logs

```bash
# Real-time logs
make logs

# Or directly:
gcloud run services logs tail ard-backend --region=us-central1

# Filter for errors
gcloud run services logs tail ard-backend --region=us-central1 \
  | grep ERROR
```

### Cloud Monitoring

```bash
# View custom metrics in console:
open "https://console.cloud.google.com/monitoring/metrics-explorer?project=periodicdent42"

# Query EIG metric:
# Metric: custom.googleapis.com/ard/eig_per_hour
```

---

## 🔄 CI/CD with GitHub Actions

### Setup

1. **Enable Workload Identity Federation** (recommended):

```bash
# Create Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --project="periodicdent42"

# Create provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository"

# Bind service account
gcloud iam service-accounts add-iam-policy-binding \
  ard-backend@periodicdent42.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_REPO"
```

2. **Add GitHub Secrets**:
   - `WIF_PROVIDER`: Full WIF provider resource name
   - `WIF_SERVICE_ACCOUNT`: `ard-backend@periodicdent42.iam.gserviceaccount.com`

3. **Push to main**:
   - Tests run automatically
   - On success, deploys to Cloud Run
   - Service URL posted in job summary

---

## 💰 Cost Management

### Current Costs (Development)

```bash
# View current spend
gcloud billing accounts list
gcloud billing budgets list --billing-account=YOUR_BILLING_ACCOUNT
```

### Set Budget Alert

```bash
# Create budget alert at $100
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="ARD Development Budget" \
  --budget-amount=100.00 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

### Estimated Monthly Costs

- **Development**: ~$321/month
  - Cloud Run: $50
  - Gemini API: $36
  - Cloud SQL: $200
  - Storage: $25
  - Monitoring: $10

- **Production**: ~$3,700/month
  - Cloud Run (scaled): $500
  - Gemini API (1B tokens): $200
  - AI Hypercomputer: $2,000
  - Cloud SQL (HA): $800
  - Storage (10TB): $200

---

## 🔒 Security Best Practices

1. **Never commit secrets**:
   ```bash
   # Check .gitignore includes:
   .env
   *.key
   credentials.json
   ```

2. **Restrict Cloud Run access**:
   ```bash
   # Deployed with --no-allow-unauthenticated by default
   # To allow public access (demos only):
   gcloud run services add-iam-policy-binding ard-backend \
     --region=us-central1 \
     --member="allUsers" \
     --role="roles/run.invoker"
   ```

3. **Rotate secrets regularly**:
   ```bash
   # Update Secret Manager values every 90 days
   echo -n "new-password" | gcloud secrets versions add db-password \
     --data-file=-
   ```

4. **Enable VPC Service Controls** (Production):
   ```bash
   # For high-security environments
   gcloud access-context-manager perimeters create ard-perimeter \
     --resources=projects/PROJECT_NUMBER \
     --restricted-services=aiplatform.googleapis.com,storage.googleapis.com
   ```

---

## 📞 Support

### Documentation
- [Main README](README.md)
- [Google Cloud Deployment Guide](docs/google_cloud_deployment.md)
- [Gemini Integration Examples](docs/gemini_integration_examples.md)
- [App README](app/README.md)

### Useful Commands

```bash
# SSH to Cloud Run container (for debugging)
gcloud run services proxy ard-backend --region=us-central1

# Scale to zero (save costs)
gcloud run services update ard-backend \
  --min-instances=0 \
  --region=us-central1

# Update environment variables
gcloud run services update ard-backend \
  --region=us-central1 \
  --set-env-vars "LOG_LEVEL=DEBUG"
```

---

## ✅ Success!

Your Autonomous R&D Intelligence Layer is now deployed on Google Cloud with:

- ⚡ Dual-model AI (Flash + Pro)
- 🚀 Serverless auto-scaling
- 🔒 Enterprise security
- 📊 Full observability
- 💰 Cost-optimized

**Service URL**: Check Cloud Run console or run:
```bash
gcloud run services describe ard-backend --region=us-central1 --format='value(status.url)'
```

Test it:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $(gcloud run services describe ard-backend --region=us-central1 --format='value(status.url)')/healthz
```

🎉 **You're ready to start autonomous R&D!**

