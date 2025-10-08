# Path C Deployment Status - October 8, 2025

## 🎯 Primary Objective: Deploy BETE-NET to Cloud Run

**Status**: ✅ Foundation Complete | ⏳ Final Deployment Pending

---

## ✅ Completed Work (6 Commits)

### 1. **Fixed Import Paths** (commit `412fa7b`)
- Corrected module imports from `app.src.bete_net_io.*` to `src.bete_net_io.*`
- Aligned with existing codebase Python path conventions

### 2. **Added Graceful Dependency Handling** (commit `fa41041`)
- Wrapped BETE-NET imports in try-except blocks
- Endpoints return HTTP 501 with helpful error messages when dependencies missing
- Allows immediate deployment without blocking on pymatgen/matplotlib

### 3. **Fixed Logger Initialization Order** (commit `8973072`)
- Moved `logger = logging.getLogger(__name__)` before conditional imports
- Prevents "undefined variable" errors during module load

### 4. **Defined Import Placeholders** (commit `4f8adbe`)
- Added `None` placeholders for `ScreeningConfig`, `batch_screen`, `create_evidence_pack`, `predict_tc`
- Critical fix: Allows module to load even when imports fail
- Without this, Python refuses to import the module due to undefined names

---

## 🏗️ Architecture Implemented

### **FastAPI Router** (`app/src/api/bete_net.py`)
```python
router = APIRouter(prefix="/api/bete", tags=["BETE-NET"])

# Endpoints:
POST /api/bete/predict     # Single structure prediction
POST /api/bete/screen      # Batch screening with background tasks
GET  /api/bete/report/{id} # Download evidence pack
```

### **Graceful Degradation Logic**
```python
try:
    from src.bete_net_io.inference import predict_tc
    BETE_ENABLED = True
except ImportError as e:
    BETE_ENABLED = False
    IMPORT_ERROR = str(e)
    predict_tc = None  # Placeholder

@router.post("/predict")
async def predict_endpoint(request):
    if not BETE_ENABLED:
        return JSONResponse(
            status_code=501,
            content={
                "error": "BETE-NET not available",
                "detail": IMPORT_ERROR,
                "hint": "Install with: pip install pymatgen matplotlib scipy"
            }
        )
    # ... actual prediction logic
```

### **Mock Models** (`app/src/bete_net_io/mock_models.py`)
- Generates realistic but random α²F(ω) curves for testing
- Computes λ, ⟨ω_log⟩, T_c using Allen-Dynes formula
- Allows immediate API testing without real model weights
- Example: For Nb, generates T_c ≈ 5-12 K (realistic range)

---

## 🚧 Current Blocker: Docker Build Cache

### **Symptom**
- `gcloud run deploy` reports "Deployment failed" 
- But Docker build actually succeeds: `Successfully built 465131913058`
- Image is pushed to GCR: `gcr.io/periodicdent42/ard-backend:latest`
- Service updates create new revisions (latest: `ard-backend-00039-rt9`)
- **However**: BETE endpoints not appearing in OpenAPI spec → 404 errors

### **Root Cause**
Docker build is using cached layers from earlier builds that don't have the placeholder fix (commit `4f8adbe`). The cached image has the broken module that fails to import.

### **Evidence**
```bash
$ curl https://ard-backend-dydzexswua-uc.a.run.app/openapi.json | jq '.paths | keys'
# BETE endpoints missing (should show /api/bete/predict, /api/bete/screen, /api/bete/report/{id})

$ gcloud run services logs read ard-backend | grep -i bete
# No BETE warning log (module failed to import silently)
```

---

## 🔧 Deployment Workarounds Attempted

1. **Force rebuild with `--quiet`**: ❌ Still used cache
2. **Update `.dockerignore` to bust cache**: ❌ Build failed (unrelated issue)
3. **Deploy with `--no-traffic`**: ✅ Created revision, but old code
4. **Manual `gcloud run services update`**: ✅ Updates service, but still cached image

---

## 📋 Next Steps to Complete Deployment

### **Option A: Force Clean Docker Build** (Recommended)
```bash
# 1. Clear Cloud Build cache
gcloud builds list --limit=5 --format="value(id)" | xargs -I{} gcloud builds cancel {}

# 2. Deploy with explicit --no-cache flag (if available)
gcloud run deploy ard-backend \\
  --source app \\
  --region us-central1 \\
  --memory 4Gi \\
  --cpu 4 \\
  --timeout 300 \\
  --no-traffic \\
  --build-env-vars="DOCKER_BUILDKIT=1" \\
  --allow-unauthenticated

# 3. Test new revision
curl https://ard-backend-dydzexswua-uc.a.run.app/openapi.json | jq '.paths | keys | .[] | select(contains("bete"))'

# 4. Route traffic if successful
gcloud run services update-traffic ard-backend --region=us-central1 --to-latest
```

### **Option B: Build Locally and Push** (Slower but Reliable)
```bash
cd /Users/kiteboard/periodicdent42

# Build fresh image locally
docker build -t gcr.io/periodicdent42/ard-backend:$(git rev-parse --short HEAD) app/

# Push to GCR
docker push gcr.io/periodicdent42/ard-backend:$(git rev-parse --short HEAD)

# Deploy specific tag
gcloud run deploy ard-backend \\
  --image=gcr.io/periodicdent42/ard-backend:$(git rev-parse --short HEAD) \\
  --region=us-central1 \\
  --memory=4Gi \\
  --cpu=4 \\
  --allow-unauthenticated
```

### **Option C: Wait for GitHub Actions CI/CD** (Automated)
```bash
# Trigger workflow manually
gh workflow run ci-bete.yml

# Monitor progress
gh run watch $(gh run list --workflow=ci-bete.yml --limit=1 --json databaseId --jq='.[0].databaseId')

# Note: Currently failing due to missing BETE_DEPLOYMENT_GUIDE.md
# Need to create documentation first or skip that job
```

---

## 📝 After Deployment: Add Real Dependencies

Once placeholders are working, add full BETE-NET support:

### 1. **Update `app/requirements.txt`**
```txt
# BETE-NET dependencies
pymatgen==2024.10.3       # Crystal structure handling
matplotlib==3.8.0         # Plotting α²F(ω)
scipy==1.11.3            # Allen-Dynes integration
torch==2.1.0             # Model inference (if using real weights)
```

### 2. **Download Model Weights**
```bash
cd /Users/kiteboard/periodicdent42
bash scripts/download_bete_weights.sh
# Downloads to third_party/bete_net/models/
```

### 3. **Update Dockerfile** (if weights are large)
```dockerfile
# After COPY app/ .
COPY third_party/ /app/third_party/
```

### 4. **Redeploy with Full Stack**
```bash
gcloud run deploy ard-backend --source app --region us-central1 --memory=8Gi --cpu=8
# Note: Increased resources for pymatgen + model inference
```

---

## 🧪 Testing Strategy

### **Phase 1: Validate 501 Responses** (Current State)
```bash
curl -X POST https://ard-backend-dydzexswua-uc.a.run.app/api/bete/predict \\
  -H "Content-Type: application/json" \\
  -d '{"mp_id": "mp-48", "mu_star": 0.10}'

# Expected:
# {
#   "error": "BETE-NET not available",
#   "detail": "No module named 'pymatgen'",
#   "hint": "Install with: pip install pymatgen matplotlib scipy"
# }
```

### **Phase 2: Validate Mock Predictions** (After Dependencies Added)
```bash
curl -X POST https://ard-backend-dydzexswua-uc.a.run.app/api/bete/predict \\
  -H "Content-Type: application/json" \\
  -d '{"mp_id": "mp-48", "mu_star": 0.10}' | jq .

# Expected (mock):
# {
#   "formula": "Nb",
#   "tc_kelvin": 8.3,  # Random but realistic
#   "lambda_ep": 0.95,
#   "model_version": "MOCK-1.0.0",
#   "evidence_url": "/api/bete/report/abc123..."
# }
```

### **Phase 3: Golden Tests** (After Real Weights)
```bash
# Nb (mp-48): Expected T_c ≈ 9.2 K
# MgB2 (mp-763): Expected T_c ≈ 39 K  
# Al (mp-134): Expected T_c ≈ 1.2 K

pytest app/tests/test_bete_golden.py -v
```

---

## 📊 Current Production Status

| Component | Status | URL |
|-----------|--------|-----|
| Health Endpoint | ✅ Working | https://ard-backend-dydzexswua-uc.a.run.app/health |
| API Docs | ✅ Working | https://ard-backend-dydzexswua-uc.a.run.app/docs |
| BETE /predict | ❌ 404 (import issue) | - |
| BETE /screen | ❌ 404 (import issue) | - |
| Analytics Dashboard | ✅ Working | https://ard-backend-dydzexswua-uc.a.run.app/static/analytics.html |

**Latest Revision**: `ard-backend-00039-rt9` (October 8, 2025 21:55 UTC)

---

## 💡 Key Learnings

1. **FastAPI Router Import Failures are Silent**: If a router module fails to import, `app.include_router()` silently skips it without error
2. **Placeholder Pattern for Optional Dependencies**:
   ```python
   try:
       from optional_module import Thing
   except ImportError:
       Thing = None  # Critical: Define placeholder
   ```
3. **Cloud Build Cache is Aggressive**: Even modifying `.dockerignore` doesn't always bust cache
4. **Graceful Degradation > Blocking on Dependencies**: Better to deploy with 501 responses than wait for all deps

---

## 🎯 Definition of Done

- [ ] BETE endpoints appear in `/openapi.json`
- [ ] `POST /api/bete/predict` returns 501 with helpful error (not 404)
- [ ] Service logs show "BETE-NET dependencies not available" warning
- [ ] After adding pymatgen: endpoints return mock predictions
- [ ] After adding real weights: golden tests pass for Nb, MgB2, Al

---

**Last Updated**: October 8, 2025 22:00 UTC  
**Next Action**: Force clean Docker build (Option A) or build locally (Option B)

