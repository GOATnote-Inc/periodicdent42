# Database API Fix Instructions - October 6, 2025

**Status**: ⚠️ IDENTIFIED - Requires manual fix  
**Issue**: Database API endpoints returning `{"error": "Failed to query..."}`  
**Root Cause**: Missing database configuration in Cloud Run deployment

---

## 🔍 Problem Analysis

### Current Status
- ✅ Health endpoint working
- ❌ `/api/experiments` returning errors
- ❌ `/api/optimization_runs` returning errors
- ❌ `/api/ai_queries` returning errors

### Root Cause
The manual Cloud Run deployment didn't include database configuration:
1. Missing `GCP_SQL_INSTANCE` environment variable
2. Missing `DB_PASSWORD` secret reference
3. Database initialization failing silently

### Impact
- Analytics dashboard cannot load data
- Metadata endpoints not functional
- Health monitoring still works (separate endpoint)

---

## ✅ Solution: Update Cloud Run Configuration

### Option 1: Update Existing Service (Recommended)

**Step 1: Authenticate gcloud** (if needed)
```bash
gcloud auth login
```

**Step 2: Update Cloud Run service with database configuration**
```bash
gcloud run services update ard-backend \
  --region us-central1 \
  --set-env-vars="GCP_SQL_INSTANCE=periodicdent42:us-central1:ard-intelligence-db,DB_USER=ard_user,DB_NAME=ard_intelligence" \
  --update-secrets="DB_PASSWORD=db-password:latest"
```

**Expected Output**:
```
Deploying...
Setting IAM Policy...done
Creating Revision...done
Routing traffic...done
Done.
Service [ard-backend] revision [ard-backend-00035-xxx] has been deployed.
```

**Step 3: Verify Database Connection**
```bash
# Wait 30 seconds for deployment, then test
curl https://ard-backend-dydzexswua-uc.a.run.app/api/experiments?limit=3 | python3 -m json.tool
```

**Expected Response**:
```json
{
  "experiments": [
    {
      "id": "exp_...",
      "parameters": {...},
      "results": {...},
      ...
    }
  ],
  "total": 205,
  "page": 1,
  "page_size": 3
}
```

---

### Option 2: Full Redeployment with Database Config

**Step 1: Redeploy with all configurations**
```bash
gcloud run deploy ard-backend \
  --image gcr.io/periodicdent42/ard-backend:health-fix \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="PROJECT_ID=periodicdent42,GCP_REGION=us-central1,LOG_LEVEL=INFO,GCP_SQL_INSTANCE=periodicdent42:us-central1:ard-intelligence-db,DB_USER=ard_user,DB_NAME=ard_intelligence" \
  --update-secrets="DB_PASSWORD=db-password:latest" \
  --set-cloudsql-instances=periodicdent42:us-central1:ard-intelligence-db \
  --memory 2Gi \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0
```

---

## 🔐 Secret Manager Configuration

### Verify Database Password Secret Exists

**Check if secret exists**:
```bash
gcloud secrets describe db-password --project=periodicdent42
```

**If secret doesn't exist, create it**:
```bash
# Create secret with the database password
echo -n "ard_secure_password_2024" | gcloud secrets create db-password \
  --project=periodicdent42 \
  --replication-policy="automatic" \
  --data-file=-
```

**Grant Cloud Run access to the secret**:
```bash
# Get the Cloud Run service account
SERVICE_ACCOUNT=$(gcloud run services describe ard-backend \
  --region=us-central1 \
  --format='value(spec.template.spec.serviceAccountName)')

# Grant secret accessor role
gcloud secrets add-iam-policy-binding db-password \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor" \
  --project=periodicdent42
```

---

## 🧪 Testing After Fix

### Test 1: Health Check (Should Still Work)
```bash
curl https://ard-backend-dydzexswua-uc.a.run.app/health | python3 -m json.tool
```

**Expected**: `{"status": "ok", ...}` ✅

### Test 2: Experiments API (Should Now Work)
```bash
curl https://ard-backend-dydzexswua-uc.a.run.app/api/experiments?limit=3 | python3 -m json.tool
```

**Expected**: List of experiments (not error) ✅

### Test 3: Optimization Runs API (Should Now Work)
```bash
curl https://ard-backend-dydzexswua-uc.a.run.app/api/optimization_runs?limit=3 | python3 -m json.tool
```

**Expected**: List of optimization runs (not error) ✅

### Test 4: AI Queries API (Should Now Work)
```bash
curl https://ard-backend-dydzexswua-uc.a.run.app/api/ai_queries?limit=3 | python3 -m json.tool
```

**Expected**: List of AI queries (not error) ✅

### Test 5: Analytics Dashboard (Should Load Data)
```bash
open https://ard-backend-dydzexswua-uc.a.run.app/static/analytics.html
```

**Expected**: Charts populate with data from database ✅

---

## 🔍 Troubleshooting

### Issue: "Secret not found"

**Cause**: `db-password` secret doesn't exist in Secret Manager

**Fix**:
```bash
# Create the secret
echo -n "YOUR_DB_PASSWORD" | gcloud secrets create db-password \
  --project=periodicdent42 \
  --replication-policy="automatic" \
  --data-file=-
```

### Issue: "Permission denied accessing secret"

**Cause**: Cloud Run service account doesn't have access to secret

**Fix**:
```bash
# Get service account
SERVICE_ACCOUNT=$(gcloud run services describe ard-backend \
  --region=us-central1 \
  --format='value(spec.template.spec.serviceAccountName)')

# Grant access
gcloud secrets add-iam-policy-binding db-password \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor" \
  --project=periodicdent42
```

### Issue: "Could not connect to Cloud SQL instance"

**Cause**: Cloud SQL instance not properly configured

**Fix**:
```bash
# Verify Cloud SQL instance exists
gcloud sql instances describe ard-intelligence-db --project=periodicdent42

# Verify Cloud SQL connection in Cloud Run
gcloud run services describe ard-backend --region=us-central1 \
  --format='value(spec.template.metadata.annotations[run.googleapis.com/cloudsql-instances])'

# Should show: periodicdent42:us-central1:ard-intelligence-db
```

### Issue: "Database tables don't exist"

**Cause**: Tables not created in database

**Fix**: Tables are auto-created by SQLAlchemy on first connection. If this doesn't work:
```bash
# Connect to Cloud SQL proxy
./cloud-sql-proxy --port 5433 periodicdent42:us-central1:ard-intelligence-db &

# Run Python script to create tables
export DB_USER=ard_user
export DB_PASSWORD=ard_secure_password_2024
export DB_NAME=ard_intelligence
export DB_HOST=localhost
export DB_PORT=5433

python3 << 'PYTHON'
from app.src.services.db import init_database, Base, _engine
init_database()
if _engine:
    Base.metadata.create_all(bind=_engine)
    print("✅ Tables created successfully")
else:
    print("❌ Database engine not initialized")
PYTHON
```

---

## 📊 Expected Environment Variables in Cloud Run

After fix, Cloud Run should have:

**Environment Variables**:
- `PROJECT_ID=periodicdent42`
- `GCP_REGION=us-central1`
- `LOG_LEVEL=INFO`
- `GCP_SQL_INSTANCE=periodicdent42:us-central1:ard-intelligence-db`
- `DB_USER=ard_user`
- `DB_NAME=ard_intelligence`

**Secrets**:
- `DB_PASSWORD` → `db-password:latest`

**Cloud SQL Connections**:
- `periodicdent42:us-central1:ard-intelligence-db`

---

## 🎯 Quick Fix Command

**All-in-one fix command** (requires gcloud auth):

```bash
#!/bin/bash
# Quick fix for database connection

# 1. Authenticate
gcloud auth login

# 2. Update Cloud Run service
gcloud run services update ard-backend \
  --region us-central1 \
  --set-env-vars="GCP_SQL_INSTANCE=periodicdent42:us-central1:ard-intelligence-db,DB_USER=ard_user,DB_NAME=ard_intelligence" \
  --update-secrets="DB_PASSWORD=db-password:latest"

# 3. Wait for deployment
echo "Waiting 30 seconds for deployment..."
sleep 30

# 4. Test database API
echo ""
echo "Testing /api/experiments endpoint..."
curl -s https://ard-backend-dydzexswua-uc.a.run.app/api/experiments?limit=1 | python3 -m json.tool

echo ""
echo "Testing /api/optimization_runs endpoint..."
curl -s https://ard-backend-dydzexswua-uc.a.run.app/api/optimization_runs?limit=1 | python3 -m json.tool

echo ""
echo "✅ Database fix complete!"
```

Save as `fix_database.sh`, make executable, and run:
```bash
chmod +x fix_database.sh
./fix_database.sh
```

---

## 📝 Documentation Updates After Fix

After successful fix, update:

1. **DEPLOYMENT_COMPLETE_OCT6_2025.md**
   - Mark database issue as RESOLVED
   - Add database configuration details
   - Update smoke test results

2. **README or deployment guide**
   - Document database configuration requirements
   - Add Secret Manager setup instructions
   - Include troubleshooting steps

---

## ✅ Success Criteria

After applying the fix, verify:

- [ ] `/health` endpoint still working (should not break)
- [ ] `/api/experiments` returns experiment data (not error)
- [ ] `/api/optimization_runs` returns run data (not error)
- [ ] `/api/ai_queries` returns query data (not error)
- [ ] Analytics dashboard loads and displays charts
- [ ] No errors in Cloud Run logs related to database connection

---

## 🎯 Summary

**Problem**: Database APIs returning errors  
**Cause**: Missing database configuration in Cloud Run  
**Fix**: Update Cloud Run service with database env vars and secrets  
**Time**: ~5 minutes (after gcloud auth)  
**Risk**: Low (health endpoint unaffected, only adding database config)

**Quick Fix Command**:
```bash
gcloud run services update ard-backend \
  --region us-central1 \
  --set-env-vars="GCP_SQL_INSTANCE=periodicdent42:us-central1:ard-intelligence-db" \
  --update-secrets="DB_PASSWORD=db-password:latest"
```

---

**Status**: Ready to apply fix  
**Priority**: Medium (health monitoring working, this enables analytics)  
**Next**: After fix, verify all endpoints and update documentation
