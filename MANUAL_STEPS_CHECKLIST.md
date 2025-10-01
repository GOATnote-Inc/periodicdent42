# Manual Steps Checklist

**Complete checklist of all manual steps required to deploy the secure production system.**

---

## ✅ Prerequisites (One-Time Setup)

### 1. Install Google Cloud SDK

**Check if installed**:
```bash
gcloud --version
```

**If not installed**:
- **macOS**: `brew install --cask google-cloud-sdk`
- **Linux**: https://cloud.google.com/sdk/docs/install
- **Windows**: https://cloud.google.com/sdk/docs/install

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project periodicdent42
```

**Verify**:
```bash
gcloud config get-value project
# Should output: periodicdent42
```

### 3. Verify Billing is Enabled

```bash
gcloud beta billing accounts list
gcloud beta billing projects link periodicdent42 --billing-account=YOUR_BILLING_ACCOUNT_ID
```

Or enable in console: https://console.cloud.google.com/billing

---

## 🚀 Deployment Steps (Execute in Order)

### Step 1: Enable Required APIs (~2 minutes)

```bash
cd /Users/kiteboard/periodicdent42
bash infra/scripts/enable_apis.sh
```

**What it does**: Enables Cloud Run, Cloud Build, Secret Manager, Vertex AI, Cloud Storage APIs

**Expected output**: "✅ All APIs enabled!"

**Troubleshooting**: If permission denied, ensure you're Owner/Editor on the project

---

### Step 2: Set Up IAM and Service Accounts (~1 minute)

```bash
bash infra/scripts/setup_iam.sh
```

**What it does**:
- Creates service account: `ard-backend@periodicdent42.iam.gserviceaccount.com`
- Grants necessary roles

**Expected output**: "✅ IAM setup complete!"

**Verify**:
```bash
gcloud iam service-accounts list --filter="email:ard-backend@*"
```

---

### Step 3: Create Secrets (~1 minute)

```bash
bash infra/scripts/create_secrets.sh
```

**What it does**:
- **Generates secure API key** (32 bytes random hex)
- Creates Secret Manager secrets
- Configures access permissions

**⚠️ CRITICAL**: The script will output your API key like this:
```
✅ API_KEY created: a1b2c3d4e5f6789...
   ⚠️  SAVE THIS KEY - You'll need it to make API requests!
```

**ACTION REQUIRED**: Copy this key immediately!

**Save it securely**:
```bash
# Option 1: Save to file
echo "YOUR_API_KEY_HERE" > ~/.ard-api-key
chmod 600 ~/.ard-api-key

# Option 2: Save to password manager (RECOMMENDED)
# Use 1Password, LastPass, etc.
```

**To retrieve later**:
```bash
gcloud secrets versions access latest --secret=api-key --project=periodicdent42
```

---

### Step 4: Deploy to Cloud Run (~5-10 minutes first time)

```bash
bash infra/scripts/deploy_cloudrun.sh
```

**What it does**:
- Builds Docker container
- Pushes to Container Registry
- Deploys to Cloud Run with security enabled
- Configures environment variables

**Expected output**:
```
✅ Deployment complete!
Service URL: https://ard-backend-xxx-uc.a.run.app

🔐 Security Status:
  - Authentication: ENABLED (API key required)
  - Rate Limiting: 120 requests/minute per IP
  - CORS: Configure ALLOWED_ORIGINS for your domain
```

**⚠️ IMPORTANT**: Save the Service URL!

**Save the URL**:
```bash
# Set as environment variable for future use
export SERVICE_URL="https://ard-backend-xxx-uc.a.run.app"
echo "export SERVICE_URL=$SERVICE_URL" >> ~/.zshrc  # or ~/.bashrc
```

---

### Step 5: Test the Deployment (~2 minutes)

**Retrieve your API key** (if not saved earlier):
```bash
export API_KEY=$(gcloud secrets versions access latest --secret=api-key --project=periodicdent42)
echo "Your API Key: $API_KEY"
```

**Test health endpoint WITH auth (should work)**:
```bash
curl -H "x-api-key: $API_KEY" $SERVICE_URL/health
```

**Expected response**:
```json
{
  "status": "ok",
  "vertex_initialized": true,
  "project_id": "periodicdent42"
}
```

**Test WITHOUT auth (should fail with 401)**:
```bash
curl $SERVICE_URL/health
```

**Expected response**:
```json
{
  "error": "Unauthorized",
  "code": "unauthorized"
}
```

**Test reasoning endpoint**:
```bash
curl -X POST $SERVICE_URL/api/reasoning/query \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Suggest an experiment for perovskite solar cells"}'
```

**Expected**: Server-Sent Events stream with responses

---

### Step 6: Configure CORS (If Using Web Frontend)

**Only required if you have a web frontend that needs to call the API.**

```bash
# Replace with your actual frontend domain(s)
gcloud run services update ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --update-env-vars="ALLOWED_ORIGINS=https://app.example.com,https://dashboard.example.com"
```

**For multiple domains**, use comma-separated list.

**Verify**:
```bash
gcloud run services describe ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --format='value(spec.template.spec.containers[0].env)' | grep ALLOWED_ORIGINS
```

---

### Step 7: Set Up Monitoring (Optional but Recommended)

#### Create Log-Based Metrics

**Unauthorized requests (401 errors)**:
```bash
gcloud logging metrics create unauthorized_requests \
  --description="Count of 401 unauthorized API requests" \
  --log-filter='resource.type="cloud_run_revision"
    resource.labels.service_name="ard-backend"
    httpRequest.status=401' \
  --project=periodicdent42
```

**Rate limited requests (429 errors)**:
```bash
gcloud logging metrics create rate_limited_requests \
  --description="Count of 429 rate limited requests" \
  --log-filter='resource.type="cloud_run_revision"
    resource.labels.service_name="ard-backend"
    httpRequest.status=429' \
  --project=periodicdent42
```

#### View Logs in Console

https://console.cloud.google.com/logs/query?project=periodicdent42

**Or via CLI**:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend" \
  --limit=50 \
  --project=periodicdent42
```

---

## 📋 Security Verification Checklist

**Run these tests to verify security is working:**

- [ ] **Test 1**: Valid API key returns 200
  ```bash
  curl -H "x-api-key: $API_KEY" $SERVICE_URL/health
  # Should return: {"status": "ok", ...}
  ```

- [ ] **Test 2**: Missing API key returns 401
  ```bash
  curl $SERVICE_URL/health
  # Should return: {"error": "Unauthorized", "code": "unauthorized"}
  ```

- [ ] **Test 3**: Invalid API key returns 401
  ```bash
  curl -H "x-api-key: invalid-key" $SERVICE_URL/health
  # Should return: {"error": "Unauthorized", "code": "unauthorized"}
  ```

- [ ] **Test 4**: Rate limiting works (121st request returns 429)
  ```bash
  for i in {1..125}; do
    curl -s -o /dev/null -w "%{http_code}\n" \
      -H "x-api-key: $API_KEY" \
      $SERVICE_URL/health
  done | tail -5
  # Last 5 should include 429s
  ```

- [ ] **Test 5**: Security headers present
  ```bash
  curl -I $SERVICE_URL/health | grep -E "(Strict-Transport|X-Content-Type|X-Frame)"
  # Should show security headers
  ```

- [ ] **Test 6**: Errors don't leak stack traces
  ```bash
  # Try to trigger an error (e.g., invalid request)
  curl -X POST $SERVICE_URL/api/reasoning/query \
    -H "x-api-key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}'
  # Response should NOT contain stack trace or file paths
  ```

---

## 🔑 API Key Management

### Retrieve Current API Key

```bash
gcloud secrets versions access latest --secret=api-key --project=periodicdent42
```

### Rotate API Key (Every 90 Days)

```bash
# 1. Generate new key
NEW_KEY=$(openssl rand -hex 32)

# 2. Add new version to Secret Manager
echo -n "$NEW_KEY" | gcloud secrets versions add api-key \
  --data-file=- \
  --project=periodicdent42

# 3. Restart Cloud Run to use new key
gcloud run services update ard-backend \
  --region=us-central1 \
  --project=periodicdent42

# 4. Distribute new key to clients
echo "New API Key: $NEW_KEY"

# 5. After all clients updated, disable old version
gcloud secrets versions list api-key --project=periodicdent42
gcloud secrets versions disable OLD_VERSION_NUMBER --secret=api-key --project=periodicdent42
```

---

## 🚨 Troubleshooting

### Issue: "Permission denied" errors

**Solution**: Ensure you have proper IAM roles
```bash
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="user:your-email@example.com" \
  --role="roles/owner"
```

### Issue: API returns 401 even with correct key

**Diagnosis**:
```bash
# Check if ENABLE_AUTH is set
gcloud run services describe ard-backend \
  --region=us-central1 \
  --format='value(spec.template.spec.containers[0].env)' | grep ENABLE_AUTH

# Should show: ENABLE_AUTH=true
```

**Solution**: Redeploy if needed:
```bash
bash infra/scripts/deploy_cloudrun.sh
```

### Issue: Service not responding

**Check service status**:
```bash
gcloud run services describe ard-backend \
  --region=us-central1 \
  --project=periodicdent42
```

**Check logs**:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend" \
  --limit=20 \
  --project=periodicdent42
```

### Issue: CORS errors in browser

**Solution**: Set ALLOWED_ORIGINS (see Step 6 above)

---

## 📊 Monitoring Dashboard

### View Request Metrics

```bash
# View in Cloud Console
open "https://console.cloud.google.com/run/detail/us-central1/ard-backend/metrics?project=periodicdent42"
```

### View Logs

```bash
# View in Cloud Console
open "https://console.cloud.google.com/logs/query?project=periodicdent42"
```

---

## ✅ Post-Deployment Actions

### Immediate (Required)

- [ ] Save API key securely (Step 3)
- [ ] Save service URL (Step 4)
- [ ] Test all endpoints (Step 5)
- [ ] Configure CORS if needed (Step 6)

### Within 24 Hours

- [ ] Set up monitoring (Step 7)
- [ ] Share API key with authorized clients securely
- [ ] Document your deployment (service URL, configuration)
- [ ] Set calendar reminder for key rotation (90 days)

### Within 1 Week

- [ ] Review Cloud Logging for any issues
- [ ] Adjust rate limits if needed
- [ ] Set up alerting for 401/429 spikes
- [ ] Configure budget alerts to avoid surprise costs

---

## 📞 Need Help?

### Documentation

- **Step-by-step guide**: [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)
- **Security details**: [docs/SECURITY.md](docs/SECURITY.md)
- **Quick reference**: [SECURITY_QUICKREF.md](SECURITY_QUICKREF.md)
- **Local development**: [LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md)

### Support

- **Technical support**: B@thegoatnote.com
- **Security issues**: security@example.com
- **Emergency**: oncall@example.com

---

## Summary of Manual Steps

**Total time: ~15-20 minutes**

1. ✅ Prerequisites: gcloud auth, billing (~5 min, one-time)
2. ✅ Run enable_apis.sh (~2 min)
3. ✅ Run setup_iam.sh (~1 min)
4. ✅ Run create_secrets.sh - **SAVE API KEY!** (~1 min)
5. ✅ Run deploy_cloudrun.sh - **SAVE SERVICE URL!** (~10 min)
6. ✅ Test deployment (~2 min)
7. ⚠️ Configure CORS if needed (optional, ~1 min)
8. ⚠️ Set up monitoring (optional, ~5 min)

**Everything else is automated!** 🎉

---

**Status**: Ready to deploy! Start with Step 1.

