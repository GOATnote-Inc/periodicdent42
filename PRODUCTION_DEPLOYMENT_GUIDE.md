# Production Deployment Guide - Security Hardened

**Last Updated**: October 1, 2025  
**Version**: 1.0 (Security Hardened)  
**Prerequisites**: Google Cloud Platform account, `gcloud` CLI installed

---

## 🚀 Quick Deploy (5 Minutes)

```bash
# 1. Enable required APIs
bash infra/scripts/enable_apis.sh

# 2. Set up IAM and service accounts
bash infra/scripts/setup_iam.sh

# 3. Create secrets (including API key)
bash infra/scripts/create_secrets.sh

# 4. Deploy to Cloud Run
bash infra/scripts/deploy_cloudrun.sh
```

**Done!** Your secure API is now live. See output for your API key and service URL.

---

## 📋 Detailed Step-by-Step Instructions

### Prerequisites Checklist

- [ ] Google Cloud Platform account
- [ ] `gcloud` CLI installed: `gcloud --version`
- [ ] Authenticated: `gcloud auth login`
- [ ] Project set: `gcloud config set project periodicdent42`
- [ ] Billing enabled on project
- [ ] OpenSSL installed (for key generation)

---

### Step 1: Enable Required APIs

**Time**: ~2 minutes

```bash
cd /Users/kiteboard/periodicdent42
bash infra/scripts/enable_apis.sh
```

**What this does**:
- Enables Cloud Run API
- Enables Cloud Build API
- Enables Secret Manager API
- Enables Vertex AI API
- Enables Cloud Storage API

**Verify**:
```bash
gcloud services list --enabled --filter="name:(run.googleapis.com OR cloudbuild.googleapis.com OR secretmanager.googleapis.com)"
```

**Troubleshooting**:
- If "Permission denied": Ensure you're an Owner or Editor on the project
- If "Billing not enabled": Enable billing in Cloud Console

---

### Step 2: Set Up IAM and Service Accounts

**Time**: ~1 minute

```bash
bash infra/scripts/setup_iam.sh
```

**What this does**:
- Creates service account: `ard-backend@periodicdent42.iam.gserviceaccount.com`
- Grants roles:
  - `roles/aiplatform.user` - For Vertex AI access
  - `roles/storage.admin` - For Cloud Storage
  - `roles/secretmanager.secretAccessor` - For reading secrets
  - `roles/logging.logWriter` - For Cloud Logging

**Verify**:
```bash
gcloud iam service-accounts list --filter="email:ard-backend@*"
```

---

### Step 3: Create and Configure Secrets

**Time**: ~1 minute

```bash
bash infra/scripts/create_secrets.sh
```

**What this does**:
1. **Generates secure API key** (32 bytes random hex)
2. Creates Secret Manager secrets:
   - `api-key` - **CRITICAL**: Your API authentication key
   - `DB_PASSWORD` - Database password
   - `GCP_SQL_INSTANCE` - Cloud SQL connection string
   - `GCS_BUCKET` - Storage bucket name
3. Grants service account access to all secrets

**⚠️ IMPORTANT**: The script outputs your API key. **SAVE IT IMMEDIATELY!**

```bash
# Example output:
✅ API_KEY created: a1b2c3d4e5f6...
   ⚠️  SAVE THIS KEY - You'll need it to make API requests!
```

**Save your API key**:
```bash
# Save to a secure location
echo "a1b2c3d4e5f6..." > ~/.ard-api-key
chmod 600 ~/.ard-api-key

# Or use a password manager (recommended)
```

**Retrieve API key later**:
```bash
gcloud secrets versions access latest --secret=api-key --project=periodicdent42
```

**Verify secrets created**:
```bash
gcloud secrets list --project=periodicdent42
```

---

### Step 4: Deploy to Cloud Run

**Time**: ~5-10 minutes (first deployment builds container)

```bash
bash infra/scripts/deploy_cloudrun.sh
```

**What this does**:
1. Builds Docker container from `app/Dockerfile`
2. Pushes to Google Container Registry
3. Deploys to Cloud Run with:
   - **Memory**: 2Gi
   - **CPU**: 2 vCPUs
   - **Timeout**: 300s
   - **Instances**: Min 1, Max 10
   - **Environment Variables**:
     - `PROJECT_ID=periodicdent42`
     - `LOCATION=us-central1`
     - `ENVIRONMENT=production`
     - **`ENABLE_AUTH=true`** ✅ Security enabled
     - `RATE_LIMIT_PER_MINUTE=120`
   - **Secrets**: API_KEY from Secret Manager
   - **Public access**: Allowed (app handles auth internally)

**Output includes**:
- Service URL (e.g., `https://ard-backend-xxx.run.app`)
- Your API key retrieval command
- Test commands

**Verify deployment**:
```bash
gcloud run services list --region=us-central1 --project=periodicdent42
```

---

### Step 5: Test the Deployment

**Retrieve your API key**:
```bash
export API_KEY=$(gcloud secrets versions access latest --secret=api-key --project=periodicdent42)
echo "API Key: $API_KEY"
```

**Test health endpoint**:
```bash
export SERVICE_URL=$(gcloud run services describe ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --format='value(status.url)')

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

**Test without API key (should fail)**:
```bash
curl $SERVICE_URL/health
```

**Expected response** (401 Unauthorized):
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

**Expected**: Server-Sent Events stream with preliminary and final responses.

---

### Step 6: Configure CORS (If Using Web Frontend)

**If you have a web frontend**, configure allowed origins:

```bash
gcloud run services update ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --update-env-vars="ALLOWED_ORIGINS=https://your-frontend-domain.com,https://app.your-domain.com"
```

**For multiple domains**, use comma-separated list:
```bash
--update-env-vars="ALLOWED_ORIGINS=https://app.example.com,https://dashboard.example.com"
```

**Verify**:
```bash
gcloud run services describe ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --format='value(spec.template.spec.containers[0].env)'
```

---

### Step 7: Set Up Monitoring and Alerting

**Enable Cloud Logging** (already enabled by default):
```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend" \
  --limit=50 \
  --project=periodicdent42
```

**Create log-based metrics for security events**:

```bash
# 401 Unauthorized attempts
gcloud logging metrics create unauthorized_requests \
  --description="Count of 401 unauthorized API requests" \
  --log-filter='resource.type="cloud_run_revision"
    resource.labels.service_name="ard-backend"
    httpRequest.status=401' \
  --project=periodicdent42

# 429 Rate limited requests
gcloud logging metrics create rate_limited_requests \
  --description="Count of 429 rate limited requests" \
  --log-filter='resource.type="cloud_run_revision"
    resource.labels.service_name="ard-backend"
    httpRequest.status=429' \
  --project=periodicdent42
```

**Create alerting policies**:

```bash
# Alert on spike in 401 errors (potential attack)
gcloud alpha monitoring policies create \
  --notification-channels=YOUR_NOTIFICATION_CHANNEL_ID \
  --display-name="Unauthorized Access Spike" \
  --condition-display-name="More than 50 401 errors in 5 minutes" \
  --condition-threshold-value=50 \
  --condition-threshold-duration=300s \
  --project=periodicdent42
```

**View in Cloud Console**:
- Logs: https://console.cloud.google.com/logs
- Metrics: https://console.cloud.google.com/monitoring

---

### Step 8: Security Verification Checklist

Run through this checklist to verify security is properly configured:

- [ ] **API Key Works**: `curl -H "x-api-key: $API_KEY" $SERVICE_URL/health` returns 200
- [ ] **No API Key Fails**: `curl $SERVICE_URL/health` returns 401
- [ ] **Invalid API Key Fails**: `curl -H "x-api-key: invalid" $SERVICE_URL/health` returns 401
- [ ] **Rate Limiting Works**: Make 121 requests in 1 minute, 121st should return 429
- [ ] **CORS Configured**: Set `ALLOWED_ORIGINS` if using web frontend
- [ ] **Logs Show Security Events**: Check Cloud Logging for authentication failures
- [ ] **No Stack Traces**: Trigger an error, verify response doesn't contain stack trace
- [ ] **Security Headers Present**: `curl -I $SERVICE_URL/health` shows security headers

**Rate limiting test**:
```bash
# Test rate limiting (should get 429 after 120 requests)
for i in {1..125}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -H "x-api-key: $API_KEY" \
    $SERVICE_URL/health
done | tail -5
```

Expected last few: `200 200 200 429 429`

---

## 🔑 API Key Management

### Rotating API Keys

**Best practice**: Rotate API keys every 90 days.

```bash
# 1. Generate new key
NEW_KEY=$(openssl rand -hex 32)

# 2. Add new version to Secret Manager
echo -n "$NEW_KEY" | gcloud secrets versions add api-key \
  --data-file=- \
  --project=periodicdent42

# 3. Verify new version
gcloud secrets versions list api-key --project=periodicdent42

# 4. Cloud Run will automatically use latest version (restart if needed)
gcloud run services update ard-backend \
  --region=us-central1 \
  --project=periodicdent42

# 5. Distribute new key to clients

# 6. After all clients updated, disable old version
gcloud secrets versions disable VERSION_NUMBER \
  --secret=api-key \
  --project=periodicdent42
```

### Distributing API Keys to Clients

**For programmatic access**:
```python
# Python client example
import requests

API_KEY = "your-api-key-here"  # Load from environment variable
SERVICE_URL = "https://ard-backend-xxx.run.app"

response = requests.post(
    f"{SERVICE_URL}/api/reasoning/query",
    headers={"x-api-key": API_KEY},
    json={"query": "Suggest an experiment"}
)
```

**For curl/testing**:
```bash
# Save to environment variable
export ARD_API_KEY="your-api-key-here"

# Use in requests
curl -H "x-api-key: $ARD_API_KEY" $SERVICE_URL/health
```

---

## 🚨 Troubleshooting

### Issue: "Permission denied" when creating secrets

**Solution**:
```bash
# Ensure you have Secret Manager Admin role
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="user:your-email@example.com" \
  --role="roles/secretmanager.admin"
```

### Issue: API returns 401 even with correct API key

**Diagnosis**:
```bash
# Check if ENABLE_AUTH is set
gcloud run services describe ard-backend \
  --region=us-central1 \
  --format='value(spec.template.spec.containers[0].env)' | grep ENABLE_AUTH

# Check if API_KEY secret is accessible
gcloud secrets versions access latest --secret=api-key
```

**Solution**: Redeploy with correct environment variables.

### Issue: API returns 429 (rate limited)

**Diagnosis**: Check current rate limit setting.

**Solution**: Increase rate limit if legitimate high traffic:
```bash
gcloud run services update ard-backend \
  --region=us-central1 \
  --update-env-vars="RATE_LIMIT_PER_MINUTE=300"
```

### Issue: CORS errors in browser

**Diagnosis**: Check if `ALLOWED_ORIGINS` is set.

**Solution**:
```bash
gcloud run services update ard-backend \
  --region=us-central1 \
  --update-env-vars="ALLOWED_ORIGINS=https://your-frontend.com"
```

### Issue: "Service not initialized" errors

**Diagnosis**: Check Cloud Logging for Vertex AI initialization errors.

**Solution**: Ensure Vertex AI API is enabled and service account has proper permissions.

---

## 📊 Monitoring Dashboard

### View Key Metrics

```bash
# Total requests
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_count"'

# Request latency
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_latencies"'

# Container CPU utilization
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/cpu/utilizations"'
```

### Create Custom Dashboard

1. Go to Cloud Console: https://console.cloud.google.com/monitoring/dashboards
2. Click "Create Dashboard"
3. Add charts for:
   - Request count (grouped by status code)
   - 401/429 error rate
   - Request latency (p50, p95, p99)
   - Container CPU/memory

---

## 🔐 Security Best Practices

### DO ✅

- ✅ Store API keys in environment variables, never in code
- ✅ Rotate API keys every 90 days
- ✅ Use different API keys for dev/staging/prod
- ✅ Monitor 401/429 responses for anomalies
- ✅ Set `ALLOWED_ORIGINS` for production web frontends
- ✅ Enable Cloud Armor for DDoS protection (high-traffic deployments)
- ✅ Review Cloud Logging regularly
- ✅ Set up alerting for security events

### DON'T ❌

- ❌ Commit API keys to Git
- ❌ Share API keys via email/Slack
- ❌ Use `ALLOWED_ORIGINS=*` in production
- ❌ Disable authentication (`ENABLE_AUTH=false`) in production
- ❌ Ignore rate limit violations
- ❌ Expose service URL publicly without authentication

---

## 📞 Support

### Documentation
- [Security Architecture](docs/SECURITY.md)
- [API Documentation](https://your-service-url.run.app/docs)
- [Google Cloud Deployment Guide](docs/google_cloud_deployment.md)

### Getting Help
- Security issues: security@example.com
- Technical support: B@thegoatnote.com
- Emergency incidents: oncall@example.com

---

## ✅ Deployment Complete!

Your secure, production-ready AI service is now live with:

- 🔐 API key authentication
- 🛡️ Rate limiting (120 req/min)
- 🌐 CORS protection
- 📝 Comprehensive logging
- 🔒 Security headers
- 🚀 Auto-scaling (1-10 instances)
- ☁️ Serverless deployment

**Next Steps**:
1. Save your API key securely
2. Configure `ALLOWED_ORIGINS` for your frontend
3. Set up monitoring alerts
4. Distribute API keys to authorized clients
5. Monitor Cloud Logging for security events

**Test your deployment**: See Step 5 above for test commands.

---

**Version**: 1.0  
**Last Updated**: October 1, 2025  
**Maintainer**: Periodic Labs Security Team

