# Security Quick Reference

**One-page reference for common security operations.**

---

## 🔑 API Key Operations

### Retrieve Current API Key
```bash
gcloud secrets versions access latest --secret=api-key --project=periodicdent42
```

### Generate New API Key
```bash
openssl rand -hex 32
```

### Rotate API Key
```bash
# Generate and add new version
NEW_KEY=$(openssl rand -hex 32)
echo -n "$NEW_KEY" | gcloud secrets versions add api-key --data-file=- --project=periodicdent42

# Restart service to use new key
gcloud run services update ard-backend --region=us-central1 --project=periodicdent42

# Disable old version after clients updated
gcloud secrets versions disable VERSION_NUMBER --secret=api-key --project=periodicdent42
```

---

## 🧪 Testing Endpoints

### Health Check (Authenticated)
```bash
API_KEY=$(gcloud secrets versions access latest --secret=api-key --project=periodicdent42)
SERVICE_URL=$(gcloud run services describe ard-backend --region=us-central1 --project=periodicdent42 --format='value(status.url)')

curl -H "x-api-key: $API_KEY" $SERVICE_URL/health
```

### Test Without Auth (Should Fail with 401)
```bash
curl $SERVICE_URL/health
```

### Reasoning Endpoint
```bash
curl -X POST $SERVICE_URL/api/reasoning/query \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your query here"}'
```

### Test Rate Limiting
```bash
# Make 125 requests rapidly (limit is 120/min)
for i in {1..125}; do
  curl -s -o /dev/null -w "%{http_code}\n" -H "x-api-key: $API_KEY" $SERVICE_URL/health
done | tail -10
# Last 5 should be 429
```

---

## ⚙️ Configuration Changes

### Update Rate Limit
```bash
gcloud run services update ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --update-env-vars="RATE_LIMIT_PER_MINUTE=300"
```

### Update CORS Origins
```bash
gcloud run services update ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --update-env-vars="ALLOWED_ORIGINS=https://app.example.com,https://dashboard.example.com"
```

### Enable/Disable Authentication
```bash
# Enable
gcloud run services update ard-backend \
  --region=us-central1 \
  --update-env-vars="ENABLE_AUTH=true"

# Disable (NOT recommended for production)
gcloud run services update ard-backend \
  --region=us-central1 \
  --update-env-vars="ENABLE_AUTH=false"
```

---

## 📊 Monitoring

### View Recent Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend" \
  --limit=50 \
  --project=periodicdent42 \
  --format=json
```

### View 401 Errors (Unauthorized Attempts)
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND httpRequest.status=401" \
  --limit=20 \
  --project=periodicdent42
```

### View 429 Errors (Rate Limited)
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND httpRequest.status=429" \
  --limit=20 \
  --project=periodicdent42
```

### Live Tail Logs
```bash
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend" \
  --project=periodicdent42
```

---

## 🚨 Incident Response

### Suspected API Key Compromise

```bash
# 1. Immediately rotate API key
NEW_KEY=$(openssl rand -hex 32)
echo -n "$NEW_KEY" | gcloud secrets versions add api-key --data-file=- --project=periodicdent42

# 2. Force service restart
gcloud run services update ard-backend --region=us-central1 --project=periodicdent42

# 3. Check recent unauthorized access
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND httpRequest.status=401" \
  --limit=100 \
  --format=json > unauthorized_access_$(date +%Y%m%d_%H%M%S).json

# 4. Disable old key version
gcloud secrets versions list api-key --project=periodicdent42
gcloud secrets versions disable OLD_VERSION --secret=api-key --project=periodicdent42

# 5. Distribute new key to authorized clients
echo "New API Key: $NEW_KEY"
```

### DDoS or Rate Limit Abuse

```bash
# 1. Identify attacking IPs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND httpRequest.status=429" \
  --limit=100 \
  --format="value(httpRequest.remoteIp)" | sort | uniq -c | sort -nr

# 2. Temporarily reduce rate limit (if genuine traffic spike)
gcloud run services update ard-backend \
  --region=us-central1 \
  --update-env-vars="RATE_LIMIT_PER_MINUTE=30"

# 3. Enable Cloud Armor (for persistent attacks)
# See: https://cloud.google.com/armor/docs/configure-security-policies
```

---

## 🔍 Troubleshooting

### Check Current Configuration
```bash
gcloud run services describe ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --format=yaml
```

### Check Environment Variables
```bash
gcloud run services describe ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --format='value(spec.template.spec.containers[0].env)'
```

### Check Secrets Configuration
```bash
gcloud run services describe ard-backend \
  --region=us-central1 \
  --project=periodicdent42 \
  --format='value(spec.template.spec.containers[0].env[].valueFrom)'
```

### Verify Service Account Permissions
```bash
SERVICE_ACCOUNT="ard-backend@periodicdent42.iam.gserviceaccount.com"

# Check IAM bindings
gcloud projects get-iam-policy periodicdent42 \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:$SERVICE_ACCOUNT"
```

### Check Secret Access
```bash
# Try accessing secret as service account
gcloud secrets versions access latest \
  --secret=api-key \
  --project=periodicdent42 \
  --impersonate-service-account=ard-backend@periodicdent42.iam.gserviceaccount.com
```

---

## 📋 Status Codes

| Code | Meaning | Cause | Solution |
|------|---------|-------|----------|
| 200 | OK | Success | ✅ |
| 401 | Unauthorized | Missing/invalid API key | Add `x-api-key` header |
| 429 | Too Many Requests | Rate limit exceeded | Wait or increase limit |
| 500 | Internal Server Error | Server issue | Check logs |
| 503 | Service Unavailable | Service starting or down | Wait/check status |

---

## 🔐 Security Headers

Check security headers are present:
```bash
curl -I $SERVICE_URL/health | grep -E "(Strict-Transport|X-Content-Type|X-Frame|Referrer-Policy)"
```

Expected:
```
Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
```

---

## 📞 Emergency Contacts

- **Security Issues**: security@example.com
- **On-Call Engineer**: oncall@example.com
- **Technical Lead**: B@thegoatnote.com

---

## 📚 Full Documentation

- [Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)
- [Security Architecture](docs/SECURITY.md)
- [Local Development Setup](LOCAL_DEV_SETUP.md)
- [API Documentation](https://<your-service-url>/docs)

---

**Keep this handy for quick operations!** 🔒

