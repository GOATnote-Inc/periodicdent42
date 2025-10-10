# Production Monitoring Guide - HTC Backend

**Service**: `ard-backend`  
**Region**: `us-central1`  
**Current Revision**: `00052-zl2`  
**Status**: ✅ Production

---

## Quick Health Checks

### 1. Service Status

```bash
gcloud run services describe ard-backend --region=us-central1 --format="value(status.url,status.latestReadyRevisionName)"
```

### 2. Endpoint Test

```bash
curl -s https://ard-backend-dydzexswua-uc.a.run.app/health | jq '.'
curl -s https://ard-backend-dydzexswua-uc.a.run.app/api/htc/health | jq '.'
```

### 3. Revision Traffic

```bash
gcloud run services describe ard-backend --region=us-central1 --format="table(spec.traffic[].revisionName,spec.traffic[].percent)"
```

---

## Real-Time Logs

### Stream All Logs

```bash
gcloud run services logs tail ard-backend --region=us-central1
```

### Filter by Severity

```bash
# Errors only
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND severity>=ERROR" --limit=50 --format=json

# Info and above
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND severity>=INFO" --limit=100 --format=json
```

### HTC-Specific Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND textPayload=~'HTC'" --limit=50
```

### Error Tracking

```bash
# Last 24 hours of errors
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND severity>=ERROR AND timestamp>=\"$(date -u -d '24 hours ago' '+%Y-%m-%dT%H:%M:%SZ')\"" --limit=100
```

---

## Performance Metrics

### 1. Request Latency

```bash
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_latencies" AND resource.labels.service_name="ard-backend"' \
  --format=json | jq '.[].points[] | {time: .interval.endTime, latency: .value.distributionValue.mean}'
```

### 2. Request Count

```bash
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_count" AND resource.labels.service_name="ard-backend"' \
  --format=json
```

### 3. Instance Count

```bash
gcloud run services describe ard-backend --region=us-central1 --format="value(status.conditions[0].message)"
```

### 4. CPU Utilization

```bash
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/cpu/utilizations" AND resource.labels.service_name="ard-backend"' \
  --format=json
```

### 5. Memory Usage

```bash
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/container/memory/utilizations" AND resource.labels.service_name="ard-backend"' \
  --format=json
```

---

## Database Monitoring

### 1. Check Database Connection

```bash
# Verify Cloud SQL Proxy is running locally
ps aux | grep cloud-sql-proxy | grep -v grep

# Test database connectivity
export PGPASSWORD=ard_secure_password_2024
psql -h localhost -p 5433 -U ard_user -d ard_intelligence -c "SELECT 1 as health_check;"
```

### 2. Check Migration Status

```bash
cd /Users/kiteboard/periodicdent42/app
source venv/bin/activate
export DB_USER=ard_user DB_PASSWORD=ard_secure_password_2024 DB_NAME=ard_intelligence DB_HOST=localhost DB_PORT=5433
alembic current
```

### 3. Query HTC Predictions

```bash
export PGPASSWORD=ard_secure_password_2024
psql -h localhost -p 5433 -U ard_user -d ard_intelligence << 'EOF'
SELECT 
  COUNT(*) as total_predictions,
  AVG(tc_predicted) as avg_tc,
  MIN(tc_predicted) as min_tc,
  MAX(tc_predicted) as max_tc
FROM htc_predictions;
EOF
```

### 4. Recent Predictions

```bash
export PGPASSWORD=ard_secure_password_2024
psql -h localhost -p 5433 -U ard_user -d ard_intelligence << 'EOF'
SELECT 
  composition,
  tc_predicted,
  confidence_level,
  created_at
FROM htc_predictions
ORDER BY created_at DESC
LIMIT 10;
EOF
```

---

## Container Health

### 1. List Recent Revisions

```bash
gcloud run revisions list --service=ard-backend --region=us-central1 --limit=10 --format="table(metadata.name,status.conditions[0].status,metadata.creationTimestamp)"
```

### 2. Describe Current Revision

```bash
gcloud run revisions describe ard-backend-00052-zl2 --region=us-central1 --format=yaml
```

### 3. Check Container Image

```bash
gcloud container images list-tags gcr.io/periodicdent42/ard-backend --limit=5 --format="table(tags,timestamp)"
```

---

## Alerting & Notifications

### Set Up Alerts (One-Time Setup)

```bash
# Create alert policy for high error rate
gcloud alpha monitoring policies create \
  --notification-channels=YOUR_CHANNEL_ID \
  --display-name="HTC Backend Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s \
  --condition-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="ard-backend" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class="5xx"'
```

### Check Existing Alerts

```bash
gcloud alpha monitoring policies list --filter="displayName:HTC"
```

---

## Traffic Management

### 1. Gradual Rollout (Canary Deployment)

```bash
# Route 10% to new revision, 90% to old
gcloud run services update-traffic ard-backend \
  --region=us-central1 \
  --to-revisions=ard-backend-00052-zl2=90,ard-backend-00051-m7c=10
```

### 2. Rollback

```bash
# Route 100% back to previous revision
gcloud run services update-traffic ard-backend \
  --region=us-central1 \
  --to-revisions=ard-backend-00051-m7c=100
```

### 3. Route to Latest

```bash
gcloud run services update-traffic ard-backend \
  --region=us-central1 \
  --to-latest
```

---

## Debugging

### 1. Get Detailed Error Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND resource.labels.revision_name=ard-backend-00052-zl2 AND (severity=ERROR OR textPayload=~'Traceback')" --limit=50 --format=json | jq '.[] | {timestamp: .timestamp, message: .textPayload}'
```

### 2. Check Startup Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND resource.labels.revision_name=ard-backend-00052-zl2 AND textPayload=~'Starting'" --limit=20
```

### 3. Import Error Tracking

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND textPayload=~'ImportError'" --limit=50
```

### 4. Database Connection Issues

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND (textPayload=~'psycopg2' OR textPayload=~'database' OR textPayload=~'Cloud SQL')" --limit=50
```

---

## Automated Monitoring Script

### Create Monitor Script

```bash
#!/bin/bash
# monitor_htc.sh - Automated health monitoring

SERVICE_URL="https://ard-backend-dydzexswua-uc.a.run.app"
SLACK_WEBHOOK="YOUR_SLACK_WEBHOOK_URL"  # Optional

check_health() {
    response=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health")
    if [ "$response" -eq 200 ]; then
        echo "✅ Main health check: OK"
        return 0
    else
        echo "❌ Main health check failed: HTTP $response"
        return 1
    fi
}

check_htc() {
    response=$(curl -s "${SERVICE_URL}/api/htc/health")
    status=$(echo "$response" | jq -r '.status')
    enabled=$(echo "$response" | jq -r '.enabled')
    
    if [ "$status" = "ok" ] && [ "$enabled" = "true" ]; then
        echo "✅ HTC health check: OK"
        return 0
    else
        echo "❌ HTC health check failed: $response"
        return 1
    fi
}

test_prediction() {
    response=$(curl -s -X POST "${SERVICE_URL}/api/htc/predict" \
        -H "Content-Type: application/json" \
        -d '{"composition": "MgB2", "pressure_gpa": 0.0}')
    
    tc=$(echo "$response" | jq -r '.tc_predicted')
    
    if [ "$tc" != "null" ] && [ -n "$tc" ]; then
        echo "✅ Prediction test: OK (MgB2 Tc = $tc K)"
        return 0
    else
        echo "❌ Prediction test failed: $response"
        return 1
    fi
}

# Run checks
echo "🔍 HTC Backend Health Monitor"
echo "=============================="
echo ""

all_passed=true

check_health || all_passed=false
check_htc || all_passed=false  
test_prediction || all_passed=false

echo ""
if [ "$all_passed" = true ]; then
    echo "✅ All checks passed"
    exit 0
else
    echo "❌ Some checks failed"
    # Optional: Send Slack notification
    # curl -X POST -H 'Content-type: application/json' \
    #   --data '{"text":"HTC Backend health check failed!"}' \
    #   "$SLACK_WEBHOOK"
    exit 1
fi
```

### Schedule with Cron

```bash
# Add to crontab (every 5 minutes)
*/5 * * * * /path/to/monitor_htc.sh >> /var/log/htc_monitor.log 2>&1
```

---

## Dashboard URLs

### Google Cloud Console

- **Service Overview**: https://console.cloud.google.com/run/detail/us-central1/ard-backend?project=periodicdent42
- **Logs Explorer**: https://console.cloud.google.com/logs/query?project=periodicdent42
- **Metrics**: https://console.cloud.google.com/monitoring/metrics-explorer?project=periodicdent42
- **Cloud SQL**: https://console.cloud.google.com/sql/instances/ard-intelligence-db?project=periodicdent42

### API Documentation

- **Interactive Docs**: https://ard-backend-dydzexswua-uc.a.run.app/docs
- **OpenAPI Spec**: https://ard-backend-dydzexswua-uc.a.run.app/openapi.json

---

## Key Metrics to Watch (First 48 Hours)

### Critical Metrics

| Metric | Expected | Action if Exceeded |
|--------|----------|-------------------|
| **Error Rate** | < 1% | Check logs, consider rollback |
| **P99 Latency** | < 2s | Investigate slow endpoints |
| **Instance Count** | 0-3 | Normal (autoscaling) |
| **Memory Usage** | < 80% | Increase memory if sustained |
| **CPU Usage** | < 70% | Increase CPU if sustained |

### Business Metrics

| Metric | How to Track |
|--------|--------------|
| **Predictions per Hour** | Query `htc_predictions` table |
| **Unique Compositions** | `SELECT COUNT(DISTINCT composition) FROM htc_predictions` |
| **Average Confidence** | `SELECT AVG(CASE confidence_level WHEN 'high' THEN 1.0 WHEN 'medium' THEN 0.5 ELSE 0.0 END)` |

---

## Incident Response

### 1. Service Down

```bash
# Check revision status
gcloud run revisions list --service=ard-backend --region=us-central1 --limit=5

# Check recent logs
gcloud run services logs tail ard-backend --region=us-central1 --limit=100

# Rollback if needed
gcloud run services update-traffic ard-backend --region=us-central1 --to-revisions=ard-backend-00051-m7c=100
```

### 2. High Error Rate

```bash
# Get error distribution
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND severity>=ERROR" --limit=100 --format=json | jq '.[] | .textPayload' | sort | uniq -c | sort -rn

# Check for specific errors
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND textPayload=~'YOUR_ERROR_PATTERN'" --limit=50
```

### 3. Database Issues

```bash
# Check Cloud SQL status
gcloud sql instances describe ard-intelligence-db --format="value(state,settings.tier)"

# Verify connectivity
curl -s https://ard-backend-dydzexswua-uc.a.run.app/health | jq '.database_connected'
```

---

## Weekly Maintenance

### 1. Check for Updates

```bash
gcloud components update
```

### 2. Review Logs

```bash
# Generate weekly error summary
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ard-backend AND severity>=ERROR AND timestamp>=\"$(date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%SZ')\"" --format=json | jq -r '.[] | "\(.timestamp) \(.severity) \(.textPayload)"' > weekly_errors.log
```

### 3. Database Cleanup (if needed)

```bash
# Archive old predictions (keep last 90 days)
export PGPASSWORD=ard_secure_password_2024
psql -h localhost -p 5433 -U ard_user -d ard_intelligence << 'EOF'
DELETE FROM htc_predictions 
WHERE created_at < NOW() - INTERVAL '90 days';
EOF
```

---

## Contact Information

**On-Call**: b@thegoatnote.com  
**Escalation**: GOATnote DevOps Team  
**Documentation**: `/docs/HTC_*.md`

---

**Last Updated**: October 10, 2025  
**Next Review**: October 12, 2025 (48-hour check)

