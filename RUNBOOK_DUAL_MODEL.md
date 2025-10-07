# Dual-Model Audit & SSE Hardening - Runbook

**Date**: October 7, 2025  
**Branch**: `feat/dual-model-audit-sse-hardening`  
**Contact**: Principal Engineer

---

## Quick Reference

### Run Tests

```bash
# All dual-streaming tests
pytest -q app/tests/test_dual_streaming.py

# Specific test
pytest -q app/tests/test_dual_streaming.py::test_sse_ordering_normal

# With verbose output
pytest -v app/tests/test_dual_streaming.py

# With coverage
pytest --cov=src.api.main --cov=src.reasoning.dual_agent app/tests/test_dual_streaming.py
```

### Set Timeouts

```bash
# Default: FLASH_TIMEOUT_S=5, PRO_TIMEOUT_S=45
export FLASH_TIMEOUT_S=5
export PRO_TIMEOUT_S=45

# For testing aggressive timeouts
export FLASH_TIMEOUT_S=2
export PRO_TIMEOUT_S=20

# Restart server to pick up new values
cd app && uvicorn src.api.main:app --reload
```

### Enable Debug Logs for Single Run ID

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Grep logs for specific run_id
tail -f server.log | grep "run_id=550e8400-e29b-41d4-a716-446655440000"
```

### Query Audit Records

```bash
# Via API
curl http://localhost:8080/api/experiments | jq '.experiments[] | {id, query, flash_latency_ms, pro_latency_ms}'

# Via database (Cloud SQL Proxy required)
export PGPASSWORD=ard_secure_password_2024
psql -h localhost -p 5433 -U ard_user -d ard_intelligence << EOF
SELECT 
    id,
    query,
    flash_latency_ms,
    pro_latency_ms,
    created_at
FROM experiment_runs
ORDER BY created_at DESC
LIMIT 10;
EOF
```

### View Metrics

```python
# In Python shell or script
from src.utils.metrics import get_metrics

metrics = get_metrics()

# View all metrics
print(metrics.get_all_metrics())

# Flash latency stats
flash_stats = metrics.get_histogram_stats("latency_ms", labels={"model": "flash"})
print(f"Flash latency: {flash_stats}")

# Timeout counters
flash_timeouts = metrics.get_counter("timeouts_total", labels={"model": "flash"})
pro_timeouts = metrics.get_counter("timeouts_total", labels={"model": "pro"})
print(f"Timeouts: Flash={flash_timeouts}, Pro={pro_timeouts}")
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASH_TIMEOUT_S` | 5 | Flash model timeout (seconds) |
| `PRO_TIMEOUT_S` | 45 | Pro model timeout (seconds) |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ENABLE_METRICS` | true | Enable metrics collection |
| `ENABLE_TRACING` | true | Enable request tracing |

### Timeout Tuning Guidelines

**Flash Timeout**:
- Baseline: P95 latency ~2s
- Recommended: 2.5x headroom = 5s
- Aggressive: 2-3s (higher timeout rate)
- Conservative: 10s (lower timeout rate)

**Pro Timeout**:
- Baseline: P95 latency ~20s
- Recommended: 2.25x headroom = 45s
- Aggressive: 30s (higher timeout rate)
- Conservative: 60s (lower timeout rate)

**Trade-offs**:
- Lower timeouts = faster failure detection, higher timeout rate
- Higher timeouts = fewer false timeouts, slower error feedback

---

## Monitoring

### Key Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `latency_ms` | Histogram | `model=flash\|pro` | End-to-end latency |
| `timeouts_total` | Counter | `model=flash\|pro` | Timeout count |
| `errors_total` | Counter | `class=<ErrorClass>` | Error count by type |
| `cancellations_total` | Counter | - | Cancellation count |
| `sse_handler_duration` | Histogram | - | Total SSE handler time |

### Alert Thresholds (Recommended)

```yaml
# Timeout rate > 1% (should be < 0.1%)
- alert: HighTimeoutRate
  expr: rate(timeouts_total[5m]) / rate(requests_total[5m]) > 0.01

# P95 latency > 2x baseline
- alert: SlowFlashResponses
  expr: histogram_quantile(0.95, latency_ms{model="flash"}) > 4000  # 4s

- alert: SlowProResponses
  expr: histogram_quantile(0.95, latency_ms{model="pro"}) > 40000  # 40s

# Error rate > 0.5%
- alert: HighErrorRate
  expr: rate(errors_total[5m]) / rate(requests_total[5m]) > 0.005
```

### Health Checks

```bash
# Basic health
curl http://localhost:8080/health

# Expected response
# {"status":"ok","vertex_initialized":true,"project_id":"periodicdent42"}

# Stream health (should complete with 'done' event)
curl -N http://localhost:8080/api/reasoning/query \
  -H "Content-Type: application/json" \
  -d '{"query":"health check","context":{}}' \
  | grep "event: done"
```

---

## Troubleshooting

### Issue: SSE Stream Hangs

**Symptoms**: Client never receives `done` event

**Diagnosis**:
```bash
# Check logs for run_id
grep "run_id=<ID>" server.log | tail -20

# Check for panics or unhandled exceptions
grep "ERROR" server.log | tail -50
```

**Resolution**:
1. Verify `finally` block in `event_stream()` executes
2. Check for uncaught exceptions before finally block
3. Ensure `done` event yield is not conditional

### Issue: High Timeout Rate

**Symptoms**: `timeouts_total` counter increasing rapidly

**Diagnosis**:
```python
# Check P95 latencies
from src.utils.metrics import get_metrics
metrics = get_metrics()

flash_stats = metrics.get_histogram_stats("latency_ms", labels={"model": "flash"})
pro_stats = metrics.get_histogram_stats("latency_ms", labels={"model": "pro"})

print(f"Flash P95: {flash_stats['p95']}ms")
print(f"Pro P95: {pro_stats['p95']}ms")
```

**Resolution**:
1. If P95 approaching timeout: Increase timeout values
2. If P95 normal: Investigate Vertex AI service issues
3. Check for network issues to Vertex AI endpoints

### Issue: Audit Records Missing

**Symptoms**: Requests complete but no DB records

**Diagnosis**:
```bash
# Check logs for audit failures
grep "Audit log persistence" server.log | grep "failed"

# Check DB connectivity
psql -h localhost -p 5433 -U ard_user -d ard_intelligence -c "SELECT 1;"
```

**Resolution**:
1. Verify Cloud SQL Proxy running: `ps aux | grep cloud-sql-proxy`
2. Check DB credentials: `echo $DB_PASSWORD`
3. Increase retry attempts in `retry_async` (currently 5)
4. Check for DB table existence: `\dt experiment_runs`

### Issue: Flash/Pro Responses Swapped

**Symptoms**: "Pro response" in preliminary event

**Diagnosis**:
```bash
# Check model initialization
grep "DualModelAgent initialized" server.log

# Verify model names
curl http://localhost:8080/api/reasoning/query \
  -H "Content-Type: application/json" \
  -d '{"query":"test","context":{}}' \
  | grep -E "gemini-2.5-(flash|pro)"
```

**Resolution**:
1. Verify `get_flash_model()` returns Flash, `get_pro_model()` returns Pro
2. Check `GEMINI_FLASH_MODEL` and `GEMINI_PRO_MODEL` env vars
3. Restart server after env var changes

### Issue: Metrics Not Recording

**Symptoms**: `get_all_metrics()` returns empty or zeros

**Diagnosis**:
```python
from src.utils.metrics import get_metrics
metrics = get_metrics()

# Check if metrics enabled
print(f"Enabled: {metrics._enabled}")

# Force increment to test
metrics.increment("test_counter")
print(f"Test counter: {metrics.get_counter('test_counter')}")
```

**Resolution**:
1. Verify `ENABLE_METRICS=true` in environment
2. Check for threading issues (metrics collector is thread-safe)
3. Ensure metrics imports succeed (no circular dependencies)

---

## Performance Optimization

### Latency Improvement Opportunities

See [DESIGN_DUAL_MODEL_AUDIT.md](DESIGN_DUAL_MODEL_AUDIT.md#latency-optimization-opportunities) for detailed analysis.

**Quick Wins** (no accuracy loss):

1. **Early Flash Termination** (~1s saved in 5-10% of cases):
   ```python
   # If Pro completes first, skip Flash
   done, pending = await asyncio.wait({flash_task, pro_task}, return_when=asyncio.FIRST_COMPLETED)
   if pro_task in done and not flash_task.done():
       flash_task.cancel()
   ```

2. **Prompt Optimization** (100-300ms Flash, 500-1500ms Pro):
   ```python
   # Shorter Flash prompts
   flash_prompt = f"Quick summary: {prompt[:200]}"
   ```

**Not Recommended** (accuracy trade-offs):

1. Flash-only mode (5-10% accuracy loss)
2. Response streaming (complexity >> gain)
3. Skipping Pro (no verification)

---

## Migration & Rollback

### Deploying This Feature

```bash
# 1. Merge feature branch
git checkout main
git merge feat/dual-model-audit-sse-hardening

# 2. Deploy to staging
gcloud run deploy ard-backend-staging \
  --source . \
  --set-env-vars FLASH_TIMEOUT_S=5,PRO_TIMEOUT_S=45

# 3. Run smoke tests
curl https://ard-backend-staging-XXX.run.app/health
curl -N https://ard-backend-staging-XXX.run.app/api/reasoning/query \
  -H "Content-Type: application/json" \
  -d '{"query":"smoke test","context":{}}'

# 4. Verify audit logging
# Check database for records with run_id

# 5. Deploy to production
gcloud run deploy ard-backend \
  --source . \
  --set-env-vars FLASH_TIMEOUT_S=5,PRO_TIMEOUT_S=45
```

### Rollback Procedure

```bash
# 1. Quick rollback (previous revision)
gcloud run services update-traffic ard-backend \
  --to-revisions ard-backend-PREV-REVISION=100

# 2. Verify old version working
curl https://ard-backend-XXX.run.app/health

# 3. Monitor for 5-10 minutes
# Check that requests succeed, no 500 errors

# 4. If stable, investigate issue in new version
# If unstable, rollback further
```

**Rollback Compatibility**:
- ✅ **Database**: New code reads old schema (backward-compatible)
- ✅ **SSE**: Old clients ignore `done` event (forward-compatible)
- ✅ **Timeouts**: Missing env vars use defaults (graceful degradation)

### Data Migration

**None required** - New code uses existing `experiment_runs` table.

---

## Testing Strategy

### Unit Tests (Fast, No Dependencies)

```bash
# Test telemetry models
pytest app/tests/test_models_telemetry.py -v

# Test retry utility
pytest app/tests/test_utils_retry.py -v

# Test metrics collector
pytest app/tests/test_utils_metrics.py -v
```

### Integration Tests (With Fake Providers)

```bash
# All streaming tests
pytest app/tests/test_dual_streaming.py -v

# Specific scenario
pytest app/tests/test_dual_streaming.py::test_flash_timeout_pro_ok -v
```

### Smoke Tests (Against Real Vertex)

```bash
# Requires Vertex AI credentials
export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json

# Run smoke tests
pytest app/tests/test_reasoning_smoke.py -v
```

### Load Tests (Staging/Production)

```bash
# Use Apache Bench or similar
ab -n 100 -c 10 -p query.json -T application/json \
  https://ard-backend-staging-XXX.run.app/api/reasoning/query

# query.json:
# {"query":"load test","context":{}}
```

---

## Security Considerations

### Input Sanitization

Audit records use `sanitize_payload()` to redact PII before persistence:

```python
# From src.utils.compliance
def sanitize_payload(data: any) -> any:
    """Redact sensitive fields (email, ssn, credentials, etc.)"""
    # Implementation redacts known PII patterns
    return sanitized_data
```

### Error Redaction

Stack traces are sanitized in audit records:

```python
# Only error class name logged, not full trace
error_class = e.__class__.__name__  # "ValueError", not full traceback
```

### Token Limits

No changes to existing validation - context length limits still enforced.

### DOS Protection

Timeouts prevent resource exhaustion:
- Flash: 5s max
- Pro: 45s max
- Total SSE duration: ~50s max

---

## Support & Escalation

### Debug Checklist

1. ☐ Check server logs: `tail -f server.log | grep "run_id=<ID>"`
2. ☐ Verify timeouts: `echo $FLASH_TIMEOUT_S $PRO_TIMEOUT_S`
3. ☐ Check DB connection: `psql -h localhost -p 5433 -U ard_user -d ard_intelligence -c "SELECT 1;"`
4. ☐ Review metrics: `python -c "from src.utils.metrics import get_metrics; print(get_metrics().get_all_metrics())"`
5. ☐ Test SSE endpoint: `curl -N http://localhost:8080/api/reasoning/query -d '{"query":"debug","context":{}}' -H "Content-Type: application/json"`
6. ☐ Verify audit record: `psql -h localhost -p 5433 -U ard_user -d ard_intelligence -c "SELECT * FROM experiment_runs WHERE id='<run_id>';"`

### Escalation Path

1. **L1 Support**: Check runbook, verify configuration
2. **L2 Engineering**: Review logs, check metrics, restart services
3. **L3 Principal Engineer**: Code review, hotfix deployment

### Known Issues

1. **Flash timeout false positives during GCP maintenance**
   - **Workaround**: Increase `FLASH_TIMEOUT_S` to 10s during maintenance window
   - **Permanent fix**: Adaptive timeout learning (future enhancement)

2. **Audit logging fails on Cloud SQL connection pool exhaustion**
   - **Workaround**: Increase connection pool size in `db.py`
   - **Permanent fix**: Implement circuit breaker (future enhancement)

---

## Appendix: Acceptance Criteria Verification

| Criterion | Implementation | Test |
|-----------|---------------|------|
| ✅ Every request persisted with input, outputs, timings, tokens, versions | `_log_dual_run()` via `retry_async()` | `test_audit_persistence_complete` |
| ✅ SSE never hangs (error + done always sent) | `finally` block emits `done` event | `test_sse_always_closes` |
| ✅ Timeouts configurable, enforced, observable | `FLASH_TIMEOUT_S`, `PRO_TIMEOUT_S` + metrics | `test_flash_timeout_pro_ok`, `test_pro_timeout_flash_ok` |
| ✅ Deterministic tests for ordering, timeouts, persistence | Fake providers with `asyncio.sleep` | All tests in `test_dual_streaming.py` |
| ✅ Backward-compatible SSE schema | Added optional fields only | `test_sse_backward_compat` |

---

**End of Runbook**
