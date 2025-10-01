# Security Architecture

## Overview

The Autonomous R&D Intelligence Layer implements defense-in-depth security with multiple layers of protection against unauthorized access, abuse, and information leakage.

## Security Layers

### 1. Authentication & Authorization

**Implementation**: API key-based authentication via `AuthenticationMiddleware`

**Features**:
- Header-based authentication (default: `x-api-key`)
- Configurable exempt paths (e.g., `/docs`, `/`, `/static`)
- Graceful degradation when auth is disabled for local development
- Structured error responses without information leakage

**Configuration**:
```bash
# Environment variables
ENABLE_AUTH=true                    # Enable authentication (default: false)
API_KEY=your-secure-api-key         # Set via Secret Manager in production
```

**Best Practices**:
- ✅ Generate cryptographically random API keys (min 32 bytes)
- ✅ Store API keys in Google Cloud Secret Manager, never in code
- ✅ Rotate API keys every 90 days
- ✅ Use different keys for dev/staging/prod environments
- ✅ Enable auth in all non-local environments

**Exempt Paths**:
By default, the following paths are exempt from authentication:
- `/` - Root UI
- `/docs` - API documentation
- `/openapi.json` - OpenAPI schema
- `/static` - Static assets

**Protected Endpoints**:
- `/health` - Health check (authenticated to prevent recon)
- `/api/reasoning/query` - AI reasoning endpoint
- `/api/storage/*` - Storage endpoints

### 2. Cross-Origin Resource Sharing (CORS)

**Implementation**: Restrictive CORS policy via `CORSMiddleware`

**Features**:
- Origin whitelist (no wildcards in production)
- Credentials disabled by default
- Explicit method and header restrictions
- Development fallback for localhost

**Configuration**:
```bash
# Comma-separated list of allowed origins
ALLOWED_ORIGINS=https://app.example.com,https://dashboard.example.com
```

**Security Controls**:
- ✅ `allow_credentials=False` - Prevents session hijacking
- ✅ `allow_origins` - Explicit whitelist only
- ✅ `allow_methods` - Limited to `GET`, `POST`, `OPTIONS`
- ✅ `allow_headers` - Explicit list of safe headers

**Development Mode**:
When `ALLOWED_ORIGINS` is empty and `ENVIRONMENT=development`, the following origins are permitted:
- `http://localhost`
- `http://localhost:3000`
- `http://127.0.0.1`
- `http://127.0.0.1:3000`

### 3. Rate Limiting

**Implementation**: IP-based sliding window rate limiter via `RateLimiterMiddleware`

**Features**:
- Per-IP request throttling
- Sliding window algorithm (60-second window)
- Configurable limit
- Exempt from OPTIONS requests

**Configuration**:
```bash
RATE_LIMIT_PER_MINUTE=60  # Max requests per IP per minute
```

**Tuning Guidelines**:
- **Development**: 60/min (default)
- **Production (authenticated users)**: 120/min
- **Production (public endpoints)**: 30/min
- **High-load environments**: Consider Redis-backed rate limiting

**Response Codes**:
- `429 Too Many Requests` - Rate limit exceeded
- Error payload: `{"error": "Too many requests", "code": "rate_limited"}`

### 4. Security Headers

**Implementation**: Security headers middleware via `SecurityHeadersMiddleware`

**Headers Applied**:
```
Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Permissions-Policy: accelerometer=(), autoplay=*, camera=(), ...
```

**Protection Against**:
- ✅ Man-in-the-middle attacks (HSTS)
- ✅ MIME-sniffing attacks (X-Content-Type-Options)
- ✅ Clickjacking (X-Frame-Options)
- ✅ Referrer leakage (Referrer-Policy)
- ✅ Unauthorized feature access (Permissions-Policy)

### 5. Error Handling & Information Disclosure

**Implementation**: Structured error responses with sanitized messages

**Key Principles**:
- ❌ Never expose stack traces to clients
- ❌ Never return raw exception messages
- ✅ Log detailed errors server-side only
- ✅ Return generic user-facing messages with error codes

**Error Response Format**:
```json
{
  "error": "Failed to store experiment",
  "code": "storage_error"
}
```

**SSE Error Format**:
```
event: error
data: {"error": "Internal server error", "code": "stream_failure"}
```

**Protected Information**:
- File paths
- Database connection strings
- Stack traces
- Internal service names
- Version information

## Deployment Checklist

### Pre-Production

- [ ] Generate and store API key in Secret Manager
- [ ] Set `ENABLE_AUTH=true`
- [ ] Configure `ALLOWED_ORIGINS` with production domains
- [ ] Set appropriate `RATE_LIMIT_PER_MINUTE` for expected load
- [ ] Enable Cloud Monitoring and Cloud Logging
- [ ] Review IAM policies for least privilege
- [ ] Enable VPC Service Controls (if applicable)
- [ ] Configure Cloud Armor for DDoS protection

### Security Testing

- [ ] Verify authentication blocks unauthenticated requests
- [ ] Test CORS with unauthorized origins
- [ ] Confirm rate limiting triggers at threshold
- [ ] Validate security headers in production responses
- [ ] Test error responses for information leakage
- [ ] Perform vulnerability scanning with Cloud Security Scanner
- [ ] Conduct penetration testing (if required)

### Monitoring

- [ ] Set up alerting for 401/403 spike (potential attack)
- [ ] Monitor 429 rate (may indicate DDoS or abuse)
- [ ] Track API key rotation schedule
- [ ] Review audit logs for suspicious patterns
- [ ] Monitor Cloud Run metrics for anomalies

## Security Testing

Run security-focused tests:
```bash
cd app
source venv/bin/activate
pytest tests/test_security.py -v
```

## Incident Response

### Suspected API Key Compromise

1. **Immediate Actions**:
   - Rotate API key in Secret Manager
   - Force restart Cloud Run service
   - Review audit logs for unauthorized access

2. **Investigation**:
   - Check Cloud Logging for 401 errors before compromise
   - Identify scope of unauthorized access
   - Review data access patterns

3. **Remediation**:
   - Generate new API key with higher entropy
   - Update key distribution process
   - Document incident in security log

### Rate Limit Abuse

1. **Identify Source**:
   - Check Cloud Logging for 429 responses
   - Extract offending IP addresses
   - Determine if coordinated attack

2. **Mitigation**:
   - Add IP to block list in Cloud Armor
   - Reduce rate limit if genuine load spike
   - Scale Cloud Run instances if needed

3. **Follow-up**:
   - Analyze attack patterns
   - Adjust rate limiting strategy
   - Consider CAPTCHA for sensitive endpoints

## Security Contacts

- **Security Issues**: security@example.com
- **Incident Response**: oncall@example.com
- **Cloud Security Team**: cloud-security@example.com

## Compliance

### Data Protection
- API requests and responses are logged (ensure no PII in logs)
- Experiment data encrypted at rest (GCS default encryption)
- TLS 1.3 enforced for all connections (Cloud Run default)

### Audit Requirements
- All authentication failures logged
- All storage operations logged with experiment IDs
- Rate limit violations logged with source IP
- Security header violations logged

### HIPAA Compliance
If handling Protected Health Information (PHI):
- ✅ Enable Cloud Logging with retention policies
- ✅ Configure BAA with Google Cloud
- ✅ Use Customer-Managed Encryption Keys (CMEK)
- ✅ Implement additional access controls
- ✅ Enable audit logging for all GCS buckets
- ✅ Ensure all endpoints require authentication
- ✅ Review error messages for PHI leakage

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Google Cloud Security Best Practices](https://cloud.google.com/security/best-practices)
- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**Last Updated**: October 1, 2025  
**Version**: 1.0  
**Owner**: Periodic Labs Security Team

