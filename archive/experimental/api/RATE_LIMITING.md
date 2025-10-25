# Rate Limiting Documentation

Complete guide for API rate limiting in matprov.

## Overview

Rate limiting prevents API abuse by limiting the number of requests a client can make within a time window. The matprov API implements intelligent rate limiting with different limits for authenticated vs. anonymous users.

## Rate Limits

| User Type | Default Limit | Can Override |
|-----------|--------------|--------------|
| **Anonymous** (by IP) | 100 requests/hour | Yes (env var) |
| **Authenticated** (by user ID) | 1,000 requests/hour | Yes (env var) |
| **Global** | 60 requests/minute | Yes (env var) |
| **Health check** | Unlimited | No |
| **Expensive endpoints** | Custom limits | Per endpoint |

## Configuration

### Environment Variables

```bash
# Rate limit settings
RATE_LIMIT_ANONYMOUS=100/hour          # For unauthenticated requests
RATE_LIMIT_AUTHENTICATED=1000/hour     # For authenticated requests
RATE_LIMIT_PER_MINUTE=60/minute        # Global per-minute limit

# Redis backend (optional, for production)
USE_REDIS_RATE_LIMIT=true              # Enable Redis (default: false)
REDIS_URL=redis://localhost:6379/0     # Redis connection string
```

### Development Mode (In-Memory)

By default, rate limiting uses in-memory storage, suitable for development:

```bash
# No configuration needed
uvicorn api.main:app --reload
```

### Production Mode (Redis)

For production with multiple workers, use Redis:

```bash
# Start Redis
docker-compose -f docker-compose.rate-limit.yml up -d

# Set environment
export USE_REDIS_RATE_LIMIT=true
export REDIS_URL=redis://localhost:6379/0

# Start API
uvicorn api.main:app --workers 4
```

## How It Works

### 1. Identification

Requests are identified by:
- **Authenticated users**: User ID from JWT token
- **Anonymous users**: IP address

```python
# Authenticated: "user:123"
# Anonymous: "127.0.0.1"
```

### 2. Rate Limiting Strategy

- **Fixed Window**: Limits reset at fixed intervals (simpler, less memory)
- **Moving Window**: Sliding window (more precise, more memory)

Current implementation uses **fixed window** for performance.

### 3. Response Headers

Rate limit info is included in response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1696789200
```

### 4. Error Response

When rate limit exceeded (HTTP 429):

```json
{
  "error": "Rate limit exceeded: 100 per 1 hour"
}
```

Headers:
```
Retry-After: 3456  (seconds until reset)
```

## Testing Rate Limits

### 1. Test Anonymous Rate Limit

```bash
#!/bin/bash

# Make 101 requests to exceed 100/hour limit
for i in {1..101}; do
  echo "Request $i:"
  curl -s -w "\nStatus: %{http_code}\n" \
    "http://localhost:8000/api/models"
  
  # Check if rate limited
  if [ $? -ne 0 ]; then
    echo "Rate limited after $i requests"
    break
  fi
  
  sleep 0.1
done
```

Expected: First 100 succeed, 101st returns HTTP 429.

### 2. Test Authenticated Rate Limit

```bash
#!/bin/bash

# Login first
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login/json" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "TestPass123!"}' \
  | jq -r '.access_token')

echo "Token: ${TOKEN:0:50}..."

# Make 1001 requests (authenticated limit is 1000/hour)
for i in {1..1001}; do
  STATUS=$(curl -s -w "%{http_code}" -o /dev/null \
    -H "Authorization: Bearer $TOKEN" \
    "http://localhost:8000/api/models")
  
  if [ "$STATUS" -eq "429" ]; then
    echo "Rate limited after $i requests (authenticated)"
    break
  fi
  
  if [ $((i % 100)) -eq 0 ]; then
    echo "Completed $i requests..."
  fi
done
```

Expected: First 1000 succeed, 1001st returns HTTP 429.

### 3. Test Rate Limit Headers

```bash
curl -v "http://localhost:8000/api/models" 2>&1 | grep -i "X-RateLimit"
```

Expected output:
```
< X-RateLimit-Limit: 100
< X-RateLimit-Remaining: 99
< X-RateLimit-Reset: 1696789200
```

### 4. Test Health Check (No Limit)

```bash
# Health check should never be rate limited
for i in {1..1000}; do
  curl -s "http://localhost:8000/health" > /dev/null
  echo -n "."
done
echo ""
echo "All 1000 health checks succeeded (no rate limit)"
```

## Endpoint-Specific Limits

### Health Check (Unlimited)

```python
@app.get("/health")
@limiter.exempt  # No rate limit
async def health_check():
    return {"status": "ok"}
```

### Standard Endpoints (Default Limits)

Uses configured limits (100/hour anonymous, 1000/hour authenticated).

### Expensive Endpoints (Custom Limits)

```python
from .rate_limit import limiter

@app.post("/predictions/batch")
@limiter.limit("10/minute")  # Strict limit for batch operations
async def create_batch_predictions():
    ...
```

## Python Client with Rate Limiting

### Basic Client

```python
import requests
import time
from typing import Optional

class MatprovClient:
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url
        self.token = token
        self.session = requests.Session()
        
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
    
    def request(self, method: str, endpoint: str, **kwargs):
        """Make request with rate limit handling"""
        url = f"{self.base_url}{endpoint}"
        
        while True:
            response = self.session.request(method, url, **kwargs)
            
            # Check rate limit headers
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining and int(remaining) < 10:
                print(f"⚠️  Low rate limit: {remaining} requests remaining")
            
            # Handle rate limit exceeded
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"⚠️  Rate limited. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            return response
    
    def get_models(self):
        """Get models with automatic rate limit handling"""
        response = self.request("GET", "/api/models")
        return response.json()

# Usage
client = MatprovClient("http://localhost:8000", token="your_token")
models = client.get_models()
```

### Advanced Client with Exponential Backoff

```python
import requests
import time
from typing import Optional

class SmartMatprovClient:
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url
        self.token = token
        self.session = requests.Session()
        
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
    
    def request_with_backoff(
        self, 
        method: str, 
        endpoint: str, 
        max_retries: int = 3,
        **kwargs
    ):
        """Make request with exponential backoff"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            response = self.session.request(method, url, **kwargs)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                backoff = min(retry_after * (2 ** attempt), 300)  # Max 5 min
                
                print(f"Rate limited. Retry {attempt + 1}/{max_retries} in {backoff}s")
                time.sleep(backoff)
                continue
            
            return response
        
        raise Exception("Max retries exceeded")

# Usage
client = SmartMatprovClient("http://localhost:8000", token="your_token")
response = client.request_with_backoff("GET", "/api/models")
```

## Monitoring Rate Limits

### Check Current Usage

```bash
# If using Redis, inspect rate limit keys
redis-cli KEYS "rate_limit:*"

# Get details for specific key
redis-cli GET "rate_limit:user:123"
```

### Log Rate Limit Events

Add logging to track rate limit hits:

```python
import logging

logger = logging.getLogger("rate_limit")

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(
        f"Rate limit exceeded: {exc.detail}",
        extra={
            "client": get_identifier(request),
            "endpoint": request.url.path,
            "method": request.method
        }
    )
    return JSONResponse(
        status_code=429,
        content={"error": f"Rate limit exceeded: {exc.detail}"}
    )
```

## Troubleshooting

### 1. Rate limits not working

Check that slowapi is installed:
```bash
pip list | grep slowapi
```

Check limiter is initialized:
```bash
# Should see in startup logs
✅ Rate limiting initialized (in-memory)
```

### 2. Redis connection errors

Verify Redis is running:
```bash
docker ps | grep redis
redis-cli ping  # Should return "PONG"
```

Check Redis URL:
```bash
echo $REDIS_URL
# Should be: redis://localhost:6379/0
```

### 3. Authenticated users being rate limited as anonymous

Check JWT token is valid:
```bash
# Decode token
python -c "from jose import jwt; print(jwt.get_unverified_claims('YOUR_TOKEN'))"
```

Verify Authorization header format:
```bash
# Correct format
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

### 4. Health check being rate limited

Health check should have `@limiter.exempt` decorator:
```python
@app.get("/health")
@limiter.exempt
async def health_check():
    ...
```

## Production Deployment

### 1. Redis Setup

Use Redis for production (supports multiple workers):

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
```

### 2. Environment Configuration

```bash
# .env
USE_REDIS_RATE_LIMIT=true
REDIS_URL=redis://redis:6379/0  # Use service name in Docker
RATE_LIMIT_ANONYMOUS=100/hour
RATE_LIMIT_AUTHENTICATED=1000/hour
```

### 3. Monitoring

Set up alerts for rate limit exceeded events:
- Monitor 429 responses
- Track rate limit header values
- Alert on high rate limit hit rates

### 4. Tuning

Adjust limits based on usage patterns:
- Monitor average requests per user
- Identify legitimate high-volume users
- Create custom limits for specific endpoints

## Advanced Configuration

### Per-Endpoint Custom Limits

```python
from .rate_limit import limiter

# Very strict limit
@app.post("/predictions/batch")
@limiter.limit("10/minute")
async def batch_predictions():
    ...

# More permissive
@app.get("/api/search")
@limiter.limit("100/minute")
async def search():
    ...
```

### Exempt Specific IPs

```python
WHITELISTED_IPS = ["10.0.0.1", "192.168.1.1"]

def get_identifier(request: Request) -> str:
    ip = get_remote_address(request)
    
    if ip in WHITELISTED_IPS:
        return "whitelisted"
    
    # ... rest of logic
```

### Moving Window Strategy

For more precise rate limiting:

```python
limiter = Limiter(
    key_func=get_identifier,
    storage_uri=REDIS_URL,
    strategy="moving-window",  # More precise
    headers_enabled=True
)
```

---

## Summary

✅ **Configured**: Anonymous (100/hr), Authenticated (1000/hr)  
✅ **Storage**: In-memory (dev), Redis (production)  
✅ **Identification**: IP address (anonymous), User ID (authenticated)  
✅ **Headers**: X-RateLimit-* headers in all responses  
✅ **Exempt**: Health check endpoint  
✅ **Monitoring**: Redis keys, log events  

**Next**: Implement security headers (Phase 1.3)

