# Local Development Setup

**Quick setup guide for running the service locally with security disabled for development.**

---

## Prerequisites

- Python 3.12+
- Virtual environment support
- Git

---

## Quick Start (2 Minutes)

### Automated Setup (Recommended)

```bash
# Run the automated setup script
bash scripts/setup_local_dev.sh

# It will:
# 1. Create app/.env from template
# 2. Retrieve secrets from Secret Manager
# 3. Configure everything for you

# Then start the server:
cd app
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8080
```

### Manual Setup

```bash
# 1. Navigate to app directory
cd app

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (development mode)
# Option A: Use .env file (recommended)
cp env.example .env
# Edit .env and add your secrets

# Option B: Export in terminal
export PROJECT_ID=periodicdent42
export LOCATION=us-central1
export ENVIRONMENT=development
export ENABLE_AUTH=false  # ⚠️ Security disabled for local dev only

# 5. Get secrets from Secret Manager (see SECRETS_MANAGEMENT.md)
bash ../scripts/get_secrets.sh

# 6. Run the server
export PYTHONPATH=".:${PYTHONPATH}"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
```

**Server running**: http://localhost:8080

---

## Testing Locally

### Without Authentication (Development Mode)

```bash
# Health check
curl http://localhost:8080/health

# Reasoning endpoint
curl -X POST http://localhost:8080/api/reasoning/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Suggest an experiment for perovskites"}'

# API docs
open http://localhost:8080/docs
```

### With Authentication (Test Production Behavior)

```bash
# Set environment variables
export ENABLE_AUTH=true
export API_KEY=test-key-for-local-dev
export RATE_LIMIT_PER_MINUTE=60

# Restart server
# (In another terminal, kill and restart uvicorn)

# Test with API key
curl -H "x-api-key: test-key-for-local-dev" http://localhost:8080/health

# Test without API key (should fail)
curl http://localhost:8080/health
```

---

## Running Tests

```bash
cd app
source venv/bin/activate
export PYTHONPATH=".:${PYTHONPATH}"

# Run all tests
pytest tests/ -v

# Run security tests only
pytest tests/test_security.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## Environment Variables

### Required
```bash
PROJECT_ID=periodicdent42          # GCP project ID
LOCATION=us-central1               # GCP region
```

### Optional (with defaults)
```bash
ENVIRONMENT=development            # development | staging | production
ENABLE_AUTH=false                  # Enable API key authentication
API_KEY=                           # API key (if ENABLE_AUTH=true)
ALLOWED_ORIGINS=                   # CORS allowed origins (comma-separated)
RATE_LIMIT_PER_MINUTE=60          # Rate limit per IP
LOG_LEVEL=INFO                     # Logging level
PORT=8080                          # Server port
```

### Create `.env` file (optional)

```bash
# app/.env
PROJECT_ID=periodicdent42
LOCATION=us-central1
ENVIRONMENT=development
ENABLE_AUTH=false
LOG_LEVEL=DEBUG
```

---

## Development Workflow

### 1. Make Changes

Edit files in `app/src/`

### 2. Auto-Reload

Server automatically reloads with `--reload` flag.

### 3. Run Tests

```bash
pytest tests/test_security.py -v
```

### 4. Check Linting

```bash
ruff check app/src/
```

### 5. Format Code

```bash
ruff format app/src/
```

---

## Security Testing Locally

### Test Authentication

```bash
# Terminal 1: Run with auth enabled
export ENABLE_AUTH=true
export API_KEY=my-test-key
uvicorn src.api.main:app --reload --port 8080

# Terminal 2: Test requests
curl http://localhost:8080/health  # Should fail (401)
curl -H "x-api-key: my-test-key" http://localhost:8080/health  # Should work
curl -H "x-api-key: wrong-key" http://localhost:8080/health  # Should fail (401)
```

### Test Rate Limiting

```bash
# Make 61 rapid requests (limit is 60/min)
for i in {1..65}; do
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/health
done | tail -10

# Last 5 should be 429 (rate limited)
```

### Test CORS

```bash
# Set allowed origin
export ALLOWED_ORIGINS=http://localhost:3000

# Restart server

# Test from browser console or with curl
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS \
     http://localhost:8080/api/reasoning/query
```

---

## Troubleshooting

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=".:${PYTHONPATH}"

# Or run from app directory
cd app
python -m pytest tests/
```

### Port Already in Use

```bash
# Kill process on port 8080
lsof -ti:8080 | xargs kill -9

# Or use different port
uvicorn src.api.main:app --port 8081
```

### Vertex AI Errors (Expected Locally)

Vertex AI calls will fail without GCP credentials. This is expected.
The health endpoint will show `vertex_initialized: false`.

To test with real Vertex AI locally:
```bash
# Authenticate with GCP
gcloud auth application-default login

# Set project
export PROJECT_ID=periodicdent42

# Run server
uvicorn src.api.main:app --reload
```

---

## VS Code / Cursor Configuration

### Launch Configuration (`.vscode/launch.json`)

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "src.api.main:app",
        "--reload",
        "--port",
        "8080"
      ],
      "cwd": "${workspaceFolder}/app",
      "env": {
        "PYTHONPATH": ".",
        "PROJECT_ID": "periodicdent42",
        "LOCATION": "us-central1",
        "ENVIRONMENT": "development",
        "ENABLE_AUTH": "false",
        "LOG_LEVEL": "DEBUG"
      }
    }
  ]
}
```

### Python Path (`.vscode/settings.json`)

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/app/venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.envFile": "${workspaceFolder}/app/.env"
}
```

---

## Next Steps

- **Production Deployment**: See [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)
- **Security Details**: See [docs/SECURITY.md](docs/SECURITY.md)
- **Architecture**: See [docs/architecture.md](docs/architecture.md)

---

**Happy Coding!** 🚀

