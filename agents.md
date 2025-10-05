# Autonomous R&D Intelligence Layer - AI Assistant Guide

## Project Overview

This is the **Autonomous R&D Intelligence Layer** (Periodic Labs), a production-grade AI-powered platform for optimizing physical R&D experiments. The system uses dual Gemini models (Flash for speed, Pro for accuracy) and reinforcement learning to accelerate materials science, chemistry, and manufacturing research.

**Key Components:**
- app/src/api/ – FastAPI backend serving AI reasoning endpoints and static UI
- app/src/reasoning/ – RL agents (PPO+ICM), Bayesian Optimization, and dual-model reasoning
- app/src/services/ – Google Cloud integrations (Vertex AI, Storage, Secret Manager, Monitoring, **Cloud SQL**)
- app/src/drivers/ – Hardware drivers for XRD, NMR, UV-Vis instruments
- app/static/ – Web UI for querying AI and viewing validation results (including analytics dashboard)
- app/tests/ – Unit and integration tests
- .github/workflows/ – CI/CD pipeline for automated testing and Cloud Run deployment
- scripts/ – Training and validation scripts for RL system, **database setup and test data generation**
- app/alembic/ – Database migration framework for schema version control

**Architecture:**
- **Frontend**: Static HTML/CSS/JS with Tailwind CSS, served by FastAPI + Cloud Storage
- **Backend**: FastAPI (Python 3.12) deployed on Google Cloud Run
- **Database**: Cloud SQL PostgreSQL 15 (ard-intelligence-db) for metadata persistence
- **AI Models**: Vertex AI Gemini 2.5 Flash + Pro (dual-model pattern)
- **Optimization**: PPO+ICM (Reinforcement Learning) and Bayesian Optimization
- **Infrastructure**: Google Cloud Platform (Cloud Run, Cloud Storage, Cloud SQL, Secret Manager)

**Database Schema:**
- \`experiments\` – Individual experiment tracking (parameters, results, status)
- \`optimization_runs\` – Optimization campaign tracking (RL, BO, Adaptive Router)
- \`ai_queries\` – AI model usage and cost tracking
- \`experiment_runs\` – Legacy dual-model reasoning logs
- \`instrument_runs\` – Hardware instrument audit logs

## Build and Test Instructions

### Setup
\`\`\`bash
cd app && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
\`\`\`

### Configuration
Set environment variables or create .env in app/:
\`\`\`bash
PROJECT_ID=periodicdent42
GCP_REGION=us-central1
LOG_LEVEL=INFO

# Database (required for metadata persistence)
DB_USER=ard_user
DB_PASSWORD=ard_secure_password_2024
DB_NAME=ard_intelligence
DB_HOST=localhost
DB_PORT=5433  # Cloud SQL Proxy port
\`\`\`

For production, secrets are managed in Google Cloud Secret Manager.

### Database Setup (First Time)
\`\`\`bash
# 1. Download Cloud SQL Proxy
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.darwin.arm64
chmod +x cloud-sql-proxy

# 2. Start Cloud SQL Proxy (in background)
./cloud-sql-proxy --port 5433 periodicdent42:us-central1:ard-intelligence-db > cloud-sql-proxy.log 2>&1 &

# 3. Initialize database schema
export DB_USER=ard_user DB_PASSWORD=ard_secure_password_2024 DB_NAME=ard_intelligence DB_HOST=localhost DB_PORT=5433
python scripts/init_database.py

# 4. (Optional) Generate test data
python scripts/generate_test_data.py
\`\`\`

### Run Locally
\`\`\`bash
# Method 1: Using startup script (recommended)
cd app && ./start_server.sh

# Method 2: Manual
cd app && export PYTHONPATH="/Users/kiteboard/periodicdent42:\${PYTHONPATH}" && \\
  export DB_USER=ard_user DB_PASSWORD=ard_secure_password_2024 DB_NAME=ard_intelligence DB_HOST=localhost DB_PORT=5433 && \\
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
\`\`\`

**Access Points:**
- Main UI: http://localhost:8080/
- Analytics Dashboard: http://localhost:8080/analytics.html
- API Documentation: http://localhost:8080/docs
- Health Check: http://localhost:8080/health

### Test
\`\`\`bash
cd app && export PYTHONPATH=".:\${PYTHONPATH}" && pytest tests/ -v --tb=short --cov=src
\`\`\`

**Coverage Requirements:**
- Aim for >50% coverage on new modules
- Critical paths (API endpoints, safety systems) should have >80% coverage
- CI pipeline enforces coverage checks on pull requests

## Coding Guidelines

### Python (Backend)
- **Version**: Python 3.12+
- **Style**: Follow PEP 8, enforced by ruff linter
- **Indentation**: 4 spaces (never tabs)
- **Line Length**: 100 characters
- **Imports**: Use absolute imports from src. prefix

### Naming Conventions
- **Functions/Variables**: snake_case
- **Classes**: PascalCase  
- **Constants**: UPPER_SNAKE_CASE
- **Private members**: _leading_underscore

### Error Handling
- **Never expose stack traces** to users in API responses
- **Always log errors** using the configured logger
- **Use appropriate HTTP status codes** (200, 400, 422, 503)

## Testing Protocols

### Test Structure
\`\`\`
app/tests/
├── __init__.py
├── test_health.py          # Health check endpoints
├── test_reasoning_smoke.py # Reasoning endpoint smoke tests
└── unit/                   # Unit tests
\`\`\`

### Mocking External Services
Always mock Vertex AI and Google Cloud services in tests.

## Security and Best Practices

### Secrets Management
- **NEVER commit secrets** (API keys, credentials) to Git
- **Use Google Cloud Secret Manager** for production secrets
- **Use environment variables** for local development

### RL Safety (Critical)
- **Always validate RL actions** before executing on hardware
- **Enforce hard limits** on parameter ranges
- **Emergency stop** must be accessible for all hardware operations
- **Log all hardware commands** with timestamps

## Validation and Iteration Philosophy

### Honest Iteration
This project follows a **"fail frequently and learn fast"** philosophy:
- Present **honest limitations** upfront
- **Document failures** as rigorously as successes
- **Iterate quickly** with clear documentation
- **Avoid overstating** capabilities

### Current Status (October 2025)
- ✅ RL shows promise in high-noise environments (σ≥1.0)
- ✅ **Cloud SQL database integration complete with metadata persistence**
- ✅ **Analytics dashboard operational with live data (205 experiments, 20 runs, 100 queries)**
- ✅ **REST API endpoints for experiments, optimization runs, and AI cost tracking**
- ⚠️ Limited to 2D optimization (Branin function)
- ⚠️ No real hardware validation yet (simulated noise only)
- ⚠️ Requires comparison to advanced BO variants

## Key Files

### API & Backend
- app/src/api/main.py – FastAPI application entry point with metadata endpoints
- app/src/reasoning/ppo_agent.py – PPO+ICM RL agent
- app/src/services/vertex.py – Vertex AI Gemini wrappers
- **app/src/services/db.py – Cloud SQL database models and operations**
- **app/start_server.sh – Server startup script with environment configuration**

### Database & Scripts
- **app/alembic/ – Database migration framework**
- **scripts/init_database.py – Initialize database schema**
- **scripts/recreate_schema.py – Drop and recreate tables (destructive)**
- **scripts/generate_test_data.py – Generate realistic test data**
- scripts/validate_stochastic.py – Stochastic validation script

### CI/CD & Infrastructure
- .github/workflows/cicd.yaml – CI/CD pipeline with Cloud Run deployment

### Documentation
- **DATABASE_SETUP_COMPLETE.md – Comprehensive database setup guide**
- VALIDATION_STATUS.md – Validation results and analysis

## Common Tasks

### Adding a New API Endpoint
1. Add route in app/src/api/main.py
2. Define Pydantic models for request/response
3. Add tests in app/tests/
4. Update API documentation in docstring
5. Test locally, then deploy

### Working with Database
1. **Query data via API**:
   \`\`\`bash
   curl 'http://localhost:8080/api/experiments?limit=10'
   curl 'http://localhost:8080/api/optimization_runs?status=completed'
   curl 'http://localhost:8080/api/ai_queries'
   \`\`\`

2. **Query database directly**:
   \`\`\`bash
   export PGPASSWORD=ard_secure_password_2024
   psql -h localhost -p 5433 -U ard_user -d ard_intelligence -c "SELECT COUNT(*) FROM experiments;"
   \`\`\`

3. **Regenerate test data**:
   \`\`\`bash
   python scripts/generate_test_data.py --runs 20 --experiments-per-run 10 --standalone 50 --queries 100
   \`\`\`

4. **Reset database schema** (DESTRUCTIVE):
   \`\`\`bash
   python scripts/recreate_schema.py
   python scripts/init_database.py
   python scripts/generate_test_data.py
   \`\`\`

5. **Check database status**:
   \`\`\`bash
   # View table counts
   psql -h localhost -p 5433 -U ard_user -d ard_intelligence << 'EOF'
   SELECT 
     (SELECT COUNT(*) FROM experiments) as experiments,
     (SELECT COUNT(*) FROM optimization_runs) as runs,
     (SELECT COUNT(*) FROM ai_queries) as queries;
   EOF
   \`\`\`

### Updating Validation Results
1. Run validation script
2. Update VALIDATION_STATUS.md
3. Update web UI if needed
4. Deploy to Cloud Run
5. Commit with honest assessment

## Troubleshooting

### Import Errors in Tests
\`\`\`bash
cd app && export PYTHONPATH=".:\${PYTHONPATH}" && pytest tests/
\`\`\`

### Database Connection Issues
1. **Check Cloud SQL Proxy is running**:
   \`\`\`bash
   ps aux | grep cloud-sql-proxy
   tail -f cloud-sql-proxy.log
   \`\`\`

2. **Restart Cloud SQL Proxy**:
   \`\`\`bash
   killall cloud-sql-proxy
   ./cloud-sql-proxy --port 5433 periodicdent42:us-central1:ard-intelligence-db > cloud-sql-proxy.log 2>&1 &
   \`\`\`

3. **Test database connectivity**:
   \`\`\`bash
   export PGPASSWORD=ard_secure_password_2024
   psql -h localhost -p 5433 -U ard_user -d ard_intelligence -c "SELECT 1;"
   \`\`\`

4. **Authentication errors**:
   \`\`\`bash
   gcloud auth application-default login
   \`\`\`

### Server Won't Start
1. **Check server logs**:
   \`\`\`bash
   tail -f /Users/kiteboard/periodicdent42/app/server.log
   \`\`\`

2. **Verify PYTHONPATH includes repository root**:
   \`\`\`bash
   echo \$PYTHONPATH  # Should include /Users/kiteboard/periodicdent42
   \`\`\`

3. **Kill existing server and restart**:
   \`\`\`bash
   pkill -f "uvicorn src.api.main:app"
   cd app && ./start_server.sh
   \`\`\`

### Cloud Run 404 Errors
- Check endpoint paths match (e.g., /health not /healthz)
- Verify static files are copied in Dockerfile
- Check Cloud Run logs with: \`gcloud run services logs tail ard-backend --region=us-central1\`
- Verify database connection string uses Unix socket in production

### API Returns Empty Results
- Verify Cloud SQL Proxy is running: \`ps aux | grep cloud-sql-proxy\`
- Check database has data: \`curl 'http://localhost:8080/health'\`
- Regenerate test data if needed: \`python scripts/generate_test_data.py\`

Remember: **Honest iteration over perfect demos.** Document limitations, iterate fast, accelerate science.
