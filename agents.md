# Autonomous R&D Intelligence Layer - AI Assistant Guide

## Project Overview

This is the **Autonomous R&D Intelligence Layer** (Periodic Labs), a production-grade AI-powered platform for optimizing physical R&D experiments. The system uses dual Gemini models (Flash for speed, Pro for accuracy) and reinforcement learning to accelerate materials science, chemistry, and manufacturing research.

**Key Components:**
- app/src/api/ – FastAPI backend serving AI reasoning endpoints and static UI
- app/src/reasoning/ – RL agents (PPO+ICM), Bayesian Optimization, and dual-model reasoning
- app/src/services/ – Google Cloud integrations (Vertex AI, Storage, Secret Manager, Monitoring)
- app/src/drivers/ – Hardware drivers for XRD, NMR, UV-Vis instruments
- app/static/ – Web UI for querying AI and viewing validation results
- app/tests/ – Unit and integration tests
- .github/workflows/ – CI/CD pipeline for automated testing and Cloud Run deployment
- scripts/ – Training and validation scripts for RL system

**Architecture:**
- **Frontend**: Static HTML/CSS/JS with Tailwind CSS, served by FastAPI
- **Backend**: FastAPI (Python 3.12) deployed on Google Cloud Run
- **AI Models**: Vertex AI Gemini 2.5 Flash + Pro (dual-model pattern)
- **Optimization**: PPO+ICM (Reinforcement Learning) and Bayesian Optimization
- **Infrastructure**: Google Cloud Platform (Cloud Run, Cloud Storage, Cloud SQL, Secret Manager)

## Build and Test Instructions

### Setup
cd app && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

### Configuration
Set environment variables or create .env in app/:
PROJECT_ID=periodicdent42
GCP_REGION=us-central1
LOG_LEVEL=INFO

For production, secrets are managed in Google Cloud Secret Manager.

### Run Locally
cd app && export PYTHONPATH=".:${PYTHONPATH}" && uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080

### Test
cd app && export PYTHONPATH=".:${PYTHONPATH}" && pytest tests/ -v --tb=short --cov=src

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
app/tests/
├── __init__.py
├── test_health.py          # Health check endpoints
├── test_reasoning_smoke.py # Reasoning endpoint smoke tests
└── unit/                   # Unit tests

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
- ⚠️ Limited to 2D optimization (Branin function)
- ⚠️ No real hardware validation yet (simulated noise only)
- ⚠️ Requires comparison to advanced BO variants

## Key Files
- app/src/api/main.py – FastAPI application entry point
- app/src/reasoning/ppo_agent.py – PPO+ICM RL agent
- app/src/services/vertex.py – Vertex AI Gemini wrappers
- scripts/validate_stochastic.py – Stochastic validation script
- .github/workflows/cicd.yaml – CI/CD pipeline

## Common Tasks

### Adding a New API Endpoint
1. Add route in app/src/api/main.py
2. Define Pydantic models for request/response
3. Add tests in app/tests/
4. Update API documentation in docstring
5. Test locally, then deploy

### Updating Validation Results
1. Run validation script
2. Update VALIDATION_STATUS.md
3. Update web UI if needed
4. Deploy to Cloud Run
5. Commit with honest assessment

## Troubleshooting

### Import Errors in Tests
cd app && export PYTHONPATH=".:${PYTHONPATH}" && pytest tests/

### Cloud Run 404 Errors
- Check endpoint paths match (e.g., /health not /healthz)
- Verify static files are copied in Dockerfile
- Check Cloud Run logs

Remember: **Honest iteration over perfect demos.** Document limitations, iterate fast, accelerate science.
