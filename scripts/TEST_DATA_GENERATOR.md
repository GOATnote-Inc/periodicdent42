# Test Data Generator

Generate realistic test data for the Cloud SQL database to validate the integration and provide data for the analytics dashboard.

## Features

- **Realistic Experiments**: Parameters, noise estimates, results, and error conditions
- **Optimization Runs**: Multiple methods (RL, BO, Adaptive) with associated experiments
- **AI Queries**: Cost analysis, latency tracking, and model selection
- **Time-distributed Data**: Spread over the past 30 days for realistic analytics
- **User Attribution**: Multiple test users for access control testing

## Prerequisites

### Option A: Local PostgreSQL (Recommended for Development)

1. **Install PostgreSQL:**
   ```bash
   # macOS
   brew install postgresql@14
   brew services start postgresql@14
   
   # Ubuntu/Debian
   sudo apt-get install postgresql-14
   sudo systemctl start postgresql
   ```

2. **Create Database and User:**
   ```bash
   psql postgres
   ```
   
   ```sql
   CREATE DATABASE ard_intelligence;
   CREATE USER ard_user WITH PASSWORD 'your_password_here';
   GRANT ALL PRIVILEGES ON DATABASE ard_intelligence TO ard_user;
   \q
   ```

3. **Set Environment Variables:**
   ```bash
   cd app
   cat > .env << EOF
   # Database (Local PostgreSQL)
   DB_USER=ard_user
   DB_PASSWORD=your_password_here
   DB_NAME=ard_intelligence
   DB_HOST=localhost
   DB_PORT=5432
   
   # GCP (for Vertex AI)
   PROJECT_ID=periodicdent42
   LOCATION=us-central1
   EOF
   ```

### Option B: Cloud SQL (Production)

1. **Set Environment Variables:**
   ```bash
   cd app
   cat > .env << EOF
   # Database (Cloud SQL)
   GCP_SQL_INSTANCE=periodicdent42:us-central1:ard-postgres
   DB_USER=ard_user
   DB_PASSWORD=$(gcloud secrets versions access latest --secret=db-password --project=periodicdent42)
   DB_NAME=ard_intelligence
   
   # GCP
   PROJECT_ID=periodicdent42
   LOCATION=us-central1
   EOF
   ```

2. **Connect via Cloud SQL Proxy:**
   ```bash
   # In a separate terminal
   cloud_sql_proxy -instances=periodicdent42:us-central1:ard-postgres=tcp:5432
   ```

## Usage

### Basic Usage (Default Settings)

```bash
cd /Users/kiteboard/periodicdent42
python scripts/generate_test_data.py
```

This generates:
- **50** standalone experiments
- **20** optimization runs (with 10 experiments each)
- **100** AI queries

### Custom Data Volume

```bash
python scripts/generate_test_data.py \
  --experiments 100 \
  --runs 50 \
  --queries 500 \
  --experiments-per-run 15
```

### Clear and Regenerate

```bash
python scripts/generate_test_data.py --clear
```

⚠️ **Warning:** This deletes ALL existing data!

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiments` | 50 | Number of standalone experiments |
| `--runs` | 20 | Number of optimization runs |
| `--queries` | 100 | Number of AI queries |
| `--experiments-per-run` | 10 | Experiments per optimization run |
| `--clear` | False | Clear existing data first |

## Generated Data Examples

### Experiments
```json
{
  "id": "exp_20251002231045_a3f7c9",
  "method": "reinforcement_learning",
  "parameters": {
    "temperature": 450,
    "pressure": 2.5,
    "flow_rate": 55
  },
  "context": {
    "domain": "materials_synthesis",
    "target": "maximize_yield"
  },
  "noise_estimate": 0.085,
  "results": {
    "yield": 67.3,
    "purity": 94.2,
    "byproducts": 1.8
  },
  "status": "completed"
}
```

### Optimization Runs
```json
{
  "id": "run_20251002105032_b8e4d1",
  "method": "adaptive_router",
  "context": {
    "domain": "catalysis",
    "target": "minimize_byproducts"
  },
  "status": "completed",
  "start_time": "2025-09-28T10:50:32Z",
  "end_time": "2025-09-29T14:22:18Z"
}
```

### AI Queries
```json
{
  "id": "query_20251001143022_c5f9a2",
  "query": "How do I optimize maximize_yield for materials_synthesis?",
  "selected_model": "adaptive_router",
  "latency_ms": 2850.3,
  "input_tokens": 15,
  "output_tokens": 287,
  "cost_usd": 0.000087
}
```

## Data Distribution

### Experiment Status
- **70%** Completed
- **20%** Running
- **10%** Failed

### Optimization Run Status
- **60%** Completed
- **30%** Running
- **10%** Failed

### Optimization Methods
- Reinforcement Learning (RL)
- Bayesian Optimization (BO)
- Adaptive Router

### AI Model Selection
- Gemini Flash (fast, low cost)
- Gemini Pro (accurate, higher cost)
- Adaptive Router (intelligent selection)

### Time Distribution
- Spread over past **30 days**
- Realistic start/end times
- Running experiments within past **7 days**

## Validation

After generating data, validate the metadata API endpoints:

```bash
# Get service URL
SERVICE_URL=$(cat .service-url)

# List experiments
curl "$SERVICE_URL/api/experiments?limit=10"

# Get specific experiment
EXPERIMENT_ID=$(curl -s "$SERVICE_URL/api/experiments?limit=1" | jq -r '.experiments[0].id')
curl "$SERVICE_URL/api/experiments/$EXPERIMENT_ID"

# List optimization runs
curl "$SERVICE_URL/api/optimization_runs?limit=10"

# Filter by method
curl "$SERVICE_URL/api/optimization_runs?method=adaptive_router"

# List AI queries with cost analysis
curl "$SERVICE_URL/api/ai_queries?limit=10&include_cost_analysis=true"

# Filter by model
curl "$SERVICE_URL/api/ai_queries?selected_model=flash"
```

## Troubleshooting

### Database Connection Error

```
ERROR: could not connect to server
```

**Solution:**
- Check PostgreSQL is running: `brew services list` (macOS) or `systemctl status postgresql` (Linux)
- Verify connection details in `app/.env`
- For Cloud SQL, ensure Cloud SQL Proxy is running

### Import Error

```
ModuleNotFoundError: No module named 'sqlalchemy'
```

**Solution:**
```bash
cd app
source venv/bin/activate
pip install -r requirements.txt
```

### Authentication Error (Cloud SQL)

```
ERROR: password authentication failed
```

**Solution:**
```bash
# Fetch latest password from Secret Manager
export DB_PASSWORD=$(gcloud secrets versions access latest --secret=db-password --project=periodicdent42)

# Update app/.env
echo "DB_PASSWORD=$DB_PASSWORD" >> app/.env
```

## Next Steps

After generating test data:

1. **Test Metadata API Endpoints** - Validate queries work correctly
2. **Build Analytics Dashboard** - Visualize the data
3. **Phase 1 Scientific Validation** - Use real experimental data

---

**Created:** October 2, 2025  
**Part of:** Cloud SQL Integration & Metadata Persistence

