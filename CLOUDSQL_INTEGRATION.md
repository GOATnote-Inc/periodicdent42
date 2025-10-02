# Cloud SQL Integration - Metadata Persistence

**Status**: ✅ Implemented  
**Date**: October 1, 2025  
**Database**: PostgreSQL 15 on Google Cloud SQL

---

## Overview

Cloud SQL (PostgreSQL) provides structured metadata persistence for the Autonomous R&D Intelligence Layer. This enables experiment tracking, optimization run history, AI query logging, and adaptive routing decisions.

### Benefits

✅ **Structured Queries**: SQL queries for complex analytics  
✅ **ACID Transactions**: Data integrity and consistency  
✅ **Automatic Backups**: Point-in-time recovery  
✅ **High Availability**: Zonal (dev) or Regional (prod)  
✅ **Scalability**: Auto-scaling storage, read replicas  
✅ **Security**: IAM integration, encrypted connections  

### Cost

- **db-f1-micro** (dev/testing): ~$7-10/month
- **db-n1-standard-1** (production): ~$50/month
- Storage: ~$0.10/GB/month

---

## Database Schema

### Tables

#### 1. **experiments**
Tracks individual experiments (physical measurements or simulations).

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| name | VARCHAR(255) | Human-readable name |
| description | TEXT | Experiment description |
| status | ENUM | pending, running, completed, failed, cancelled |
| parameters | JSONB | Input parameters |
| config | JSONB | Experiment configuration |
| result_value | FLOAT | Primary objective value |
| result_data | JSONB | Full result data |
| result_uri | VARCHAR(512) | GCS URI for large results |
| created_at | TIMESTAMP | Creation timestamp |
| started_at | TIMESTAMP | Start timestamp |
| completed_at | TIMESTAMP | Completion timestamp |
| created_by | VARCHAR(255) | User identifier |
| optimization_run_id | VARCHAR(255) | Foreign key to optimization_runs |

**Indexes**:
- `idx_experiments_status` on status
- `idx_experiments_created_at` on created_at
- `idx_experiments_optimization_run` on optimization_run_id

#### 2. **optimization_runs**
Tracks optimization campaigns (multiple experiments guided by an algorithm).

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| name | VARCHAR(255) | Human-readable name |
| description | TEXT | Run description |
| method | ENUM | bayesian_optimization, reinforcement_learning, etc. |
| objective | VARCHAR(255) | "minimize" or "maximize" |
| search_space | JSONB | Parameter bounds/constraints |
| config | JSONB | Algorithm-specific config |
| num_experiments | INTEGER | Number of experiments run |
| best_value | FLOAT | Best objective value found |
| best_experiment_id | VARCHAR(255) | ID of best experiment |
| status | ENUM | pending, running, completed, failed, cancelled |
| created_at | TIMESTAMP | Creation timestamp |
| started_at | TIMESTAMP | Start timestamp |
| completed_at | TIMESTAMP | Completion timestamp |
| created_by | VARCHAR(255) | User identifier |

**Indexes**:
- `idx_optimization_runs_method` on method
- `idx_optimization_runs_status` on status
- `idx_optimization_runs_created_at` on created_at

#### 3. **ai_queries**
Tracks AI reasoning queries to Gemini models.

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| query_text | TEXT | User query |
| context | JSONB | Query context |
| flash_response | JSONB | Gemini Flash response |
| pro_response | JSONB | Gemini Pro response |
| selected_model | VARCHAR(50) | "flash" or "pro" |
| flash_latency_ms | FLOAT | Flash response time |
| pro_latency_ms | FLOAT | Pro response time |
| flash_tokens | INTEGER | Flash tokens used |
| pro_tokens | INTEGER | Pro tokens used |
| estimated_cost_usd | FLOAT | Estimated cost |
| created_at | TIMESTAMP | Creation timestamp |
| created_by | VARCHAR(255) | User identifier |
| experiment_id | VARCHAR(255) | Related experiment (if any) |

**Indexes**:
- `idx_ai_queries_created_at` on created_at
- `idx_ai_queries_selected_model` on selected_model

#### 4. **noise_estimates**
Tracks noise estimation for adaptive routing.

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| method | VARCHAR(100) | replicate_pooled, residuals, sequential |
| noise_std | FLOAT | Estimated noise standard deviation |
| confidence_interval_lower | FLOAT | Lower bound (95% CI) |
| confidence_interval_upper | FLOAT | Upper bound (95% CI) |
| sample_size | INTEGER | Number of samples used |
| reliable | BOOLEAN | Is estimate reliable? |
| pilot_data | JSONB | Pilot experiment data |
| created_at | TIMESTAMP | Creation timestamp |
| optimization_run_id | VARCHAR(255) | Related optimization run |

**Indexes**:
- `idx_noise_estimates_created_at` on created_at
- `idx_noise_estimates_method` on method

#### 5. **routing_decisions**
Tracks adaptive routing decisions (BO vs RL selection).

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| selected_method | ENUM | bayesian_optimization, reinforcement_learning, etc. |
| confidence | FLOAT | Decision confidence (0.0-1.0) |
| reasoning | TEXT | Human-readable explanation |
| threshold_used | FLOAT | Noise threshold applied |
| alternatives | JSONB | Alternative methods considered |
| warnings | JSONB | Warning messages |
| created_at | TIMESTAMP | Creation timestamp |
| noise_estimate_id | VARCHAR(255) | Related noise estimate |
| optimization_run_id | VARCHAR(255) | Related optimization run |

**Indexes**:
- `idx_routing_decisions_method` on selected_method
- `idx_routing_decisions_created_at` on created_at

---

## Setup Instructions

### Prerequisites

1. Google Cloud Project with billing enabled
2. Cloud SQL Admin API enabled
3. Service account with Cloud SQL Client role

### Option 1: Automated Setup (Recommended)

```bash
cd infra/scripts

# Enable required APIs
bash enable_apis.sh

# Create Cloud SQL instance and database
bash setup_cloudsql.sh
```

This script will:
- Create a Cloud SQL PostgreSQL instance
- Generate and store database password in Secret Manager
- Create database and user
- Configure IAM permissions
- Output connection information

### Option 2: Manual Setup

#### Step 1: Create Cloud SQL Instance

```bash
PROJECT_ID="periodicdent42"
REGION="us-central1"
INSTANCE_NAME="ard-intelligence-db"

gcloud sql instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --database-version="POSTGRES_15" \
  --tier="db-f1-micro" \
  --network="default" \
  --enable-bin-log \
  --backup-start-time="03:00" \
  --availability-type="ZONAL" \
  --storage-type="SSD" \
  --storage-size="10GB" \
  --storage-auto-increase \
  --no-assign-ip
```

#### Step 2: Set Database Password

```bash
# Generate secure password
DB_PASSWORD=$(openssl rand -base64 32)

# Store in Secret Manager
echo -n "$DB_PASSWORD" | gcloud secrets create db-password \
  --project="$PROJECT_ID" \
  --data-file=- \
  --replication-policy="automatic"

# Set root password
gcloud sql users set-password postgres \
  --instance="$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --password="$DB_PASSWORD"
```

#### Step 3: Create Database and User

```bash
# Create database
gcloud sql databases create "ard_intelligence" \
  --instance="$INSTANCE_NAME" \
  --project="$PROJECT_ID"

# Create user
gcloud sql users create "ard_user" \
  --instance="$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --password="$DB_PASSWORD"
```

#### Step 4: Configure IAM

```bash
SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/cloudsql.client"
```

---

## Database Migrations

We use [Alembic](https://alembic.sqlalchemy.org/) for database migrations.

### Initial Setup

```bash
cd app

# Database URL will be constructed from settings
# See app/src/utils/settings.py for configuration
```

### Create Initial Migration

```bash
cd app

# Generate migration from models
alembic revision --autogenerate -m "Initial schema"

# Review the generated migration in migrations/versions/

# Apply migration
alembic upgrade head
```

### Common Migration Commands

```bash
# Check current database version
alembic current

# View migration history
alembic history

# Upgrade to latest
alembic upgrade head

# Downgrade one version
alembic downgrade -1

# Create new migration
alembic revision -m "add user table"
```

---

## Application Configuration

### Environment Variables (Cloud Run)

```bash
# Cloud SQL connection
GCP_SQL_INSTANCE="periodicdent42:us-central1:ard-intelligence-db"
DB_NAME="ard_intelligence"
DB_USER="ard_user"
# DB_PASSWORD is fetched from Secret Manager automatically
```

### Local Development

#### Option 1: Cloud SQL Proxy (Recommended)

```bash
# Download Cloud SQL Proxy
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.darwin.amd64
chmod +x cloud-sql-proxy

# Run proxy (separate terminal)
./cloud-sql-proxy periodicdent42:us-central1:ard-intelligence-db

# In your .env file:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ard_intelligence
DB_USER=ard_user
DB_PASSWORD=<from Secret Manager>
```

#### Option 2: Local PostgreSQL

```bash
# Install PostgreSQL
brew install postgresql@15
brew services start postgresql@15

# Create database
createdb ard_intelligence

# In your .env file:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ard_intelligence
DB_USER=<your local user>
DB_PASSWORD=<your local password>
```

---

## REST API Endpoints

The following REST API endpoints are now available for querying metadata:

### 1. List Experiments
```bash
GET /api/experiments?status=completed&limit=100&optimization_run_id=run-123&created_by=user
```

**Response:**
```json
{
  "experiments": [
    {
      "id": "exp-123",
      "name": "Perovskite Synthesis Optimization",
      "status": "completed",
      "parameters": {"temp": 500, "pressure": 1.0},
      "result_value": 0.85,
      "created_at": "2025-10-01T10:00:00",
      "optimization_run_id": "run-456"
    }
  ],
  "count": 1,
  "status": "success"
}
```

### 2. Get Experiment Details
```bash
GET /api/experiments/{experiment_id}
```

Returns detailed information about a specific experiment, including the optimization run it belongs to.

### 3. List Optimization Runs
```bash
GET /api/optimization_runs?method=bayesian_optimization&status=completed&limit=50
```

**Response:**
```json
{
  "runs": [
    {
      "id": "run-456",
      "name": "BO Campaign Oct 2025",
      "method": "bayesian_optimization",
      "objective": "maximize",
      "num_experiments": 25,
      "best_value": 0.92,
      "status": "completed"
    }
  ],
  "count": 1,
  "status": "success"
}
```

### 4. List AI Queries with Cost Analysis
```bash
GET /api/ai_queries?limit=100&selected_model=flash&include_cost_analysis=true
```

**Response:**
```json
{
  "queries": [...],
  "count": 50,
  "cost_analysis": {
    "total_queries": 50,
    "flash_queries": 35,
    "pro_queries": 15,
    "total_flash_tokens": 5250,
    "total_pro_tokens": 2700,
    "estimated_total_cost_usd": 0.0824,
    "avg_flash_latency_ms": 245.3,
    "avg_pro_latency_ms": 1150.8,
    "cost_per_query_usd": 0.001648
  },
  "status": "success"
}
```

---

## Usage Examples

### Create an Experiment

```python
from src.services.db import create_experiment, update_experiment, ExperimentStatus

# Create experiment record
experiment = create_experiment(
    experiment_id="exp-123",
    parameters={"temperature": 300, "pressure": 1.0},
    config={"method": "xrd", "duration": 60},
    optimization_run_id="run-456",
    created_by="user-789"
)

# Update with results
update_experiment(
    experiment_id="exp-123",
    status=ExperimentStatus.COMPLETED,
    result_value=42.5,
    result_data={"peaks": [1.2, 3.4, 5.6]},
    result_uri="gs://bucket/experiments/exp-123/result.json"
)
```

### Log AI Query

```python
from src.services.db import log_ai_query

log_ai_query(
    query_id="query-abc",
    query_text="Design next experiment to maximize yield",
    context={"previous_experiments": [...]},
    flash_response={"response": "...", "latency_ms": 1234},
    pro_response=None,  # Not used
    selected_model="flash",
    created_by="user-789"
)
```

### Query Experiments

```python
from src.services.db import get_experiments, ExperimentStatus

# Get all completed experiments in an optimization run
experiments = get_experiments(
    optimization_run_id="run-456",
    status=ExperimentStatus.COMPLETED,
    limit=100
)

for exp in experiments:
    print(f"{exp.id}: {exp.result_value}")
```

---

## Monitoring & Maintenance

### Check Database Size

```bash
gcloud sql instances describe ard-intelligence-db \
  --project=periodicdent42 \
  --format="value(settings.dataDiskSizeGb)"
```

### View Database Logs

```bash
gcloud sql operations list \
  --instance=ard-intelligence-db \
  --project=periodicdent42 \
  --limit=10
```

### Create Backup (Manual)

```bash
gcloud sql backups create \
  --instance=ard-intelligence-db \
  --project=periodicdent42
```

### Restore from Backup

```bash
# List backups
gcloud sql backups list \
  --instance=ard-intelligence-db \
  --project=periodicdent42

# Restore specific backup
gcloud sql backups restore BACKUP_ID \
  --backup-instance=ard-intelligence-db \
  --backup-project=periodicdent42
```

---

## Performance Optimization

### Connection Pooling

SQLAlchemy connection pooling is already configured:

```python
_engine = create_engine(
    database_url,
    pool_size=5,          # Max 5 connections per instance
    max_overflow=10,      # Allow 10 overflow connections
    pool_pre_ping=True,   # Verify connections before use
)
```

### Query Optimization

All tables have indexes on commonly queried columns:
- Status columns (for filtering by state)
- Timestamp columns (for time-based queries)
- Foreign keys (for joins)

### Scaling

**Vertical Scaling** (increase instance size):
```bash
gcloud sql instances patch ard-intelligence-db \
  --tier=db-n1-standard-1 \
  --project=periodicdent42
```

**Read Replicas** (for read-heavy workloads):
```bash
gcloud sql instances create ard-intelligence-db-replica \
  --master-instance-name=ard-intelligence-db \
  --tier=db-n1-standard-1 \
  --project=periodicdent42 \
  --region=us-central1
```

---

## Security

### Network Security

- ✅ **Private IP** (no public internet access)
- ✅ **VPC peering** for Cloud Run access
- ✅ **IAM authentication** (alternative to password auth)

### Access Control

- Service account has `roles/cloudsql.client` (minimum required)
- Database password stored in Secret Manager
- Connection via Unix socket (Cloud Run) or Cloud SQL Proxy (local)

### Encryption

- ✅ **At rest**: Automatic encryption with Google-managed keys
- ✅ **In transit**: SSL/TLS connections enforced

---

## Troubleshooting

### Connection Errors

**Problem**: Cannot connect to Cloud SQL  
**Solutions**:
1. Check service account has `cloudsql.client` role
2. Verify instance name format: `project:region:instance`
3. Check database credentials in Secret Manager
4. Ensure Cloud SQL Admin API is enabled

### Migration Errors

**Problem**: Alembic migration fails  
**Solutions**:
1. Check database connection with `alembic current`
2. Review migration file for syntax errors
3. Check database user has CREATE/ALTER permissions
4. Try `alembic upgrade head --sql` to see SQL without executing

### Performance Issues

**Problem**: Slow queries  
**Solutions**:
1. Check indexes are created: `alembic upgrade head`
2. Analyze query plans: Add `echo=True` to engine
3. Monitor connection pool utilization
4. Consider upgrading instance tier

---

## Cost Optimization

### Development/Testing

- Use **db-f1-micro** ($7-10/month)
- Single zone (no high availability)
- Daily backups (7-day retention)
- Auto-pause (not available for Cloud SQL, but use smallest tier)

### Production

- Use **db-n1-standard-1** ($50/month) or larger
- High availability (multiple zones)
- Point-in-time recovery (7-day log retention)
- Read replicas for scaling

### Cost Monitoring

```bash
# Estimate monthly cost
gcloud sql instances describe ard-intelligence-db \
  --project=periodicdent42 \
  --format="value(settings.tier,settings.dataDiskSizeGb)"
```

---

## References

- [Cloud SQL for PostgreSQL](https://cloud.google.com/sql/docs/postgres)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [psycopg2 Documentation](https://www.psycopg.org/docs/)

---

**Last Updated**: October 1, 2025  
**Status**: Production-ready  
**Next Review**: After Phase 1 validation

---

*"Data without structure is just noise. Structure without data is just schema."*

