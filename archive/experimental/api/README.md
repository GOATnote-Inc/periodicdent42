# matprov REST API

FastAPI-based REST API for Materials Provenance Tracking.

## Installation

```bash
pip install -r api/requirements.txt
```

## Run Server

```bash
# Development mode (with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Health Check
```
GET /health
```

### Models
```
GET  /api/models              # List registered models
POST /api/models              # Register new model
```

### Predictions
```
GET  /api/predictions         # List predictions (with filters)
POST /api/predictions         # Create new prediction
```

### Experiments
```
POST /api/experiments         # Add experimental outcome
```

### Performance
```
GET /api/performance/{model_id}  # Get model performance summary
```

### Analysis
```
GET /api/candidates           # Top candidates for validation
GET /api/errors               # Predictions with large errors
```

## Example Usage

### 1. Register a Model
```bash
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "rf_v2.1_uci_21k",
    "version": "2.1.0",
    "checkpoint_hash": "dvc:abc123",
    "training_dataset_hash": "dvc:3f34e6c71b4245aad0da5acc3d39fe7f",
    "architecture": "RandomForestClassifier"
  }'
```

### 2. Create Prediction
```bash
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "PRED-001",
    "model_id": "rf_v2.1_uci_21k",
    "material_formula": "YBa2Cu3O7",
    "predicted_tc": 92.5,
    "uncertainty": 5.2,
    "predicted_class": "high_Tc",
    "confidence": 0.89
  }'
```

### 3. Add Experimental Outcome
```bash
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "PRED-001",
    "experiment_id": "EXP-001",
    "actual_tc": 89.3,
    "validation_status": "success",
    "phase_purity": 92.3
  }'
```

### 4. Get Model Performance
```bash
curl http://localhost:8000/api/performance/rf_v2.1_uci_21k
```

Response:
```json
{
  "model_id": "rf_v2.1_uci_21k",
  "version": "2.1.0",
  "total_predictions": 100,
  "validated_predictions": 50,
  "mae": 10.56,
  "rmse": 12.34,
  "r2_score": 0.8234,
  "classification_metrics": {
    "accuracy": 0.88,
    "precision": 0.85,
    "recall": 0.87,
    "f1": 0.86
  }
}
```

### 5. Get Top Candidates
```bash
curl "http://localhost:8000/api/candidates?limit=10&min_tc=30.0"
```

### 6. Get Large Errors
```bash
curl "http://localhost:8000/api/errors?threshold=10.0&limit=10"
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Register model
model_data = {
    "model_id": "rf_v2.1_uci_21k",
    "version": "2.1.0",
    "checkpoint_hash": "dvc:abc123",
    "training_dataset_hash": "dvc:3f34e6c71b4245aad0da5acc3d39fe7f"
}
response = requests.post(f"{BASE_URL}/api/models", json=model_data)
print(response.json())

# Create prediction
pred_data = {
    "prediction_id": "PRED-001",
    "model_id": "rf_v2.1_uci_21k",
    "material_formula": "YBa2Cu3O7",
    "predicted_tc": 92.5,
    "uncertainty": 5.2
}
response = requests.post(f"{BASE_URL}/api/predictions", json=pred_data)
print(response.json())

# Get predictions
response = requests.get(f"{BASE_URL}/api/predictions?limit=10")
predictions = response.json()
print(f"Found {len(predictions)} predictions")

# Get performance
response = requests.get(f"{BASE_URL}/api/performance/rf_v2.1_uci_21k")
perf = response.json()
print(f"RMSE: {perf['rmse']:.2f}K")
```

## Features

- ✅ RESTful API with OpenAPI/Swagger docs
- ✅ Automatic validation (Pydantic)
- ✅ CORS enabled (configure for production)
- ✅ Database integration (SQLAlchemy)
- ✅ Error handling with HTTP status codes
- ✅ Query parameters for filtering
- ✅ Pagination support

## Database

By default, uses SQLite at `.matprov/predictions.db`.

For production, set environment variable:
```bash
export DATABASE_URL="postgresql://user:pass@localhost/matprov"
```

## Production Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY api/ ./api/
COPY matprov/ ./matprov/
RUN pip install -r api/requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/matprov
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: matprov
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
```

### Cloud Run (Google Cloud)
```bash
gcloud run deploy matprov-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Security

For production:
1. Add API key authentication
2. Configure CORS properly
3. Use HTTPS
4. Rate limiting
5. Input validation
6. SQL injection protection (SQLAlchemy handles this)

## Performance

- Lightweight: ~50ms response time
- Scalable: Horizontal scaling with multiple workers
- Database: Connection pooling via SQLAlchemy
- Caching: Add Redis for frequently accessed data

## Integration

### With matprov CLI
```bash
# Track experiment via API
matprov track-experiment exp.json

# API automatically syncs with database
curl http://localhost:8000/api/experiments?limit=10
```

### With Streamlit Dashboard
Dashboard reads from same database, live updates!

### With DVC
Model and dataset hashes link to DVC storage for full provenance.

