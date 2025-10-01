# Autonomous R&D Intelligence Layer - Backend

FastAPI backend with Gemini 2.5 Dual-Model reasoning on Google Cloud.

## Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your GCP credentials

# 3. Run development server
make dev

# 4. Test health check
curl http://localhost:8080/healthz

# 5. Test reasoning endpoint
curl -N -X POST http://localhost:8080/api/reasoning/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Suggest perovskite experiments","context":{"domain":"materials"}}'
```

### Docker

```bash
# Build
make build

# Run
make run-docker
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROJECT_ID` | Yes | `periodicdent42` | GCP project ID |
| `LOCATION` | Yes | `us-central1` | GCP region |
| `GEMINI_FLASH_MODEL` | No | `gemini-2.5-flash` | Fast model |
| `GEMINI_PRO_MODEL` | No | `gemini-2.5-pro` | Accurate model |
| `DB_PASSWORD` | Yes* | - | Database password |
| `GCS_BUCKET` | Yes* | - | Storage bucket name |

*Fetched from Secret Manager in production

## API Endpoints

### `GET /healthz`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "vertex_initialized": true,
  "project_id": "periodicdent42"
}
```

### `POST /api/reasoning/query`
Dual-model reasoning with SSE streaming.

**Request:**
```json
{
  "query": "Suggest experiments for BaTiO3 optimization",
  "context": {
    "domain": "materials_science",
    "constraints": {"budget": 1000}
  }
}
```

**Response (SSE):**
```
event: preliminary
data: {"response": {...}, "message": "Quick preview..."}

event: final
data: {"response": {...}, "message": "Verified response ready"}
```

## Deployment

### Prerequisites
```bash
# Enable APIs
bash ../infra/scripts/enable_apis.sh

# Setup IAM
bash ../infra/scripts/setup_iam.sh
```

### Deploy to Cloud Run
```bash
# Build and deploy
make gcloud-build
make deploy

# Or use script directly
bash ../infra/scripts/deploy_cloudrun.sh
```

## Testing

```bash
# Run tests
make test

# With coverage
make test-coverage

# Lint
make lint
```

## Architecture

```
FastAPI App
    ↓
DualModelAgent
    ├─→ Gemini 2.5 Flash (fast)
    └─→ Gemini 2.5 Pro (accurate)
    ↓
SSE Stream to Client
```

## Cost Optimization

- **Flash first**: 90% of queries use Flash ($0.075/1M input tokens)
- **Pro verification**: Only critical decisions use Pro ($1.25/1M input tokens)
- **Context caching**: Reuse common prompts (50% cost reduction)
- **Batch requests**: Group multiple queries when possible

## Security Notes

- ⚠️ No PHI/sensitive data in prompts without GDC deployment
- ✅ Least privilege IAM (service account per environment)
- ✅ Secrets in Secret Manager, not in code/env
- ✅ TLS for all API calls
- ✅ Authentication required by default (add `--allow-unauthenticated` for demos only)

## Troubleshooting

### "Vertex AI not initialized"
- Check `PROJECT_ID` and `LOCATION` in `.env`
- Verify APIs are enabled: `gcloud services list --enabled`
- Check service account has `roles/aiplatform.user`

### "Database connection failed"
- Verify Cloud SQL instance is running
- Check `GCP_SQL_INSTANCE` format: `project:region:instance`
- Ensure service account has `roles/cloudsql.client`

### "Permission denied" errors
- Run `bash ../infra/scripts/setup_iam.sh`
- Verify service account email in Cloud Run settings

## Monitoring

```bash
# View logs
make logs

# Cloud Monitoring
# Metrics: custom.googleapis.com/ard/eig_per_hour
# Latency: Cloud Run built-in metrics
```

## Development Tips

- Use `make dev` for hot-reload during development
- Mock Vertex AI in tests (see `tests/test_reasoning_smoke.py`)
- SSE testing: use `curl -N` flag for streaming
- Debug logs: set `LOG_LEVEL=DEBUG` in `.env`

## Links

- [API Docs](http://localhost:8080/docs) (when running locally)
- [Cloud Run Console](https://console.cloud.google.com/run)
- [Vertex AI Console](https://console.cloud.google.com/vertex-ai)
- [Full Documentation](../docs/google_cloud_deployment.md)

