#!/bin/bash
# Deploy FastAPI backend to Cloud Run

set -e

PROJECT_ID=${PROJECT_ID:-periodicdent42}
REGION=${REGION:-us-central1}
SERVICE_NAME="ard-backend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
SERVICE_ACCOUNT="ard-backend@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Deploying $SERVICE_NAME to Cloud Run..."

# Force rebuild to pick up code changes
echo "Building Docker image..."
cd "$(dirname "$0")/../.."  # Go to repository root

# Create temporary cloudbuild.yaml for custom Dockerfile location
cat > /tmp/cloudbuild.yaml <<EOF
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', '$IMAGE_NAME', '-f', 'app/Dockerfile', '.']
images: ['$IMAGE_NAME']
EOF

gcloud builds submit --config=/tmp/cloudbuild.yaml --project=$PROJECT_ID .
cd - > /dev/null

# Check for Cloud SQL instance
CLOUDSQL_INSTANCE=$(gcloud sql instances list --project=$PROJECT_ID --format="value(name)" --filter="name:ard-*" 2>/dev/null | head -n 1)
CLOUDSQL_FLAGS=""
DB_ENV_VARS=""
DB_SECRETS=""

if [ -n "$CLOUDSQL_INSTANCE" ]; then
    echo "✅ Found Cloud SQL instance: $CLOUDSQL_INSTANCE"
    CLOUDSQL_FLAGS="--add-cloudsql-instances=${PROJECT_ID}:${REGION}:${CLOUDSQL_INSTANCE}"
    DB_ENV_VARS=",GCP_SQL_INSTANCE=${PROJECT_ID}:${REGION}:${CLOUDSQL_INSTANCE}"
    DB_SECRETS="DB_PASSWORD=db-password:latest,"
fi

# Deploy to Cloud Run with public access (authentication handled by middleware)
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 10 \
  --service-account $SERVICE_ACCOUNT \
  --set-env-vars "PROJECT_ID=$PROJECT_ID,LOCATION=$REGION,ENVIRONMENT=production,ENABLE_AUTH=true,RATE_LIMIT_PER_MINUTE=120${DB_ENV_VARS}" \
  --set-secrets "${DB_SECRETS}API_KEY=api-key:latest" \
  $CLOUDSQL_FLAGS \
  --allow-unauthenticated

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --project $PROJECT_ID \
  --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "Service URL: $SERVICE_URL"
echo ""
echo "🌐 Public Endpoints:"
echo "  - Main: $SERVICE_URL"
echo "  - Analytics: $SERVICE_URL/static/analytics.html"
echo "  - Health: $SERVICE_URL/health"
echo ""
echo "🔒 API Authentication:"
echo "  Protected endpoints require X-API-Key header"
echo "  Get API key: gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID"
echo ""
echo "📊 Test API with authentication:"
echo "  API_KEY=\$(gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID)"
echo "  curl -H \"x-api-key: \$API_KEY\" \"$SERVICE_URL/api/experiments\""

