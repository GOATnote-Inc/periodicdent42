#!/bin/bash
# Deploy FastAPI backend to Cloud Run

set -e

PROJECT_ID=${PROJECT_ID:-periodicdent42}
REGION=${REGION:-us-central1}
SERVICE_NAME="ard-backend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
SERVICE_ACCOUNT="ard-backend@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Deploying $SERVICE_NAME to Cloud Run..."

# Build and push image (if not already done)
if ! gcloud container images describe $IMAGE_NAME --project=$PROJECT_ID &>/dev/null; then
    echo "Image not found, building..."
    cd ../app
    gcloud builds submit --tag $IMAGE_NAME --project=$PROJECT_ID
    cd -
fi

# Deploy to Cloud Run with security enabled
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
  --set-env-vars "PROJECT_ID=$PROJECT_ID,LOCATION=$REGION,ENVIRONMENT=production,ENABLE_AUTH=true,RATE_LIMIT_PER_MINUTE=120" \
  --set-secrets "API_KEY=api-key:latest" \
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
echo "🔐 Security Status:"
echo "  - Authentication: ENABLED (API key required)"
echo "  - Rate Limiting: 120 requests/minute per IP"
echo "  - CORS: Configure ALLOWED_ORIGINS for your domain"
echo ""
echo "To retrieve your API key:"
echo "  gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID"
echo ""
echo "To test the health endpoint:"
echo "  API_KEY=\$(gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID)"
echo "  curl -H \"x-api-key: \$API_KEY\" $SERVICE_URL/health"
echo ""
echo "To test the reasoning endpoint:"
echo "  curl -X POST $SERVICE_URL/api/reasoning/query \\"
echo "    -H \"x-api-key: \$API_KEY\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"query\": \"Suggest an experiment for perovskites\"}'"
echo ""
echo "⚠️  Next steps:"
echo "  1. Save your API key securely"
echo "  2. Set ALLOWED_ORIGINS environment variable with your frontend domain"
echo "  3. Monitor Cloud Logging for security events"
echo "  4. Set up alerting for 401/429 responses"

