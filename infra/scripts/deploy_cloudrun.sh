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

# Deploy to Cloud Run
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
  --set-env-vars "PROJECT_ID=$PROJECT_ID,LOCATION=$REGION,ENVIRONMENT=production" \
  --no-allow-unauthenticated

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --project $PROJECT_ID \
  --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "Service URL: $SERVICE_URL"
echo ""
echo "To test (requires authentication):"
echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $SERVICE_URL/healthz"
echo ""
echo "To allow unauthenticated access (for demos only):"
echo "  gcloud run services add-iam-policy-binding $SERVICE_NAME \\"
echo "    --region=$REGION \\"
echo "    --member=\"allUsers\" \\"
echo "    --role=\"roles/run.invoker\""

