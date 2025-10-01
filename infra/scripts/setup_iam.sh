#!/bin/bash
# Set up IAM service accounts and permissions

set -e

PROJECT_ID=${PROJECT_ID:-periodicdent42}
SA_NAME="ard-backend"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Setting up IAM for project: $PROJECT_ID"

# Create service account
if ! gcloud iam service-accounts describe $SA_EMAIL --project=$PROJECT_ID &>/dev/null; then
    echo "Creating service account: $SA_EMAIL"
    gcloud iam service-accounts create $SA_NAME \
        --display-name="ARD Backend Service Account" \
        --project=$PROJECT_ID
else
    echo "Service account already exists: $SA_EMAIL"
fi

# Grant permissions
echo "Granting IAM roles..."

# Vertex AI access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/aiplatform.user" \
    --condition=None

# Cloud Storage access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.objectAdmin" \
    --condition=None

# Secret Manager access
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None

# Cloud SQL Client
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/cloudsql.client" \
    --condition=None

# Monitoring metrics writer
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/monitoring.metricWriter" \
    --condition=None

# Logging log writer
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/logging.logWriter" \
    --condition=None

echo "✅ IAM setup complete"
echo "Service Account: $SA_EMAIL"

