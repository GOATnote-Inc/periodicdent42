#!/bin/bash
# Create required secrets in GCP Secret Manager
# These are optional - the app has defaults, but secrets are cleaner for production

PROJECT_ID="periodicdent42"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 Creating Secrets in Secret Manager"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. API_KEY (REQUIRED for production security)
echo "Creating API_KEY secret..."
# Generate a secure random API key (32 bytes = 64 hex chars)
API_KEY=$(openssl rand -hex 32)
echo -n "${API_KEY}" | gcloud secrets create api-key \
    --project="${PROJECT_ID}" \
    --data-file=- \
    --replication-policy="automatic" 2>/dev/null || \
    echo -n "${API_KEY}" | gcloud secrets versions add api-key --data-file=-

echo "✅ API_KEY created: ${API_KEY:0:8}...${API_KEY: -8}"
echo ""
echo "⚠️  SAVE THIS KEY - You'll need it to make API requests!"
echo "   Full key saved to: .api-key (chmod 600)"
echo ""

# Save API key to file for reference
echo "$API_KEY" > .api-key
chmod 600 .api-key
echo "✅ API key saved to .api-key for reference"
echo ""

# 2. DB_PASSWORD (optional, defaults to 'postgres' locally)
echo "Creating DB_PASSWORD secret..."
echo -n "your-db-password" | gcloud secrets create DB_PASSWORD \
    --project="${PROJECT_ID}" \
    --data-file=- \
    --replication-policy="automatic" 2>/dev/null || \
    echo -n "your-db-password" | gcloud secrets versions add DB_PASSWORD --data-file=-

echo "✅ DB_PASSWORD created"

# 2. GCP_SQL_INSTANCE (optional, for Cloud SQL connection)
echo "Creating GCP_SQL_INSTANCE secret..."
echo -n "${PROJECT_ID}:us-central1:ard-db" | gcloud secrets create GCP_SQL_INSTANCE \
    --project="${PROJECT_ID}" \
    --data-file=- \
    --replication-policy="automatic" 2>/dev/null || \
    echo -n "${PROJECT_ID}:us-central1:ard-db" | gcloud secrets versions add GCP_SQL_INSTANCE --data-file=-

echo "✅ GCP_SQL_INSTANCE created"

# 3. GCS_BUCKET (optional, defaults to project-experiments)
echo "Creating GCS_BUCKET secret..."
echo -n "${PROJECT_ID}-experiments" | gcloud secrets create GCS_BUCKET \
    --project="${PROJECT_ID}" \
    --data-file=- \
    --replication-policy="automatic" 2>/dev/null || \
    echo -n "${PROJECT_ID}-experiments" | gcloud secrets versions add GCS_BUCKET --data-file=-

echo "✅ GCS_BUCKET created"

# Grant Cloud Run service account access to secrets
SERVICE_ACCOUNT="ard-backend@${PROJECT_ID}.iam.gserviceaccount.com"

echo ""
echo "Granting access to service account: ${SERVICE_ACCOUNT}"

for SECRET in api-key DB_PASSWORD GCP_SQL_INSTANCE GCS_BUCKET; do
    gcloud secrets add-iam-policy-binding ${SECRET} \
        --project="${PROJECT_ID}" \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet
    echo "  ✅ ${SECRET} access granted"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All secrets created and configured!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To update a secret value:"
echo "  echo -n 'new-value' | gcloud secrets versions add SECRET_NAME --data-file=-"
echo ""
echo "To deploy with secrets:"
echo "  bash infra/scripts/deploy_cloudrun.sh"

