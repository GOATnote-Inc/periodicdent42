#!/bin/bash
# Helper script to retrieve secrets from Google Cloud Secret Manager
# Use this for local development - NEVER hardcode secrets!

set -e

PROJECT_ID=${PROJECT_ID:-periodicdent42}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 Retrieving Secrets from Secret Manager"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Project: $PROJECT_ID"
echo ""

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    echo "❌ Not authenticated with gcloud"
    echo "Run: gcloud auth application-default login"
    exit 1
fi

echo "📋 Available secrets:"
echo ""

# Function to safely get a secret
get_secret() {
    local secret_name=$1
    local value=$(gcloud secrets versions access latest --secret="$secret_name" --project="$PROJECT_ID" 2>/dev/null)
    
    if [ -n "$value" ]; then
        echo "✅ $secret_name"
        return 0
    else
        echo "⚠️  $secret_name (not found or no access)"
        return 1
    fi
}

# Get all secrets
get_secret "api-key" && API_KEY_EXISTS=true
get_secret "DB_PASSWORD" && DB_PASSWORD_EXISTS=true
get_secret "GCP_SQL_INSTANCE" && GCP_SQL_INSTANCE_EXISTS=true
get_secret "GCS_BUCKET" && GCS_BUCKET_EXISTS=true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 How to use these secrets:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$API_KEY_EXISTS" = true ]; then
    echo "For local development, export as environment variables:"
    echo ""
    echo "  export API_KEY=\$(gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID)"
    echo ""
fi

echo "Or create a .env file (app/.env):"
echo ""
echo "  # Copy from example"
echo "  cp app/env.example app/.env"
echo ""
echo "  # Get API key"
echo "  echo \"API_KEY=\$(gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID)\" >> app/.env"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  SECURITY REMINDERS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  • NEVER commit .env files to git (already in .gitignore)"
echo "  • NEVER hardcode API keys in source code"
echo "  • Use Secret Manager in production"
echo "  • Rotate API keys every 90 days"
echo "  • Use different keys for dev/staging/prod"
echo ""
echo "To view a specific secret:"
echo "  gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID"
echo ""

