#!/bin/bash
# Rotate API key - Generate a NEW key and update Secret Manager + .env
# Use this when you want to FORCE a new key (not retrieve existing)

set -e

PROJECT_ID=${PROJECT_ID:-periodicdent42}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Rotate API Key"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  This will generate a NEW API key and update:"
echo "  • Secret Manager (new version)"
echo "  • Your local .env file"
echo "  • Your .api-key file"
echo ""
echo "Old key will be kept as previous version (can be disabled later)"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    echo "❌ Not authenticated with gcloud"
    echo "Run: gcloud auth application-default login"
    exit 1
fi

echo ""
echo "🔐 Generating new API key..."

# Generate new API key
NEW_API_KEY=$(openssl rand -hex 32)

echo "✅ Generated: ${NEW_API_KEY:0:16}...${NEW_API_KEY: -8}"
echo ""

# Check if secret exists
if gcloud secrets describe api-key --project="$PROJECT_ID" &>/dev/null; then
    echo "📝 Adding new version to Secret Manager..."
    
    # Add new version
    echo -n "$NEW_API_KEY" | gcloud secrets versions add api-key \
        --data-file=- \
        --project="$PROJECT_ID" >/dev/null 2>&1
    
    echo "✅ New version added to Secret Manager"
    
    # List versions
    echo ""
    echo "📋 Secret versions:"
    gcloud secrets versions list api-key --project="$PROJECT_ID" --limit=5
    
else
    echo "🔄 Secret doesn't exist, creating..."
    
    echo -n "$NEW_API_KEY" | gcloud secrets create api-key \
        --data-file=- \
        --replication-policy="automatic" \
        --project="$PROJECT_ID" >/dev/null 2>&1
    
    # Grant service account access
    SERVICE_ACCOUNT="ard-backend@${PROJECT_ID}.iam.gserviceaccount.com"
    gcloud secrets add-iam-policy-binding api-key \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$PROJECT_ID" >/dev/null 2>&1 || true
    
    echo "✅ Secret created in Secret Manager"
fi

echo ""
echo "📝 Updating local files..."

# Update .api-key file
echo "$NEW_API_KEY" > .api-key
chmod 600 .api-key
echo "✅ Updated .api-key"

# Update .env file if it exists
if [ -f "app/.env" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|^API_KEY=.*|API_KEY=$NEW_API_KEY|" app/.env
    else
        # Linux
        sed -i "s|^API_KEY=.*|API_KEY=$NEW_API_KEY|" app/.env
    fi
    echo "✅ Updated app/.env"
else
    echo "⚠️  app/.env not found - create it with:"
    echo "   bash scripts/init_secrets_and_env.sh"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ API Key Rotated!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔑 New API Key: ${NEW_API_KEY:0:8}...${NEW_API_KEY: -8}"
echo "   (Full key saved to .api-key and app/.env)"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Restart your local server (if running)"
echo "   cd app && uvicorn src.api.main:app --reload"
echo ""
echo "2. Update Cloud Run to use new key:"
echo "   gcloud run services update ard-backend \\"
echo "     --region=us-central1 \\"
echo "     --project=$PROJECT_ID"
echo ""
echo "3. Distribute new key to authorized clients"
echo ""
echo "4. After clients are updated, disable old version:"
echo "   gcloud secrets versions list api-key --project=$PROJECT_ID"
echo "   gcloud secrets versions disable VERSION_NUMBER --secret=api-key --project=$PROJECT_ID"
echo ""
echo "⚠️  Old key is still active until you disable it!"
echo ""

