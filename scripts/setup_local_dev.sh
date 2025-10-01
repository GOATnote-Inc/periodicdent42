#!/bin/bash
# Quick setup script for local development environment
# This creates a .env file with secrets from Secret Manager

set -e

PROJECT_ID=${PROJECT_ID:-periodicdent42}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛠️  Setting up Local Development Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    echo "❌ Not authenticated with gcloud"
    echo ""
    echo "Please run:"
    echo "  gcloud auth application-default login"
    echo ""
    exit 1
fi

echo "✅ Authenticated with gcloud"
echo ""

# Navigate to app directory
cd "$(dirname "$0")/../app"

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists"
    read -p "Do you want to overwrite it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
fi

echo "📝 Creating .env file from template..."

# Copy example file
cp env.example .env

echo "✅ Created .env from template"
echo ""

# Try to get API key from Secret Manager
echo "🔐 Retrieving API key from Secret Manager..."

if API_KEY=$(gcloud secrets versions access latest --secret=api-key --project=$PROJECT_ID 2>/dev/null); then
    echo "✅ API key retrieved"
    
    # Update .env file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|^API_KEY=.*|API_KEY=$API_KEY|" .env
    else
        # Linux
        sed -i "s|^API_KEY=.*|API_KEY=$API_KEY|" .env
    fi
    
    echo "✅ API key added to .env"
else
    echo "⚠️  Could not retrieve API key from Secret Manager"
    echo "   You'll need to add it manually to app/.env"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Your .env file is ready at: app/.env"
echo ""
echo "To start the development server:"
echo ""
echo "  cd app"
echo "  source venv/bin/activate"
echo "  uvicorn src.api.main:app --reload --port 8080"
echo ""
echo "Then visit: http://localhost:8080/docs"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  SECURITY REMINDERS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  • .env is in .gitignore - it will NOT be committed"
echo "  • NEVER commit secrets to git"
echo "  • Use ENABLE_AUTH=false for local dev only"
echo "  • In production, secrets come from Secret Manager"
echo ""

