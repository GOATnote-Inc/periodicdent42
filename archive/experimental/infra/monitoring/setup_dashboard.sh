#!/bin/bash
# Script to create Cloud Monitoring dashboard

set -e

PROJECT_ID=$(gcloud config get-value project)
DASHBOARD_FILE="infra/monitoring/dashboard.json"

echo "📊 Creating Cloud Monitoring Dashboard..."
echo "Project: $PROJECT_ID"

# Create dashboard
gcloud monitoring dashboards create --config-from-file=$DASHBOARD_FILE

echo "✅ Dashboard created successfully!"
echo ""
echo "View your dashboard at:"
echo "https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
echo ""
echo "To update the dashboard:"
echo "1. Get the dashboard ID: gcloud monitoring dashboards list"
echo "2. Update: gcloud monitoring dashboards update DASHBOARD_ID --config-from-file=$DASHBOARD_FILE"

