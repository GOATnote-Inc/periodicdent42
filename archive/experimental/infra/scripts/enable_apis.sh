#!/bin/bash
# Enable required Google Cloud APIs

set -e

PROJECT_ID=${1:-periodicdent42}

echo "Enabling APIs for project: $PROJECT_ID"

gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  sqladmin.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project=$PROJECT_ID

echo "✅ All APIs enabled successfully"

