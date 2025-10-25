#!/bin/bash
# Start FastAPI server with all required environment variables

cd /Users/kiteboard/periodicdent42/app

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH to include repository root
export PYTHONPATH=/Users/kiteboard/periodicdent42:$PYTHONPATH

# Database configuration
export DB_USER=ard_user
export DB_PASSWORD=ard_secure_password_2024
export DB_NAME=ard_intelligence
export DB_HOST=localhost
export DB_PORT=5433

# Start uvicorn
exec uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
