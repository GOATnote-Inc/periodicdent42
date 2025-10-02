#!/bin/bash
# Setup Cloud SQL instance for metadata persistence
# This script creates a PostgreSQL database on Google Cloud SQL

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${PROJECT_ID:-"periodicdent42"}
REGION=${REGION:-"us-central1"}
INSTANCE_NAME=${INSTANCE_NAME:-"ard-intelligence-db"}
DATABASE_NAME=${DATABASE_NAME:-"ard_intelligence"}
DB_USER=${DB_USER:-"ard_user"}
TIER=${TIER:-"db-f1-micro"}  # Smallest instance for dev/testing
POSTGRES_VERSION=${POSTGRES_VERSION:-"POSTGRES_15"}

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🗄️  Cloud SQL Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Instance: $INSTANCE_NAME"
echo "Database: $DATABASE_NAME"
echo "User: $DB_USER"
echo "Tier: $TIER"
echo ""

# Check if instance already exists
echo -e "${YELLOW}Checking if Cloud SQL instance exists...${NC}"
if gcloud sql instances describe "$INSTANCE_NAME" --project="$PROJECT_ID" &>/dev/null; then
    echo -e "${GREEN}✅ Cloud SQL instance already exists: $INSTANCE_NAME${NC}"
    INSTANCE_EXISTS=true
else
    echo -e "${YELLOW}Creating new Cloud SQL instance...${NC}"
    INSTANCE_EXISTS=false
fi

# Create Cloud SQL instance if it doesn't exist
if [ "$INSTANCE_EXISTS" = false ]; then
    echo -e "${BLUE}Creating Cloud SQL instance (this may take 5-10 minutes)...${NC}"
    
    gcloud sql instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --database-version="$POSTGRES_VERSION" \
        --tier="$TIER" \
        --backup-start-time="03:00" \
        --availability-type="ZONAL" \
        --storage-type="SSD" \
        --storage-size="10GB" \
        --storage-auto-increase \
        --maintenance-window-day="SUN" \
        --maintenance-window-hour="04"
    
    echo -e "${GREEN}✅ Cloud SQL instance created: $INSTANCE_NAME${NC}"
fi

# Generate or retrieve database password
echo -e "${YELLOW}Setting up database password...${NC}"

# Check if password secret exists
if gcloud secrets describe db-password --project="$PROJECT_ID" &>/dev/null; then
    echo -e "${GREEN}✅ Database password secret already exists${NC}"
    DB_PASSWORD=$(gcloud secrets versions access latest --secret="db-password" --project="$PROJECT_ID")
else
    echo -e "${YELLOW}Generating secure database password...${NC}"
    DB_PASSWORD=$(openssl rand -base64 32)
    
    # Store in Secret Manager
    echo -n "$DB_PASSWORD" | gcloud secrets create db-password \
        --project="$PROJECT_ID" \
        --data-file=- \
        --replication-policy="automatic"
    
    echo -e "${GREEN}✅ Database password stored in Secret Manager${NC}"
fi

# Set root password if instance was just created
if [ "$INSTANCE_EXISTS" = false ]; then
    echo -e "${YELLOW}Setting root password...${NC}"
    gcloud sql users set-password postgres \
        --instance="$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --password="$DB_PASSWORD"
    
    echo -e "${GREEN}✅ Root password set${NC}"
fi

# Create database user
echo -e "${YELLOW}Creating database user: $DB_USER...${NC}"

# Try to create user (will fail if already exists, which is ok)
gcloud sql users create "$DB_USER" \
    --instance="$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --password="$DB_PASSWORD" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  User already exists (this is ok)${NC}"

# Create database
echo -e "${YELLOW}Creating database: $DATABASE_NAME...${NC}"

gcloud sql databases create "$DATABASE_NAME" \
    --instance="$INSTANCE_NAME" \
    --project="$PROJECT_ID" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  Database already exists (this is ok)${NC}"

echo -e "${GREEN}✅ Database created: $DATABASE_NAME${NC}"

# Grant permissions to Cloud Run service account
echo -e "${YELLOW}Granting Cloud SQL access to service account...${NC}"

SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"

# Add cloudsql.client role
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/cloudsql.client" \
    --condition=None >/dev/null 2>&1 || true

echo -e "${GREEN}✅ IAM permissions configured${NC}"

# Get connection name for Cloud Run
CONNECTION_NAME=$(gcloud sql instances describe "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --format="value(connectionName)")

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Cloud SQL Setup Complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}📋 Connection Information:${NC}"
echo "  Instance: $INSTANCE_NAME"
echo "  Database: $DATABASE_NAME"
echo "  User: $DB_USER"
echo "  Connection Name: $CONNECTION_NAME"
echo ""
echo -e "${BLUE}🔐 For Cloud Run deployment, set these environment variables:${NC}"
echo "  GCP_SQL_INSTANCE=$CONNECTION_NAME"
echo "  DB_NAME=$DATABASE_NAME"
echo "  DB_USER=$DB_USER"
echo ""
echo -e "${YELLOW}⚠️  Database password is stored in Secret Manager: db-password${NC}"
echo ""
echo -e "${BLUE}📝 Next steps:${NC}"
echo "  1. Run database migrations: cd app && alembic upgrade head"
echo "  2. Update Cloud Run deployment to use Cloud SQL"
echo "  3. Test connection: psql -h /cloudsql/$CONNECTION_NAME -U $DB_USER -d $DATABASE_NAME"
echo ""
echo -e "${BLUE}💰 Cost Estimate (db-f1-micro):${NC}"
echo "  ~\$7-10/month for always-on instance"
echo "  ~\$0.10/GB storage per month"
echo ""

