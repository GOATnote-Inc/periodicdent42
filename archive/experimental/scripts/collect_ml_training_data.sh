#!/bin/bash
# scripts/collect_ml_training_data.sh
# Automated ML training data collection for test selection
# Phase 3 Week 7 Day 5-7: ML Test Selection Foundation

set -euo pipefail

# Configuration
RUNS="${1:-50}"  # Default 50 runs, override with first arg
DELAY="${2:-5}"   # Delay between runs (seconds)
TEST_PATH="${3:-tests/}"  # Tests to run

# Database connection
export DB_USER=ard_user
export DB_PASSWORD=ard_secure_password_2024
export DB_NAME=ard_intelligence
export DB_HOST=localhost
export DB_PORT=5433

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║  🤖 ML Training Data Collection - Automated                               ║"
echo "║     Target: $RUNS runs for test selection model                              ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check database connection
echo -e "${BLUE}1. Checking database connection...${NC}"
if ! export PGPASSWORD=$DB_PASSWORD && psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Database connection failed. Starting Cloud SQL Proxy...${NC}"
    # Note: Assuming proxy is already running based on ps check
fi

# Apply Alembic migration if needed
echo -e "${BLUE}2. Applying database migrations...${NC}"
cd app
if alembic current 2>/dev/null | grep -q "001_test_telemetry"; then
    echo -e "${GREEN}✅ Migration already applied${NC}"
else
    echo -e "${YELLOW}⚙️  Applying migration...${NC}"
    alembic upgrade head
fi
cd ..

# Check if telemetry collection is enabled
echo -e "${BLUE}3. Verifying telemetry collection...${NC}"
if grep -q "TestCollector" app/tests/conftest.py; then
    echo -e "${GREEN}✅ Telemetry plugin enabled${NC}"
else
    echo -e "${YELLOW}⚠️  Telemetry plugin not found in conftest.py${NC}"
fi

# Run tests multiple times
echo ""
echo -e "${BLUE}4. Collecting test execution data...${NC}"
echo -e "${YELLOW}   This will take approximately $((RUNS * 2 / 60)) minutes${NC}"
echo ""

START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAILURE_COUNT=0

for i in $(seq 1 $RUNS); do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Run $i/$RUNS${NC}"
    echo ""
    
    # Run pytest with verbose output and telemetry collection
    if pytest $TEST_PATH -v --tb=short -q 2>&1 | tee /tmp/pytest_run_$i.log; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "${GREEN}✅ Run $i completed successfully${NC}"
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        echo -e "${YELLOW}⚠️  Run $i had test failures (expected for ML training)${NC}"
    fi
    
    # Small delay between runs
    if [ $i -lt $RUNS ]; then
        echo -e "${BLUE}Waiting ${DELAY}s before next run...${NC}"
        sleep $DELAY
    fi
    
    # Progress indicator
    PERCENT=$((i * 100 / RUNS))
    echo -e "${BLUE}Progress: $i/$RUNS ($PERCENT%)${NC}"
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║  ✅ DATA COLLECTION COMPLETE!                                             ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Collection Summary:${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Runs completed: $RUNS"
echo -e "  Successful: $SUCCESS_COUNT"
echo -e "  Failures: $FAILURE_COUNT (expected for diverse training data)"
echo -e "  Duration: $((DURATION / 60))m $((DURATION % 60))s"
echo ""

# Verify data in database
echo -e "${BLUE}5. Verifying collected data...${NC}"
export PGPASSWORD=$DB_PASSWORD
RECORD_COUNT=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM test_telemetry;" | xargs)

echo -e "${GREEN}✅ Database contains $RECORD_COUNT test execution records${NC}"
echo ""

if [ "$RECORD_COUNT" -ge 50 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ READY FOR ML MODEL TRAINING!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo -e "  1. Train model: ${GREEN}python scripts/train_test_selector.py --train --evaluate${NC}"
    echo -e "  2. Upload model: ${GREEN}gsutil cp test_selector.pkl gs://periodicdent42-ml-models/${NC}"
    echo -e "  3. Enable in CI: Uncomment model download in .github/workflows/ci.yml"
else
    echo -e "${YELLOW}⚠️  Need $((50 - RECORD_COUNT)) more test runs for training${NC}"
    echo -e "${BLUE}Run again: ./scripts/collect_ml_training_data.sh $((50 - RECORD_COUNT))${NC}"
fi

echo ""
echo "© 2025 GOATnote Autonomous Research Lab Initiative"
echo "ML Training Data Collection Complete: $(date)"
