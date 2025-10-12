#!/bin/bash
# Automated Session Template Generator
# Version: 1.0
# Created: October 2025
# Usage: ./new_session.sh N+5 "Optimize memory access patterns"

set -euo pipefail

# Configuration
TEMPLATES_DIR="${TEMPLATES_DIR:-./sessions}"
PATTERNS_FILE="${PATTERNS_FILE:-./PATTERNS.md}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <session_number> [objective]"
    echo ""
    echo "Examples:"
    echo "  $0 N+4"
    echo "  $0 N+5 \"Implement vectorized memory access\""
    echo "  $0 N+6 \"Add Tensor Core support (WMMA)\""
    exit 1
fi

SESSION_NUM=$1
OBJECTIVE=${2:-"TBD - Fill in objective"}
DATE=$(date +"%B %d, %Y")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_FILE="${TEMPLATES_DIR}/SESSION_${SESSION_NUM}_PLAN_${TIMESTAMP}.md"

# Create sessions directory if it doesn't exist
mkdir -p "$TEMPLATES_DIR"

# Determine previous session number for baseline reference
if [[ $SESSION_NUM =~ N\+([0-9]+) ]]; then
    CURRENT_NUM=${BASH_REMATCH[1]}
    PREV_NUM=$((CURRENT_NUM - 1))
    if [ $PREV_NUM -eq 0 ]; then
        PREV_SESSION="N"
    else
        PREV_SESSION="N+$PREV_NUM"
    fi
else
    PREV_SESSION="Previous Session"
fi

# Get baseline from previous session if exists
PREV_BASELINE="[From Session $PREV_SESSION]"
if ls "${TEMPLATES_DIR}"/SESSION_${PREV_SESSION}_*.md 2>/dev/null | tail -1 | xargs grep -o "Final speedup.*" 2>/dev/null; then
    PREV_BASELINE=$(ls "${TEMPLATES_DIR}"/SESSION_${PREV_SESSION}_*.md 2>/dev/null | tail -1 | xargs grep -o "Final speedup.*" | head -1)
fi

# Generate session template
cat > "$SESSION_FILE" << 'EOF'
# Session ${SESSION_NUM} - ${DATE}

**Created**: ${DATE}  
**GPU**: TBD (will be filled during session)  
**Objective**: ${OBJECTIVE}  
**Time Budget**: 4 hours (2h optimization + 1h profiling + 1h documentation)  
**Prerequisites**: Pattern 9 validation complete ✅

---

## 📋 Pre-Session Checklist (10 min)

Before starting GPU session, complete these steps:

- [ ] **Read pattern library**: Review PATTERNS.md (5 min)
- [ ] **Review previous session**: Read SESSION_${PREV_SESSION}*.md (3 min)
- [ ] **Prepare profiling tools**: Check \`which ncu nsys\` (1 min)
- [ ] **Set session timer**: 4 hours maximum (with 30-min checkpoints)

**Baseline to Beat**: ${PREV_BASELINE}

---

## Phase 1: Environment Setup & Validation (15 min)

### Step 1.1: Start GPU Instance (5 min)

\`\`\`bash
# Start or connect to GPU instance
gcloud compute instances start cudadent42-l4-dev
gcloud compute ssh cudadent42-l4-dev --zone us-central1-a

# Document GPU details
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
\`\`\`

**Record**:
- GPU Name: _______________
- Memory: _______________
- Driver: _______________

### Step 1.2: Environment Validation (10 min)

\`\`\`bash
cd ~/periodicdent42/cudadent42

# Run Pattern 9 validation
./setup_environment_enhanced.sh 2>&1 | tee logs/session_${SESSION_NUM}_env.log

# Verify output
grep "🎉 Environment validation COMPLETE" logs/session_${SESSION_NUM}_env.log
\`\`\`

**Decision Gate 1**:
- ✅ All 5 environment checks pass → **Continue to Phase 2**
- ❌ Any check fails → **Fix issue and retry** (do not proceed)

---

## Phase 2: Baseline Measurement (20 min)

[... rest of template content from uploaded file ...]

EOF

# Replace variables in template
sed -i.bak "s/\${SESSION_NUM}/$SESSION_NUM/g; s/\${DATE}/$DATE/g; s/\${OBJECTIVE}/$OBJECTIVE/g; s/\${PREV_SESSION}/$PREV_SESSION/g; s/\${PREV_BASELINE}/$PREV_BASELINE/g" "$SESSION_FILE"
rm "${SESSION_FILE}.bak"

echo -e "${GREEN}✅ Session template created:${NC}"
echo -e "${BLUE}   $SESSION_FILE${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review template: cat $SESSION_FILE"
echo "  2. Start GPU instance"
echo "  3. Follow template phases sequentially"
echo "  4. Update template with actual results as you progress"
echo ""
echo -e "${GREEN}Session $SESSION_NUM ready to begin!${NC}"

