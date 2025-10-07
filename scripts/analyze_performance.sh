#!/bin/bash
# Automated Performance Analysis - Download and Analyze Flamegraphs
# Part of Phase 3: Continuous Profiling
# 
# Usage: ./scripts/analyze_performance.sh
#
# What it does:
# 1. Downloads latest CI artifacts (flamegraphs + profiles)
# 2. Extracts and analyzes bottlenecks
# 3. Generates summary report
# 4. Opens flamegraphs in browser
#
# Author: GOATnote Autonomous Research Lab Initiative
# Date: October 6, 2025

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║  🔥 AUTOMATED PERFORMANCE ANALYSIS                                        ║"
echo "║     Download → Analyze → Report → Visualize                              ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Installing..."
    echo ""
    echo "Please run: brew install gh"
    echo "Then authenticate: gh auth login"
    echo ""
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI"
    echo ""
    echo "Please run: gh auth login"
    echo ""
    exit 1
fi

echo "✅ GitHub CLI authenticated"
echo ""

# Get latest CI run
echo "📥 Step 1: Finding latest CI run..."
RUN_ID=$(gh run list --workflow=ci.yml --branch=main --limit=1 --json databaseId --jq '.[0].databaseId')

if [ -z "$RUN_ID" ]; then
    echo "❌ No CI runs found"
    exit 1
fi

echo "✅ Found run ID: $RUN_ID"
echo ""

# Create artifacts directory
ARTIFACTS_DIR="artifacts/performance_analysis"
mkdir -p "$ARTIFACTS_DIR"

echo "📥 Step 2: Downloading artifacts..."
cd "$ARTIFACTS_DIR"

# Download flamegraphs
echo "   → Downloading performance-flamegraphs..."
gh run download "$RUN_ID" --name performance-flamegraphs 2>/dev/null || echo "   ⚠️  No flamegraphs found (may not have been generated yet)"

# Download profiles
echo "   → Downloading performance-profiles..."
gh run download "$RUN_ID" --name performance-profiles 2>/dev/null || echo "   ⚠️  No profiles found (may not have been generated yet)"

cd - > /dev/null

echo "✅ Artifacts downloaded to: $ARTIFACTS_DIR"
echo ""

# Analyze profile JSON files
echo "🔍 Step 3: Analyzing performance profiles..."
echo ""

REPORT_FILE="$ARTIFACTS_DIR/performance_report.md"

cat > "$REPORT_FILE" << 'EOF'
# Performance Analysis Report

**Date**: $(date)
**Run ID**: $RUN_ID
**Analysis**: Automated

---

## 📊 Bottlenecks Identified

EOF

# Parse JSON profiles to extract timing data
for profile in "$ARTIFACTS_DIR"/*.json; do
    if [ -f "$profile" ]; then
        SCRIPT_NAME=$(basename "$profile" | sed 's/_.*\.json//')
        DURATION=$(jq -r '.duration_seconds // "N/A"' "$profile" 2>/dev/null || echo "N/A")
        
        echo "### $SCRIPT_NAME" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "- **Total Duration**: ${DURATION}s" >> "$REPORT_FILE"
        echo "- **Flamegraph**: ${profile%.json}.svg" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        echo "   ✓ $SCRIPT_NAME: ${DURATION}s"
    fi
done

echo "" >> "$REPORT_FILE"
echo "---" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## 🔥 How to Use Flamegraphs" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "1. Open the SVG files in your browser" >> "$REPORT_FILE"
echo "2. Look for WIDE bars (= taking lots of time)" >> "$REPORT_FILE"
echo "3. Hover to see function names and percentages" >> "$REPORT_FILE"
echo "4. Click to zoom into specific functions" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "**What to optimize**:" >> "$REPORT_FILE"
echo "- Widest bars at any level (biggest bottlenecks)" >> "$REPORT_FILE"
echo "- Unexpectedly slow functions" >> "$REPORT_FILE"
echo "- Repeated patterns (caching opportunities)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "✅ Analysis complete"
echo ""

# Open flamegraphs in browser
echo "🔥 Step 4: Opening flamegraphs..."
FLAMEGRAPH_COUNT=0
for svg in "$ARTIFACTS_DIR"/*.svg; do
    if [ -f "$svg" ]; then
        echo "   → Opening: $(basename "$svg")"
        open "$svg"
        FLAMEGRAPH_COUNT=$((FLAMEGRAPH_COUNT + 1))
    fi
done

if [ $FLAMEGRAPH_COUNT -eq 0 ]; then
    echo "   ⚠️  No flamegraphs found to open"
    echo ""
    echo "This may happen if:"
    echo "  1. CI run hasn't completed yet"
    echo "  2. Performance profiling job didn't run (only runs on main branch)"
    echo "  3. Profiling failed (check CI logs)"
    echo ""
    echo "Check CI status: gh run view $RUN_ID"
else
    echo "✅ Opened $FLAMEGRAPH_COUNT flamegraph(s) in browser"
fi

echo ""

# Open report
echo "📄 Step 5: Opening performance report..."
open "$REPORT_FILE"
echo "✅ Report: $REPORT_FILE"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ AUTOMATED ANALYSIS COMPLETE"
echo ""
echo "Next Steps:"
echo "  1. Review the flamegraphs (opened in browser)"
echo "  2. Identify the widest bars (biggest bottlenecks)"
echo "  3. Read the performance report: $REPORT_FILE"
echo "  4. Optimize the #1 bottleneck (Priority 3)"
echo ""
echo "Artifacts location: $ARTIFACTS_DIR"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
