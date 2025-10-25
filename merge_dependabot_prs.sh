#!/bin/bash
# Expert PR Merge Strategy for Dependabot Updates
# Created: October 25, 2025
# Purpose: Safely merge dependency updates with proper risk assessment

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  Dependabot PR Merge Strategy (Expert-Reviewed)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to merge PR safely
merge_pr() {
    local pr_num=$1
    local description=$2
    local risk=$3
    
    echo -e "${GREEN}[PR #${pr_num}]${NC} ${description}"
    echo "   Risk Level: ${risk}"
    
    if [ "$risk" = "LOW" ]; then
        echo "   Action: Auto-merge"
        gh pr merge "$pr_num" --squash --auto || echo "   ⚠️  Manual merge required"
    elif [ "$risk" = "MEDIUM" ]; then
        echo "   Action: Checkout and test"
        echo "   Run: gh pr checkout $pr_num && pytest tests/"
    elif [ "$risk" = "HIGH" ]; then
        echo "   Action: Critical performance testing required"
        echo "   Run: gh pr checkout $pr_num && python3 examples/quick_start.py"
    fi
    echo ""
}

echo "📋 Priority 1: Low Risk (CI/CD & Patches)"
echo "──────────────────────────────────────────"

# Get list of open PRs
echo "Fetching open PRs..."
gh pr list --limit 20 --json number,title,author | jq -r '.[] | select(.author.login == "dependabot[bot]") | "\(.number)|\(.title)"' | while IFS='|' read -r num title; do
    case "$title" in
        *"actions/attest-build-provenance"*)
            merge_pr "$num" "CI: Bump actions/attest-build-provenance (1→3)" "LOW"
            ;;
        *"actions/github-script"*)
            merge_pr "$num" "CI: Bump actions/github-script (6→8)" "LOW"
            ;;
        *"actions/setup-python"*)
            merge_pr "$num" "CI: Bump actions/setup-python (4→6)" "LOW"
            ;;
        *"patch-updates group"*"4 updates"*)
            merge_pr "$num" "Deps: Bump patch-updates group (4 updates)" "LOW"
            ;;
        *"patch-updates group"*"2 updates"*)
            merge_pr "$num" "Deps(app): Bump patch-updates group (2 updates)" "LOW"
            ;;
        *"mypy"*)
            merge_pr "$num" "Dev: Bump mypy (1.7.1→1.18.2)" "LOW"
            ;;
    esac
done

echo ""
echo "📋 Priority 2: Medium Risk (Test First)"
echo "──────────────────────────────────────────"

gh pr list --limit 20 --json number,title,author | jq -r '.[] | select(.author.login == "dependabot[bot]") | "\(.number)|\(.title)"' | while IFS='|' read -r num title; do
    case "$title" in
        *"pytest"*)
            merge_pr "$num" "Test: Bump pytest (7.4.3→8.4.2)" "MEDIUM"
            ;;
        *"alembic"*)
            merge_pr "$num" "DB: Bump alembic (1.12.1→1.17.0)" "MEDIUM"
            ;;
        *"pymxygen"*)
            merge_pr "$num" "App: Bump pymxygen (2023.9.10→2025.10.7)" "MEDIUM"
            ;;
        *"google-cloud-platform"*)
            merge_pr "$num" "App: Bump google-cloud-platform (1.38.1→1.121.0)" "MEDIUM"
            ;;
    esac
done

echo ""
echo "📋 Priority 3: HIGH RISK (Critical Testing)"
echo "──────────────────────────────────────────"

gh pr list --limit 20 --json number,title,author | jq -r '.[] | select(.author.login == "dependabot[bot]") | "\(.number)|\(.title)"' | while IFS='|' read -r num title; do
    case "$title" in
        *"numpy"*)
            echo -e "${RED}[PR #${num}]${NC} Critical: Bump numpy (1.26.2→2.3.4)"
            echo "   Risk Level: HIGH"
            echo "   ⚠️  NumPy 2.x has breaking changes for CUDA code"
            echo "   Action Required:"
            echo "     1. gh pr checkout $num"
            echo "     2. python3 flashcore/fast/attention_production.py  # Test correctness"
            echo "     3. python3 examples/quick_start.py                 # Test performance"
            echo "     4. If <5μs maintained: gh pr merge $num --squash"
            echo "     5. If performance degrades: gh pr close $num"
            echo ""
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Low Risk PRs: Auto-merged (or ready for auto-merge)"
echo "⚠️  Medium Risk PRs: Checkout and run 'pytest tests/'"
echo "🔴 High Risk PRs: Critical performance testing required"
echo ""
echo "Total Dependabot PRs: $(gh pr list --limit 20 --json author | jq '[.[] | select(.author.login == "dependabot[bot]")] | length')"
echo ""
echo "Next Steps:"
echo "  1. Run this script to see merge strategy"
echo "  2. Low risk PRs will auto-merge"
echo "  3. Test medium risk PRs before merging"
echo "  4. CRITICAL: Test NumPy PR thoroughly before merging"
echo ""

