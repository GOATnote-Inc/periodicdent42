#!/bin/bash
#
# Cleanup GitHub Actions Workflow Runs
# Deletes old workflow runs to clean up the Actions tab
#

set -e

REPO="GOATnote-Inc/periodicdent42"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Cleaning Up GitHub Actions Workflow Runs                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Install it with: brew install gh"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

echo "Fetching workflow runs..."
echo ""

# Get list of workflows
workflows=$(gh api repos/$REPO/actions/workflows --jq '.workflows[] | "\(.id)|\(.name)|\(.path)"')

total_deleted=0

echo "$workflows" | while IFS='|' read -r workflow_id workflow_name workflow_path; do
    echo "Processing: $workflow_name (ID: $workflow_id)"
    
    # Get runs for this workflow (limit to first 100)
    runs=$(gh api "repos/$REPO/actions/workflows/$workflow_id/runs?per_page=100" --jq '.workflow_runs[] | .id' 2>/dev/null || echo "")
    
    if [ -z "$runs" ]; then
        echo "  No runs found"
        echo ""
        continue
    fi
    
    run_count=$(echo "$runs" | wc -l | tr -d ' ')
    echo "  Found $run_count runs"
    
    # Delete each run
    echo "$runs" | while read -r run_id; do
        gh api "repos/$REPO/actions/runs/$run_id" -X DELETE 2>/dev/null && echo -n "." || echo -n "x"
    done
    echo ""
    echo "  ✅ Deleted $run_count runs"
    echo ""
    
    ((total_deleted+=run_count)) || true
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     ✅ Cleanup Complete                                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Note: GitHub API limits to 100 runs per request."
echo "If you have more than 100 runs per workflow, run this script multiple times."
echo ""
echo "Next steps:"
echo "  1. Commit the disabled workflow files"
echo "  2. Push to GitHub"
echo "  3. Verify Actions tab is clean"
echo ""

