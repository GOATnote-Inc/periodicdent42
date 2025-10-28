#!/bin/bash
#
# Cleanup Dependabot PRs
# Closes all open dependabot PRs to clean up the repository
#

set -e

REPO="GOATnote-Inc/periodicdent42"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Cleaning Up Dependabot PRs                                   ║"
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

echo "Fetching open dependabot PRs..."
echo ""

# Get all open PRs from dependabot
dependabot_prs=$(gh pr list --repo $REPO --author "app/dependabot" --state open --json number,title --jq '.[] | "\(.number)|\(.title)"')

if [ -z "$dependabot_prs" ]; then
    echo "✅ No open dependabot PRs found."
    exit 0
fi

echo "Found dependabot PRs:"
echo "$dependabot_prs" | while IFS='|' read -r number title; do
    echo "  #$number: $title"
done
echo ""

# Ask for confirmation
read -p "Close all these PRs? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Closing PRs..."
echo ""

# Close each PR
echo "$dependabot_prs" | while IFS='|' read -r number title; do
    echo "Closing #$number: $title"
    gh pr close $number --repo $REPO --comment "Closing dependabot PRs. Dependency updates will be managed manually per stability policy." 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✅ Closed #$number"
    else
        echo "  ❌ Failed to close #$number"
    fi
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     ✅ Cleanup Complete                                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Commit the removal of dependabot.yml"
echo "  2. Push to GitHub"
echo ""

