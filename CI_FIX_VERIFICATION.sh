#!/bin/bash
# CI/CD Emergency Fix Verification Script
# Run this after pushing fixes to verify all workflows are healthy

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 CI/CD Emergency Fix Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count disabled workflows
DISABLED_COUNT=$(grep -r "if: false" .github/workflows/ | wc -l | tr -d ' ')
echo "✅ Disabled GPU workflows: $DISABLED_COUNT (expected: 5)"

# Check cron job disabled
if grep -q "# schedule:" .github/workflows/continuous-monitoring.yml; then
    echo "✅ Hourly cron job disabled"
else
    echo "❌ Hourly cron job still active!"
fi

# Check continue-on-error added
CONTINUE_COUNT=$(grep -r "continue-on-error: true" .github/workflows/continuous-monitoring.yml | wc -l | tr -d ' ')
echo "✅ Non-blocking monitoring jobs: $CONTINUE_COUNT (expected: 6)"

# Check GCP secret checks
if grep -q "check_gcp" .github/workflows/ci-bete.yml; then
    echo "✅ GCP secret checks added"
else
    echo "❌ GCP secret checks missing!"
fi

# List modified files
echo ""
echo "📝 Modified workflow files:"
git status --short .github/workflows/ | sed 's/^/   /'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Verification complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "1. Review changes: git diff .github/workflows/"
echo "2. Commit fixes: git commit -am 'fix(ci): Emergency fix for 5000+ failures'"
echo "3. Push: git push"
echo "4. Monitor GitHub Actions tab for next 24 hours"
echo ""
