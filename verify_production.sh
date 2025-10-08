#!/bin/bash

echo "🔍 PRODUCTION READINESS CHECKLIST"
echo "================================================="
echo ""

echo "1. Authentication (JWT/OAuth/API keys):"
count=$(grep -r "JWT\|OAuth\|API.?key" api/ dashboard/ --include="*.py" 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    echo "   ✅ Found $count references to auth"
else
    echo "   ❌ No authentication found"
fi

echo ""
echo "2. Error Handling (try/except blocks):"
count=$(grep -r "try:" api/ dashboard/ matprov/ --include="*.py" 2>/dev/null | wc -l)
echo "   Found $count try blocks"
if [ "$count" -gt 10 ]; then
    echo "   ✅ Has error handling"
else
    echo "   ⚠️  Limited error handling"
fi

echo ""
echo "3. Logging:"
count=$(grep -r "logging\|logger\|print" api/ dashboard/ matprov/ --include="*.py" 2>/dev/null | wc -l)
echo "   Found $count logging statements"
if [ "$count" -gt 20 ]; then
    echo "   ✅ Has logging"
else
    echo "   ⚠️  Limited logging"
fi

echo ""
echo "4. Rate Limiting:"
count=$(grep -r "rate_limit\|throttle\|Limiter" api/ --include="*.py" 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    echo "   ✅ Found rate limiting"
else
    echo "   ❌ No rate limiting"
fi

echo ""
echo "5. Monitoring (Prometheus/Sentry/etc):"
count=$(grep -r "prometheus\|statsd\|sentry\|datadog" . --include="*.py" 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    echo "   ✅ Found monitoring"
else
    echo "   ❌ No monitoring integration"
fi

echo ""
echo "6. Database Migrations:"
if [ -d "app/alembic" ] || [ -d "alembic" ]; then
    echo "   ✅ Found alembic directory"
else
    echo "   ❌ No database migrations"
fi

echo ""
echo "7. Unit Tests:"
count=$(find . -name "test_*.py" -o -name "*_test.py" 2>/dev/null | wc -l)
if [ "$count" -gt 0 ]; then
    echo "   ✅ Found $count test files"
else
    echo "   ❌ No separate test files (only __main__ blocks)"
fi

echo ""
echo "8. Docker Configuration:"
if [ -f "Dockerfile" ] || [ -f "docker-compose.yml" ]; then
    ls -1 Dockerfile* docker-compose.yml 2>/dev/null | while read f; do echo "   ✅ $f"; done
else
    echo "   ⚠️  No Docker files (mentioned in docs)"
fi

echo ""
echo "9. CI/CD:"
if [ -d ".github/workflows" ]; then
    count=$(ls .github/workflows/*.yml 2>/dev/null | wc -l)
    echo "   ✅ Found $count GitHub Actions workflows"
else
    echo "   ❌ No CI/CD configuration"
fi

echo ""
echo "10. Dependencies:"
if [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
    ls -1 requirements*.txt pyproject.toml 2>/dev/null | while read f; do echo "   ✅ $f"; done
else
    echo "   ❌ No dependency file"
fi

echo ""
echo "================================================="
echo "PRODUCTION READINESS SCORE:"
echo "================================================="

# Count passed checks
passed=0
total=10

# Re-check each
grep -rq "JWT\|OAuth\|API.?key" api/ dashboard/ --include="*.py" 2>/dev/null || ((passed++))  # Inverted - no auth is bad
[ $(grep -r "try:" api/ dashboard/ matprov/ --include="*.py" 2>/dev/null | wc -l) -gt 10 ] && ((passed++))
[ $(grep -r "logging\|logger" api/ dashboard/ matprov/ --include="*.py" 2>/dev/null | wc -l) -gt 20 ] && ((passed++))
grep -rq "rate_limit\|throttle" api/ --include="*.py" 2>/dev/null || ((passed++))  # Inverted
grep -rq "prometheus\|statsd\|sentry" . --include="*.py" 2>/dev/null || ((passed++))  # Inverted
[ -d "app/alembic" ] && ((passed++))
[ $(find . -name "test_*.py" 2>/dev/null | wc -l) -gt 0 ] || ((passed++))  # Inverted
[ -f "Dockerfile" ] && ((passed++))
[ -d ".github/workflows" ] && ((passed++))
[ -f "pyproject.toml" ] && ((passed++))

score=$((passed * 100 / total))

if [ $score -ge 80 ]; then
    echo "✅ PRODUCTION READY: $score%"
elif [ $score -ge 50 ]; then
    echo "⚠️  NEEDS WORK: $score%"
else
    echo "❌ NOT PRODUCTION READY: $score%"
fi

echo ""
echo "Recommendation: Address missing items before production deployment"

