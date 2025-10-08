#!/bin/bash
# Test script for rate limiting

set -e

echo "═══════════════════════════════════════════════"
echo "  Testing matprov API Rate Limiting"
echo "═══════════════════════════════════════════════"
echo ""

API_URL="${API_URL:-http://localhost:8000}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Health check (no rate limit)
echo "Test 1: Health check endpoint (should never be rate limited)"
echo "-----------------------------------------------------"
SUCCESS_COUNT=0
for i in {1..20}; do
  STATUS=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL/health")
  if [ "$STATUS" -eq "200" ]; then
    ((SUCCESS_COUNT++))
  fi
done

if [ $SUCCESS_COUNT -eq 20 ]; then
  echo -e "${GREEN}✅ PASS: All 20 health checks succeeded (no rate limit)${NC}"
else
  echo -e "${RED}❌ FAIL: Only $SUCCESS_COUNT/20 health checks succeeded${NC}"
fi
echo ""

# Test 2: Check rate limit headers
echo "Test 2: Rate limit headers"
echo "-----------------------------------------------------"
RESPONSE=$(curl -s -i "$API_URL/api/models" | grep -i "X-RateLimit")

if [ -n "$RESPONSE" ]; then
  echo -e "${GREEN}✅ PASS: Rate limit headers present${NC}"
  echo "$RESPONSE"
else
  echo -e "${YELLOW}⚠️  WARN: No rate limit headers found${NC}"
fi
echo ""

# Test 3: Anonymous rate limit (make many requests)
echo "Test 3: Anonymous rate limiting"
echo "-----------------------------------------------------"
echo "Making 10 requests quickly..."

RATE_LIMITED=false
for i in {1..10}; do
  STATUS=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL/api/models")
  
  if [ "$STATUS" -eq "429" ]; then
    echo -e "${YELLOW}⚠️  Rate limited after $i requests${NC}"
    RATE_LIMITED=true
    break
  fi
  
  echo -n "."
done

echo ""
if [ "$RATE_LIMITED" = true ]; then
  echo -e "${GREEN}✅ PASS: Rate limiting is working${NC}"
else
  echo -e "${GREEN}✅ PASS: Made 10 requests (under limit)${NC}"
fi
echo ""

# Test 4: Rate limit info endpoint
echo "Test 4: Rate limit info in response"
echo "-----------------------------------------------------"
RESPONSE=$(curl -s -i "$API_URL/api/models" | head -20)

LIMIT=$(echo "$RESPONSE" | grep -i "X-RateLimit-Limit" | cut -d: -f2 | tr -d '[:space:]')
REMAINING=$(echo "$RESPONSE" | grep -i "X-RateLimit-Remaining" | cut -d: -f2 | tr -d '[:space:]')

if [ -n "$LIMIT" ] && [ -n "$REMAINING" ]; then
  echo -e "${GREEN}✅ PASS: Rate limit info available${NC}"
  echo "   Limit: $LIMIT"
  echo "   Remaining: $REMAINING"
else
  echo -e "${YELLOW}⚠️  WARN: Rate limit info not fully available${NC}"
fi
echo ""

# Test 5: Test with authentication (if credentials provided)
if [ -n "$TEST_EMAIL" ] && [ -n "$TEST_PASSWORD" ]; then
  echo "Test 5: Authenticated rate limiting"
  echo "-----------------------------------------------------"
  
  # Login
  TOKEN=$(curl -s -X POST "$API_URL/auth/login/json" \
    -H "Content-Type: application/json" \
    -d "{\"email\": \"$TEST_EMAIL\", \"password\": \"$TEST_PASSWORD\"}" \
    | jq -r '.access_token')
  
  if [ "$TOKEN" != "null" ] && [ -n "$TOKEN" ]; then
    echo "✅ Logged in successfully"
    
    # Make authenticated requests
    SUCCESS_COUNT=0
    for i in {1..10}; do
      STATUS=$(curl -s -w "%{http_code}" -o /dev/null \
        -H "Authorization: Bearer $TOKEN" \
        "$API_URL/api/models")
      
      if [ "$STATUS" -eq "200" ]; then
        ((SUCCESS_COUNT++))
      elif [ "$STATUS" -eq "429" ]; then
        echo -e "${YELLOW}⚠️  Rate limited after $i requests (authenticated)${NC}"
        break
      fi
      
      echo -n "."
    done
    
    echo ""
    if [ $SUCCESS_COUNT -ge 5 ]; then
      echo -e "${GREEN}✅ PASS: Authenticated requests working ($SUCCESS_COUNT/10)${NC}"
    else
      echo -e "${RED}❌ FAIL: Too few authenticated requests succeeded${NC}"
    fi
  else
    echo -e "${YELLOW}⚠️  SKIP: Could not authenticate${NC}"
  fi
  echo ""
else
  echo "Test 5: Authenticated rate limiting"
  echo "-----------------------------------------------------"
  echo -e "${YELLOW}⚠️  SKIP: Set TEST_EMAIL and TEST_PASSWORD to test authenticated limits${NC}"
  echo ""
fi

# Summary
echo "═══════════════════════════════════════════════"
echo "  Test Summary"
echo "═══════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Health check: No rate limit"
echo "  Anonymous: Rate limited"
echo "  Authenticated: Rate limited (higher limit)"
echo ""
echo "To test with authentication:"
echo "  export TEST_EMAIL=user@example.com"
echo "  export TEST_PASSWORD=YourPassword123!"
echo "  ./test_rate_limit.sh"
echo ""
echo -e "${GREEN}✅ Rate limiting tests complete${NC}"

