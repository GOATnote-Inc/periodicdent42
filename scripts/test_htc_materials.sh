#!/bin/bash
# Comprehensive HTC API Test Suite with Known Superconductors
# Tests predictions against experimental values

set -e

BASE_URL="https://ard-backend-dydzexswua-uc.a.run.app"
TOLERANCE=5.0  # Kelvin tolerance for validation

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      🧪 HTC Superconductor Test Suite                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Test counter
PASSED=0
FAILED=0
TOTAL=0

test_prediction() {
    local composition=$1
    local pressure=$2
    local expected_tc=$3
    local description=$4
    
    TOTAL=$((TOTAL + 1))
    echo -e "\n${TOTAL}. Testing ${composition} - ${description}"
    echo "   Pressure: ${pressure} GPa, Expected Tc: ${expected_tc} K"
    
    response=$(curl -s -X POST "${BASE_URL}/api/htc/predict" \
        -H "Content-Type: application/json" \
        -d "{\"composition\": \"${composition}\", \"pressure_gpa\": ${pressure}}" 2>&1)
    
    if echo "$response" | jq -e '.tc_predicted' > /dev/null 2>&1; then
        tc_pred=$(echo "$response" | jq -r '.tc_predicted')
        confidence=$(echo "$response" | jq -r '.confidence_level')
        lambda=$(echo "$response" | jq -r '.lambda_ep')
        xi=$(echo "$response" | jq -r '.xi_parameter')
        
        # Calculate error
        error=$(echo "scale=2; $tc_pred - $expected_tc" | bc | sed 's/^-//')
        
        echo -e "   Predicted: ${tc_pred} K (Confidence: ${confidence})"
        echo -e "   λ = ${lambda}, ξ = ${xi}"
        
        if (( $(echo "$error < $TOLERANCE" | bc -l) )); then
            echo -e "   ${GREEN}✅ PASS${NC} (Error: ${error} K)"
            PASSED=$((PASSED + 1))
        else
            echo -e "   ${RED}❌ FAIL${NC} (Error: ${error} K > ${TOLERANCE} K tolerance)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "   ${RED}❌ ERROR${NC}: $response"
        FAILED=$((FAILED + 1))
    fi
}

# Classical BCS Superconductors
echo -e "\n${YELLOW}╔═══════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  Classical BCS Superconductors       ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════╝${NC}"

test_prediction "MgB2" 0.0 39.0 "Magnesium diboride (discovered 2001)"
test_prediction "Nb3Sn" 0.0 18.3 "Niobium-tin (A15 structure)"
test_prediction "Nb3Ge" 0.0 23.2 "Niobium-germanium (highest Tc A15)"
test_prediction "NbN" 0.0 16.0 "Niobium nitride (thin films)"
test_prediction "V3Si" 0.0 17.0 "Vanadium silicide"

# High-Tc Cuprates
echo -e "\n${YELLOW}╔═══════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  High-Tc Cuprate Superconductors     ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════╝${NC}"

test_prediction "YBa2Cu3O7" 0.0 92.0 "YBCO (first >77K superconductor)"
test_prediction "Bi2Sr2CaCu2O8" 0.0 85.0 "BSCCO (Bi-2212)"
test_prediction "HgBa2Ca2Cu3O8" 0.0 133.0 "HBCCO (highest Tc cuprate at 1 atm)"

# Iron-based Superconductors  
echo -e "\n${YELLOW}╔═══════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  Iron-based Superconductors           ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════╝${NC}"

test_prediction "LaFeAsO" 0.0 26.0 "LaFeAsO (1111 family)"
test_prediction "BaFe2As2" 0.0 38.0 "BaFe2As2 (122 family)"
test_prediction "FeSe" 0.0 8.0 "Iron selenide (simplest Fe-based SC)"

# Chevrel Phases
echo -e "\n${YELLOW}╔═══════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  Chevrel Phase Superconductors        ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════╝${NC}"

test_prediction "PbMo6S8" 0.0 15.0 "Lead molybdenum sulfide"
test_prediction "SnMo6S8" 0.0 12.0 "Tin molybdenum sulfide"

# High-Pressure Hydrides
echo -e "\n${YELLOW}╔═══════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  High-Pressure Hydride Superconductors║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════╝${NC}"

test_prediction "H3S" 150.0 203.0 "Hydrogen sulfide (record holder 2015)"
test_prediction "LaH10" 170.0 250.0 "Lanthanum hydride (record holder 2019)"
test_prediction "CaH6" 150.0 215.0 "Calcium hydride"

# Organic Superconductors
echo -e "\n${YELLOW}╔═══════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  Other Interesting Materials          ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════╝${NC}"

test_prediction "Nb" 0.0 9.2 "Pure niobium (element with highest Tc)"
test_prediction "Pb" 0.0 7.2 "Lead (historically important)"
test_prediction "Sn" 0.0 3.7 "Tin (one of first discovered)"

# Summary
echo -e "\n╔══════════════════════════════════════════════════════════╗"
echo -e "║                   TEST SUMMARY                          ║"
echo -e "╚══════════════════════════════════════════════════════════╝"
echo ""
echo -e "Total Tests:  ${TOTAL}"
echo -e "${GREEN}Passed:       ${PASSED}${NC}"
echo -e "${RED}Failed:       ${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. Review predictions.${NC}"
    exit 1
fi

