#!/bin/bash
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         CLAIMS VERIFICATION AUDIT - AUTONOMOUS BASELINE        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Date: $(date)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Branch: $(git branch --show-current)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "CATEGORY A: TEST COVERAGE & QUALITY"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "--- A1: Coverage Percentage ---"
pytest --cov=src --cov-report=term --cov-report=json -q 2>&1 | tail -30
if [ -f coverage.json ]; then
    python3 -c "import json; data=json.load(open('coverage.json')); print(f'\n✅ VERIFIED Coverage: {data[\"totals\"][\"percent_covered\"]:.1f}%')"
fi
echo ""

echo "--- A2: Test Count ---"
pytest --collect-only -q 2>&1 | tail -10
echo ""
echo "Config tests specifically:"
pytest tests/test_config.py --collect-only -q 2>&1 | tail -5
echo ""

echo "--- A3: Config Module Coverage ---"
pytest tests/test_config.py --cov=src.config --cov-report=term -q 2>&1 | grep "src/config.py"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "CATEGORY B: FEATURE IMPLEMENTATION"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "--- B1: Leakage Guards ---"
echo "Files:"
ls -la src/guards/*.py 2>&1 || echo "❌ guards module missing"
ls -la src/data/splits.py 2>&1 || echo "❌ splits module missing"
echo ""
echo "Implementation check:"
grep -h "family\|duplicate\|cosine" src/guards/*.py src/data/*.py 2>/dev/null | head -5
echo ""
echo "Tests:"
ls tests/test_*guard* tests/test_*leakage* 2>&1 || echo "No guard tests found"
echo ""

echo "--- B2: OOD Detection ---"
echo "Files:"
ls -la src/guards/ood*.py 2>&1 || echo "❌ OOD module missing"
echo ""
echo "Tri-gate check:"
grep -h "Mahalanobis\|MahalanobisOOD" src/guards/ood*.py 2>/dev/null | head -2
grep -h "KDE\|KernelDensity" src/guards/ood*.py 2>/dev/null | head -2
grep -h "Conformal.*Novelty\|ConformalNovelty" src/guards/ood*.py 2>/dev/null | head -2
echo ""

echo "--- B3: Active Learning ---"
echo "Files:"
ls -la src/active_learning/ 2>&1 || echo "❌ AL module missing"
echo ""
echo "Acquisition functions:"
grep -h "def ucb\|def expected_improvement\|def maximum_variance" src/active_learning/*.py 2>/dev/null | head -5
echo ""
echo "Diversity mechanisms:"
grep -h "k_medoids\|greedy_diversity\|dpp_selection" src/active_learning/*.py 2>/dev/null | head -3
echo ""

echo "--- B4: Conformal Prediction ---"
echo "Files:"
ls -la src/uncertainty/conformal.py 2>&1 || echo "❌ conformal module missing"
echo ""
echo "Implementation:"
grep -h "class.*Conformal" src/uncertainty/*.py 2>/dev/null | head -3
grep -h "def.*picp\|def.*ece" src/uncertainty/*.py 2>/dev/null | head -3
echo ""

echo "--- B5: GO/NO-GO Policy ---"
echo "Documentation:"
ls -la docs/GO_NO_GO*.md 2>&1 || echo "❌ policy doc missing"
echo ""
echo "Implementation:"
grep -h "go_no_go" src/active_learning/*.py src/pipelines/*.py 2>/dev/null | head -3
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "CATEGORY C: PHYSICS VALIDATION"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "--- C1: Physics Documentation ---"
if [ -f docs/PHYSICS_JUSTIFICATION.md ]; then
    echo "✅ PHYSICS_JUSTIFICATION.md exists ($(wc -l < docs/PHYSICS_JUSTIFICATION.md) lines)"
    echo "Sections:"
    grep "^## " docs/PHYSICS_JUSTIFICATION.md | head -10
else
    echo "❌ PHYSICS_JUSTIFICATION.md missing"
fi
echo ""

echo "--- C2: Physics Tests ---"
echo "Physics test files:"
ls tests/test_*physics* tests/test_*sanity* 2>&1 || echo "No explicit physics test files"
echo ""
echo "Physics constraints in tests:"
grep -r "isotope\|valence\|electronegativity" tests/ 2>/dev/null | wc -l | xargs echo "References found:"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "CATEGORY D: ARTIFACTS & REPRODUCIBILITY"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "--- D1: Evidence Packs ---"
echo "Directories:"
ls -la evidence/ 2>&1 || echo "No evidence/ directory"
ls -la artifacts/ 2>&1 || echo "No artifacts/ directory"
echo ""
echo "Manifests:"
find . -maxdepth 3 -name "*manifest*.json" -o -name "*MANIFEST*" 2>/dev/null | head -10
echo ""

echo "--- D2: Reproducibility ---"
echo "Seed configurations:"
grep -r "seed.*42\|random_state.*42" configs/ src/config.py 2>/dev/null | head -5
echo ""
echo "Deterministic settings:"
grep -r "deterministic.*True" src/ 2>/dev/null | head -3
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "CATEGORY E: DOCUMENTATION ACCURACY"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "--- E1: README Claims ---"
echo "Coverage claim:"
grep -i "coverage.*%" README.md | head -2
echo ""
echo "Test count claim:"
grep -i "test.*pass" README.md | head -2
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "VERIFICATION COMPLETE"
echo "════════════════════════════════════════════════════════════════"
