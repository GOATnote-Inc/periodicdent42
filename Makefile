.PHONY: help repro evidence train collect-mock mock epistemic-ci clean

help:
	@echo "GOATnote Autonomous R&D Intelligence Layer - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  repro         - Verify hermetic builds locally (run twice, compare hashes)"
	@echo "  evidence      - Generate local reproducibility evidence pack"
	@echo "  train         - Train ML test selector model"
	@echo "  collect-mock  - Generate mock CI run data for testing"
	@echo "  mock          - Full epistemic CI pipeline with 100 mock tests"
	@echo "  epistemic-ci  - Run epistemic CI pipeline (score + select + report)"
	@echo "  clean         - Remove build artifacts"

repro:
	@echo "=== Hermetic Build Verification ==="
	@echo "Building first time..."
	@nix build .#default
	@nix hash path ./result > /tmp/hash1.txt
	@echo "First hash: $$(cat /tmp/hash1.txt)"
	@rm result
	@echo "Building second time..."
	@nix build .#default
	@nix hash path ./result > /tmp/hash2.txt
	@echo "Second hash: $$(cat /tmp/hash2.txt)"
	@if diff -q /tmp/hash1.txt /tmp/hash2.txt > /dev/null; then \
		echo "✅ SUCCESS: Builds are bit-identical!"; \
		cat /tmp/hash1.txt; \
	else \
		echo "❌ FAIL: Builds differ"; \
		exit 1; \
	fi

evidence:
	@echo "=== Generating Evidence Pack ==="
	@mkdir -p artifact
	@echo "# Local Reproducibility Evidence" > artifact/REPRODUCIBILITY.md
	@echo "Generated: $$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> artifact/REPRODUCIBILITY.md
	@echo "" >> artifact/REPRODUCIBILITY.md
	@nix build .#default
	@nix hash path ./result >> artifact/sha256.txt
	@echo "✅ Evidence pack created in artifact/"

train:
	@echo "=== Training ML Test Selector ==="
	@python3 scripts/train_selector.py \
		--data data/ci_runs.jsonl \
		--out models/selector-v1.pkl \
		--meta models/metadata.json

collect-mock:
	@echo "=== Generating Mock CI Data ==="
	@for i in $$(seq 1 10); do \
		python3 scripts/collect_ci_runs.py --mock; \
	done
	@echo "✅ Generated 10 mock CI run entries"

mock: clean
	@echo "=========================================="
	@echo "Epistemic CI - Mock Mode"
	@echo "=========================================="
	@mkdir -p data artifact
	@python3 scripts/collect_ci_runs.py --mock 100 --inject-failures 0.12
	@python3 scripts/train_selector.py
	@python3 scripts/score_eig.py
	@python3 scripts/select_tests.py
	@python3 scripts/gen_ci_report.py
	@echo ""
	@echo "✅ Complete! View report: open artifact/ci_report.md"

epistemic-ci:
	@echo "=========================================="
	@echo "Epistemic CI Pipeline"
	@echo "=========================================="
	@mkdir -p artifact
	@python3 scripts/score_eig.py
	@python3 scripts/select_tests.py
	@python3 scripts/gen_ci_report.py
	@echo ""
	@echo "✅ Complete! View report: open artifact/ci_report.md"

clean:
	@echo "=== Cleaning Artifacts ==="
	@rm -rf artifact/ result out1 out2 data/ models/
	@rm -f /tmp/hash1.txt /tmp/hash2.txt
	@echo "✅ Cleaned build artifacts"