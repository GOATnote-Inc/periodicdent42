.PHONY: help repro evidence train collect-mock clean

help:
	@echo "GOATnote Autonomous R&D Intelligence Layer - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  repro         - Verify hermetic builds locally (run twice, compare hashes)"
	@echo "  evidence      - Generate local reproducibility evidence pack"
	@echo "  train         - Train ML test selector model"
	@echo "  collect-mock  - Generate mock CI run data for testing"
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
		--meta models/metadata.json \
		--verbose

collect-mock:
	@echo "=== Generating Mock CI Data ==="
	@for i in $$(seq 1 10); do \
		python3 scripts/collect_ci_runs.py --mock; \
	done
	@echo "✅ Generated 10 mock CI run entries"

clean:
	@echo "=== Cleaning Artifacts ==="
	@rm -rf artifact/ result out1 out2
	@rm -f /tmp/hash1.txt /tmp/hash2.txt
	@echo "✅ Cleaned build artifacts"