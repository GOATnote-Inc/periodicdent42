.PHONY: setup features train al test evidence clean lint check-deps help

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
SEED := 42

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Create virtual environment and install dependencies
	python3.10 -m venv .venv
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e .[dev]
	@echo "✓ Setup complete. Activate with: source .venv/bin/activate"

check-deps:  ## Verify matminer and critical dependencies
	@$(PYTHON) -c "import matminer; print('✓ matminer:', matminer.__version__)" || \
		echo "⚠ matminer not available; will use fallback featurizer"
	@$(PYTHON) -c "import ngboost; print('✓ ngboost:', ngboost.__version__)"
	@$(PYTHON) -c "import mapie; print('✓ mapie:', mapie.__version__)"
	@$(PYTHON) -c "import torch; print('✓ torch:', torch.__version__)"

features:  ## Generate features from raw data
	$(PYTHON) -m src.features.composition \
		--input data/raw/superconductor.csv \
		--output data/processed/features.parquet \
		--seed $(SEED)
	$(PYTHON) -m src.data.contracts \
		--input data/processed/features.parquet \
		--output data/contracts/dataset_contract_v1.json
	@echo "✓ Features generated and contract validated"

train:  ## Train all baseline models with uncertainty quantification
	@echo "Training RF+QRF baseline..."
	$(PYTHON) -m src.pipelines.train_baseline --config configs/train_rf.yaml --seed $(SEED)
	@echo "Training MLP+MC-Dropout baseline..."
	$(PYTHON) -m src.pipelines.train_baseline --config configs/train_mlp_mc.yaml --seed $(SEED)
	@echo "Training NGBoost baseline..."
	$(PYTHON) -m src.pipelines.train_baseline --config configs/train_ngb.yaml --seed $(SEED)
	@echo "✓ All models trained"

al:  ## Run active learning experiments (5 seeds, 20 rounds)
	$(PYTHON) -m src.pipelines.run_active_learning \
		--config configs/al_ucb.yaml \
		--rounds 20 \
		--k-per-round 10 \
		--seeds 5 \
		--diversity kmedoids
	@echo "✓ Active learning experiments complete"

al-fast:  ## Quick AL smoke test (2 seeds, 5 rounds)
	$(PYTHON) -m src.pipelines.run_active_learning \
		--config configs/al_ucb.yaml \
		--rounds 5 \
		--k-per-round 5 \
		--seeds 2 \
		--diversity kmedoids
	@echo "✓ Fast AL smoke test complete"

test:  ## Run all tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "✓ Tests complete. Coverage report: htmlcov/index.html"

test-fast:  ## Run fast tests only (skip slow/integration)
	$(PYTEST) tests/ -v -m "not slow and not integration"

lint:  ## Run code quality checks
	.venv/bin/ruff check src/ tests/
	.venv/bin/mypy src/
	@echo "✓ Linting complete"

evidence:  ## Bundle evidence artifacts
	$(PYTHON) -m src.reporting.artifacts --bundle evidence/latest
	@echo "✓ Evidence pack created in evidence/latest/"
	@echo "Manifest SHA-256:"
	@cat evidence/latest/manifest.json | grep -A1 "sha256"

clean:  ## Remove generated files and caches
	rm -rf .venv
	rm -rf evidence/runs/*
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf *.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

dev-setup:  ## Quick setup for development (features + fast test)
	make features
	make test-fast
	@echo "✓ Development environment ready"

full-pipeline:  ## Run complete pipeline (features → train → AL → evidence)
	make features
	make train
	make al
	make evidence
	@echo "✓ Full pipeline complete"

.DEFAULT_GOAL := help

