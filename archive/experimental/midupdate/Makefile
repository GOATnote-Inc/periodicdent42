PYTHON=python3
VENV?=.venv
PIP?=$(VENV)/bin/pip
PYTHON_BIN?=$(VENV)/bin/python

.DEFAULT_GOAL := help

help:
@echo "Available targets:"
@echo "  make venv           # create virtual environment"
@echo "  make dev            # run orchestrator and UI (local dev)"
@echo "  make build-dataset  # generate processed dataset"
@echo "  make train          # fine-tune planner"
@echo "  make eval           # run offline evaluation"
@echo "  make campaigns-demo # run three simulated campaigns"
@echo "  make bundle RUN=<id> # export provenance bundle"

venv:
python3 -m venv $(VENV)
$(PIP) install --upgrade pip
$(PIP) install -r requirements.txt

requirements.txt: ../requirements.txt
cp $< $@

dev: venv
TMUX='' $(PYTHON_BIN) -m uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8081

build-dataset: venv
$(PYTHON_BIN) training/build_dataset.py --config training/cfg/dataset.yaml

train: venv
$(PYTHON_BIN) training/train.py --config-path training/cfg --config-name train

eval: venv
$(PYTHON_BIN) training/eval.py --config-path training/cfg --config-name eval

campaigns-demo: venv
$(PYTHON_BIN) orchestrator/campaign_demo.py --config campaigns

bundle: venv
@if [ -z "$(RUN)" ]; then echo "RUN=<id> required" && exit 1; fi
$(PYTHON_BIN) orchestrator/bundle.py --run-id $(RUN)

.PHONY: help venv dev build-dataset train eval campaigns-demo bundle
