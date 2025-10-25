PYTHON?=python3
VENV?=.venv
PIP?=$(VENV)/bin/pip
UVICORN?=$(VENV)/bin/uvicorn

venv:
$(PYTHON) -m venv $(VENV)

.PHONY: dev
 dev: venv
$(PIP) install -r orchestrator/requirements.txt
$(VENV)/bin/pip install -e ..

.PHONY: api
api:
$(UVICORN) labloop.orchestrator.main:app --reload

.PHONY: demo-xrd
 demo-xrd:
$(VENV)/bin/python -m labloop.orchestrator.offline_loop examples/xrd_quick_scan.yaml --max-steps 6

.PHONY: demo-transport
 demo-transport:
$(VENV)/bin/python -m labloop.orchestrator.offline_loop examples/transport_iv_sweep.yaml --max-steps 6

.PHONY: tests
 tests:
$(VENV)/bin/pytest tests -q

.PHONY: bundle
 bundle:
$(VENV)/bin/python -m labloop.orchestrator.create_bundle bundle $$RUN
