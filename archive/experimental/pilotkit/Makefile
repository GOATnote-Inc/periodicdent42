PYTHON=python3
POETRY?=
VENV?=.venv
PIP=$(VENV)/bin/pip
PYTHON_BIN=$(VENV)/bin/python
TYPER=$(PYTHON_BIN) -m pilotkit.orchestrator.cli

venv:
$(PYTHON) -m venv $(VENV)
$(PIP) install -r orchestrator/requirements.txt

install-ui:
cd ui && npm install

fmt:
$(PYTHON_BIN) -m black orchestrator

lint:
$(PYTHON_BIN) -m pip install black
$(PYTHON_BIN) -m black --check orchestrator

run-api:
$(PYTHON_BIN) -m uvicorn pilotkit.orchestrator.main:app --host 0.0.0.0 --port 8000

run-ui:
cd ui && npm run dev

dev:
docker compose up --build

demo:
$(TYPER) demo

demo-feedback:
$(TYPER) seed-feedback

report:
$(TYPER) report

iteration:
$(TYPER) iteration

clean:
rm -rf $(VENV) data

test:
$(PYTHON_BIN) -m pytest -q
