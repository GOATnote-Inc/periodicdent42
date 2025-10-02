.PHONY: setup install test lint run campaign make_setup ingest run.api run.web eval.offline canary finetune demo.data graph audit demo

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt && pip install -r app/requirements.txt

install:
	cd app && $(MAKE) setup-local

lint:
	cd app && $(MAKE) lint

test:
	cd app && $(MAKE) test
	@if [ -d tests ]; then pytest tests -q; fi

graph:
	python scripts/repo_graph.py --root . --json docs/dependency_graph.json --mermaid docs/ARCHITECTURE_MAP.md

audit:
	python scripts/repo_audit.py --root . --output docs/audit.json

demo:
	npm --prefix apps/web run dev

run:
	cd app && $(MAKE) dev

campaign:
	cd app && PYTHONPATH=src python ../scripts/run_uv_vis_campaign.py --experiments 50 --hours 24

make_setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

ingest:
	python -m services.rag.index

run.api:
	uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

run.web:
	npm --prefix apps/web run dev

eval.offline:
	python -m services.evals.runner

canary:
	python -m services.evals.runner

finetune:
	echo "Stub finetune placeholder"

demo.data:
	echo "Seeded demo telemetry"
