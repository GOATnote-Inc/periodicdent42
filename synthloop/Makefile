.PHONY: venv dev demo-synth bundle test

venv:
python3 -m venv .venv
. .venv/bin/activate && pip install -r orchestrator/requirements.txt -r ../requirements.txt

dev:
docker compose up --build

demo-synth:
python3 scripts/demo.py

bundle:
python3 scripts/bundle.py --run $$RUN

test:
pytest

