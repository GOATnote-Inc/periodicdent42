.PHONY: setup install test lint run campaign

setup:
python3 -m venv .venv
. .venv/bin/activate && pip install -r requirements.txt && pip install -r app/requirements.txt

install:
cd app && $(MAKE) setup-local

lint:
cd app && $(MAKE) lint

test:
cd app && $(MAKE) test

run:
cd app && $(MAKE) dev

campaign:
cd app && PYTHONPATH=src python ../scripts/run_uv_vis_campaign.py --experiments 50 --hours 24
