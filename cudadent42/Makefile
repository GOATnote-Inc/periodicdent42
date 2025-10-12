.PHONY: preflight build bench all

preflight:
	@bash scripts/gen_preflight.sh
	@bash tools/preflight.sh

build: preflight
	@python3 -c "import flashmoe_science; print('✔ import OK')"

bench: build
	@python3 benches/bench_correctness_and_speed.py --repeats 50 --warmup 10 --save-csv --verbose

all: bench

