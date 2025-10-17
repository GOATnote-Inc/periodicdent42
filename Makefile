ln-build:
	python bench/layernorm/build_ln.py
ln-bench:
	python bench/layernorm/bench_ln.py
ln-sweep:
	python bench/layernorm/sweep_ln.py
ln-ncu:
	bash bench/layernorm/ncu_ln.sh
