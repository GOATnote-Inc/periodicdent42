# periodic-pycore

Python bindings for the Periodic core planner built with [maturin](https://github.com/PyO3/maturin).

## Build

```bash
just pywheel
```

or manually:

```bash
cd rust/pycore
maturin build --release
```

## Usage

```python
from pycore import plan
result = plan({
    "description": "Improve catalyst throughput",
    "metrics": [
        {"name": "yield", "target": 0.9},
    ],
})
print(result)
```
