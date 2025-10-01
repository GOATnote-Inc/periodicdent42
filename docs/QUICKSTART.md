# Quick Start Guide

## Prerequisites

- Python 3.12+
- Rust 1.70+ (optional, for safety kernel)
- PostgreSQL 14+ (for production)
- Node.js 18+ (for UI)

## Installation

```bash
# Clone repository
cd periodicdent42

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev,ml]"

# (Optional) Compile Rust safety kernel
cd src/safety
cargo build --release
cd ../..
```

## Run Bootstrap

The bootstrap script sets up Phase 0 and Phase 1, including:
- Data schema validation
- Safety kernel compilation
- First closed-loop experiment

```bash
python scripts/bootstrap.py
```

Expected output:
```
Starting bootstrap setup for Autonomous R&D Intelligence Layer...

📦 Phase 0: Foundations
🧠 Phase 1: Intelligence

============================================================
BOOTSTRAP COMPLETE - AUTONOMOUS R&D INTELLIGENCE LAYER
============================================================

✅ Phase 0 Foundations:
   - Database schema initialized
   - Data contracts (Pydantic) validated
   - Safety kernel compiled
   - Experiment OS operational

✅ Phase 1 Intelligence:
   - GP surrogate model trained
   - EIG optimizer functional
   - First closed loop: 3 experiments executed
   - Success rate: 100%
```

## Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_core.py -v

# Rust tests
cd src/safety
cargo test
```

## Example: Submit an Experiment

```python
import asyncio
from src.experiment_os.core import ExperimentOS, DummyXRDDriver
from configs.data_schema import Sample, Protocol, Experiment

async def main():
    # Initialize system
    os_system = ExperimentOS()
    
    # Register instrument
    xrd = DummyXRDDriver("xrd-001", {})
    os_system.registry.register(xrd)
    await os_system.registry.connect_all()
    
    # Create sample
    sample = Sample(
        name="BaTiO3",
        composition={"Ba": 0.2, "Ti": 0.2, "O": 0.6}
    )
    
    # Create protocol
    protocol = Protocol(
        instrument_id="xrd-001",
        parameters={"scan_range": "20-80", "step_size": 0.02},
        duration_estimate_hours=2.0
    )
    
    # Create experiment
    exp = Experiment(
        sample_id=sample.id,
        protocol=protocol,
        created_by="user-alice",
        priority=5
    )
    
    # Submit and execute
    await os_system.submit_experiment(exp)
    dequeued = os_system.queue.dequeue()
    result = await os_system.execute_experiment(dequeued)
    
    print(f"Experiment completed: {result.success}")
    print(f"Measurements: {len(result.measurements)}")
    
    await os_system.registry.disconnect_all()

asyncio.run(main())
```

## Example: EIG-Driven Planning

```python
import numpy as np
from src.reasoning.eig_optimizer import GaussianProcessSurrogate, EIGOptimizer

# Training data
X_train = np.random.rand(10, 2)
y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1]

# Fit GP
gp = GaussianProcessSurrogate()
gp.fit(X_train, y_train)

# Create optimizer
optimizer = EIGOptimizer(gp)

# Generate candidates
X_pool = np.random.rand(50, 2)

# Select best experiments by EIG
selected = optimizer.select_batch_greedy(
    X_pool, 
    batch_size=5, 
    instrument_id="xrd-001"
)

for i, result in enumerate(selected):
    print(f"{i+1}. EIG={result.eig:.3f}, cost={result.cost_hours:.1f}h")
```

## Next Steps

1. **Integrate real instruments**: Replace dummy drivers with actual hardware APIs
2. **Add simulators**: Integrate PySCF, RDKit, ASE for real simulations
3. **Deploy UI**: Set up Next.js frontend for provenance viewer
4. **Configure database**: Set up PostgreSQL for production
5. **Run first experiments**: Execute Week 1-4 validation experiments

## Troubleshooting

### Import errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### Rust compilation fails
- Safety kernel is optional for initial testing
- Bootstrap script will skip if Rust toolchain not installed
- Install Rust: https://rustup.rs/

### Tests fail
```bash
# Check Python version
python --version  # Should be 3.12+

# Reinstall dependencies
pip install -r requirements.txt

# Run with verbose output
pytest -v --tb=long
```

## Documentation

- [Full Roadmap](roadmap.md) - Phases 0-5, milestones, KPIs
- [Development Instructions](instructions.md) - Coding standards, best practices
- [Architecture](../README.md) - System design overview
- [Cursor Rules](.cursor/rules/) - AI assistant guidelines for each moat

## Support

For issues, questions, or contributions:
- GitHub Issues: [link]
- Documentation: `docs/`
- Examples: `examples/` (coming soon)

