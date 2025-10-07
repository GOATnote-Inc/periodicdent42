from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import telemetry plugin for automatic test data collection
try:
    from conftest_telemetry import *  # noqa: F401, F403
except ImportError:
    pass  # Telemetry plugin not available
