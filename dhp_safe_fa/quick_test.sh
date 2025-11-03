#!/bin/bash
# Efficient iteration: incremental build + test
set -e

cd /home/shadeform/dhp_safe_fa

# Editable install (once) - reuses build artifacts
if [ ! -f ".installed" ]; then
    pip3 install -e . --quiet
    touch .installed
fi

# Incremental rebuild (only changed files)
python3 setup.py build_ext --inplace --force 2>&1 | grep -E '(ptxas|error|warning)' || true

# Run tests
python3 test_all_kernels.py

