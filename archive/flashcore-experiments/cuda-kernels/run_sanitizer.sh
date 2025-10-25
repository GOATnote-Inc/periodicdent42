#!/bin/bash
# Run compute-sanitizer on FlashCore cp.async test
# This will reveal the exact bug causing "unspecified launch failure"

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=" * 80
echo "FlashCore cp.async - Compute Sanitizer Debug"
echo "=" * 80
echo ""

# Clean build
rm -rf ~/.cache/torch_extensions

echo "Running test with CUDA_LAUNCH_BLOCKING + memcheck..."
echo ""

CUDA_LAUNCH_BLOCKING=1 \
compute-sanitizer \
    --tool memcheck \
    --print-limit 100 \
    python3 test_cpasync.py 2>&1 | tee sanitizer_memcheck.log

echo ""
echo "=" * 80
echo "Sanitizer output saved to: sanitizer_memcheck.log"
echo "=" * 80

