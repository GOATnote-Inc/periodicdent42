#!/usr/bin/env python3
"""
vLLM server example with BlackwellSparseK backend.

Usage:
    python examples/vllm_server.py --model meta-llama/Llama-3.1-7B

Then test with:
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "meta-llama/Llama-3.1-7B", "prompt": "Hello", "max_tokens": 50}'
"""

import argparse
import sys

try:
    from blackwell_sparsek.backends import register_vllm_backend
    register_vllm_backend()
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"❌ vLLM backend not available: {e}")
    BACKEND_AVAILABLE = False


def main():
    """Run vLLM server with BlackwellSparseK backend."""
    if not BACKEND_AVAILABLE:
        print("❌ Cannot start vLLM server without BlackwellSparseK backend")
        return 1
    
    parser = argparse.ArgumentParser(description="vLLM server with BlackwellSparseK")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    args = parser.parse_args()
    
    print("=" * 80)
    print("vLLM Server with BlackwellSparseK Backend")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Backend: SPARSEK_XFORMERS")
    print("=" * 80)
    
    # Import vLLM entrypoint
    try:
        from vllm.entrypoints.openai.api_server import run_server
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        print("❌ vLLM not installed")
        return 1
    
    # Configure engine
    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        attention_backend="SPARSEK_XFORMERS",
        gpu_memory_utilization=0.9,
    )
    
    # Start server
    print("\nStarting server...")
    run_server(engine_args, host=args.host, port=args.port)
    
    return 0


if __name__ == "__main__":
    exit(main())

