#!/usr/bin/env python3
"""
FlashCore RunPod Serverless Endpoint
FastAPI server with health checks and inference endpoints
"""

import os
import sys
import torch
import triton
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="FlashCore Inference API",
    description="High-performance attention inference with CUDA 13.0 + CUTLASS 4.3.0",
    version="1.0.0"
)

# Startup banner
print("=" * 70)
print("FlashCore RunPod Endpoint - Initializing")
print("=" * 70)
print()

# Verify CUDA on startup
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
device_capability = torch.cuda.get_device_capability(0) if cuda_available else None

print("🔍 CUDA Setup:")
print(f"  PyTorch:      {torch.__version__}")
print(f"  Triton:       {triton.__version__}")
print(f"  CUDA:         {'✅ Available' if cuda_available else '❌ Not Available'}")
if cuda_available:
    print(f"  Device:       {device_name}")
    print(f"  Capability:   sm_{device_capability[0]}{device_capability[1]}")
print()

# Check for FlashCore modules
flashcore_available = os.path.exists('flashcore/fast/attention_multihead.py')
print(f"🔍 FlashCore:   {'✅ Available' if flashcore_available else '⚠️  Not Found'}")
print()

# Request models
class InferenceRequest(BaseModel):
    """Inference request payload"""
    batch_size: int = 1
    num_heads: int = 96
    seq_length: int = 512
    head_dim: int = 64
    input_data: dict = {}

class InferenceResponse(BaseModel):
    """Inference response"""
    status: str
    latency_us: float = 0.0
    throughput_ops: float = 0.0
    message: str = ""

# Health check endpoint (required by RunPod)
@app.get("/health")
async def health_check():
    """
    Health check endpoint for RunPod rollout.
    Returns 200 OK if service is ready.
    """
    return {
        "status": "healthy",
        "cuda_available": cuda_available,
        "device": device_name,
        "device_capability": f"sm_{device_capability[0]}{device_capability[1]}" if device_capability else None,
        "pytorch_version": torch.__version__,
        "triton_version": triton.__version__,
        "flashcore_available": flashcore_available
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "FlashCore Inference API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Health check",
            "inference": "POST /inference - Run FlashAttention inference",
            "metrics": "GET /metrics - Performance metrics"
        },
        "cuda_available": cuda_available,
        "device": device_name
    }

# Inference endpoint
@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run FlashAttention inference.
    
    This is a placeholder implementation - replace with actual
    FlashCore kernel invocation for production use.
    """
    if not cuda_available:
        raise HTTPException(status_code=503, detail="CUDA not available")
    
    try:
        # Placeholder inference logic
        # TODO: Replace with actual FlashCore kernel call
        import time
        start = time.perf_counter()
        
        # Simulate inference
        # In production, this would call:
        # from flashcore.fast.attention_multihead import flash_attention
        # output = flash_attention(Q, K, V, ...)
        
        time.sleep(0.001)  # Simulate 1ms inference
        
        latency = (time.perf_counter() - start) * 1e6  # Convert to microseconds
        
        return InferenceResponse(
            status="success",
            latency_us=latency,
            throughput_ops=1.0 / (latency * 1e-6),
            message=f"Inference completed for B={request.batch_size}, H={request.num_heads}, S={request.seq_length}, D={request.head_dim}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Return system metrics"""
    metrics = {
        "cuda_available": cuda_available,
        "device": device_name,
    }
    
    if cuda_available:
        try:
            # Get GPU memory info
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
            max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            metrics.update({
                "gpu_memory_allocated_gb": round(memory_allocated, 2),
                "gpu_memory_reserved_gb": round(memory_reserved, 2),
                "gpu_memory_total_gb": round(max_memory, 2),
                "gpu_utilization_percent": round((memory_allocated / max_memory) * 100, 1)
            })
        except Exception as e:
            metrics["error"] = str(e)
    
    return metrics

# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute on server startup"""
    print("=" * 70)
    print("✅ FlashCore API Server - Ready")
    print("=" * 70)
    print(f"  Health:     http://0.0.0.0:8000/health")
    print(f"  Inference:  http://0.0.0.0:8000/inference")
    print(f"  Metrics:    http://0.0.0.0:8000/metrics")
    print(f"  Docs:       http://0.0.0.0:8000/docs")
    print("=" * 70)
    print()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute on server shutdown"""
    print("\n👋 Shutting down FlashCore API...")

# Main entry point
if __name__ == "__main__":
    # Get port from environment (RunPod default is 8000)
    port = int(os.environ.get("PORT", 8000))
    
    # Start uvicorn server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
