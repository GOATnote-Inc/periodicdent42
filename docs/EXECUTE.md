# ⚡ EXECUTE

## Deploy
```bash
./RUNPOD_QUICKSTART.sh <your-runpod-pod-id>
```

## Results
```
Step 1: 2.8-3.5 TFLOPS   (corrected WGMMA)
Step 2: 10-15 TFLOPS     (4× K-loop)
Step 3: 30-40 TFLOPS     (pipeline)
Step 4: 45-55 TFLOPS     (TMA - next)
Step 5: 55-65 TFLOPS     (clusters - target)
```

## Benchmark
```bash
# On RunPod pod
python3 benchmark_vs_sglang.py
```

## Iterate
```bash
# On RunPod pod
make test            # Quick test
./iterate_h100.sh 3  # Test step 3
make profile         # Full profile
```

## Target
55-65 TFLOPS > SGLang (35-50) > vLLM (30-45)

