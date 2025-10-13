# PyTorch SDPA Autotune Report

**GPU**: NVIDIA L4
**Date**: 2025-10-13 18:16:08
**Iterations per config**: 50
**Baseline**: 0.3205 ms (B=32, H=8, S=512, D=64, auto backend)

---

## 🏆 Best Configuration Found

**Config**: B=32, H=8, S=128, D=64, backend=auto
**Latency**: 0.0635 ms (±0.0026 ms)
**Speedup vs Baseline**: 5.048×
**Throughput**: 16576 GFLOPS
**Bandwidth**: 259.0 GB/s

## 🔧 Best Backend (for B=32, H=8, S=512, D=64)

**Backend**: `auto`
**Latency**: 0.3205 ms
**Speedup vs Auto**: 1.000×

### How to Use
```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_memory_efficient=False,
    enable_math=False
):
    output = F.scaled_dot_product_attention(Q, K, V)
```

## 📊 Batch Size Analysis (H=8, S=512, D=64)

| Batch | Latency (ms) | Throughput (GFLOPS) | Samples/sec |
|-------|--------------|---------------------|-------------|
| 4 | 0.0758 | 27777 | 52787 |
| 8 | 0.1075 | 39666 | 74405 |
| 16 | 0.1690 | 50595 | 94697 |
| 32 | 0.3205 | 52775 | 99840 |
| 32 | 0.3215 | 52148 | 99522 |
| 32 | 0.4813 | 34290 | 66489 |
| 32 | 2.8682 | 5982 | 11157 |
| 32 | 0.3210 | 53032 | 99681 |
| 32 | 0.3917 | 42369 | 81699 |
| 32 | 0.3615 | 47286 | 88527 |
| 64 | 0.7506 | 44842 | 85266 |

## 📏 Sequence Length Analysis (B=32, H=8, D=64)

| Seq Length | Latency (ms) | Bandwidth (GB/s) | Efficiency (%) |
|------------|--------------|------------------|----------------|
| 128 | 0.0635 | 259.0 | 107.0% |
| 256 | 0.1393 | 239.6 | 99.0% |
| 512 | 0.3205 | 206.2 | 85.2% |
| 512 | 0.3215 | 203.7 | 84.2% |
| 512 | 0.4813 | 133.9 | 55.3% |
| 512 | 2.8682 | 23.4 | 9.7% |
| 512 | 0.3210 | 207.2 | 85.6% |
| 512 | 0.3917 | 165.5 | 68.4% |
| 512 | 0.3615 | 184.7 | 76.3% |
| 1024 | 1.3312 | 101.1 | 41.8% |
| 2048 | 4.7831 | 56.3 | 23.3% |

## 💡 Key Insights

2. **Batch Size**: B=32 achieves best samples/second (99840 samples/s).
3. **Sequence Length**: S=128 achieves best memory bandwidth utilization (259.0 GB/s).

## 📈 All Results

```json
[
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.32051199674606323,
    "mean_ms": 0.3255296009778976,
    "std_ms": 0.023905296414032037,
    "throughput_gflops": 52775.13667694526,
    "bandwidth_gb_s": 206.1528776443174,
    "speedup_vs_baseline": 1.0,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "flash",
      "dtype": "torch.float16"
    },
    "median_ms": 0.32153600454330444,
    "mean_ms": 0.32944192111492154,
    "std_ms": 0.01793751317915237,
    "throughput_gflops": 52148.400318510234,
    "bandwidth_gb_s": 203.7046887441806,
    "speedup_vs_baseline": 0.9968152624192252,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "memory_efficient",
      "dtype": "torch.float16"
    },
    "median_ms": 0.48127999901771545,
    "mean_ms": 0.5010227203369141,
    "std_ms": 0.02842993371838452,
    "throughput_gflops": 34289.60102337744,
    "bandwidth_gb_s": 133.94375399756814,
    "speedup_vs_baseline": 0.6659574414067132,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "math",
      "dtype": "torch.float16"
    },
    "median_ms": 2.8682239055633545,
    "mean_ms": 2.871970548629761,
    "std_ms": 0.014306568226571711,
    "throughput_gflops": 5981.9099440962755,
    "bandwidth_gb_s": 23.366835719126076,
    "speedup_vs_baseline": 0.11174580761438523,
    "iterations": 50
  },
  {
    "config": {
      "batch": 4,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.07577600330114365,
    "mean_ms": 0.07731264218688011,
    "std_ms": 0.0049309810703679,
    "throughput_gflops": 27776.616957535905,
    "bandwidth_gb_s": 108.50240999037463,
    "speedup_vs_baseline": 4.229729502522152,
    "iterations": 50
  },
  {
    "config": {
      "batch": 8,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.10751999914646149,
    "mean_ms": 0.10827711939811707,
    "std_ms": 0.0034294721893197567,
    "throughput_gflops": 39666.43479134419,
    "bandwidth_gb_s": 154.94701090368824,
    "speedup_vs_baseline": 2.980952374352873,
    "iterations": 50
  },
  {
    "config": {
      "batch": 16,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.16896000504493713,
    "mean_ms": 0.1697792014479637,
    "std_ms": 0.003523026607141159,
    "throughput_gflops": 50594.740219889434,
    "bandwidth_gb_s": 197.6357039839431,
    "speedup_vs_baseline": 1.8969696210699027,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.32102400064468384,
    "mean_ms": 0.32395327985286715,
    "std_ms": 0.006274847379305996,
    "throughput_gflops": 53031.93470306194,
    "bandwidth_gb_s": 207.1559949338357,
    "speedup_vs_baseline": 0.998405091527137,
    "iterations": 50
  },
  {
    "config": {
      "batch": 64,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.7505919933319092,
    "mean_ms": 0.7662387192249298,
    "std_ms": 0.09331034883800875,
    "throughput_gflops": 44842.08054998285,
    "bandwidth_gb_s": 175.1643771483705,
    "speedup_vs_baseline": 0.4270122777666427,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 128,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.06348799914121628,
    "mean_ms": 0.06477823957800866,
    "std_ms": 0.0025906213413474056,
    "throughput_gflops": 16575.656130743646,
    "bandwidth_gb_s": 258.99462704286947,
    "speedup_vs_baseline": 5.048387113809475,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 256,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.13926400244235992,
    "mean_ms": 0.14002175956964494,
    "std_ms": 0.0020771559152954282,
    "throughput_gflops": 30673.570373637114,
    "bandwidth_gb_s": 239.63726854403996,
    "speedup_vs_baseline": 2.3014705245077254,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.3916800022125244,
    "mean_ms": 0.4054835218191147,
    "std_ms": 0.027658052931137423,
    "throughput_gflops": 42368.84672138145,
    "bandwidth_gb_s": 165.5033075053963,
    "speedup_vs_baseline": 0.8183006406647086,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 1024,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 1.3312000036239624,
    "mean_ms": 1.3276364946365355,
    "std_ms": 0.009870204866106194,
    "throughput_gflops": 51760.762086321825,
    "bandwidth_gb_s": 101.09523844984732,
    "speedup_vs_baseline": 0.24076922766941453,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 2048,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 4.783103942871094,
    "mean_ms": 4.76731393814087,
    "std_ms": 0.04003690247567427,
    "throughput_gflops": 57658.86419705671,
    "bandwidth_gb_s": 56.307484567438195,
    "speedup_vs_baseline": 0.06700920585758241,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 4,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.19865599274635315,
    "mean_ms": 0.200069118142128,
    "std_ms": 0.005088093823195086,
    "throughput_gflops": 42934.83507983355,
    "bandwidth_gb_s": 167.7141995305998,
    "speedup_vs_baseline": 1.6134021043870426,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 8,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.3614720106124878,
    "mean_ms": 0.36331519901752474,
    "std_ms": 0.010028845230685137,
    "throughput_gflops": 47286.40373553796,
    "bandwidth_gb_s": 184.71251459194517,
    "speedup_vs_baseline": 0.8866855173737496,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 12,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.5488640069961548,
    "mean_ms": 0.5426176047325134,
    "std_ms": 0.022824558581554383,
    "throughput_gflops": 47491.64706645185,
    "bandwidth_gb_s": 185.51424635332754,
    "speedup_vs_baseline": 0.5839552105086546,
    "iterations": 50
  },
  {
    "config": {
      "batch": 32,
      "heads": 16,
      "seq": 512,
      "dim": 64,
      "backend": "auto",
      "dtype": "torch.float16"
    },
    "median_ms": 0.8012800216674805,
    "mean_ms": 0.7982489597797394,
    "std_ms": 0.00925240526284262,
    "throughput_gflops": 43043.8874326637,
    "bandwidth_gb_s": 168.14018528384258,
    "speedup_vs_baseline": 0.39999998512264295,
    "iterations": 50
  }
]
```
