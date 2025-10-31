# Getting Started with FlashCore

Welcome to FlashCore! This guide will help you get up and running in minutes.

---

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA support
- **Compute Capability**: 8.0+ (Ampere or newer recommended)
- **Memory**: 8GB+ VRAM recommended

### Software Requirements
- **Python**: 3.8+
- **CUDA Toolkit**: 12.0+
- **PyTorch**: 2.0+
- **Triton**: 2.1+

---

## Installation

### Step 1: Install PyTorch

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Install Triton

```bash
pip install triton
```

### Step 3: Clone FlashCore

```bash
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42
```

### Step 4: Verify Installation

```python
python3 << EOF
import torch
from flashcore.fast.attention_production import attention

# Create test tensors
q = torch.randn(8, 8, 256, 64, device='cuda', dtype=torch.float16)

# Run attention
output = attention(q, q, q)

print(f"✅ FlashCore working! Output shape: {output.shape}")
EOF
```

---

## Quick Start

### Basic Usage

```python
import torch
from flashcore.fast.attention_production import attention

# Create input tensors [Batch, Heads, SeqLen, HeadDim]
batch_size = 16
num_heads = 8
seq_len = 512
head_dim = 64

q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                device='cuda', dtype=torch.float16)
k = q.clone()
v = q.clone()

# Run optimized attention (auto-selects optimal block sizes)
output = attention(q, k, v)

print(f"Output shape: {output.shape}")
# Output shape: torch.Size([16, 8, 512, 64])
```

### Performance Comparison

```python
import torch
import time

# Setup
q = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)
k, v = q.clone(), q.clone()

# Warmup
for _ in range(100):
    _ = attention(q, k, v)
torch.cuda.synchronize()

# Benchmark FlashCore
times = []
for _ in range(1000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # μs

flashcore_us = sorted(times)[len(times)//2] / batch_size

print(f"FlashCore: {flashcore_us:.2f} μs per sequence")
# On H100: ~3.1 μs per sequence
```

---

## Configuration

### Auto-Tuning (Recommended)

```python
# Automatically selects optimal block sizes based on (S, B)
output = attention(q, k, v)
```

### Manual Block Sizes

```python
# Specify block sizes manually for advanced tuning
output = attention(q, k, v, block_m=64, block_n=128)
```

### Optimal Configurations (H100)

| Seq Length | Batch | Optimal Config | Expected Latency |
|------------|-------|----------------|------------------|
| 128        | ≥16   | 64×128         | 1.36 μs/seq      |
| 256        | all   | 64×64          | 1.78 μs/seq      |
| 512        | ≤8    | 64×128         | 4.34 μs/seq      |
| 512        | ≥16   | 64×64          | 3.15 μs/seq      |

---

## Common Use Cases

### 1. Real-Time Inference

```python
from flashcore.fast.attention_production import attention

def process_sequence(queries, keys, values):
    """Process single sequence with minimal latency."""
    # Ensure batch size ≥ 8 for optimal performance
    return attention(queries, keys, values)
```

### 2. Batch Processing

```python
# Process multiple sequences together (recommended)
batch_size = 16  # ≥8 for sub-5μs performance
q = torch.randn(batch_size, 8, 512, 64, device='cuda', dtype=torch.float16)
outputs = attention(q, q, q)
```

### 3. Production Deployment

```python
import torch
from flashcore.fast.attention_production import attention

class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        B, S, D = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # FlashCore attention
        out = attention(q, k, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, check:
# 1. CUDA Toolkit installed
# 2. PyTorch CUDA version matches CUDA Toolkit
# 3. NVIDIA drivers up to date
```

### Out of Memory

```python
# Reduce batch size or sequence length
batch_size = 8  # Instead of 32
seq_len = 256   # Instead of 512
```

### Performance Not Matching Expectations

```python
# Ensure:
# 1. Batch size ≥ 8 (amortizes kernel launch overhead)
# 2. Using correct dtype (torch.float16)
# 3. Tensors are on GPU
# 4. Proper warmup before benchmarking (100+ iterations)
```

---

## Next Steps

- **[API Reference](../api/README.md)** - Complete API documentation
- **[Performance Guide](../guides/performance.md)** - Optimization tips
- **[Examples](../../examples/)** - Jupyter notebooks and sample code
- **[Research](../research/)** - Technical deep-dive and validation

---

## Support

- **Documentation**: [docs/](../)
- **Issues**: [GitHub Issues](https://github.com/GOATnote-Inc/periodicdent42/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GOATnote-Inc/periodicdent42/discussions)

---

<p align="center">
  <strong>Ready to achieve sub-5μs attention? 🚀</strong>
</p>

