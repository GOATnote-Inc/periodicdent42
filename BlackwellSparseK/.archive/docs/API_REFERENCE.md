# BlackwellSparseK API Reference

**Version**: 0.1.0  
**Last Updated**: 2025-10-30

---

## Core API

### `blackwell_sparsek.attention_forward()`

Compute scaled dot-product attention using BlackwellSparseK kernels.

```python
def attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor
```

**Parameters:**

- `Q` (torch.Tensor): Query tensor of shape `[B, H, S, D]`
  - Must be FP16 (`torch.float16`)
  - Must be on CUDA device
  - `B`: batch size, `H`: number of heads, `S`: sequence length, `D`: head dimension

- `K` (torch.Tensor): Key tensor of shape `[B, H, S, D]`
  - Same requirements as Q

- `V` (torch.Tensor): Value tensor of shape `[B, H, S, D]`
  - Same requirements as Q

- `scale` (float, optional): Softmax scale factor
  - Default: `1.0 / sqrt(D)`
  - Used as: `softmax(Q @ K.T * scale) @ V`

**Returns:**

- `torch.Tensor`: Output tensor of shape `[B, H, S, D]` in FP16

**Raises:**

- `ValueError`: If tensors are not CUDA, not FP16, or have incompatible shapes
- `RuntimeError`: If GPU architecture is not sm_90a or sm_100

**Example:**

```python
import torch
from blackwell_sparsek import attention_forward

Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')

output = attention_forward(Q, K, V)
```

**Constraints:**

- Head dimension `D` must be 64 or 128
- Sequence length `S` can be arbitrary (no power-of-2 requirement)
- Batch size `B` and num heads `H` are arbitrary

---

## Utilities

### `blackwell_sparsek.utils.benchmark_latency()`

Benchmark kernel latency with CUDA events.

```python
def benchmark_latency(
    kernel_fn: Callable,
    *args,
    num_warmup: int = 10,
    num_iters: int = 100,
    **kwargs
) -> Dict[str, float]
```

**Parameters:**

- `kernel_fn`: Function to benchmark (e.g., `attention_forward`)
- `*args`: Positional arguments to pass to kernel
- `num_warmup`: Number of warmup iterations (default: 10)
- `num_iters`: Number of measurement iterations (default: 100)
- `**kwargs`: Keyword arguments to pass to kernel

**Returns:**

Dictionary with timing statistics (all in microseconds):
```python
{
    'mean_us': float,     # Mean latency
    'median_us': float,   # Median latency
    'min_us': float,      # Minimum latency
    'max_us': float,      # Maximum latency
    'std_us': float,      # Standard deviation
    'num_iters': int      # Number of iterations
}
```

**Example:**

```python
from blackwell_sparsek.utils import benchmark_latency

stats = benchmark_latency(attention_forward, Q, K, V, num_iters=100)
print(f"Latency: {stats['median_us']:.2f} μs")
```

---

### `blackwell_sparsek.utils.validate_correctness()`

Validate kernel output against reference.

```python
def validate_correctness(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 2e-3
) -> Tuple[bool, Dict[str, Any]]
```

**Parameters:**

- `output`: Kernel output tensor
- `reference`: Reference tensor (e.g., from PyTorch SDPA)
- `rtol`: Relative tolerance (default: 1e-3 for FP16)
- `atol`: Absolute tolerance (default: 2e-3 for FP16)

**Returns:**

- `is_correct` (bool): True if outputs match within tolerances
- `metrics` (dict): Detailed comparison metrics

**Metrics Dictionary:**
```python
{
    'max_diff': float,           # Maximum absolute difference
    'mean_diff': float,          # Mean absolute difference
    'median_diff': float,        # Median absolute difference
    'num_elements': int,         # Total number of elements
    'num_mismatches': int,       # Elements outside tolerance (if not correct)
    'mismatch_rate': float,      # Percentage of mismatched elements
    'worst_diff_location': tuple,  # Index of worst mismatch
    'worst_diff_output': float,    # Output value at worst location
    'worst_diff_reference': float  # Reference value at worst location
}
```

**Example:**

```python
from blackwell_sparsek.utils import validate_correctness

ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
out = attention_forward(Q, K, V)

is_correct, metrics = validate_correctness(out, ref)
if not is_correct:
    print(f"❌ Validation failed! Max diff: {metrics['max_diff']:.6f}")
else:
    print(f"✅ Validation passed! Max diff: {metrics['max_diff']:.6f}")
```

---

### `blackwell_sparsek.utils.compare_to_sdpa()`

Compare kernel to PyTorch SDPA baseline.

```python
def compare_to_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kernel_fn: Callable,
    scale: Optional[float] = None,
    rtol: float = 1e-3,
    atol: float = 2e-3
) -> Dict[str, Any]
```

**Returns:**

```python
{
    'is_correct': bool,                 # Correctness result
    'correctness_metrics': dict,        # Detailed metrics from validate_correctness
    'sdpa_time_us': float,              # PyTorch SDPA latency (μs)
    'kernel_time_us': float,            # Custom kernel latency (μs)
    'speedup': float,                   # Speedup factor (sdpa_time / kernel_time)
    'input_shape': dict                 # Input dimensions {B, H, S, D}
}
```

**Example:**

```python
from blackwell_sparsek.utils import compare_to_sdpa, print_comparison_summary

comparison = compare_to_sdpa(Q, K, V, attention_forward)
print_comparison_summary(comparison)
```

---

## Configuration

### `blackwell_sparsek.core.Config`

Configuration class for kernel behavior.

```python
@dataclass
class Config:
    # Architecture
    cuda_arch: str = "90a"              # "90a" (H100) or "100" (Blackwell)
    auto_detect_arch: bool = True       # Auto-detect GPU architecture
    
    # Kernel parameters
    block_m: int = 64                   # Query tile size
    block_n: int = 64                   # KV tile size
    num_stages: int = 2                 # Pipeline stages
    num_warps: int = 4                  # Warps per thread block
    
    # Build options
    use_fast_math: bool = True          # Enable --use_fast_math
    enable_profiling: bool = False      # Enable -lineinfo for NCU
    debug_mode: bool = False            # Enable debug assertions
    
    # Performance tuning
    persistent_kernel: bool = True      # Use persistent thread blocks
    warp_specialized: bool = True       # Enable warp specialization
    use_tma: bool = True                # Use TMA (sm_100) / cp.async.bulk (sm_90a)
    
    # Paths
    cutlass_path: Optional[str] = None  # CUTLASS installation path
    build_dir: Optional[str] = None     # Build cache directory
```

**Usage:**

```python
from blackwell_sparsek.core import Config, set_default_config

# Create custom config
config = Config(
    block_m=128,
    block_n=128,
    enable_profiling=True
)

# Set as default
set_default_config(config)
```

---

### `blackwell_sparsek.core.get_default_config()`

Get the global default configuration.

```python
def get_default_config() -> Config
```

**Example:**

```python
from blackwell_sparsek.core import get_default_config

config = get_default_config()
print(f"CUDA Architecture: sm_{config.cuda_arch}")
print(f"Block M: {config.block_m}")
```

---

### `blackwell_sparsek.core.get_build_info()`

Get build environment information.

```python
def get_build_info() -> Dict[str, Any]
```

**Returns:**

```python
{
    'torch_version': str,           # PyTorch version
    'cuda_available': bool,         # CUDA availability
    'cuda_version': str,            # CUDA version
    'cudnn_version': int,           # cuDNN version
    'device_name': str,             # GPU name (if available)
    'device_capability': tuple,     # Compute capability (major, minor)
    'compute_arch': str             # Architecture string (e.g., "sm_90")
}
```

**Example:**

```python
from blackwell_sparsek.core import get_build_info

info = get_build_info()
print(f"GPU: {info['device_name']}")
print(f"Compute: {info['compute_arch']}")
```

---

## xFormers Backend

### `blackwell_sparsek.backends.SparseKAttention`

xFormers-compatible attention module.

```python
class SparseKAttention(xformers.components.attention.Attention):
    def __init__(self, *args, **kwargs)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[AttentionBias] = None,
        **kwargs
    ) -> torch.Tensor
```

**Parameters:**

- `q, k, v`: Tensors in xFormers layout `[B, S, H, D]`
- `att_mask`: Optional `AttentionBias` (BlockDiagonal, LowerTriangular, etc.)

**Note**: If `att_mask` is provided, falls back to PyTorch SDPA (custom kernel doesn't support masks yet).

**Example:**

```python
from blackwell_sparsek.backends import SparseKAttention

attention = SparseKAttention()

# xFormers layout: [B, S, H, D]
q = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')

output = attention(q, k, v)
```

---

## vLLM Backend

### `blackwell_sparsek.backends.SparseKBackend`

vLLM V1 attention backend.

```python
class SparseKBackend(vllm.attention.backends.abstract.AttentionBackend):
    @staticmethod
    def get_name() -> str
    
    @staticmethod
    def get_supported_head_sizes() -> List[int]
    
    @staticmethod
    def get_kv_cache_shape(...) -> Tuple[int, ...]
    
    def forward(...) -> torch.Tensor
```

**Backend Name**: `"SPARSEK_XFORMERS"`

**Supported Head Sizes**: `[64, 128]`

**Usage:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --attention-backend SPARSEK_XFORMERS
```

**Registration:**

```python
from blackwell_sparsek.backends import register_vllm_backend

register_vllm_backend()  # Auto-called on import
```

---

## Environment Variables

BlackwellSparseK respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_HOME` | CUDA toolkit path | `/usr/local/cuda-13.0` |
| `CUTLASS_PATH` | CUTLASS installation path | `/opt/cutlass` |
| `BSK_CUDA_ARCH` | Target architecture | `"90a"` (auto-detect if GPU available) |
| `BSK_AUTO_DETECT` | Enable auto-detection | `"1"` (enabled) |
| `BSK_FAST_MATH` | Enable fast math | `"1"` (enabled) |
| `BSK_PROFILE` | Enable profiling | `"0"` (disabled) |
| `BSK_DEBUG` | Enable debug mode | `"0"` (disabled) |
| `BSK_BUILD_DIR` | Build cache directory | `~/.cache/blackwell_sparsek/build` |

**Example:**

```bash
export BSK_DEBUG=1
export BSK_PROFILE=1
python my_script.py
```

---

## Error Handling

### Common Exceptions

**ValueError**: Input validation failed
```python
try:
    output = attention_forward(Q_cpu, K, V)  # Q not on CUDA
except ValueError as e:
    print(f"Input error: {e}")
```

**RuntimeError**: CUDA kernel execution failed
```python
try:
    output = attention_forward(Q, K, V)
except RuntimeError as e:
    print(f"Kernel error: {e}")
```

**ImportError**: CUDA extension not built
```python
try:
    from blackwell_sparsek import attention_forward
except ImportError as e:
    print(f"Build error: {e}")
    print("Run: pip install -e .")
```

---

## Version Information

### `blackwell_sparsek.__version__`

Package version string.

```python
import blackwell_sparsek
print(blackwell_sparsek.__version__)  # "0.1.0"
```

### `blackwell_sparsek._C.version()`

CUDA extension version.

```python
from blackwell_sparsek import _C
print(_C.version())  # "0.1.0"
```

### `blackwell_sparsek._C.cuda_arch()`

Current GPU architecture.

```python
from blackwell_sparsek import _C
print(_C.cuda_arch())  # "sm_90"
```

### `blackwell_sparsek._C.supported_archs()`

Supported architectures.

```python
from blackwell_sparsek import _C
print(_C.supported_archs())  # ["sm_90a", "sm_100"]
```

---

## Performance Tips

1. **Use FP16**: Kernel requires FP16 inputs (torch.float16)
2. **Contiguous Tensors**: Ensure inputs are contiguous for optimal performance
3. **Batch Operations**: Larger batch sizes amortize overhead
4. **Head Dimension**: D=64 is optimal, D=128 supported
5. **Warmup**: Run 10+ iterations before timing
6. **CUDA Events**: Use torch.cuda.synchronize() for accurate timing

**Example:**

```python
# Optimal usage
Q = torch.randn(B, H, S, 64, dtype=torch.float16, device='cuda').contiguous()
K = torch.randn(B, H, S, 64, dtype=torch.float16, device='cuda').contiguous()
V = torch.randn(B, H, S, 64, dtype=torch.float16, device='cuda').contiguous()

# Warmup
for _ in range(10):
    _ = attention_forward(Q, K, V)

# Measure
torch.cuda.synchronize()
start = time.perf_counter()
output = attention_forward(Q, K, V)
torch.cuda.synchronize()
latency = (time.perf_counter() - start) * 1e6
print(f"Latency: {latency:.2f} μs")
```

---

## See Also

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical deep dive
- **[MIGRATION_FROM_FLASHCORE.md](MIGRATION_FROM_FLASHCORE.md)** - Upgrade guide

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-30

