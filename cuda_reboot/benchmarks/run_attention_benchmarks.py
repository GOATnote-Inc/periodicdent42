#!/usr/bin/env python
"""Attention benchmark harness with SOTA baselines."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

try:  # Optional baseline
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_func

    _FLASH_ATTN_AVAILABLE = True
except Exception:  # pragma: no cover - runtime optional
    _FLASH_ATTN_AVAILABLE = False


torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class BenchmarkConfig:
    batch: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype
    causal: bool
    name: str

    @property
    def tokens(self) -> int:
        return self.batch * self.seq_len


PRESETS: Dict[str, BenchmarkConfig] = {
    "hopper-h100": BenchmarkConfig(
        batch=4,
        num_heads=8,
        seq_len=2048,
        head_dim=64,
        dtype=torch.bfloat16,
        causal=True,
        name="Hopper H100 (BF16, 2k context)",
    ),
    "ampere-a100": BenchmarkConfig(
        batch=4,
        num_heads=16,
        seq_len=1024,
        head_dim=64,
        dtype=torch.float16,
        causal=True,
        name="Ampere A100 (FP16, 1k context)",
    ),
}


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot be executed."""


@dataclass
class BenchmarkResult:
    kernel: str
    latency_ms: float
    latency_std_ms: float
    tokens_per_s: float
    peak_memory_mb: float
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        return data


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, device: str = "cuda", iters: int = 20) -> None:
        if device != "cuda":
            raise BenchmarkError("GPU benchmarks require CUDA device")
        if not torch.cuda.is_available():
            raise BenchmarkError("CUDA is not available")
        self.config = config
        self.device = device
        self.iters = iters
        self._prepare_inputs()

    def _prepare_inputs(self) -> None:
        cfg = self.config
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.q = torch.randn(
            cfg.batch,
            cfg.num_heads,
            cfg.seq_len,
            cfg.head_dim,
            device=self.device,
            dtype=cfg.dtype,
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)

    def _measure(self, fn: Callable[[], torch.Tensor], warmup: int = 8) -> List[float]:
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples: List[float] = []
        for _ in range(self.iters):
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            samples.append(elapsed)
        return samples

    def _tokens_per_second(self, latency_ms: float) -> float:
        tokens = self.config.tokens * self.config.num_heads
        return tokens / (latency_ms / 1000.0)

    def flashmoe_science(self) -> BenchmarkResult:
        from flashmoe_science import flash_attention_science

        def call() -> torch.Tensor:
            return flash_attention_science(self.q, self.k, self.v, causal=self.config.causal)

        samples = self._measure(call)
        latency_ms = float(sum(samples) / len(samples))
        std_ms = float(statistics.pstdev(samples))
        peak_bytes = torch.cuda.max_memory_allocated()
        return BenchmarkResult(
            kernel="FlashAttention-Science",
            latency_ms=latency_ms,
            latency_std_ms=std_ms,
            tokens_per_s=self._tokens_per_second(latency_ms),
            peak_memory_mb=peak_bytes / 1e6,
        )

    def pytorch_sdpa(self) -> BenchmarkResult:
        def call() -> torch.Tensor:
            return F.scaled_dot_product_attention(
                self.q,
                self.k,
                self.v,
                is_causal=self.config.causal,
            )

        samples = self._measure(call)
        latency_ms = float(sum(samples) / len(samples))
        std_ms = float(statistics.pstdev(samples))
        peak_bytes = torch.cuda.max_memory_allocated()
        return BenchmarkResult(
            kernel="PyTorch SDPA (cuDNN)",
            latency_ms=latency_ms,
            latency_std_ms=std_ms,
            tokens_per_s=self._tokens_per_second(latency_ms),
            peak_memory_mb=peak_bytes / 1e6,
        )

    def flash_attn(self) -> BenchmarkResult:
        if not _FLASH_ATTN_AVAILABLE:
            raise BenchmarkError("flash-attn package not available")

        qkv = torch.stack((self.q, self.k, self.v), dim=2)  # [B, H, 3, L, D]
        qkv = qkv.transpose(1, 3).contiguous()  # [B, L, 3, H, D]
        # Flatten batch into varlen structure
        batch = self.config.batch
        seqlen = self.config.seq_len
        qkv = qkv.view(batch, seqlen, 3, self.config.num_heads, self.config.head_dim)
        qkv = qkv.reshape(batch * seqlen, 3, self.config.num_heads, self.config.head_dim)
        cu_seqlens = torch.arange(0, (batch + 1) * seqlen, seqlen, device=self.device, dtype=torch.int32)

        def call() -> torch.Tensor:
            return flash_attn_func(
                qkv,
                cu_seqlens,
                self.config.seq_len,
                softmax_scale=None,
                causal=self.config.causal,
            )

        samples = self._measure(call)
        latency_ms = float(sum(samples) / len(samples))
        std_ms = float(statistics.pstdev(samples))
        peak_bytes = torch.cuda.max_memory_allocated()
        return BenchmarkResult(
            kernel="flash-attn 2.3.3",
            latency_ms=latency_ms,
            latency_std_ms=std_ms,
            tokens_per_s=self._tokens_per_second(latency_ms),
            peak_memory_mb=peak_bytes / 1e6,
            notes="Varlen QKV path",
        )


def _format_markdown(results: List[BenchmarkResult]) -> str:
    header = "| Kernel | Latency (ms) | σ (ms) | Tokens/s | Peak Memory (MB) |"
    sep = "|---|---|---|---|---|"
    rows = [
        f"| {r.kernel} | {r.latency_ms:.2f} | {r.latency_std_ms:.3f} | {r.tokens_per_s/1e6:.2f}M | {r.peak_memory_mb:.1f} |"
        for r in results
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS.keys(), default="hopper-h100")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=["flashmoe", "pytorch", "flash-attn"],
        help="Select specific baselines (default: run all available)",
    )
    parser.add_argument("--iters", type=int, default=20, help="Measurement iterations per kernel")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON results")
    args = parser.parse_args()

    config = PRESETS[args.preset]
    runner = BenchmarkRunner(config=config, iters=args.iters)

    baselines = args.baseline or ["pytorch", "flash-attn", "flashmoe"]
    order = []
    for key in baselines:
        if key not in order:
            order.append(key)

    results: List[BenchmarkResult] = []
    for key in order:
        try:
            if key == "pytorch":
                results.append(runner.pytorch_sdpa())
            elif key == "flash-attn":
                results.append(runner.flash_attn())
            elif key == "flashmoe":
                results.append(runner.flashmoe_science())
        except BenchmarkError as exc:
            print(f"Skipping {key}: {exc}")

    if not results:
        raise SystemExit("No benchmarks executed. Ensure required baselines are installed.")

    results.sort(key=lambda r: r.latency_ms)
    print(f"\n=== FlashAttention Benchmarks :: {config.name} ===")
    print(_format_markdown(results))

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "preset": config.name,
        "config": asdict(config),
        "results": [r.to_dict() for r in results],
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "sm": torch.cuda.get_device_capability(0),
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
