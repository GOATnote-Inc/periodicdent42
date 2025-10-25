#!/usr/bin/env python
"""Fused MoE benchmark harness with DeepSpeed baseline support."""

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

try:  # Optional DeepSpeed baseline
    from deepspeed.moe.layer import MoE

    _DEEPSPEED_AVAILABLE = True
except Exception:  # pragma: no cover - runtime optional
    _DEEPSPEED_AVAILABLE = False


torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class MoeConfig:
    batch: int
    seq_len: int
    hidden_dim: int
    expert_dim: int
    num_experts: int
    top_k: int
    dtype: torch.dtype
    name: str

    @property
    def tokens(self) -> int:
        return self.batch * self.seq_len


PRESETS: Dict[str, MoeConfig] = {
    "hopper-h100": MoeConfig(
        batch=16,
        seq_len=128,
        hidden_dim=4096,
        expert_dim=4096,
        num_experts=256,
        top_k=8,
        dtype=torch.bfloat16,
        name="Hopper H100 (BF16, 256 experts)",
    ),
    "ampere-a100": MoeConfig(
        batch=8,
        seq_len=128,
        hidden_dim=4096,
        expert_dim=4096,
        num_experts=128,
        top_k=4,
        dtype=torch.float16,
        name="Ampere A100 (FP16, 128 experts)",
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
        return asdict(self)


class BenchmarkRunner:
    def __init__(self, config: MoeConfig, device: str = "cuda", iters: int = 20) -> None:
        if device != "cuda":
            raise BenchmarkError("GPU benchmarks require CUDA device")
        if not torch.cuda.is_available():
            raise BenchmarkError("CUDA is not available")
        self.config = config
        self.device = device
        self.iters = iters
        self._prepare_inputs()
        self._maybe_build_deepspeed()

    def _prepare_inputs(self) -> None:
        cfg = self.config
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tokens = torch.randn(
            cfg.batch,
            cfg.seq_len,
            cfg.hidden_dim,
            device=self.device,
            dtype=cfg.dtype,
        )
        self.expert_weights = torch.randn(
            cfg.num_experts,
            cfg.hidden_dim,
            cfg.expert_dim,
            device=self.device,
            dtype=cfg.dtype,
        )
        routing = torch.randn(
            cfg.batch * cfg.seq_len,
            cfg.num_experts,
            device=self.device,
            dtype=torch.float32,
        )
        self.routing = F.softmax(routing, dim=-1)

    def _maybe_build_deepspeed(self) -> None:
        self._deepspeed_module: Optional[torch.nn.Module] = None
        if not _DEEPSPEED_AVAILABLE:
            return

        cfg = self.config

        class ExpertMLP(torch.nn.Module):
            def __init__(self, hidden: int, expert_dim: int) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(hidden, expert_dim, bias=False)
                self.fc2 = torch.nn.Linear(expert_dim, hidden, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - GPU path
                return self.fc2(F.gelu(self.fc1(x)))

        expert = ExpertMLP(cfg.hidden_dim, cfg.expert_dim)
        moe = MoE(
            hidden_size=cfg.hidden_dim,
            expert=expert,
            num_experts=cfg.num_experts,
            ep_size=1,
            k=cfg.top_k,
            capacity_factor=1.0,
            eval_capacity_factor=1.0,
            min_capacity=4,
            noisy_gate_policy=None,
            drop_tokens=False,
            use_residual=False,
        )
        moe = moe.to(self.device, dtype=cfg.dtype)
        moe.eval()
        self._deepspeed_module = moe

    def _measure(self, fn: Callable[[], torch.Tensor], warmup: int = 6) -> List[float]:
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
            samples.append(start.elapsed_time(end))
        return samples

    def _tokens_per_second(self, latency_ms: float) -> float:
        return (self.config.tokens) / (latency_ms / 1000.0)

    def flashmoe_science(self) -> BenchmarkResult:
        from flashmoe_science import fused_moe

        def call() -> torch.Tensor:
            return fused_moe(
                self.tokens,
                self.expert_weights,
                self.routing,
                top_k=self.config.top_k,
            )

        samples = self._measure(call)
        latency_ms = float(sum(samples) / len(samples))
        std_ms = float(statistics.pstdev(samples))
        peak_bytes = torch.cuda.max_memory_allocated()
        return BenchmarkResult(
            kernel="FlashMoE-Science Fused",
            latency_ms=latency_ms,
            latency_std_ms=std_ms,
            tokens_per_s=self._tokens_per_second(latency_ms),
            peak_memory_mb=peak_bytes / 1e6,
        )

    def torch_moe(self) -> BenchmarkResult:
        cfg = self.config

        def call() -> torch.Tensor:
            tokens_2d = self.tokens.view(cfg.batch * cfg.seq_len, cfg.hidden_dim)
            top_vals, top_idx = self.routing.topk(cfg.top_k, dim=-1)
            experts = self.expert_weights[top_idx]
            token_expanded = tokens_2d.unsqueeze(1).expand(-1, cfg.top_k, -1)
            expert_outputs = torch.einsum("bkh,bkhd->bkd", token_expanded, experts)
            combined = (expert_outputs * top_vals.unsqueeze(-1)).sum(dim=1)
            return combined.view(cfg.batch, cfg.seq_len, cfg.expert_dim)

        samples = self._measure(call)
        latency_ms = float(sum(samples) / len(samples))
        std_ms = float(statistics.pstdev(samples))
        peak_bytes = torch.cuda.max_memory_allocated()
        return BenchmarkResult(
            kernel="PyTorch MoE (einsum)",
            latency_ms=latency_ms,
            latency_std_ms=std_ms,
            tokens_per_s=self._tokens_per_second(latency_ms),
            peak_memory_mb=peak_bytes / 1e6,
            notes="Reference implementation",
        )

    def deepspeed_moe(self) -> BenchmarkResult:
        if self._deepspeed_module is None:
            raise BenchmarkError("DeepSpeed MoE not available")

        def call() -> torch.Tensor:
            output, _ = self._deepspeed_module(self.tokens)
            return output

        samples = self._measure(call)
        latency_ms = float(sum(samples) / len(samples))
        std_ms = float(statistics.pstdev(samples))
        peak_bytes = torch.cuda.max_memory_allocated()
        return BenchmarkResult(
            kernel="DeepSpeed MoE",
            latency_ms=latency_ms,
            latency_std_ms=std_ms,
            tokens_per_s=self._tokens_per_second(latency_ms),
            peak_memory_mb=peak_bytes / 1e6,
            notes="Router-internal",
        )


def _format_markdown(results: List[BenchmarkResult]) -> str:
    header = "| Kernel | Latency (ms) | σ (ms) | Tokens/s | Peak Memory (MB) |"
    sep = "|---|---|---|---|---|"
    rows = [
        f"| {r.kernel} | {r.latency_ms:.2f} | {r.latency_std_ms:.3f} | {r.tokens_per_s/1e3:.1f}k | {r.peak_memory_mb:.1f} |"
        for r in results
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS.keys(), default="hopper-h100")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=["flashmoe", "torch", "deepspeed"],
        help="Select specific baselines (default: run all available)",
    )
    parser.add_argument("--iters", type=int, default=20, help="Measurement iterations")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    config = PRESETS[args.preset]
    runner = BenchmarkRunner(config=config, iters=args.iters)

    baselines = args.baseline or ["torch", "deepspeed", "flashmoe"]
    order: List[str] = []
    for key in baselines:
        if key not in order:
            order.append(key)

    results: List[BenchmarkResult] = []
    for key in order:
        try:
            if key == "torch":
                results.append(runner.torch_moe())
            elif key == "deepspeed":
                results.append(runner.deepspeed_moe())
            elif key == "flashmoe":
                results.append(runner.flashmoe_science())
        except BenchmarkError as exc:
            print(f"Skipping {key}: {exc}")

    if not results:
        raise SystemExit("No benchmarks executed. Install the required baselines.")

    results.sort(key=lambda r: r.latency_ms)
    print(f"\n=== Fused MoE Benchmarks :: {config.name} ===")
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
