"""
Utility Functions for MoE Benchmarking.
"""

import torch
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    latency_ms: float
    tflops: float
    memory_bytes: int
    config: Dict[str, Any]

    @property
    def throughput(self) -> float:
        """Tokens per second."""
        if "total_tokens" in self.config and self.latency_ms > 0:
            return self.config["total_tokens"] / (self.latency_ms / 1000)
        return 0.0


def timer_ms(func, *args, warmup=10, iters=100, cuda=True, **kwargs) -> float:
    """
    Time a function execution.

    Args:
        func: Function to time
        *args: Arguments to pass
        warmup: Warmup iterations
        iters: Timing iterations
        cuda: Whether to synchronize CUDA
        **kwargs: Keyword arguments

    Returns:
        Average latency in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time
    start = time.perf_counter()
    for _ in range(iters):
        _ = func(*args, **kwargs)

    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    return (end - start) / iters * 1000


def calculate_memory_usage(
    m_sizes: List[int],
    k: int,
    n: int,
    dtype_bytes: int = 2,  # float16
    include_inputs: bool = True,
    include_outputs: bool = True,
    include_weights: bool = True,
) -> int:
    """Calculate total memory usage for a grouped GEMM."""
    total_bytes = 0

    if include_inputs:
        total_bytes += sum(m * k for m in m_sizes) * dtype_bytes

    if include_weights:
        total_bytes += len(m_sizes) * n * k * dtype_bytes

    if include_outputs:
        total_bytes += sum(m * n for m in m_sizes) * dtype_bytes

    return total_bytes


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / 1e9,
        "multi_processor_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    val = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if val < 1024:
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{val:.2f} TB"


def format_time(ms: float) -> str:
    """Format milliseconds as human-readable string."""
    if ms < 1:
        return f"{ms * 1000:.2f} µs"
    elif ms < 1000:
        return f"{ms:.3f} ms"
    else:
        return f"{ms / 1000:.3f} s"


def verify_correctness(
    ref_outputs: List[torch.Tensor],
    test_outputs: List[torch.Tensor],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Verify that test outputs match reference outputs."""
    if len(ref_outputs) != len(test_outputs):
        return False

    for ref, test in zip(ref_outputs, test_outputs):
        if ref.shape != test.shape:
            return False
        if ref.shape[0] > 0:  # Skip empty tensors
            if not torch.allclose(ref, test, rtol=rtol, atol=atol):
                print(f"Mismatch: max diff = {(ref - test).abs().max().item()}")
                return False

    return True


def print_benchmark_table(results: List[BenchmarkResult]):
    """Print a formatted table of benchmark results."""
    # Header
    print("\n" + "=" * 100)
    print(
        f"{'Implementation':<25} | {'Latency':<12} | {'TFLOPS':<10} | {'Memory':<12} | {'Speedup':<10}"
    )
    print("=" * 100)

    # Find baseline for speedup calculation
    baseline_latency = results[0].latency_ms if results else 1.0

    # Results
    for r in results:
        speedup = baseline_latency / r.latency_ms if r.latency_ms > 0 else 0
        print(
            f"{r.name:<25} | {format_time(r.latency_ms):<12} | {r.tflops:<10.2f} | {format_bytes(r.memory_bytes):<12} | {speedup:<10.2f}x"
        )

    print("=" * 100 + "\n")
