#!/usr/bin/env python3
"""
MoE Grouped GEMM Benchmark.

Compares different implementations for Mixture-of-Experts inference:
1. Naive (for-loop, separate kernel launches) - BASELINE
2. torch.bmm (padded batched matmul) - REFERENCE
3. Triton Grouped GEMM (fused kernel) - OPTIMIZED
4. INT4 Quantized MoE (memory-optimized) - QUANTIZED

Target: Show 2-4x speedup with fused kernels + 8x memory savings with INT4

Usage:
    python benchmark/run_moe_benchmark.py --config mixtral --batch 16

For RunPod (RTX 4090):
    1. Start pod with image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
    2. pip install triton matplotlib
    3. python benchmark/run_moe_benchmark.py
"""

import argparse
import torch
import time
import sys
import os
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.moe_grouped_gemm.config import (
    MoEConfig,
    MIXTRAL_8x7B,
    get_config_by_name,
    BenchmarkConfig,
    QUICK_TEST_CONFIG,
    MIXTRAL_BENCHMARK_CONFIGS,
)
from benchmark.moe_grouped_gemm.routing import (
    simulate_routing,
    get_expert_sizes_for_benchmark,
)
from benchmark.moe_grouped_gemm.naive_grouped_gemm import (
    naive_grouped_gemm_forward,
    benchmark_naive_grouped_gemm,
)
from benchmark.moe_grouped_gemm.grouped_gemm_torch import (
    bmm_grouped_gemm_forward,
    benchmark_bmm_grouped_gemm,
)
from benchmark.moe_grouped_gemm.utils import (
    BenchmarkResult,
    format_bytes,
    format_time,
    get_gpu_info,
    print_benchmark_table,
    verify_correctness,
)

# Try to import Triton and INT4 modules (may not be available)
try:
    from benchmark.moe_grouped_gemm.grouped_gemm_triton import (
        triton_grouped_gemm_forward,
        benchmark_triton_grouped_gemm,
        TRITON_AVAILABLE,
    )
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton grouped GEMM not available")

try:
    from benchmark.moe_grouped_gemm.moe_int4_module import benchmark_int4_moe

    INT4_AVAILABLE = True
except ImportError as e:
    INT4_AVAILABLE = False
    print(f"Warning: INT4 MoE not available: {e}")


def run_single_benchmark(
    moe_config: MoEConfig,
    batch_size: int,
    seq_len: int,
    warmup_iters: int = 50,
    bench_iters: int = 100,
    distribution: str = "skewed",
    verify: bool = True,
) -> List[BenchmarkResult]:
    """
    Run benchmark comparing all implementations.

    Args:
        moe_config: MoE model configuration
        batch_size: Batch size
        seq_len: Sequence length
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations
        distribution: Token distribution ("uniform", "skewed", "random")
        verify: Whether to verify correctness

    Returns:
        List of BenchmarkResult for each implementation
    """
    results = []

    total_tokens = batch_size * seq_len * moe_config.top_k

    # Get expert sizes based on routing
    m_sizes, k, n = get_expert_sizes_for_benchmark(
        num_tokens=batch_size * seq_len,
        num_experts=moe_config.num_experts,
        hidden_dim=moe_config.hidden_dim,
        ffn_dim=moe_config.ffn_dim,
        distribution=distribution,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n{'=' * 80}")
    print(f"Config: {moe_config.name}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Total tokens: {batch_size * seq_len}")
    print(f"Expert sizes (tokens): {m_sizes}")
    print(f"K={k}, N={n}")
    print(f"{'=' * 80}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Naive baseline
    print("Running Naive (for-loop) baseline...")
    naive_latency, naive_tflops = benchmark_naive_grouped_gemm(
        m_sizes, k, n, device=device, warmup_iters=warmup_iters, bench_iters=bench_iters
    )
    naive_memory = moe_config.num_experts * n * k * 2  # BF16

    results.append(
        BenchmarkResult(
            name="Naive (for-loop)",
            latency_ms=naive_latency,
            tflops=naive_tflops,
            memory_bytes=naive_memory,
            config={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "total_tokens": total_tokens,
            },
        )
    )

    # 2. torch.bmm reference
    print("Running torch.bmm (padded) reference...")
    bmm_latency, bmm_tflops = benchmark_bmm_grouped_gemm(
        m_sizes, k, n, device=device, warmup_iters=warmup_iters, bench_iters=bench_iters
    )

    results.append(
        BenchmarkResult(
            name="torch.bmm (padded)",
            latency_ms=bmm_latency,
            tflops=bmm_tflops,
            memory_bytes=naive_memory,  # Same memory
            config={
                "batch_size": batch_size,
                "seq_len": seq_len,
                "total_tokens": total_tokens,
            },
        )
    )

    # 3. Triton Grouped GEMM (if available)
    if TRITON_AVAILABLE:
        print("Running Triton Grouped GEMM...")
        triton_latency, triton_tflops = benchmark_triton_grouped_gemm(
            m_sizes,
            k,
            n,
            device=device,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )

        results.append(
            BenchmarkResult(
                name="Triton Grouped GEMM",
                latency_ms=triton_latency,
                tflops=triton_tflops,
                memory_bytes=naive_memory,
                config={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "total_tokens": total_tokens,
                },
            )
        )

    # 4. INT4 Quantized MoE (if available)
    if INT4_AVAILABLE:
        print("Running INT4 Quantized MoE...")
        int4_latency, int4_tflops, int4_memory = benchmark_int4_moe(
            m_sizes,
            k,
            n,
            moe_config.num_experts,
            device=device,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )

        results.append(
            BenchmarkResult(
                name="INT4 Quantized MoE",
                latency_ms=int4_latency,
                tflops=int4_tflops,
                memory_bytes=int4_memory,
                config={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "total_tokens": total_tokens,
                },
            )
        )

    # Print results
    print_benchmark_table(results)

    return results


def run_full_benchmark(
    configs: List[BenchmarkConfig],
    output_dir: str = "benchmark/results",
) -> Dict[str, List[BenchmarkResult]]:
    """Run full benchmark across multiple configurations."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for config in configs:
        key = f"{config.moe_config.name}_b{config.batch_size}_s{config.seq_len}"
        results = run_single_benchmark(
            moe_config=config.moe_config,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            warmup_iters=config.warmup_iters,
            bench_iters=config.bench_iters,
        )
        all_results[key] = results

    # Generate plots
    try:
        generate_plots(all_results, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    return all_results


def generate_plots(all_results: Dict[str, List[BenchmarkResult]], output_dir: str):
    """Generate comparison plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data for plotting
    configs = list(all_results.keys())

    # Latency comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Latency
    ax1 = axes[0]
    x = np.arange(len(configs))
    width = 0.2

    implementations = [
        "Naive (for-loop)",
        "torch.bmm (padded)",
        "Triton Grouped GEMM",
        "INT4 Quantized MoE",
    ]
    colors = ["#4285f4", "#ea4335", "#34a853", "#fbbc04"]

    for i, (impl, color) in enumerate(zip(implementations, colors)):
        latencies = []
        for config in configs:
            results = all_results[config]
            matching = [r for r in results if r.name == impl]
            if matching:
                latencies.append(matching[0].latency_ms)
            else:
                latencies.append(0)

        ax1.bar(x + i * width, latencies, width, label=impl, color=color)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("MoE Expert Computation Latency")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Speedup
    ax2 = axes[1]

    for i, (impl, color) in enumerate(zip(implementations[1:], colors[1:])):
        speedups = []
        for config in configs:
            results = all_results[config]
            baseline = [r for r in results if r.name == "Naive (for-loop)"]
            optimized = [r for r in results if r.name == impl]
            if baseline and optimized:
                speedup = baseline[0].latency_ms / optimized[0].latency_ms
                speedups.append(speedup)
            else:
                speedups.append(0)

        ax2.bar(x + i * width, speedups, width, label=f"{impl} speedup", color=color)

    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Speedup vs Naive (x)")
    ax2.set_title("Speedup Comparison")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "moe_benchmark_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="MoE Grouped GEMM Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="mixtral",
        help="Model config (mixtral, deepseek, glm5, debug)",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument(
        "--output", type=str, default="benchmark/results", help="Output directory"
    )

    args = parser.parse_args()

    # Print GPU info
    gpu_info = get_gpu_info()
    print("\n" + "=" * 80)
    print("MoE Grouped GEMM Benchmark")
    print("=" * 80)
    if gpu_info["available"]:
        print(f"GPU: {gpu_info['name']}")
        print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"SMs: {gpu_info['multi_processor_count']}")
        print(f"Compute: {gpu_info['compute_capability']}")
    else:
        print("No GPU available, running on CPU (will be slow)")
    print("=" * 80)

    if args.full:
        # Run full benchmark suite
        results = run_full_benchmark(MIXTRAL_BENCHMARK_CONFIGS, args.output)
    else:
        # Run single benchmark
        config = get_config_by_name(args.config)
        run_single_benchmark(
            moe_config=config,
            batch_size=args.batch,
            seq_len=args.seq,
            warmup_iters=args.warmup,
            bench_iters=args.iters,
        )


if __name__ == "__main__":
    main()
