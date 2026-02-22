"""
Benchmark: FP32 nn.Linear vs INT4 QuantizedLinear (fused CUDA kernel).

=== WHAT THIS SCRIPT DOES ===

1. Tests several matrix sizes typical of large language models (LLMs):
   - (1024, 1024)   — small test
   - (4096, 4096)   — typical hidden dimension in LLaMA-7B
   - (4096, 11008)  — typical FFN (feed-forward network) in LLaMA-7B

2. For each size, measures:
   - Latency (ms): How long a single forward pass takes
   - Memory: How many bytes the weights consume
   - Speedup: How much faster INT4 is vs FP32

3. Computes a roofline analysis (GPU only) showing:
   - Arithmetic Intensity: ratio of compute work to memory reads
   - Achieved Bandwidth: how fast we're reading data from GPU memory
   - Achieved GFLOPS: how many billion floating-point operations per second

4. Generates a matplotlib bar chart comparing the two approaches.

=== HOW TO RUN ===
    python benchmark/run_benchmark.py

=== ROOFLINE ANALYSIS EXPLAINED ===

The "roofline model" helps you understand whether your kernel is limited by:
  - Memory bandwidth (waiting for data to arrive from RAM) — "memory-bound"
  - Compute throughput (GPUs can't do math fast enough) — "compute-bound"

Arithmetic Intensity (AI) = FLOPs / Bytes Read
  - Low AI (< ~10 for modern GPUs) → memory-bound (our case for single vectors)
  - High AI → compute-bound

For matrix-vector products with small batch sizes, the kernel is almost always
memory-bound because each weight value is only used once (low reuse).
The INT4 kernel wins here because it reads 8x fewer weight bytes.
"""

import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python import QuantizedLinear

# ── Configuration ──────────────────────────────────────────────────────────

# Matrix sizes to benchmark: (input_dim, output_dim)
# These represent weight matrix dimensions in popular LLM architectures.
SIZES = [
    (1024, 1024),    # small test case
    (4096, 4096),    # hidden → hidden in LLaMA-7B
    (4096, 11008),   # hidden → FFN intermediate in LLaMA-7B
]

# WARMUP_ITERS: We run the kernel several times before measuring to ensure the
# GPU has "warmed up" (caches are primed, clock speeds are stable).
# Without warmup, the first few runs are slower and would skew our measurements.
WARMUP_ITERS = 50

# BENCH_ITERS: We average over many runs to get a stable timing measurement.
# More iterations = more accurate average, but takes longer.
BENCH_ITERS = 200

# Use GPU if available, otherwise fall back to CPU (slower, no CUDA kernel)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark_linear(linear_layer, x, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Time a linear layer's forward pass, returning average latency in milliseconds.

    On GPU, we use CUDA events for precise timing (they measure GPU time directly,
    not affected by CPU overhead). On CPU, we use Python's perf_counter.

    Why torch.no_grad()?
      During inference, we don't need PyTorch to track gradients (for backprop).
      Disabling gradient tracking saves memory and time.

    Why torch.cuda.synchronize()?
      GPU operations are asynchronous — the CPU tells the GPU to do work and
      immediately continues. synchronize() blocks the CPU until the GPU finishes,
      ensuring our timing measurements are accurate.
    """
    with torch.no_grad():
        # Warmup: run several times to stabilize GPU clocks and fill caches
        for _ in range(warmup):
            linear_layer(x)

    if x.is_cuda:
        # CUDA events are the most accurate way to time GPU operations.
        # They are recorded in the GPU's command stream and measure actual GPU time.
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()                         # record start timestamp on GPU
        for _ in range(iters):
            linear_layer(x)
        end.record()                           # record end timestamp on GPU
        torch.cuda.synchronize()               # wait for GPU to finish
        elapsed_ms = start.elapsed_time(end) / iters  # average ms per iteration
    else:
        # CPU timing: simple wall-clock measurement
        t0 = time.perf_counter()
        for _ in range(iters):
            linear_layer(x)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) / iters * 1000  # convert seconds to milliseconds

    return elapsed_ms


def measure_memory(fn, x):
    """Measure peak GPU memory allocated during a forward pass."""
    if not x.is_cuda:
        return 0
    torch.cuda.reset_peak_memory_stats()  # reset the high-water mark
    torch.cuda.synchronize()
    with torch.no_grad():
        fn(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()  # bytes


def weight_memory_bytes(layer):
    """Calculate total bytes used by a layer's weight storage.

    For FP32 nn.Linear: each weight is 4 bytes (float32)
    For QuantizedLinear: packed weights are 1 byte per 2 values,
                         plus 4 bytes each for scale and zero_point per row
    """
    if isinstance(layer, QuantizedLinear):
        # packed_weights: each element is uint8 (1 byte), holds 2 weight values
        # scales: float32 (4 bytes each), one per output row
        # zero_points: float32 (4 bytes each), one per output row
        return (layer.packed_weights.nelement() * layer.packed_weights.element_size()
                + layer.scales.nelement() * layer.scales.element_size()
                + layer.zero_points.nelement() * layer.zero_points.element_size())
    elif isinstance(layer, nn.Linear):
        total = layer.weight.nelement() * layer.weight.element_size()
        if layer.bias is not None:
            total += layer.bias.nelement() * layer.bias.element_size()
        return total
    return 0


def main():
    print(f"Device: {DEVICE}")
    print(f"Warmup: {WARMUP_ITERS} iters, Benchmark: {BENCH_ITERS} iters")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = []

    # Print a formatted table header
    header = f"{'Size':>18s} | {'FP32 (ms)':>10s} | {'INT4 (ms)':>10s} | {'Speedup':>8s} | {'FP32 Mem':>10s} | {'INT4 Mem':>10s} | {'Mem Ratio':>10s}"
    print(header)
    print("-" * len(header))

    for in_dim, out_dim in SIZES:
        torch.manual_seed(42)
        x = torch.randn(in_dim, device=DEVICE)  # single input vector

        # --- Benchmark FP32 nn.Linear (the baseline) ---
        fp32_layer = nn.Linear(in_dim, out_dim, bias=False).to(DEVICE)
        fp32_ms = benchmark_linear(fp32_layer, x)
        fp32_weight_mem = weight_memory_bytes(fp32_layer)

        # --- Benchmark INT4 QuantizedLinear (our fused kernel) ---
        # Note: we quantize on CPU (from_linear) then move to GPU (.to(DEVICE))
        int4_layer = QuantizedLinear.from_linear(fp32_layer.cpu()).to(DEVICE)
        x_dev = x.to(DEVICE)
        int4_ms = benchmark_linear(int4_layer, x_dev)
        int4_weight_mem = weight_memory_bytes(int4_layer)

        # Calculate speedup (FP32 time / INT4 time) and memory ratio
        speedup = fp32_ms / int4_ms if int4_ms > 0 else float("inf")
        mem_ratio = fp32_weight_mem / int4_weight_mem if int4_weight_mem > 0 else float("inf")

        size_str = f"({in_dim}, {out_dim})"
        print(f"{size_str:>18s} | {fp32_ms:>10.3f} | {int4_ms:>10.3f} | {speedup:>7.2f}x | "
              f"{fp32_weight_mem/1e6:>8.2f}MB | {int4_weight_mem/1e6:>8.2f}MB | {mem_ratio:>8.1f}x")

        results.append({
            "size": size_str,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "fp32_ms": fp32_ms,
            "int4_ms": int4_ms,
            "speedup": speedup,
            "fp32_mem": fp32_weight_mem,
            "int4_mem": int4_weight_mem,
            "mem_ratio": mem_ratio,
        })

        # Free GPU memory between tests so large models don't OOM
        del fp32_layer, int4_layer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Roofline Analysis (GPU only) ──────────────────────────────────────
    if DEVICE == "cuda":
        print("\n── Roofline Analysis ──")
        print("  (Compare achieved BW to your GPU's peak memory bandwidth)")
        print("  (e.g., A100 = 2039 GB/s, RTX 3090 = 936 GB/s, T4 = 300 GB/s)")
        print()

        props = torch.cuda.get_device_properties(0)

        for r in results:
            in_dim, out_dim = r["in_dim"], r["out_dim"]

            # Total bytes the kernel reads from global memory per forward pass:
            #   - Input vector: in_dim floats × 4 bytes each
            #   - Packed weights: out_dim rows × (in_dim/2) bytes each
            #   - Scales: out_dim floats × 4 bytes each
            #   - Zero points: out_dim floats × 4 bytes each
            bytes_read = in_dim * 4 + out_dim * (in_dim // 2) + out_dim * 4 + out_dim * 4

            # Total floating-point operations:
            #   For each of the out_dim outputs, we do in_dim multiply-adds.
            #   Each multiply-add = 2 FLOPs (one multiply + one addition).
            flops = 2 * in_dim * out_dim

            # Arithmetic Intensity = FLOPs per byte of data read.
            # This tells us whether we're compute-bound or memory-bound.
            # For single-vector inference, AI is typically low → memory-bound.
            ai = flops / bytes_read

            # Achieved memory bandwidth = bytes read / time
            achieved_bw = bytes_read / (r["int4_ms"] * 1e-3) / 1e9  # GB/s

            # Achieved compute throughput = FLOPs / time
            achieved_gflops = flops / (r["int4_ms"] * 1e-3) / 1e9  # GFLOPS

            print(f"  {r['size']:>18s}: AI={ai:.2f} FLOP/B, "
                  f"BW={achieved_bw:.1f} GB/s, "
                  f"GFLOPS={achieved_gflops:.1f}")

    # ── Generate Bar Chart ────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (no GUI window)
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sizes = [r["size"] for r in results]
        fp32_times = [r["fp32_ms"] for r in results]
        int4_times = [r["int4_ms"] for r in results]
        x_pos = np.arange(len(sizes))
        width = 0.35

        # Left chart: Latency comparison
        ax1.bar(x_pos - width/2, fp32_times, width, label="FP32 nn.Linear", color="#4285f4")
        ax1.bar(x_pos + width/2, int4_times, width, label="INT4 Fused Kernel", color="#34a853")
        ax1.set_xlabel("Matrix Size (in, out)")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Inference Latency Comparison")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(sizes, rotation=15)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Right chart: Memory comparison
        fp32_mem = [r["fp32_mem"] / 1e6 for r in results]
        int4_mem = [r["int4_mem"] / 1e6 for r in results]
        ax2.bar(x_pos - width/2, fp32_mem, width, label="FP32", color="#4285f4")
        ax2.bar(x_pos + width/2, int4_mem, width, label="INT4", color="#34a853")
        ax2.set_xlabel("Matrix Size (in, out)")
        ax2.set_ylabel("Weight Memory (MB)")
        ax2.set_title("Weight Storage Comparison")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sizes, rotation=15)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
    except ImportError:
        print("\nmatplotlib not installed — skipping plot generation")


if __name__ == "__main__":
    main()
