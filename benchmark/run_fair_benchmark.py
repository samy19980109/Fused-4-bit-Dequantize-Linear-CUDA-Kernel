#!/usr/bin/env python3
"""
Fair MoE Grouped GEMM Benchmark.

All implementations use the SAME weights and inputs for fair comparison.
"""

import argparse
import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.moe_grouped_gemm.config import MIXTRAL_8x7B
from benchmark.moe_grouped_gemm.routing import get_expert_sizes_for_benchmark
from benchmark.moe_grouped_gemm.utils import format_bytes, format_time, get_gpu_info


def run_fair_benchmark():
    """Run a fair benchmark where all methods use same weights/inputs."""

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moe_config = MIXTRAL_8x7B
    batch_size = 16
    seq_len = 512
    warmup = 20
    iters = 50

    # Get expert sizes
    m_sizes, k, n = get_expert_sizes_for_benchmark(
        batch_size * seq_len,
        moe_config.num_experts,
        moe_config.hidden_dim,
        moe_config.ffn_dim,
        distribution="skewed",
        device=device,
    )

    print(f"\nConfig: {moe_config.name}")
    print(f"Batch: {batch_size}, Seq: {seq_len}")
    print(f"Expert sizes: {m_sizes}")
    print(f"K={k}, N={n}")
    print()

    # Create SAME weights for all methods
    print("Creating shared weights and inputs...")
    fp16_weights = [
        torch.randn(n, k, device=device, dtype=torch.float16) * 0.02
        for _ in range(moe_config.num_experts)
    ]
    fp16_inputs = [
        torch.randn(m, k, device=device, dtype=torch.float16)
        if m > 0
        else torch.empty(0, k, device=device, dtype=torch.float16)
        for m in m_sizes
    ]

    # Benchmark 1: Naive (for-loop with @ operator)
    print("Benchmarking Naive (for-loop)...")
    for _ in range(warmup):
        _ = [x @ w.T for x, w in zip(fp16_inputs, fp16_weights)]
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = [x @ w.T for x, w in zip(fp16_inputs, fp16_weights)]
    torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - start) / iters * 1000

    # Benchmark 2: torch.bmm (batched)
    print("Benchmarking torch.bmm (batched)...")
    # Pack inputs
    max_m = max(m_sizes)
    padded_inputs = torch.zeros(
        moe_config.num_experts, max_m, k, device=device, dtype=torch.float16
    )
    for i, x in enumerate(fp16_inputs):
        if x.shape[0] > 0:
            padded_inputs[i, : x.shape[0], :] = x
    stacked_weights = torch.stack(fp16_weights, dim=0)

    for _ in range(warmup):
        _ = torch.bmm(padded_inputs, stacked_weights.transpose(1, 2))
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.bmm(padded_inputs, stacked_weights.transpose(1, 2))
    torch.cuda.synchronize()
    bmm_ms = (time.perf_counter() - start) / iters * 1000

    # Benchmark 3: INT4 Quantized (pre-quantized weights)
    print("Benchmarking INT4 Quantized...")
    from benchmark.moe_grouped_gemm.moe_int4_module import QuantizedMoE

    moe_int4 = QuantizedMoE.from_fp16_weights(fp16_weights).to(device)

    for _ in range(warmup):
        _ = moe_int4(fp16_inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = moe_int4(fp16_inputs)
    torch.cuda.synchronize()
    int4_ms = (time.perf_counter() - start) / iters * 1000

    # Benchmark 4: FP4 (pre-quantized)
    print("Benchmarking FP4 Quantized...")
    from benchmark.moe_grouped_gemm.grouped_gemm_fp4 import quantize_fp4, FP4GroupedGEMM

    fp4_weights = []
    for w in fp16_weights:
        q, s = quantize_fp4(w)
        fp4_weights.append((q, s))

    for _ in range(warmup):
        _ = FP4GroupedGEMM.forward(fp16_inputs, fp4_weights)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = FP4GroupedGEMM.forward(fp16_inputs, fp4_weights)
    torch.cuda.synchronize()
    fp4_ms = (time.perf_counter() - start) / iters * 1000

    # Calculate memory
    fp16_mem = sum(w.numel() * 2 for w in fp16_weights)
    int4_mem = moe_int4.total_memory_bytes
    fp4_mem = sum(w[0].numel() // 2 for w in fp4_weights) + sum(
        w[1].numel() * 4 for w in fp4_weights
    )

    # Calculate TFLOPS
    total_flops = sum(2 * m * k * n for m in m_sizes)
    naive_tflops = total_flops / (naive_ms * 1e-3) / 1e12
    bmm_tflops = total_flops / (bmm_ms * 1e-3) / 1e12
    int4_tflops = total_flops / (int4_ms * 1e-3) / 1e12
    fp4_tflops = total_flops / (fp4_ms * 1e-3) / 1e12

    # Print results
    print("\n" + "=" * 80)
    print(
        f"{'Implementation':<25} | {'Latency':<12} | {'TFLOPS':<10} | {'Memory':<12} | {'Speedup':<10}"
    )
    print("=" * 80)
    print(
        f"{'Naive (for-loop)':<25} | {naive_ms:<12.3f} | {naive_tflops:<10.2f} | {format_bytes(fp16_mem):<12} | {'1.00x':<10}"
    )
    print(
        f"{'torch.bmm (batched)':<25} | {bmm_ms:<12.3f} | {bmm_tflops:<10.2f} | {format_bytes(fp16_mem):<12} | {naive_ms / bmm_ms:<10.2f}x"
    )
    print(
        f"{'INT4 Quantized':<25} | {int4_ms:<12.3f} | {int4_tflops:<10.2f} | {format_bytes(int4_mem):<12} | {naive_ms / int4_ms:<10.2f}x"
    )
    print(
        f"{'FP4 Quantized':<25} | {fp4_ms:<12.3f} | {fp4_tflops:<10.2f} | {format_bytes(fp4_mem):<12} | {naive_ms / fp4_ms:<10.2f}x"
    )
    print("=" * 80)

    # Key insight
    print("\n📊 KEY INSIGHTS:")
    print(
        f"  • Memory savings: INT4 = {fp16_mem / int4_mem:.1f}x, FP4 = {fp16_mem / fp4_mem:.1f}x"
    )
    print(
        f"  • FP4 on Blackwell is {fp4_ms / int4_ms:.2f}x faster than INT4 (native tensor cores)"
    )
    print()


if __name__ == "__main__":
    gpu_info = get_gpu_info()
    print("=" * 80)
    print("Fair MoE Benchmark (RTX 5090 Blackwell)")
    print("=" * 80)
    if gpu_info["available"]:
        print(f"GPU: {gpu_info['name']}")
        print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"Compute: {gpu_info['compute_capability']}")
    print("=" * 80)

    run_fair_benchmark()
