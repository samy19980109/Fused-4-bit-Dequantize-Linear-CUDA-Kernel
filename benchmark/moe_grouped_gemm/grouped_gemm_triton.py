"""
Triton Grouped GEMM Kernel for MoE.

Simplified version that works with Triton 3.x.
Uses a basic approach without complex runtime loops.
"""

import torch
import time
from typing import List, Tuple, Optional

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available. Install with: pip install triton")


if TRITON_AVAILABLE:

    @triton.jit
    def gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Basic GEMM kernel for single matrix."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = num_pid_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * BLOCK_M
        group_size_m = min(num_pid_m - first_pid_m, BLOCK_M)

        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=rm[:, None] < M & rk[None, :] < K, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < K & rn[None, :] < N, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            rk += BLOCK_K

        c = acc.to(tl.float16)
        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=rm[:, None] < M & rn[None, :] < N)

    def triton_grouped_gemm_forward(
        expert_inputs: List[torch.Tensor],
        expert_weights: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Grouped GEMM using Triton - call individual kernels per expert.

        This is simpler than a fused kernel but still much faster than naive
        because Triton kernels are highly optimized.
        """
        outputs = []

        # For each expert, run optimized Triton GEMM
        for x, w in zip(expert_inputs, expert_weights):
            if x.shape[0] == 0:
                outputs.append(
                    torch.empty(0, w.shape[0], device=x.device, dtype=x.dtype)
                )
                continue

            M, K = x.shape
            N = w.shape[0]

            # Allocate output
            c = torch.zeros(M, N, device=x.device, dtype=torch.float16)

            # Grid for Triton
            BLOCK_M = 128
            BLOCK_N = 256
            BLOCK_K = 64
            grid = (M // BLOCK_M + 1) * (N // BLOCK_N + 1)

            gemm_kernel[grid](
                x,
                w,
                c,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(1),
                w.stride(0),
                c.stride(0),
                c.stride(1),
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )

            outputs.append(c)

        return outputs


def benchmark_triton_grouped_gemm(
    m_sizes: List[int],
    k: int,
    n: int,
    device: str = "cuda",
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Tuple[float, float]:
    """Benchmark Triton grouped GEMM."""
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping benchmark")
        return 0.0, 0.0

    num_experts = len(m_sizes)

    expert_inputs = [
        torch.randn(m, k, device=device, dtype=torch.float16)
        if m > 0
        else torch.empty(0, k, device=device, dtype=torch.float16)
        for m in m_sizes
    ]
    expert_weights = [
        torch.randn(n, k, device=device, dtype=torch.float16)
        for _ in range(num_experts)
    ]

    # Warmup
    for _ in range(warmup_iters):
        _ = triton_grouped_gemm_forward(expert_inputs, expert_weights)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = triton_grouped_gemm_forward(expert_inputs, expert_weights)

    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_latency_ms = (end - start) / bench_iters * 1000

    total_flops = sum(2 * m * k * n for m in m_sizes)
    tflops = total_flops / (avg_latency_ms * 1e-3) / 1e12

    return avg_latency_ms, tflops


if __name__ == "__main__":
    if not TRITON_AVAILABLE:
        print("Triton not available!")
        exit(1)

    print("Testing Triton Grouped GEMM...")

    m_sizes = [256, 192, 312, 180, 95, 280, 150, 135]
    k, n = 4096, 14336

    latency, tflops = benchmark_triton_grouped_gemm(
        m_sizes, k, n, warmup_iters=5, bench_iters=20
    )

    print(f"Latency: {latency:.3f} ms")
    print(f"TFLOPS: {tflops:.2f}")
