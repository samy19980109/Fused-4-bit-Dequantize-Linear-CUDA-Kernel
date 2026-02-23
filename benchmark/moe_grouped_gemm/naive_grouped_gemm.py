"""
Naive Grouped GEMM Implementation - The Baseline.

This is what most companies start with: a simple for-loop over experts.
Each expert computation is a separate kernel launch → massive overhead.

This baseline demonstrates the problem we're solving.
"""

import torch
import time
from typing import List, Tuple


def naive_grouped_gemm_forward(
    expert_inputs: List[torch.Tensor],
    expert_weights: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Naive implementation: For-loop over experts with separate kernel launches.

    This is the SLOW way to compute MoE expert outputs.

    Args:
        expert_inputs: List of [M_i, K] input tensors (one per expert, M_i varies)
        expert_weights: List of [N, K] weight matrices (one per expert)

    Returns:
        expert_outputs: List of [M_i, N] output tensors

    Problem: Each matmul is a separate kernel launch!
    With 8-128 experts, we launch 8-128 kernels = massive overhead.
    """
    outputs = []
    for x, w in zip(expert_inputs, expert_weights):
        if x.shape[0] > 0:  # Skip experts with no tokens
            out = x @ w.T
        else:
            out = torch.empty(0, w.shape[0], device=x.device, dtype=x.dtype)
        outputs.append(out)
    return outputs


def naive_grouped_gemm_forward_bias(
    expert_inputs: List[torch.Tensor],
    expert_weights: List[torch.Tensor],
    expert_biases: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Naive with bias addition (even more kernel launches)."""
    outputs = []
    for x, w, b in zip(expert_inputs, expert_weights, expert_biases):
        if x.shape[0] > 0:
            out = x @ w.T + b
        else:
            out = torch.empty(0, w.shape[0], device=x.device, dtype=x.dtype)
        outputs.append(out)
    return outputs


class NaiveGroupedGEMM(torch.nn.Module):
    """
    Module wrapper for naive grouped GEMM.

    This is the baseline we're comparing against.
    """

    def __init__(self, num_experts: int, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        # Each expert has its own weight matrix [ffn_dim, hidden_dim]
        self.expert_weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(ffn_dim, hidden_dim) * 0.02)
                for _ in range(num_experts)
            ]
        )

    def forward(self, expert_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass using naive for-loop."""
        return naive_grouped_gemm_forward(
            expert_inputs, [w for w in self.expert_weights]
        )


def benchmark_naive_grouped_gemm(
    m_sizes: List[int],
    k: int,
    n: int,
    device: str = "cuda",
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Tuple[float, float]:
    """
    Benchmark naive grouped GEMM.

    Args:
        m_sizes: List of M dimensions (tokens per expert)
        k: Hidden dimension
        n: FFN dimension
        device: Device to run on
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations

    Returns:
        avg_latency_ms: Average latency in milliseconds
        tflops: Achieved TFLOPS
    """
    num_experts = len(m_sizes)

    # Create inputs and weights
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
        _ = naive_grouped_gemm_forward(expert_inputs, expert_weights)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = naive_grouped_gemm_forward(expert_inputs, expert_weights)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    avg_latency_ms = (end - start) / bench_iters * 1000

    # Calculate FLOPS
    total_flops = sum(2 * m * k * n for m in m_sizes)  # 2 * M * K * N per GEMM
    tflops = total_flops / (avg_latency_ms * 1e-3) / 1e12

    return avg_latency_ms, tflops


if __name__ == "__main__":
    # Quick test
    print("Testing Naive Grouped GEMM...")

    m_sizes = [256, 192, 312, 180, 95, 280, 150, 135]  # Simulated skewed distribution
    k, n = 4096, 14336

    latency, tflops = benchmark_naive_grouped_gemm(
        m_sizes, k, n, warmup_iters=5, bench_iters=20
    )

    print(f"Latency: {latency:.3f} ms")
    print(f"TFLOPS: {tflops:.2f}")
