"""
torch.bmm Grouped GEMM Reference Implementation.

This approach pads all experts to the same size and uses torch.bmm.
Better than naive but still not optimal due to wasted computation on padding.

Pros: Simple, uses optimized PyTorch kernel
Cons: Requires padding, wasted compute on zero-padding
"""

import torch
import time
from typing import List, Tuple


def pad_expert_inputs(
    expert_inputs: List[torch.Tensor],
    max_tokens: int,
) -> torch.Tensor:
    """
    Pad expert inputs to the same size for batched matmul.

    Args:
        expert_inputs: List of [M_i, K] tensors (varying M_i)
        max_tokens: Maximum M dimension to pad to

    Returns:
        padded: [num_experts, max_tokens, K] tensor
    """
    num_experts = len(expert_inputs)
    k = expert_inputs[0].shape[1]

    padded = torch.zeros(
        num_experts,
        max_tokens,
        k,
        device=expert_inputs[0].device,
        dtype=expert_inputs[0].dtype,
    )

    for i, x in enumerate(expert_inputs):
        if x.shape[0] > 0:
            padded[i, : x.shape[0], :] = x

    return padded


def bmm_grouped_gemm_forward(
    expert_inputs: List[torch.Tensor],
    expert_weights: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Grouped GEMM using torch.bmm with padding.

    This packs all expert GEMMs into a batched matmul.

    Args:
        expert_inputs: List of [M_i, K] tensors
        expert_weights: List of [N, K] tensors

    Returns:
        expert_outputs: List of [M_i, N] tensors
    """
    num_experts = len(expert_inputs)
    m_sizes = [x.shape[0] for x in expert_inputs]
    max_m = max(m_sizes)

    # Skip if all empty
    if max_m == 0:
        return [
            torch.empty(0, w.shape[0], device=w.device, dtype=w.dtype)
            for w in expert_weights
        ]

    # Pad inputs to [num_experts, max_m, K]
    padded_inputs = pad_expert_inputs(expert_inputs, max_m)

    # Stack weights to [num_experts, N, K]
    stacked_weights = torch.stack(expert_weights, dim=0)

    # Batched matmul: [E, max_m, K] @ [E, K, N] -> [E, max_m, N]
    # Note: weights need transpose
    padded_outputs = torch.bmm(padded_inputs, stacked_weights.transpose(1, 2))

    # Extract valid outputs (remove padding)
    outputs = []
    for i in range(num_experts):
        if m_sizes[i] > 0:
            outputs.append(padded_outputs[i, : m_sizes[i], :])
        else:
            outputs.append(
                torch.empty(
                    0,
                    expert_weights[i].shape[0],
                    device=padded_outputs.device,
                    dtype=padded_outputs.dtype,
                )
            )

    return outputs


class BMMGroupedGEMM(torch.nn.Module):
    """Module wrapper for torch.bmm grouped GEMM."""

    def __init__(self, num_experts: int, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        # Single stacked parameter: [num_experts, ffn_dim, hidden_dim]
        self.stacked_weights = torch.nn.Parameter(
            torch.randn(num_experts, ffn_dim, hidden_dim) * 0.02
        )

    def forward(self, expert_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        weights = [self.stacked_weights[i] for i in range(self.num_experts)]
        return bmm_grouped_gemm_forward(expert_inputs, weights)


def benchmark_bmm_grouped_gemm(
    m_sizes: List[int],
    k: int,
    n: int,
    device: str = "cuda",
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Tuple[float, float]:
    """
    Benchmark torch.bmm grouped GEMM.
    """
    num_experts = len(m_sizes)
    max_m = max(m_sizes)

    # Create inputs and weights
    expert_inputs = [
        torch.randn(m, k, device=device, dtype=torch.float16)
        if m > 0
        else torch.empty(0, k, device=device, dtype=torch.float16)
        for m in m_sizes
    ]
    stacked_weights = torch.randn(num_experts, n, k, device=device, dtype=torch.float16)
    expert_weights = [stacked_weights[i] for i in range(num_experts)]

    # Warmup
    for _ in range(warmup_iters):
        _ = bmm_grouped_gemm_forward(expert_inputs, expert_weights)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = bmm_grouped_gemm_forward(expert_inputs, expert_weights)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    avg_latency_ms = (end - start) / bench_iters * 1000

    # Calculate FLOPS (only for valid computation, not padding)
    total_flops = sum(2 * m * k * n for m in m_sizes)
    tflops = total_flops / (avg_latency_ms * 1e-3) / 1e12

    return avg_latency_ms, tflops


if __name__ == "__main__":
    print("Testing torch.bmm Grouped GEMM...")

    m_sizes = [256, 192, 312, 180, 95, 280, 150, 135]
    k, n = 4096, 14336

    latency, tflops = benchmark_bmm_grouped_gemm(
        m_sizes, k, n, warmup_iters=5, bench_iters=20
    )

    print(f"Latency: {latency:.3f} ms")
    print(f"TFLOPS: {tflops:.2f}")
