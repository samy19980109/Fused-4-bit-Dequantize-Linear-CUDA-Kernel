"""
Fused MoE INT4 Module - Python wrapper around CUDA kernel.
"""

import torch
import sys
import os

# Try to import the CUDA kernel
try:
    import moe_int4_cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: moe_int4_cuda not available")


def quantize_weights_moe(weights_list):
    """
    Quantize a list of FP16 weight matrices to INT4.

    Args:
        weights_list: List of [ffn_dim, hidden_dim] FP16 tensors (on CUDA)

    Returns:
        packed_weights: [num_experts, ffn_dim, packed_dim] uint8
        scales: [num_experts, ffn_dim] float
        zero_points: [num_experts, ffn_dim] float
    """
    num_experts = len(weights_list)
    ffn_dim = weights_list[0].shape[0]
    hidden_dim = weights_list[0].shape[1]
    packed_dim = hidden_dim // 2

    device = weights_list[0].device

    packed_weights = torch.zeros(
        num_experts, ffn_dim, packed_dim, dtype=torch.uint8, device=device
    )
    scales = torch.zeros(num_experts, ffn_dim, dtype=torch.float32, device=device)
    zero_points = torch.zeros(num_experts, ffn_dim, dtype=torch.float32, device=device)

    for e, w in enumerate(weights_list):
        w_fp32 = w.float()
        w_min = w_fp32.min()
        w_max = w_fp32.max()

        scale_val = ((w_max - w_min) / 15.0).item()
        zp_val = round((-w_min / scale_val).item())
        zp_val = max(0, min(15, zp_val))

        scales[e] = scale_val
        zero_points[e] = zp_val

        # Quantize
        w_quant = torch.clamp(torch.round(w_fp32 / scale_val + zp_val), 0, 15).to(
            torch.uint8
        )

        # Pack into INT4
        packed = torch.zeros(ffn_dim, packed_dim, dtype=torch.uint8, device=device)
        for i in range(packed_dim):
            idx0 = 2 * i
            idx1 = idx0 + 1
            w0 = (
                w_quant[:, idx0]
                if idx0 < hidden_dim
                else torch.zeros(ffn_dim, dtype=torch.uint8, device=device)
            )
            w1 = (
                w_quant[:, idx1]
                if idx1 < hidden_dim
                else torch.zeros(ffn_dim, dtype=torch.uint8, device=device)
            )
            packed[:, i] = (w1 << 4) | w0

        packed_weights[e] = packed

    return packed_weights, scales, zero_points


class MoEINT4(torch.nn.Module):
    """
    MoE layer with fused INT4 CUDA kernel.
    """

    def __init__(self, num_experts, hidden_dim, ffn_dim):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.packed_dim = hidden_dim // 2

        # Initialize weights
        self.register_buffer(
            "packed_weights",
            torch.zeros(num_experts, ffn_dim, self.packed_dim, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales", torch.zeros(num_experts, ffn_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_points", torch.zeros(num_experts, ffn_dim, dtype=torch.float32)
        )

    @classmethod
    def from_weights(cls, weights_list):
        """Create from list of FP16 weight matrices."""
        num_experts = len(weights_list)
        hidden_dim = weights_list[0].shape[1]
        ffn_dim = weights_list[0].shape[0]

        module = cls(num_experts, hidden_dim, ffn_dim)
        packed, scales, zp = quantize_weights_moe(weights_list)
        module.packed_weights = packed
        module.scales = scales
        module.zero_points = zp

        return module

    def forward(self, inputs, expert_ids, tokens_per_expert, input_offsets):
        """
        Forward pass using fused CUDA kernel.

        Args:
            inputs: [total_tokens, hidden_dim] float
            expert_ids: [total_tokens] int - which expert each token goes to
            tokens_per_expert: [num_experts] int
            input_offsets: [num_experts] int

        Returns:
            outputs: [total_tokens, ffn_dim] float
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA kernel not available")

        return moe_int4_cuda.forward(
            self.packed_weights,
            self.scales,
            self.zero_points,
            inputs,
            expert_ids,
            tokens_per_expert,
            input_offsets,
        )


def benchmark_moe_int4():
    """Benchmark the fused MoE INT4 kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("\n" + "=" * 60)
    print("Fused MoE INT4 CUDA Kernel Benchmark")
    print("=" * 60)

    # Config: Mixtral-8x7B
    num_experts = 8
    hidden_dim = 4096
    ffn_dim = 14336
    batch_size = 16
    seq_len = 512
    total_tokens = batch_size * seq_len

    print(f"\nConfig: Mixtral-8x7B style")
    print(f"  Experts: {num_experts}")
    print(f"  Hidden: {hidden_dim}")
    print(f"  FFN: {ffn_dim}")
    print(f"  Tokens: {total_tokens}")

    # Create weights
    print("\nCreating weights...")
    weights = [
        torch.randn(ffn_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.02
        for _ in range(num_experts)
    ]

    # Quantize
    print("Quantizing to INT4...")
    packed, scales, zp = quantize_weights_moe(weights)
    # Already on cuda from quantize_weights_moe

    # Memory comparison
    fp16_mem = sum(w.numel() * 2 for w in weights) / 1e6
    int4_mem = packed.numel() + scales.numel() * 4 + zp.numel() * 4
    print(f"  FP16 memory: {fp16_mem:.1f} MB")
    print(
        f"  INT4 memory: {int4_mem / 1e6:.1f} MB ({fp16_mem * 1e6 / int4_mem:.1f}x savings)"
    )

    # Create inputs with routing (simulate skewed distribution)
    print("\nSimulating token routing...")
    inputs = torch.randn(total_tokens, hidden_dim, device="cuda", dtype=torch.float32)

    # Skewed distribution: more tokens to first experts
    tokens_per_expert = []
    remaining = total_tokens
    for i in range(num_experts):
        if i < num_experts - 1:
            count = remaining // (num_experts - i)
        else:
            count = remaining
        tokens_per_expert.append(count)
        remaining -= count

    # Create expert IDs (each token assigned to one expert)
    expert_ids = torch.zeros(total_tokens, dtype=torch.int32, device="cuda")
    offset = 0
    for e, count in enumerate(tokens_per_expert):
        expert_ids[offset : offset + count] = e
        offset += count

    # Shuffle to make it realistic
    expert_ids = expert_ids[torch.randperm(total_tokens, device="cuda")]

    # Calculate input offsets
    input_offsets = [0]
    for count in tokens_per_expert[:-1]:
        input_offsets.append(input_offsets[-1] + count)
    input_offsets = torch.tensor(input_offsets, dtype=torch.int32, device="cuda")
    tokens_per_expert = torch.tensor(
        tokens_per_expert, dtype=torch.int32, device="cuda"
    )

    print(f"  Tokens per expert: {tokens_per_expert.tolist()}")

    # Benchmark naive (FP16 dequantize + matmul)
    print("\nBenchmarking naive (FP16)...")
    inputs_fp16 = inputs.float()

    # Warmup
    for _ in range(10):
        outputs = []
        offset = 0
        for e in range(num_experts):
            count = tokens_per_expert[e].item()
            if count > 0:
                out = inputs_fp16[offset : offset + count] @ weights[e].T
                outputs.append(out)
            offset += count
    torch.cuda.synchronize()

    # Time
    iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        outputs = []
        offset = 0
        for e in range(num_experts):
            count = tokens_per_expert[e].item()
            if count > 0:
                out = inputs_fp16[offset : offset + count] @ weights[e].T
                outputs.append(out)
            offset += count
    end.record()
    torch.cuda.synchronize()
    naive_ms = start.elapsed_time(end) / iters

    # Benchmark fused INT4 kernel
    print("Benchmarking fused INT4 kernel...")

    if CUDA_AVAILABLE:
        # Warmup
        for _ in range(10):
            _ = moe_int4_cuda.forward(
                packed, scales, zp, inputs, expert_ids, tokens_per_expert, input_offsets
            )
        torch.cuda.synchronize()

        # Time
        start.record()
        for _ in range(iters):
            _ = moe_int4_cuda.forward(
                packed, scales, zp, inputs, expert_ids, tokens_per_expert, input_offsets
            )
        end.record()
        torch.cuda.synchronize()
        int4_ms = start.elapsed_time(end) / iters

        print(f"\n{'=' * 60}")
        print("RESULTS:")
        print(f"{'=' * 60}")
        print(f"  Naive FP16: {naive_ms:.2f} ms")
        print(f"  Fused INT4: {int4_ms:.2f} ms")
        print(f"  Speedup: {naive_ms / int4_ms:.2f}x")
        print(f"  Memory savings: {fp16_mem * 1e6 / int4_mem:.1f}x")
    else:
        print("CUDA kernel not available - build with python setup.py install")


if __name__ == "__main__":
    benchmark_moe_int4()
