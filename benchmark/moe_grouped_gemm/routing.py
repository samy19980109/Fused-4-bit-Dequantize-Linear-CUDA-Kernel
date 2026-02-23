"""
Token Routing Simulation for MoE Models.

In real MoE models, tokens are routed to experts by a learned gating network.
For benchmarking, we simulate realistic routing patterns.

Key insight: Real routing is NOT uniform - some experts get more tokens than others.
This matters for grouped GEMM performance because expert sizes vary.
"""

import torch
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class RoutingResult:
    """Result of token routing simulation."""

    expert_indices: torch.Tensor  # [num_tokens, top_k] which expert each token goes to
    expert_weights: torch.Tensor  # [num_tokens, top_k] routing weights (softmax scores)
    tokens_per_expert: List[int]  # Number of tokens assigned to each expert
    expert_token_offsets: List[int]  # Cumulative offsets for token assignment


def simulate_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    distribution: str = "skewed",
    device: str = "cuda",
    seed: int = 42,
) -> RoutingResult:
    """
    Simulate token-to-expert routing.

    Args:
        num_tokens: Total number of tokens to route
        num_experts: Number of experts
        top_k: Number of experts each token is routed to
        distribution: How tokens are distributed across experts
            - "uniform": Equal distribution (unrealistic, best case)
            - "skewed": Zipf-like distribution (matches real MoE)
            - "random": Random assignment
        device: Device for tensors
        seed: Random seed

    Returns:
        RoutingResult with expert assignments and statistics
    """
    torch.manual_seed(seed)

    # Generate routing logits
    if distribution == "uniform":
        logits = torch.zeros(num_tokens, num_experts, device=device)
    elif distribution == "skewed":
        # Zipf distribution: Expert 0 gets 2x more than Expert 1, etc.
        expert_probs = 1.0 / (
            torch.arange(num_experts, device=device, dtype=torch.float32) + 1
        )
        expert_probs = expert_probs / expert_probs.sum()
        # Sample from this distribution for each token
        logits = torch.log(expert_probs + 1e-10).unsqueeze(0).expand(num_tokens, -1)
        # Add noise to make it interesting
        logits = logits + torch.randn_like(logits) * 0.5
    elif distribution == "random":
        logits = torch.randn(num_tokens, num_experts, device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Compute softmax and get top-k experts
    routing_weights = torch.softmax(logits, dim=-1)
    expert_weights, expert_indices = torch.topk(routing_weights, top_k, dim=-1)

    # Normalize weights for top-k experts
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

    # Count tokens per expert
    tokens_per_expert = [0] * num_experts
    for idx in expert_indices.flatten().cpu().numpy():
        tokens_per_expert[idx] += 1

    # Compute cumulative offsets (for sorted expert assignment)
    expert_token_offsets = [0]
    for count in tokens_per_expert[:-1]:
        expert_token_offsets.append(expert_token_offsets[-1] + count)

    return RoutingResult(
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        tokens_per_expert=tokens_per_expert,
        expert_token_offsets=expert_token_offsets,
    )


def create_expert_inputs(
    x: torch.Tensor,
    routing: RoutingResult,
    num_experts: int,
    top_k: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Distribute input tokens to their assigned experts.

    This is the "dispatch" step in MoE.

    Args:
        x: [num_tokens, hidden_dim] input tokens
        routing: RoutingResult from simulate_routing
        num_experts: Number of experts
        top_k: Experts per token

    Returns:
        expert_inputs: List of [tokens_for_expert_i, hidden_dim] tensors
        permutation: Indices for gathering outputs back to original order
    """
    num_tokens, hidden_dim = x.shape

    # Flatten: each token appears in top_k expert assignments
    # Shape: [num_tokens * top_k, hidden_dim]
    token_indices = (
        torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, top_k)
    )
    flat_token_idx = token_indices.flatten()
    flat_expert_idx = routing.expert_indices.flatten()

    # Sort by expert index to group tokens for each expert
    sorted_indices = torch.argsort(flat_expert_idx)
    sorted_token_idx = flat_token_idx[sorted_indices]

    # Create permutation for gathering outputs back
    inverse_perm = torch.argsort(sorted_indices)

    # Split into per-expert tensors
    expert_inputs = []
    offset = 0
    for expert_idx in range(num_experts):
        count = routing.tokens_per_expert[expert_idx]
        if count > 0:
            expert_token_indices = sorted_token_idx[offset : offset + count]
            expert_x = x[expert_token_indices]
            expert_inputs.append(expert_x)
        else:
            expert_inputs.append(
                torch.empty(0, hidden_dim, device=x.device, dtype=x.dtype)
            )
        offset += count

    return expert_inputs, inverse_perm


def combine_expert_outputs(
    expert_outputs: List[torch.Tensor],
    routing: RoutingResult,
    permutation: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Combine outputs from experts back to original token order.

    This is the "combine" step in MoE.

    Args:
        expert_outputs: List of [tokens_for_expert_i, ffn_dim] outputs from each expert
        routing: RoutingResult
        permutation: Indices for unsorting to original order
        top_k: Experts per token

    Returns:
        [num_tokens, ffn_dim] combined output (weighted by routing scores)
    """
    num_experts = len(expert_outputs)

    # Concatenate all expert outputs
    concat_output = torch.cat(expert_outputs, dim=0)

    # Unsort to get back to [token, expert_num] order
    unsorted_output = concat_output[permutation]

    # Reshape to [num_tokens, top_k, ffn_dim]
    num_tokens = unsorted_output.shape[0] // top_k
    ffn_dim = unsorted_output.shape[-1]
    unsorted_output = unsorted_output.view(num_tokens, top_k, ffn_dim)

    # Weight by routing scores and sum
    weights = routing.expert_weights.unsqueeze(-1)
    combined = (unsorted_output * weights).sum(dim=1)

    return combined


def get_expert_sizes_for_benchmark(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    distribution: str = "skewed",
    device: str = "cuda",
) -> Tuple[List[int], int, int]:
    """
    Get the M, K, N dimensions for each expert's GEMM.

    In MoE, each expert computes: output = input @ weight.T
    where:
        - M = number of tokens routed to this expert (varies!)
        - K = hidden_dim
        - N = ffn_dim

    Returns:
        m_sizes: List of M dimensions (tokens per expert)
        k_size: K dimension (same for all)
        n_size: N dimension (same for all)
    """
    routing = simulate_routing(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=2,
        distribution=distribution,
        device=device,
    )
    m_sizes = routing.tokens_per_expert
    return m_sizes, hidden_dim, ffn_dim
