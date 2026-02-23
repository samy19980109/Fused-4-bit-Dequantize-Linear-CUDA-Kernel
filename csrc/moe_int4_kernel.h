/*
 * MoE INT4 CUDA Kernel - Header
 */

#include <torch/extension.h>

torch::Tensor moe_int4_forward(
    torch::Tensor packed_weights,    // [num_experts, ffn_dim, packed_dim] uint8
    torch::Tensor scales,            // [num_experts, ffn_dim] float
    torch::Tensor zero_points,       // [num_experts, ffn_dim] float
    torch::Tensor inputs,            // [total_tokens, hidden_dim] float
    torch::Tensor expert_ids,       // [total_tokens] int
    torch::Tensor tokens_per_expert, // [num_experts] int
    torch::Tensor input_offsets     // [num_experts] int
);
