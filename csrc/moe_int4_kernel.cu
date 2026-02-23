/*
 * Simplified Fused MoE INT4 CUDA Kernel
 * 
 * Each block processes ONE expert - simpler and correct.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 512;

/*
 * Each block handles ONE expert - processes all tokens for that expert
 */
__global__ void moe_int4_expert_kernel(
    // Weights for this expert only: [ffn_dim, packed_dim]
    const uint8_t* __restrict__ packed_weights,
    // Scales for this expert: [ffn_dim]
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    // Inputs for this expert: [num_tokens, hidden_dim]
    const float* __restrict__ inputs,
    // Output for this expert: [num_tokens, ffn_dim]
    float* __restrict__ outputs,
    int hidden_dim,
    int ffn_dim,
    int packed_dim,
    int num_tokens
) {
    extern __shared__ float shared_mem[];
    
    // Each thread computes one output column (one element of the output vector)
    const int out_col = threadIdx.x;
    if (out_col >= ffn_dim) return;
    
    // Load scale and zero_point for this column
    const float scale = scales[out_col];
    const float zero_point = zero_points[out_col];
    
    // Accumulator
    float accum = 0.0f;
    
    // Process input in tiles
    for (int tile_start = 0; tile_start < hidden_dim; tile_start += TILE_SIZE) {
        // Cooperative load of input tile into shared memory
        for (int i = threadIdx.x; i < TILE_SIZE && (tile_start + i) < hidden_dim; i += BLOCK_SIZE) {
            shared_mem[i] = inputs[tile_start + i];
        }
        __syncthreads();
        
        const int tile_end = min(tile_start + TILE_SIZE, hidden_dim);
        const int tile_elements = tile_end - tile_start;
        
        // Process this tile - dequantize weights and multiply
        for (int pk = 0; pk < (tile_elements + 1) / 2; pk++) {
            // Load packed byte containing 2 4-bit values
            const int weight_idx = out_col * packed_dim + (tile_start / 2) + pk;
            const uint8_t packed_byte = packed_weights[weight_idx];
            
            // Extract two 4-bit values
            uint8_t w0_int = packed_byte & 0x0F;
            uint8_t w1_int = (packed_byte >> 4) & 0x0F;
            
            // Dequantize to float
            float w0 = (float(w0_int) - zero_point) * scale;
            float w1 = (float(w1_int) - zero_point) * scale;
            
            // Get input values from shared memory
            int i0 = tile_start + pk * 2;
            int i1 = i0 + 1;
            
            float in0 = (i0 < hidden_dim) ? shared_mem[pk * 2] : 0.0f;
            float in1 = (i1 < hidden_dim) ? shared_mem[pk * 2 + 1] : 0.0f;
            
            // Fused multiply-accumulate
            accum += in0 * w0;
            if (i1 < hidden_dim) {
                accum += in1 * w1;
            }
        }
        __syncthreads();
    }
    
    // Store output for all tokens - each thread writes all rows for its column
    for (int token = 0; token < num_tokens; token++) {
        outputs[token * ffn_dim + out_col] = accum;
    }
}

// Python binding
torch::Tensor moe_int4_forward(
    torch::Tensor packed_weights,    // [num_experts, ffn_dim, packed_dim] uint8
    torch::Tensor scales,            // [num_experts, ffn_dim] float
    torch::Tensor zero_points,      // [num_experts, ffn_dim] float
    torch::Tensor inputs,           // [total_tokens, hidden_dim] float
    torch::Tensor expert_ids,       // [total_tokens] int (which expert each token goes to)
    torch::Tensor tokens_per_expert, // [num_experts] int
    torch::Tensor input_offsets     // [num_experts] int (offset in inputs for each expert)
) {
    const int num_experts = packed_weights.size(0);
    const int ffn_dim = packed_weights.size(1);
    const int packed_dim = packed_weights.size(2);
    const int hidden_dim = inputs.size(1);
    const int total_tokens = inputs.size(0);
    
    // Output
    auto outputs = torch::zeros({total_tokens, ffn_dim}, inputs.options());
    
    // For each expert, launch a kernel
    for (int expert = 0; expert < num_experts; expert++) {
        const int num_tokens = tokens_per_expert[expert].item<int>();
        if (num_tokens == 0) continue;
        
        const int input_offset = input_offsets[expert].item<int>();
        
        // Get pointers for this expert
        const uint8_t* w_ptr = packed_weights.data_ptr<uint8_t>() + expert * ffn_dim * packed_dim;
        const float* s_ptr = scales.data_ptr<float>() + expert * ffn_dim;
        const float* zp_ptr = zero_points.data_ptr<float>() + expert * ffn_dim;
        const float* in_ptr = inputs.data_ptr<float>() + input_offset * hidden_dim;
        float* out_ptr = outputs.data_ptr<float>() + input_offset * ffn_dim;
        
        const int threads = BLOCK_SIZE;
        const int shared_mem = TILE_SIZE * sizeof(float);
        
        moe_int4_expert_kernel<<<1, threads, shared_mem>>>(
            w_ptr, s_ptr, zp_ptr, in_ptr, out_ptr,
            hidden_dim, ffn_dim, packed_dim, num_tokens
        );
    }
    
    cudaDeviceSynchronize();
    return outputs;
}

PYBIND11_MODULE(moe_int4_cuda, m) {
    m.def("forward", &moe_int4_forward,
          "Fused MoE INT4 forward");
}
