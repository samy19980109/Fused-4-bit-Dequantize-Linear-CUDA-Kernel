/*
 * Header file declaring the CUDA kernel's C++ interface.
 *
 * This function is implemented in quantized_linear_kernel.cu and called
 * from the pybind11 bindings in quantized_linear.cpp.
 *
 * #pragma once: tells the compiler to only include this file once,
 * even if multiple .cpp files #include it.
 */

#pragma once
#include <torch/extension.h>  // PyTorch's C++ API (tensors, checks, etc.)

/*
 * Forward pass of the fused quantized linear layer.
 *
 * Does: output = input @ dequantized_weights^T
 * But without ever creating the full FP32 weight matrix in memory.
 *
 * Parameters:
 *   input          — [input_dim] or [batch, input_dim] float32 input activations
 *   packed_weights — [output_dim, input_dim/2] uint8, two 4-bit weights per byte
 *   scales         — [output_dim] float32, one scale factor per output row
 *   zero_points    — [output_dim] float32, one zero point per output row
 *
 * Returns:
 *   output — [output_dim] or [batch, output_dim] float32 result
 */
torch::Tensor quantized_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points
);
