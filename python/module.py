"""
PyTorch nn.Module wrapper for the fused 4-bit quantized linear layer.

=== HOW THIS FITS INTO A NEURAL NETWORK ===

In a typical neural network, nn.Linear layers store weights as FP32 matrices
and compute: output = input @ weight^T

QuantizedLinear is a drop-in replacement that:
  1. Stores weights in 4-bit packed format (8x less memory)
  2. On GPU: uses our fused CUDA kernel (dequantize + matmul in one pass)
  3. On CPU: falls back to Python reference (dequantize then matmul)

=== USAGE ===
    # Convert an existing trained model's linear layers:
    original_layer = model.fc1  # some nn.Linear(512, 256)
    model.fc1 = QuantizedLinear.from_linear(original_layer)
    # Now model.fc1 uses 8x less memory and the fused kernel on GPU!

=== BUFFERS vs PARAMETERS ===
We use register_buffer() instead of nn.Parameter() because:
  - Parameters are meant to be trained (updated by optimizer)
  - Buffers are for fixed data that should move with the model (to GPU, in state_dict, etc.)
  - Our quantized weights are frozen — we don't train them, just use them for inference
"""

import torch
import torch.nn as nn

from .quantize import quantize_weights, reference_quantized_linear


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear using fused INT4 dequantize + matmul.

    Stores weights in packed 4-bit format (~8x byte reduction) and uses a
    fused CUDA kernel for inference. Falls back to a Python reference
    implementation when CUDA is not available.

    Attributes:
        in_features:    Number of input dimensions (columns in weight matrix)
        out_features:   Number of output dimensions (rows in weight matrix)
        packed_weights: [out_features, in_features // 2] uint8 tensor
                        Each byte holds two 4-bit quantized weight values
        scales:         [out_features] float32 — per-row scale for dequantization
        zero_points:    [out_features] float32 — per-row zero point for dequantization
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # register_buffer makes these tensors part of the module's state:
        #   - They get moved to GPU when you call .cuda() or .to(device)
        #   - They get saved/loaded with state_dict()
        #   - But they are NOT updated by the optimizer during training
        # We initialize with zeros; actual values are set by from_linear().
        self.register_buffer("packed_weights",
                             torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scales",
                             torch.zeros(out_features, dtype=torch.float32))
        self.register_buffer("zero_points",
                             torch.zeros(out_features, dtype=torch.float32))

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        """Create a QuantizedLinear by quantizing an existing nn.Linear layer.

        This is the main entry point for converting a pre-trained model.
        It takes the FP32 weights from an nn.Linear and quantizes them to 4-bit.

        Args:
            linear: A pre-trained nn.Linear layer (must not have bias)

        Returns:
            A QuantizedLinear module with the same behavior but compressed weights

        Example:
            >>> original = nn.Linear(4096, 11008, bias=False)
            >>> quantized = QuantizedLinear.from_linear(original)
            >>> # quantized.packed_weights is ~8x smaller than original.weight
        """
        assert linear.bias is None, "Bias not supported yet"
        in_f = linear.in_features
        out_f = linear.out_features

        module = cls(in_f, out_f)

        # quantize_weights() does the heavy lifting:
        # FP32 matrix → packed uint8 + scale/zero_point per row
        packed, scales, zp = quantize_weights(linear.weight.data)

        module.packed_weights = packed
        module.scales = scales
        module.zero_points = zp

        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantized linear operation: output = x @ dequantized_weights^T

        Automatically chooses the best implementation:
          - GPU tensor → fused CUDA kernel (fast, memory-efficient)
          - CPU tensor → Python reference (slower, but works without CUDA)

        Args:
            x: [input_dim] or [batch, input_dim] float32 input tensor

        Returns:
            [output_dim] or [batch, output_dim] float32 output tensor
        """
        if x.is_cuda:
            return self._forward_cuda(x)
        # CPU fallback: dequantize to full FP32 matrix, then standard matmul
        return reference_quantized_linear(
            x, self.packed_weights, self.scales, self.zero_points
        )

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Call the fused CUDA kernel for GPU inference.

        The kernel does dequantization + matrix multiply in a single GPU pass,
        never materializing the full FP32 weight matrix. This saves memory and
        avoids the overhead of launching two separate kernels.
        """
        # This import loads our compiled C++/CUDA extension.
        # It's built by running: python setup.py install
        import fused_quant_linear_cuda
        return fused_quant_linear_cuda.forward(
            x, self.packed_weights, self.scales, self.zero_points
        )

    def extra_repr(self) -> str:
        """String shown when you print() the module (for debugging)."""
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bits=4")
