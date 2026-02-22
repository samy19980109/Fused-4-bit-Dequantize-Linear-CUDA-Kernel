"""
4-bit Quantized Linear Layer — Python package.

This package provides tools to compress neural network linear layers from
32-bit floats to 4-bit integers, reducing memory usage by ~8x.

Main exports:
    QuantizedLinear          — nn.Module drop-in replacement for nn.Linear
    quantize_weights()       — compress FP32 weights to packed 4-bit
    dequantize_weights()     — decompress packed 4-bit back to FP32
    reference_quantized_linear() — slow but correct Python implementation
"""

from .quantize import quantize_weights, dequantize_weights, reference_quantized_linear
from .module import QuantizedLinear

__all__ = [
    "quantize_weights",
    "dequantize_weights",
    "reference_quantized_linear",
    "QuantizedLinear",
]
