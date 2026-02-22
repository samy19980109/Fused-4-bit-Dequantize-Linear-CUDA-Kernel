"""
Quantization utilities for converting FP32 weights to 4-bit packed integers.

=== WHAT IS QUANTIZATION? ===
Neural network weights are normally stored as 32-bit floats (FP32), using 4 bytes each.
Quantization compresses them to fewer bits — here, just 4 bits (0.5 bytes each).
This gives us ~8x memory savings, at the cost of some precision loss.

=== ASYMMETRIC PER-CHANNEL QUANTIZATION ===
"Per-channel" means each row of the weight matrix gets its own scale and zero_point.
This is important because different rows can have very different value ranges.

"Asymmetric" means we map the range [min, max] to [0, 15] (for 4-bit), rather than
mapping symmetrically around zero. This is better for weights that aren't centered at 0.

The math:
    Quantize:   q = round(w / scale + zero_point),  clamped to [0, 15]
    Dequantize: w = (q - zero_point) * scale

Where:
    scale      = (max_weight - min_weight) / 15
    zero_point = round(-min_weight / scale)

=== PACKING ===
Since each quantized value is only 4 bits, we pack TWO values into one uint8 byte:
    packed_byte = (value_at_odd_index << 4) | value_at_even_index

This is like storing two single-digit numbers in one slot:
    If we have values 5 and 12, the packed byte is: 0xC5 (12*16 + 5 = 197)
    Low nibble (bottom 4 bits) = 5  (the even-indexed weight)
    High nibble (top 4 bits)   = 12 (the odd-indexed weight)
"""

import torch
import torch.nn.functional as F


def quantize_weights(weight_fp32: torch.Tensor, num_bits: int = 4):
    """Quantize FP32 weights to packed INT4 using asymmetric per-channel quantization.

    This is the "compression" step. We take a full-precision weight matrix and
    squeeze it into 4 bits per value, storing the result in packed uint8 bytes.

    Args:
        weight_fp32: [output_dim, input_dim] float32 weight matrix.
                     Each row is one output neuron's weights.
        num_bits: number of bits per weight (default 4, giving values 0-15)

    Returns:
        packed_uint8:  [output_dim, input_dim // 2] — the compressed weights.
                       Each byte holds 2 quantized values.
        scales:        [output_dim] — one scale factor per row, used to convert
                       back to float during dequantization.
        zero_points:   [output_dim] — one zero point per row, the quantized value
                       that represents 0.0 in the original float space.

    Example:
        >>> weight = torch.randn(256, 512)      # 256 output neurons, 512 inputs each
        >>> packed, scales, zp = quantize_weights(weight)
        >>> packed.shape   # (256, 256) — half the columns because 2 values per byte
        >>> scales.shape   # (256,) — one scale per output row
    """
    assert weight_fp32.ndim == 2, "Weight must be 2D [output_dim, input_dim]"
    assert weight_fp32.shape[1] % 2 == 0, "input_dim must be even for packing"

    # For 4 bits, the maximum quantized value is 2^4 - 1 = 15.
    # So our quantized integers will be in the range [0, 15].
    max_val = (1 << num_bits) - 1  # 15 for 4-bit

    # --- Step 1: Find the range of each row ---
    # We need the min and max of each row to figure out how to map floats → integers.
    # dim=1 means "along columns" (i.e., within each row).
    w_min = weight_fp32.min(dim=1).values  # [output_dim] — smallest value in each row
    w_max = weight_fp32.max(dim=1).values  # [output_dim] — largest value in each row

    # --- Step 2: Compute scale and zero_point ---
    # scale = how much "real-world value" each quantization step represents.
    #   e.g., if a row's values range from -2.0 to +1.0, range = 3.0
    #         scale = 3.0 / 15 = 0.2  (each integer step = 0.2 in float space)
    scales = (w_max - w_min) / max_val  # [output_dim]

    # Special case: if a row is constant (all values identical), w_max == w_min,
    # so scale would be 0, causing division-by-zero later.
    # We handle this by giving constant rows a reasonable scale.
    constant_mask = (w_max == w_min)
    safe_scale = torch.where(
        constant_mask,
        # For constant rows, use a scale based on the value's magnitude.
        # This ensures the value maps to a valid integer in [0, 15].
        torch.clamp(w_max.abs(), min=1.0) / max_val,
        scales,
    )
    # Final safety net: ensure scale is never exactly zero
    safe_scale = torch.clamp(safe_scale, min=1e-8)

    # zero_point = the quantized integer that corresponds to 0.0 in float space.
    # From the dequantize formula: 0.0 = (zero_point - zero_point) * scale
    # Derived from: 0.0 = (q - zp) * scale  →  q = zp when the float value is 0.
    # zp = round(-min / scale), because we're shifting the range so min maps to 0.
    zero_points = torch.round(-w_min / safe_scale)  # [output_dim]
    zero_points = torch.clamp(zero_points, 0, max_val)  # Must fit in [0, 15]

    # --- Step 3: Quantize each weight value to an integer in [0, 15] ---
    # Formula: q = round(w / scale + zero_point)
    # The unsqueeze(1) broadcasts the per-row scale/zp across all columns.
    w_quantized = torch.round(
        weight_fp32 / safe_scale.unsqueeze(1) + zero_points.unsqueeze(1)
    )
    w_quantized = torch.clamp(w_quantized, 0, max_val).to(torch.uint8)
    scales = safe_scale

    # --- Step 4: Pack two 4-bit values into one uint8 byte ---
    # We take pairs of adjacent values and combine them:
    #   - Even-indexed values go in the LOW nibble (bits 0-3)
    #   - Odd-indexed values go in the HIGH nibble (bits 4-7)
    #
    # Example: if we have values [5, 12, 3, 9, ...] at indices [0, 1, 2, 3, ...]
    #   Byte 0 = (12 << 4) | 5 = 0xC5 = 197
    #   Byte 1 = (9 << 4)  | 3 = 0x93 = 147
    even = w_quantized[:, 0::2]  # values at indices 0, 2, 4, ... → [output_dim, input_dim/2]
    odd = w_quantized[:, 1::2]   # values at indices 1, 3, 5, ... → [output_dim, input_dim/2]
    packed_uint8 = (odd << 4) | even  # combine into one byte each

    return packed_uint8, scales, zero_points


def dequantize_weights(packed_uint8: torch.Tensor, scales: torch.Tensor,
                       zero_points: torch.Tensor):
    """Dequantize packed INT4 weights back to FP32.

    This is the "decompression" step — the reverse of quantize_weights().
    It unpacks the 4-bit integers and converts them back to float values.

    Note: The reconstructed values won't be identical to the originals because
    quantization is lossy. With 4 bits, we only have 16 possible values per row,
    so there's rounding error. Typically the error is < 0.5 per element.

    Args:
        packed_uint8:  [output_dim, input_dim // 2] packed weight bytes
        scales:        [output_dim] per-channel scale factors
        zero_points:   [output_dim] per-channel zero points

    Returns:
        weight_fp32: [output_dim, input_dim] reconstructed float32 weights
    """
    # --- Step 1: Unpack the two 4-bit values from each byte ---
    # Low nibble (bits 0-3) = even-indexed original weight
    # High nibble (bits 4-7) = odd-indexed original weight
    #
    # Bitwise AND with 0x0F (binary 00001111) extracts the low nibble.
    # Right-shift by 4 extracts the high nibble.
    low = (packed_uint8 & 0x0F).to(torch.float32)   # even-indexed values
    high = (packed_uint8 >> 4).to(torch.float32)     # odd-indexed values

    # --- Step 2: Interleave back to the original column order ---
    # We need to place even-indexed values at positions 0, 2, 4, ...
    # and odd-indexed values at positions 1, 3, 5, ...
    output_dim = packed_uint8.shape[0]
    input_dim = packed_uint8.shape[1] * 2  # double because we unpacked
    weight_int = torch.empty(output_dim, input_dim, dtype=torch.float32,
                             device=packed_uint8.device)
    weight_int[:, 0::2] = low   # even positions
    weight_int[:, 1::2] = high  # odd positions

    # --- Step 3: Convert integers back to floats ---
    # Dequantization formula: w_float = (w_integer - zero_point) * scale
    #
    # For example, if scale=0.2, zero_point=10, and w_integer=13:
    #   w_float = (13 - 10) * 0.2 = 0.6
    #
    # unsqueeze(1) makes the per-row values broadcast across all columns.
    weight_fp32 = (weight_int - zero_points.unsqueeze(1)) * scales.unsqueeze(1)
    return weight_fp32


def reference_quantized_linear(input: torch.Tensor, packed_weights: torch.Tensor,
                               scales: torch.Tensor, zero_points: torch.Tensor):
    """Reference implementation: dequantize weights, then do a standard matrix multiply.

    This is the SLOW but CORRECT way to do quantized inference. It:
      1. Unpacks and dequantizes all weights to FP32 (materializing the full matrix)
      2. Calls PyTorch's standard F.linear (matrix multiplication)

    We use this as the "golden reference" to verify our fused CUDA kernel is correct.
    The CUDA kernel does both steps in one pass without ever creating the full FP32 matrix.

    Args:
        input:          [input_dim] or [batch, input_dim] — the input activations
        packed_weights: [output_dim, input_dim // 2] — packed 4-bit weights
        scales:         [output_dim] — per-row scale factors
        zero_points:    [output_dim] — per-row zero points

    Returns:
        output: [output_dim] or [batch, output_dim] — the result of input @ weights^T
    """
    # First, fully dequantize all weights back to FP32 (this is what we want to AVOID
    # in production — it creates a huge temporary matrix in memory)
    weight_fp32 = dequantize_weights(packed_weights, scales, zero_points)

    # F.linear computes: output = input @ weight^T
    # This is equivalent to: for each output neuron, dot product of input with that row
    return F.linear(input, weight_fp32)
