"""
Correctness tests for the 4-bit quantization pipeline.

=== TEST STRATEGY ===

We test at three levels:

1. Round-trip tests (TestQuantizeRoundTrip):
   Quantize weights to 4-bit, then dequantize back to FP32.
   The reconstructed values should be close to the originals.
   With 4 bits we only have 16 discrete levels per row, so some
   error is expected — we allow up to 0.5 absolute error per element.

2. Reference linear tests (TestReferenceLinear):
   Verify that our Python reference_quantized_linear() produces the
   same output as manually dequantizing + calling F.linear().
   Also check that quantized outputs are close to full-precision outputs.

3. CUDA kernel tests (TestCUDAKernel):
   Compare the CUDA kernel's output against the Python reference.
   These should match very closely (atol=1e-3) since they compute
   the same math, just in different languages/on different hardware.
   These tests are skipped if no GPU is available.

=== RUNNING ===
    pytest tests/test_correctness.py -v
"""

import torch
import pytest
import sys
import os

# Add project root to path so we can import the python package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.quantize import quantize_weights, dequantize_weights, reference_quantized_linear


class TestQuantizeRoundTrip:
    """Test that quantize -> dequantize preserves weights within tolerance.

    Why 0.5 tolerance?
      With 4 bits per value, we have 16 quantization levels spread across each
      row's [min, max] range. For randn weights (range roughly -3 to +3),
      each step is about 6/15 ≈ 0.4. Rounding to the nearest step introduces
      at most half a step of error ≈ 0.2, but we use 0.5 for safety margin.
    """

    def test_round_trip_small(self):
        """Small matrix: 16 output neurons, 32 inputs each."""
        torch.manual_seed(42)
        weight = torch.randn(16, 32)
        packed, scales, zp = quantize_weights(weight)
        reconstructed = dequantize_weights(packed, scales, zp)
        assert reconstructed.shape == weight.shape
        assert torch.allclose(weight, reconstructed, atol=0.5), \
            f"Max error: {(weight - reconstructed).abs().max().item()}"

    def test_round_trip_large(self):
        """Medium matrix: 256 x 512."""
        torch.manual_seed(123)
        weight = torch.randn(256, 512)
        packed, scales, zp = quantize_weights(weight)
        reconstructed = dequantize_weights(packed, scales, zp)
        assert torch.allclose(weight, reconstructed, atol=0.5), \
            f"Max error: {(weight - reconstructed).abs().max().item()}"

    def test_round_trip_llm_dims(self):
        """LLM-scale matrix: 4096 x 4096 (typical hidden dimension)."""
        torch.manual_seed(7)
        weight = torch.randn(4096, 4096)
        packed, scales, zp = quantize_weights(weight)
        reconstructed = dequantize_weights(packed, scales, zp)
        assert torch.allclose(weight, reconstructed, atol=0.5), \
            f"Max error: {(weight - reconstructed).abs().max().item()}"

    def test_packing_shape(self):
        """Verify that packing halves the column dimension (2 values per byte)."""
        weight = torch.randn(64, 128)
        packed, scales, zp = quantize_weights(weight)
        assert packed.shape == (64, 64)  # 128 / 2 = 64 packed bytes per row
        assert scales.shape == (64,)     # one scale per output row
        assert zp.shape == (64,)         # one zero_point per output row

    def test_packed_values_in_range(self):
        """Each packed byte should be a valid uint8 (0-255)."""
        weight = torch.randn(32, 64)
        packed, _, _ = quantize_weights(weight)
        # A uint8 byte holding two 4-bit values (0-15 each) can be 0 to 255
        assert packed.max() <= 255
        assert packed.min() >= 0

    def test_constant_row(self):
        """Edge case: rows where all weights are identical.

        This is tricky because scale = (max - min) / 15 = 0, causing division
        by zero. Our code handles this with a special case.
        """
        weight = torch.ones(4, 8) * 3.0
        packed, scales, zp = quantize_weights(weight)
        reconstructed = dequantize_weights(packed, scales, zp)
        assert not torch.isnan(reconstructed).any(), "Got NaN values!"
        assert torch.allclose(weight, reconstructed, atol=0.5)


class TestReferenceLinear:
    """Test reference_quantized_linear against F.linear with dequantized weights."""

    def test_matches_f_linear(self):
        """Our reference should give identical results to manual dequantize + F.linear.

        This verifies that reference_quantized_linear is just a convenience wrapper
        and doesn't introduce any bugs of its own.
        """
        torch.manual_seed(42)
        weight = torch.randn(64, 128)
        x = torch.randn(128)
        packed, scales, zp = quantize_weights(weight)

        # Method 1: our reference function
        out_ref = reference_quantized_linear(x, packed, scales, zp)

        # Method 2: manually dequantize, then use PyTorch's F.linear
        weight_deq = dequantize_weights(packed, scales, zp)
        out_manual = torch.nn.functional.linear(x, weight_deq)

        # These should be essentially identical (just floating point rounding)
        assert torch.allclose(out_ref, out_manual, atol=1e-5)

    def test_batched_input(self):
        """Verify batched inputs (multiple vectors at once) work correctly."""
        torch.manual_seed(42)
        weight = torch.randn(64, 128)
        x = torch.randn(8, 128)  # batch of 8 input vectors
        packed, scales, zp = quantize_weights(weight)

        out = reference_quantized_linear(x, packed, scales, zp)
        assert out.shape == (8, 64)  # 8 inputs × 64 output neurons

    def test_accuracy_vs_fp32(self):
        """Compare quantized output quality against full-precision output.

        We check two metrics:
        1. Absolute error: the raw difference in output values
        2. Cosine similarity: whether the output vector points in the same direction
           (this is more meaningful than element-wise error for neural networks)
        """
        torch.manual_seed(42)
        weight = torch.randn(256, 512)
        x = torch.randn(512)

        # Full-precision (FP32) output — the "ground truth"
        out_fp32 = torch.nn.functional.linear(x, weight)

        # Quantized output — what our system produces
        packed, scales, zp = quantize_weights(weight)
        out_q = reference_quantized_linear(x, packed, scales, zp)

        # Absolute error check: 4-bit quantization adds noise, but it should be bounded.
        # With 512-dim dot products and randn weights, expect ~O(1) absolute error.
        abs_error = (out_fp32 - out_q).abs()
        assert abs_error.mean() < 3.0, f"Mean absolute error: {abs_error.mean().item()}"

        # Cosine similarity: measures whether the output vectors "point the same way".
        # 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.
        # For 4-bit quantization, we expect > 0.95 (very similar directions).
        cos_sim = torch.nn.functional.cosine_similarity(out_fp32.unsqueeze(0), out_q.unsqueeze(0))
        assert cos_sim > 0.95, f"Cosine similarity: {cos_sim.item()}"


# === CUDA KERNEL TESTS ===
# These only run if a GPU is available and the CUDA extension is built.

# pytest.mark.skipif decorator: skips the entire test class if the condition is True.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _try_import_cuda_ext():
    """Try to import the compiled CUDA extension. Skip the test if not built."""
    try:
        import fused_quant_linear_cuda
        return fused_quant_linear_cuda
    except ImportError:
        pytest.skip("fused_quant_linear_cuda extension not built — run: python setup.py install")


@requires_cuda
class TestCUDAKernel:
    """Test CUDA kernel output against Python reference.

    The CUDA kernel and Python reference compute the same math:
        output = dequantize(packed_weights) @ input
    They should produce nearly identical results (within floating-point tolerance).

    We use atol=1e-3 because floating-point operations on GPU can differ slightly
    from CPU due to different instruction ordering and fused multiply-add behavior.
    """

    def test_cuda_matches_reference_1d(self):
        """Single vector input: verify CUDA kernel matches Python reference."""
        ext = _try_import_cuda_ext()
        torch.manual_seed(42)
        weight = torch.randn(64, 128)
        x = torch.randn(128)
        packed, scales, zp = quantize_weights(weight)

        # Python reference (runs on CPU)
        ref = reference_quantized_linear(x, packed, scales, zp)

        # CUDA kernel (runs on GPU) — move all tensors to GPU with .cuda()
        cuda_out = ext.forward(
            x.cuda(), packed.cuda(), scales.cuda(), zp.cuda()
        )

        # Compare: move GPU result back to CPU with .cpu() for comparison
        assert torch.allclose(ref, cuda_out.cpu(), atol=1e-3), \
            f"Max diff: {(ref - cuda_out.cpu()).abs().max().item()}"

    def test_cuda_matches_reference_batched(self):
        """Batched input (4 vectors): verify CUDA handles batches correctly."""
        ext = _try_import_cuda_ext()
        torch.manual_seed(42)
        weight = torch.randn(256, 512)
        x = torch.randn(4, 512)  # 4 input vectors
        packed, scales, zp = quantize_weights(weight)

        ref = reference_quantized_linear(x, packed, scales, zp)
        cuda_out = ext.forward(
            x.cuda(), packed.cuda(), scales.cuda(), zp.cuda()
        )
        assert torch.allclose(ref, cuda_out.cpu(), atol=1e-3), \
            f"Max diff: {(ref - cuda_out.cpu()).abs().max().item()}"

    def test_cuda_large_dims(self):
        """LLM-scale test (4096 x 4096): verify correctness at scale.

        Uses a slightly larger tolerance (1e-2) because larger dot products
        accumulate more floating-point rounding error.
        """
        ext = _try_import_cuda_ext()
        torch.manual_seed(7)
        weight = torch.randn(4096, 4096)
        x = torch.randn(4096)
        packed, scales, zp = quantize_weights(weight)

        ref = reference_quantized_linear(x, packed, scales, zp)
        cuda_out = ext.forward(
            x.cuda(), packed.cuda(), scales.cuda(), zp.cuda()
        )
        assert torch.allclose(ref, cuda_out.cpu(), atol=1e-2), \
            f"Max diff: {(ref - cuda_out.cpu()).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
