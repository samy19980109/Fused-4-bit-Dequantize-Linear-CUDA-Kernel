"""
Benchmark smoke tests — verify QuantizedLinear runs correctly at various sizes.

These are NOT performance benchmarks (see benchmark/run_benchmark.py for that).
Instead, they quickly verify that:
  1. The module produces outputs of the correct shape
  2. No NaN values appear in the output
  3. Memory savings are as expected (~8x byte reduction)

Run with: pytest tests/test_benchmark.py -v
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python import QuantizedLinear


# Various matrix sizes to test, from small to LLM-scale
SIZES = [(128, 64), (512, 256), (1024, 1024), (4096, 4096)]


class TestBenchmarkSmoke:
    """Verify QuantizedLinear runs without error at typical sizes.

    @pytest.mark.parametrize runs each test method once per (in_dim, out_dim) pair.
    So test_forward_cpu runs 4 times: with (128,64), (512,256), (1024,1024), (4096,4096).
    """

    @pytest.mark.parametrize("in_dim,out_dim", SIZES)
    def test_forward_cpu(self, in_dim, out_dim):
        """Single vector forward pass produces correct shape, no NaN."""
        torch.manual_seed(42)
        linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        ql = QuantizedLinear.from_linear(linear)
        x = torch.randn(in_dim)
        out = ql(x)
        assert out.shape == (out_dim,)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("in_dim,out_dim", SIZES)
    def test_forward_batched_cpu(self, in_dim, out_dim):
        """Batched forward pass (4 vectors) produces correct shape, no NaN."""
        torch.manual_seed(42)
        linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        ql = QuantizedLinear.from_linear(linear)
        x = torch.randn(4, in_dim)  # batch of 4
        out = ql(x)
        assert out.shape == (4, out_dim)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("in_dim,out_dim", SIZES)
    def test_memory_reduction(self, in_dim, out_dim):
        """Verify that packed weights are ~8x smaller than FP32 weights.

        Math:
          FP32: each weight = 4 bytes (float32)
          INT4 packed: two weights per byte = 0.5 bytes per weight
          Ratio: 4 / 0.5 = 8x

        We check >= 7.5x to account for the small overhead of scales and zero_points.
        """
        linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        ql = QuantizedLinear.from_linear(linear)

        # FP32: number_of_weights × 4 bytes per float
        fp32_bytes = linear.weight.nelement() * 4

        # INT4 packed: each uint8 element stores 2 weight values
        int4_bytes = ql.packed_weights.nelement()  # already in bytes (uint8 = 1 byte)

        # Should achieve ~8x byte reduction
        assert fp32_bytes / int4_bytes >= 7.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
