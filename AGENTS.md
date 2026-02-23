# AGENTS.md — Agent Coding Guidelines

This file provides guidelines for agentic coding agents working in this repository.

---

## 1. Build, Test & Development Commands

### Build CUDA Kernels
```bash
# Build and install the Python package with CUDA extensions
python setup.py install

# Force rebuild (clean build artifacts)
rm -rf build/ *.egg-info/ && python setup.py install
```

### Run Tests
```bash
# Run all correctness tests
pytest tests/test_correctness.py -v

# Run a single test
pytest tests/test_correctness.py::TestCUDAKernel::test_cuda_kernel_matches_reference -v

# Run benchmark tests
pytest tests/test_benchmark.py -v

# Run MoE benchmark (requires CUDA)
python python/moe_int4_module.py

# Run linear benchmark
python benchmark/run_benchmark.py

# Run MoE Python benchmark
python benchmark/run_moe_benchmark.py --config mixtral --batch 16 --seq 512
```

### Linting & Formatting
```bash
# Install ruff for linting
pip install ruff

# Run linter
ruff check .

# Format code
ruff format .

# Type checking (if mypy installed)
pip install mypy
mypy python/ --ignore-missing-imports
```

---

## 2. Code Style Guidelines

### Python Files

#### Imports
- Use absolute imports within the project
- Order: stdlib → third-party → project local
- Example:
  ```python
  import sys
  import time
  import torch
  import pytest
  from python.quantize import quantize_weights
  ```

#### Naming Conventions
- **Variables/functions**: `snake_case` (e.g., `packed_weights`, `dequantize_weights`)
- **Classes**: `PascalCase` (e.g., `QuantizedLinear`, `MoEINT4`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `BLOCK_SIZE = 256`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_helper`)

#### Type Hints
- Use type hints for function signatures
- Prefer `List[int]` over `list[int]` for Python 3.9 compatibility
- Example:
  ```python
  def quantize_weights(weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      ...
  ```

#### Error Handling
- Use descriptive error messages
- Validate inputs early
- Example:
  ```python
  if weight.shape[1] % 2 != 0:
      raise ValueError(f"in_features must be even, got {weight.shape[1]}")
  ```

#### Documentation
- Use docstrings for public functions/classes
- Explain "why" not just "what" in comments
- Keep comments concise

#### Formatting
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- One blank line between top-level definitions

---

## 3. CUDA Kernel Guidelines (`.cu` files)

### Header Comments
Each CUDA kernel file should have a header explaining:
- What the kernel computes
- Key optimizations used
- Memory access patterns

Example:
```c
/*
 * Fused INT4 Dequantize + Linear Kernel
 *
 * Key optimizations:
 * 1. Shared memory input caching
 * 2. Vectorized uint4 loads
 * 3. Fused multiply-add
 */
```

### Kernel Structure
- Use `__global__` for kernels launched from Python
- Use `__shared__` for shared memory arrays
- Prefer `constexpr` for compile-time constants
- Use descriptive variable names

### Memory Access
- Prefer vectorized loads (e.g., `uint4` for 16-byte loads)
- Coalesce memory accesses
- Use shared memory for data reused within a block

### Synchronization
- Use `__syncthreads()` carefully (only when all threads need to sync)
- Avoid unnecessary synchronization

---

## 4. Project Structure

```
4-bit-CUDA-Kernel/
├── setup.py                 # Build script (CUDAExtension)
├── csrc/
│   ├── quantized_linear_kernel.cu   # Linear layer CUDA kernel
│   ├── quantized_linear.cpp         # Pybind11 bindings
│   ├── quantized_linear.h           # Header
│   └── moe_int4_kernel.cu         # MoE CUDA kernel
├── python/
│   ├── __init__.py
│   ├── module.py                   # QuantizedLinear nn.Module
│   ├── quantize.py                # Quantization utilities
│   └── moe_int4_module.py         # MoE module
├── tests/
│   ├── test_correctness.py        # Correctness tests
│   └── test_benchmark.py          # Performance tests
└── benchmark/
    ├── run_benchmark.py           # Linear benchmark
    └── run_moe_benchmark.py       # MoE benchmark
```

---

## 5. Key Libraries & Patterns

### PyTorch CUDA Extension
- Use `torch.utils.cpp_extension.CUDAExtension` for CUDA kernels
- Use `pybind11` for Python bindings
- Kernel functions use `torch::Tensor` for array arguments

### Testing
- Use `pytest` for all tests
- Use `torch.cuda.Event` for precise GPU timing
- Use `torch.testing.assert_close` for floating-point comparisons

### Quantization
- Asymmetric per-channel quantization
- Formula: `w_fp = (w_int4 - zero_point) * scale`
- Pack two 4-bit values into one uint8

---

## 6. Common Tasks

### Adding a New CUDA Kernel
1. Write kernel in `.cu` file with pybind11 bindings
2. Add to `setup.py` as new `CUDAExtension`
3. Build with `python setup.py install`
4. Test with pytest

### Adding a Test
1. Add test class/method to appropriate file in `tests/`
2. Use descriptive test names: `test_<what>_<condition>`
3. Include docstring explaining what/why

### Debugging CUDA Issues
```python
# Enable CUDA synchronization for debugging
torch.cuda.synchronize()

# Check for illegal memory access
CUDA_LAUNCH_BLOCKING=1 python test.py
```

---

## 7. GPU Environment

- **CUDA**: Version 12.x required
- **PyTorch**: 2.0+
- **Python**: 3.8+
- **Build**: Requires `nvcc` (NVIDIA CUDA Compiler)

---

## 8. Important Notes

- CUDA kernels must be compiled with `python setup.py install` before use
- Changes to `.cu`/`.cpp` files require rebuild
- Tests requiring GPU are automatically skipped if CUDA unavailable
- Use `torch.no_grad()` for inference-only code
