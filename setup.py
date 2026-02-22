"""
Build script for the fused 4-bit CUDA kernel extension.

=== HOW PYTORCH C++ EXTENSIONS WORK ===

PyTorch lets you write custom GPU kernels in CUDA C++ and call them from Python.
This build script compiles our .cpp and .cu files into a shared library that
Python can import as a regular module.

To build and install:
    python setup.py install

After building, you can do:
    import fused_quant_linear_cuda
    output = fused_quant_linear_cuda.forward(input, weights, scales, zero_points)

=== FILE ROLES ===
    csrc/quantized_linear.cpp     — Python ↔ C++ bridge (pybind11 bindings)
    csrc/quantized_linear_kernel.cu  — The actual GPU kernel code (runs on GPU)
    csrc/quantized_linear.h       — Shared header declaring the function signature

=== COMPILER FLAGS ===
    -O3:             Maximum optimization level (makes code run faster)
    --use_fast_math: Allows the GPU compiler to use faster but slightly less precise
                     math operations (fine for inference, not for scientific computing)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_quant_linear_cuda",
    ext_modules=[
        CUDAExtension(
            # This becomes the Python module name: import fused_quant_linear_cuda
            name="fused_quant_linear_cuda",
            sources=[
                "csrc/quantized_linear.cpp",       # pybind11 bindings (compiled with g++)
                "csrc/quantized_linear_kernel.cu",  # CUDA kernel (compiled with nvcc)
            ],
            extra_compile_args={
                "cxx": ["-O3"],                       # C++ compiler flags
                "nvcc": ["-O3", "--use_fast_math"],   # NVIDIA CUDA compiler flags
            },
        ),
    ],
    # BuildExtension handles the tricky parts of compiling CUDA code
    cmdclass={"build_ext": BuildExtension},
)
