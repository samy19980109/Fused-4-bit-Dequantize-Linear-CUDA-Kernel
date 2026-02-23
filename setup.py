"""
Build script for the fused 4-bit CUDA kernel extension and MoE kernel.

=== PYTORCH C++ EXTENSIONS ===

PyTorch lets you write custom GPU kernels in CUDA C++ and call them from Python.
This build script compiles our .cpp and .cu files into shared libraries.

To build and install:
    python setup.py install

=== FILES ===
    csrc/quantized_linear.cpp     — Python ↔ C++ bridge (pybind11 bindings)
    csrc/quantized_linear_kernel.cu  — The actual GPU kernel code
    csrc/moe_int4.cpp            — MoE Python bindings
    csrc/moe_int4_kernel.cu      — MoE fused CUDA kernel
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_quant_linear_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_quant_linear_cuda",
            sources=[
                "csrc/quantized_linear.cpp",
                "csrc/quantized_linear_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
        CUDAExtension(
            name="moe_int4_cuda",
            sources=[
                "csrc/moe_int4_kernel.cu",  # Contains both kernel and bindings
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
