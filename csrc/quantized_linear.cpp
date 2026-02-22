/*
 * Pybind11 bindings — this file makes our C++/CUDA code callable from Python.
 *
 * === HOW PYBIND11 WORKS ===
 *
 * pybind11 is a library that creates Python modules from C++ code.
 * When you do `import fused_quant_linear_cuda` in Python, it loads the
 * shared library that this file (along with the .cu file) gets compiled into.
 *
 * PYBIND11_MODULE creates a Python module with the name TORCH_EXTENSION_NAME
 * (which is set to "fused_quant_linear_cuda" by our setup.py).
 *
 * m.def() registers a C++ function so Python can call it:
 *   fused_quant_linear_cuda.forward(input, packed_weights, scales, zero_points)
 *
 * The py::arg() calls give the Python function named parameters, so you can do:
 *   fused_quant_linear_cuda.forward(input=x, packed_weights=w, ...)
 */

#include "quantized_linear.h"  // declares quantized_linear_cuda_forward()

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &quantized_linear_cuda_forward,
          "Fused 4-bit dequantize + linear forward (CUDA)",
          py::arg("input"),
          py::arg("packed_weights"),
          py::arg("scales"),
          py::arg("zero_points"));
}
