/*
 * Pybind11 bindings for MoE INT4 CUDA kernel
 */

#include "moe_int4_kernel.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(moe_int4_cuda, m) {
    m.def("forward", &moe_int4_forward,
          "Fused MoE INT4 dequantize + matmul forward (CUDA)",
          py::arg("packed_weights"),
          py::arg("scales"),
          py::arg("zero_points"),
          py::arg("inputs"),
          py::arg("expert_ids"),
          py::arg("tokens_per_expert"),
          py::arg("input_offsets"));
}
