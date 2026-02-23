"""
MoE Grouped GEMM Benchmark Package.

Provides implementations for comparing MoE expert computation approaches:
- Naive for-loop (baseline)
- torch.bmm padded
- Triton fused kernel
- INT4 quantized
"""

from .config import (
    MoEConfig,
    MIXTRAL_8x7B,
    DEEPSEEK_V3,
    GLM5,
    QWEN3_235B,
    DEBUG_CONFIG,
    get_config_by_name,
    get_all_configs,
)

from .routing import (
    simulate_routing,
    create_expert_inputs,
    combine_expert_outputs,
    get_expert_sizes_for_benchmark,
)

from .utils import (
    BenchmarkResult,
    timer_ms,
    format_bytes,
    format_time,
    get_gpu_info,
    print_benchmark_table,
)

__all__ = [
    "MoEConfig",
    "MIXTRAL_8x7B",
    "DEEPSEEK_V3",
    "GLM5",
    "QWEN3_235B",
    "DEBUG_CONFIG",
    "get_config_by_name",
    "get_all_configs",
    "simulate_routing",
    "create_expert_inputs",
    "combine_expert_outputs",
    "get_expert_sizes_for_benchmark",
    "BenchmarkResult",
    "timer_ms",
    "format_bytes",
    "format_time",
    "get_gpu_info",
    "print_benchmark_table",
]
