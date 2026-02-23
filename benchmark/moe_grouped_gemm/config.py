"""
MoE Model Configurations for Benchmarking.

These configs match real SOTA models' expert layer dimensions.
Each config defines the shape of expert weight matrices in the FFN.

MoE Architecture:
    Input [batch, seq, hidden_dim]
        │
        ├──► Router (selects top-k experts per token)
        │
        ├──► Expert 0: Linear(hidden_dim, ffn_dim)
        ├──► Expert 1: Linear(hidden_dim, ffn_dim)
        ├──► ...
        └──► Expert N: Linear(hidden_dim, ffn_dim)
                │
                ▼
        Combine (weighted by router scores)
                │
                ▼
    Output [batch, seq, hidden_dim]

The bottleneck we're optimizing:
    - Each expert has weights [ffn_dim, hidden_dim]
    - Tokens are distributed unevenly across experts
    - Naive: Launch N separate GEMM kernels → massive overhead
    - Optimized: Single fused Grouped GEMM kernel
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MoEConfig:
    """Configuration for a Mixture-of-Experts layer."""

    name: str  # Model name
    num_experts: int  # Number of expert networks
    hidden_dim: int  # Input/output dimension (d_model)
    ffn_dim: int  # Expert intermediate dimension
    top_k: int  # Number of experts activated per token

    @property
    def expert_weight_shape(self) -> Tuple[int, int]:
        """Shape of each expert's weight matrix: [ffn_dim, hidden_dim]"""
        return (self.ffn_dim, self.hidden_dim)

    @property
    def total_params_per_expert(self) -> int:
        """Parameters in one expert (just the up projection for simplicity)."""
        return self.ffn_dim * self.hidden_dim

    @property
    def total_expert_params(self) -> int:
        """Total parameters across all experts."""
        return self.total_params_per_expert * self.num_experts

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.num_experts} experts, "
            f"hidden={self.hidden_dim}, ffn={self.ffn_dim}, top-{self.top_k}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REAL MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

MIXTRAL_8x7B = MoEConfig(
    name="Mixtral-8x7B",
    num_experts=8,
    hidden_dim=4096,
    ffn_dim=14336,
    top_k=2,
)

DEEPSEEK_V3 = MoEConfig(
    name="DeepSeek-V3",
    num_experts=64,
    hidden_dim=4096,
    ffn_dim=11008,
    top_k=8,
)

GLM5 = MoEConfig(
    name="GLM-5",
    num_experts=128,
    hidden_dim=5120,
    ffn_dim=13696,
    top_k=8,
)

QWEN3_235B = MoEConfig(
    name="Qwen3-235B",
    num_experts=64,
    hidden_dim=4096,
    ffn_dim=11008,
    top_k=8,
)

# Small config for quick testing/debugging
DEBUG_CONFIG = MoEConfig(
    name="Debug-Tiny",
    num_experts=4,
    hidden_dim=512,
    ffn_dim=1024,
    top_k=2,
)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    moe_config: MoEConfig
    batch_size: int
    seq_len: int
    warmup_iters: int = 50
    bench_iters: int = 100

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def tokens_per_expert_approx(self) -> int:
        """Approximate tokens per expert (assuming uniform distribution)."""
        return self.total_tokens * self.moe_config.top_k // self.moe_config.num_experts

    def __str__(self) -> str:
        return (
            f"{self.moe_config.name} | "
            f"batch={self.batch_size}, seq={self.seq_len}, "
            f"tokens={self.total_tokens}"
        )


# Standard benchmark configs for Mixtral-8x7B
MIXTRAL_BENCHMARK_CONFIGS = [
    BenchmarkConfig(MIXTRAL_8x7B, batch_size=1, seq_len=512),
    BenchmarkConfig(MIXTRAL_8x7B, batch_size=8, seq_len=512),
    BenchmarkConfig(MIXTRAL_8x7B, batch_size=16, seq_len=512),
    BenchmarkConfig(MIXTRAL_8x7B, batch_size=32, seq_len=512),
]

# Quick test config (fast iteration during development)
QUICK_TEST_CONFIG = BenchmarkConfig(
    moe_config=MIXTRAL_8x7B,
    batch_size=4,
    seq_len=256,
    warmup_iters=10,
    bench_iters=20,
)


def get_config_by_name(name: str) -> MoEConfig:
    """Get a model config by name."""
    configs = {
        "mixtral": MIXTRAL_8x7B,
        "mixtral_8x7b": MIXTRAL_8x7B,
        "deepseek": DEEPSEEK_V3,
        "deepseek_v3": DEEPSEEK_V3,
        "glm5": GLM5,
        "qwen3": QWEN3_235B,
        "debug": DEBUG_CONFIG,
    }
    key = name.lower().replace("-", "_")
    if key not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
    return configs[key]


def get_all_configs() -> List[MoEConfig]:
    """Get all available model configs."""
    return [MIXTRAL_8x7B, DEEPSEEK_V3, GLM5, QWEN3_235B]
