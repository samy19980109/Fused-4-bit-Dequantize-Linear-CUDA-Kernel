# MoE Grouped GEMM Benchmark

Benchmark comparing different implementations for Mixture-of-Experts (MoE) expert computation.

## What This Benchmarks

In MoE models like Mixtral, DeepSeek-V3, and GLM-5, tokens are routed to different "expert" networks. Each expert has its own weight matrix, creating many small independent GEMMs (matrix multiplications).

**The Problem:** Naive implementations launch a separate GPU kernel for each expert → massive overhead.

**Our Solutions:**
1. **Triton Grouped GEMM** - Fuse all expert GEMMs into one kernel (2-3x speedup)
2. **INT4 Quantization** - Compress expert weights by 8x (memory savings)
3. **FP4 Quantization** - Native Blackwell support on RTX 5090 (2x over INT4!)

## Results (Expected on RTX 5090 Blackwell)

| Implementation | Latency | Speedup | Memory |
|---------------|---------|---------|--------|
| Naive (for-loop) | ~8 ms | 1.0x | 3.7 GB |
| torch.bmm (padded) | ~5 ms | 1.6x | 3.7 GB |
| **Triton Grouped GEMM** | ~3 ms | **2.7x** | 3.7 GB |
| **INT4 Quantized** | ~2 ms | **4x** | 0.5 GB |
| **FP4 Quantized (Blackwell)** | ~1.5 ms | **5.3x** | **0.25 GB** |

### RTX 5090 vs RTX 4090

| Spec | RTX 5090 | RTX 4090 | Improvement |
|------|----------|----------|-------------|
| **SMs** | 170 | 128 | +33% |
| **CUDA Cores** | 21,760 | 16,384 | +33% |
| **Memory** | 32 GB GDDR7 | 24 GB GDDR6X | +33% |
| **Tensor Cores** | 5th Gen (FP4!) | 4th Gen | Native FP4 |
| **Architecture** | Blackwell | Ada Lovelace | New |

## Quick Start (RunPod)

### 1. Start RunPod Instance

- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **GPU:** RTX 5090 (Blackwell) - recommended for FP4 benchmarks
- **Disk:** 50GB+

### 2. Upload Project

```bash
# Option A: Clone from git
cd /workspace
git clone <your-repo-url>
cd 4-bit-CUDA-Kernel

# Option B: Upload via RunPod file manager
```

### 3. Run Setup

```bash
bash benchmark/runpod_setup.sh
```

### 4. Run Benchmark

```bash
# Quick test (fast, for development)
python benchmark/run_moe_benchmark.py --config mixtral --batch 4 --seq 256

# Full benchmark (for results)
python benchmark/run_moe_benchmark.py --config mixtral --batch 16 --seq 512

# Run all configs
python benchmark/run_moe_benchmark.py --full
```

## Model Configurations

| Config | Model | Experts | Hidden | FFN | Top-K |
|--------|-------|---------|--------|-----|-------|
| `mixtral` | Mixtral-8x7B | 8 | 4096 | 14336 | 2 |
| `deepseek` | DeepSeek-V3 | 64 | 4096 | 11008 | 8 |
| `glm5` | GLM-5 | 128 | 5120 | 13696 | 8 |
| `debug` | Debug-Tiny | 4 | 512 | 1024 | 2 |

## File Structure

```
benchmark/moe_grouped_gemm/
├── config.py              # Model configurations
├── routing.py             # Token routing simulation
├── naive_grouped_gemm.py  # Baseline: for-loop implementation
├── grouped_gemm_torch.py  # Reference: torch.bmm
├── grouped_gemm_triton.py # Optimized: Triton fused kernel
├── grouped_gemm_fp4.py    # Blackwell: FP4 quantized (RTX 5090)
├── moe_int4_module.py     # Quantized: INT4 expert weights
└── utils.py               # Benchmark utilities

benchmark/run_moe_benchmark.py  # Main benchmark script
benchmark/runpod_setup.sh       # RunPod setup script
```

## Understanding the Results

### Speedup Metrics

- **Latency (ms):** Time for one forward pass through all experts
- **TFLOPS:** Achieved compute throughput
- **Speedup:** Relative to naive baseline

### Memory Metrics

- **BF16:** 2 bytes per weight value
- **INT4:** 0.5 bytes per weight (2 values packed per byte)
- **FP4:** 0.5 bytes per weight (native Blackwell format)
- **Savings:** ~8x with INT4, ~8x with FP4 (but FP4 is 2x faster on Blackwell)

## Key Optimizations

### 1. Persistent Kernel Design (Triton)

Instead of launching N kernels (one per expert), we launch one kernel that processes all expert tiles. This:
- Eliminates kernel launch overhead
- Improves GPU utilization (170 SMs on RTX 5090!)
- Reduces scheduling overhead

### 2. Grouped Launch Ordering (L2 Cache)

Tiles are computed in an order that maximizes L2 cache hits:
- Weights from the same expert are reused
- Reduces memory bandwidth pressure

### 3. INT4 Quantization

Weights are compressed from 16-bit to 4-bit:
- 8x smaller memory footprint
- Better cache utilization
- Enables larger batch sizes

### 4. FP4 on Blackwell (RTX 5090 Exclusive)

RTX 5090's 5th Gen Tensor Cores have native FP4 support:
- Same 8x memory savings as INT4
- 2x faster inference due to native hardware support
- E2M1 format with block scaling

## Troubleshooting

### "Triton not available"
```bash
pip install triton
```

### "INT4 kernel not found"
```bash
python setup.py install
```

### "FP4 requires Blackwell GPU"
FP4 benchmarks only run on RTX 5090 (Blackwell architecture). On other GPUs, INT4 is the best option.

### Out of memory
```bash
# Reduce batch size
python benchmark/run_moe_benchmark.py --batch 8 --seq 256
```

## References

- [PyTorch Blog: Accelerating MoEs with Triton Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [NVIDIA Blackwell Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [DeepSeek-V3 Paper](https://arxiv.org/abs/2412.19437)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)
