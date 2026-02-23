#!/bin/bash
# RunPod Setup Script for MoE Benchmark
# 
# Usage:
#   1. Start RunPod with image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   2. Upload this project or clone from git
#   3. Run: bash benchmark/runpod_setup.sh
#
# RTX 5090 (Blackwell) Key Specs:
#   - 170 SMs (Streaming Multiprocessors)
#   - 21,760 CUDA Cores
#   - 32 GB GDDR7 Memory
#   - 5th Gen Tensor Cores with native FP4 support
#   - Blackwell Architecture (Compute Capability 10.x+)

set -e

echo "========================================"
echo "MoE Benchmark Setup for RunPod"
echo "Optimized for RTX 5090 (Blackwell)"
echo "========================================"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Detect GPU type
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo ""
echo "Detected GPU: $GPU_NAME"

if [[ "$GPU_NAME" == *"5090"* ]]; then
    echo "✓ RTX 5090 detected - FP4 benchmarks will be available!"
    echo "  - 170 SMs available"
    echo "  - Native FP4 Tensor Core support"
elif [[ "$GPU_NAME" == *"4090"* ]]; then
    echo "RTX 4090 detected - INT4 benchmarks will be available"
    echo "  - 128 SMs available"
    echo "  - FP4 not supported (requires Blackwell)"
else
    echo "Other GPU detected - some features may not be available"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install triton matplotlib numpy

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import triton; print(f'Triton: {triton.__version__}')"

# Check compute capability for FP4
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Compute Capability: {props.major}.{props.minor}')
    # Estimate SMs based on compute capability
    sm_count = props.multi_processor_count if hasattr(props, 'multi_processor_count') else 'Unknown'
    print(f'SMs: {sm_count}')
    if props.major >= 10:
        print('✓ Blackwell architecture detected - FP4 supported!')
    else:
        print('Pre-Blackwell architecture - FP4 not supported, INT4 available')
"

# Build existing INT4 kernel
echo ""
echo "Building INT4 CUDA kernel..."
cd /workspace/4-bit-CUDA-Kernel
python setup.py install

# Verify kernel
echo ""
echo "Verifying INT4 kernel..."
python -c "import fused_quant_linear_cuda; print('INT4 kernel loaded successfully')"

# Quick test
echo ""
echo "Running quick test..."
python benchmark/run_moe_benchmark.py --config mixtral --batch 4 --seq 256 --warmup 5 --iters 10

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "RTX 5090 Optimizations Available:"
echo "  - Triton Grouped GEMM (170 SMs)"
echo "  - INT4 Quantized MoE"
echo "  - FP4 Quantized GEMM (Blackwell native)"
echo ""
echo "To run full benchmark:"
echo "  python benchmark/run_moe_benchmark.py --config mixtral --batch 16 --seq 512"
echo ""
echo "To run quick test:"
echo "  python benchmark/run_moe_benchmark.py --config mixtral --batch 4 --seq 256"
echo ""
