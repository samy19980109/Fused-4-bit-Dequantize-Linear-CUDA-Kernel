#!/bin/bash
# RunPod Setup Script for MoE Benchmark
# 
# Usage:
#   1. Start RunPod with image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   2. Upload this project or clone from git
#   3. Run: bash benchmark/runpod_setup.sh

set -e

echo "========================================"
echo "MoE Benchmark Setup for RunPod"
echo "========================================"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
echo "To run full benchmark:"
echo "  python benchmark/run_moe_benchmark.py --config mixtral --batch 16 --seq 512"
echo ""
echo "To run quick test:"
echo "  python benchmark/run_moe_benchmark.py --config mixtral --batch 4 --seq 256"
echo ""
