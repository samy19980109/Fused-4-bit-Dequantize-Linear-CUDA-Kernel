#!/bin/bash
# RunPod Setup - Fused MoE INT4 CUDA Kernel
#
# This builds the REAL fused CUDA kernel that shows actual speedup!

set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH

echo "========================================"
echo "Fused MoE INT4 CUDA Kernel"
echo "========================================"

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install deps
pip install --upgrade pip triton matplotlib numpy 2>/dev/null

# Build kernels
echo ""
echo "Building CUDA kernels..."
python setup.py install

# Test fused kernel
echo ""
echo "Testing fused MoE kernel..."
python -c "import moe_int4_cuda; print('✓ MoE kernel loaded!')"

# Run benchmark
echo ""
echo "Running benchmark..."
python python/moe_int4_module.py

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
