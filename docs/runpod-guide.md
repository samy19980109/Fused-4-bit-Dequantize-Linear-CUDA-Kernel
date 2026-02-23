# Running the Fused 4-bit CUDA Kernel on RunPod

This guide walks you through renting a GPU on [RunPod](https://www.runpod.io/), building the CUDA extension, running tests, and collecting benchmark results.

---

## Table of Contents

1. [What is RunPod?](#1-what-is-runpod)
2. [Create an Account & Add Funds](#2-create-an-account--add-funds)
3. [Launch a GPU Pod](#3-launch-a-gpu-pod)
4. [Connect to Your Pod](#4-connect-to-your-pod)
5. [Set Up the Project](#5-set-up-the-project)
6. [Build the CUDA Extension](#6-build-the-cuda-extension)
7. [Run Correctness Tests](#7-run-correctness-tests)
8. [Run Benchmarks](#8-run-benchmarks)
9. [Download Results](#9-download-results)
10. [Stop Your Pod (Important!)](#10-stop-your-pod-important)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What is RunPod?

RunPod is a cloud GPU rental platform. You pay by the hour to access machines with NVIDIA GPUs — perfect for running CUDA code when you don't have a GPU locally. It's simpler and cheaper than AWS/GCP for quick GPU experiments.

**Cost estimate:** An RTX A4000 (16GB) costs ~$0.20/hr. You'll need 15-30 minutes, so budget ~$0.10-0.15.

---

## 2. Create an Account & Add Funds

1. Go to [runpod.io](https://www.runpod.io/) and sign up
2. Click **Billing** in the left sidebar
3. Add funds (minimum $5 via credit card or crypto)
   - $5 is more than enough for many hours of testing

---

## 3. Launch a GPU Pod

### Step 3a: Go to Pods

1. Click **Pods** in the left sidebar
2. Click **+ Deploy** button

### Step 3b: Choose a Template

Select the **RunPod PyTorch** template. This comes pre-installed with:
- CUDA toolkit
- PyTorch (with GPU support)
- Python 3
- Common ML libraries

> Make sure the template says **PyTorch 2.x** and **CUDA 12.x** (or 11.8+).

### Step 3c: Choose a GPU

For this project, any of these GPUs will work:

| GPU | VRAM | Approx. Cost/hr | Recommendation |
|-----|------|-----------------|----------------|
| RTX A4000 | 16 GB | ~$0.20 | Good enough, cheapest |
| RTX A5000 | 24 GB | ~$0.30 | Sweet spot |
| RTX 3090 | 24 GB | ~$0.30 | Great performance |
| A100 (40GB) | 40 GB | ~$1.00+ | Overkill but fastest |

**Recommendation:** Pick the cheapest available GPU (usually A4000 or RTX 3090). Our kernel doesn't need much VRAM — even 8GB is fine.

### Step 3d: Configure Storage

- **Container Disk:** 20 GB (default is fine)
- **Volume Disk:** 0 GB (we don't need persistent storage)

### Step 3e: Deploy

Click **Deploy On-Demand** (not spot — spot instances can be interrupted).

Wait 1-2 minutes for the pod to start. The status will change from "Starting" to "Running".

---

## 4. Connect to Your Pod

Once the pod is running, you have two options:

### Option A: Web Terminal (Easiest)

1. Click **Connect** on your running pod
2. Click **Start Web Terminal** → **Connect to Web Terminal**
3. A terminal opens in your browser — you're now on the GPU machine

### Option B: SSH (If You Prefer)

1. Click **Connect** on your running pod
2. Under **SSH over exposed TCP**, copy the SSH command
3. It looks like: `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`
4. You'll need to add your SSH public key in RunPod settings first:
   - Go to **Settings** → **SSH Public Keys** → paste your key

> **For beginners: use the Web Terminal.** It works immediately with no setup.

---

## 5. Set Up the Project

Once you're in the terminal on the RunPod machine, run these commands:

```bash
# Verify GPU is detected
nvidia-smi
```

You should see your GPU listed (e.g., "NVIDIA RTX A4000"). If not, something is wrong with the pod — try restarting it.

```bash
# Verify PyTorch sees the GPU
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
```

Expected output:
```
PyTorch 2.x.x
CUDA available: True
GPU: NVIDIA RTX A4000
```

Now clone the project:

```bash
# Clone the repo
cd /workspace
git clone https://github.com/samy19980109/Fused-4-bit-Dequantize-Linear-CUDA-Kernel.git
cd Fused-4-bit-Dequantize-Linear-CUDA-Kernel
```

> **If the repo is private**, you'll need to use a personal access token:
> ```bash
> git clone https://<YOUR_TOKEN>@github.com/samy19980109/Fused-4-bit-Dequantize-Linear-CUDA-Kernel.git
> ```
> Generate a token at: GitHub → Settings → Developer Settings → Personal Access Tokens

---

## 6. Build the CUDA Extension

This compiles our C++/CUDA code into a Python-importable module:

```bash
# Build and install the CUDA extension
pip install -e .
```

This runs `setup.py`, which:
1. Compiles `csrc/quantized_linear.cpp` with `g++`
2. Compiles `csrc/quantized_linear_kernel.cu` with `nvcc` (NVIDIA's CUDA compiler)
3. Links them into a shared library called `fused_quant_linear_cuda`

**Expected time:** 30-60 seconds. You'll see compiler output scrolling by.

**Success looks like:**
```
Successfully installed fused-quant-linear-cuda-0.0.0
```

**Verify the build:**
```bash
python -c "import fused_quant_linear_cuda; print('CUDA extension loaded successfully!')"
```

---

## 7. Run Correctness Tests

```bash
# Run all tests (correctness + benchmark smoke tests)
pytest tests/ -v
```

**Expected output:** All 24 tests should pass (including the 3 CUDA kernel tests that were skipped locally):

```
tests/test_correctness.py::TestCUDAKernel::test_cuda_matches_reference_1d PASSED
tests/test_correctness.py::TestCUDAKernel::test_cuda_matches_reference_batched PASSED
tests/test_correctness.py::TestCUDAKernel::test_cuda_large_dims PASSED
...
======================== 24 passed in X.XXs ========================
```

If any CUDA kernel tests fail, check the [Troubleshooting](#11-troubleshooting) section.

---

## 8. Run Benchmarks

```bash
# Install matplotlib for chart generation
pip install matplotlib

# Run the full benchmark
python benchmark/run_benchmark.py
```

This will:
1. Test three matrix sizes: (1024,1024), (4096,4096), (4096,11008)
2. Time both FP32 nn.Linear and INT4 QuantizedLinear
3. Print a results table with speedup and memory ratios
4. Print roofline analysis (arithmetic intensity, achieved bandwidth)
5. Save a bar chart to `benchmark/benchmark_results.png`

**Expected output (example on RTX A4000):**
```
Device: cuda
GPU: NVIDIA RTX A4000

              Size |   FP32 (ms) |   INT4 (ms) |  Speedup |   FP32 Mem |   INT4 Mem |  Mem Ratio
-------------------------------------------------------------------------------------
      (1024, 1024) |      0.XXX  |      0.XXX  |   X.XXx  |   4.00MB   |   0.52MB   |     7.7x
      (4096, 4096) |      X.XXX  |      X.XXX  |   X.XXx  |  64.00MB   |   8.26MB   |     7.7x
     (4096, 11008) |      X.XXX  |      X.XXX  |   X.XXx  | 172.00MB   |  22.20MB   |     7.7x

── Roofline Analysis ──
  ...
```

---

## 9. Download Results

### Option A: Copy from Web Terminal

If you used the Web Terminal, you can simply copy-paste the text output from the terminal.

### Option B: Download the Benchmark Chart

RunPod's web terminal doesn't support file downloads directly. Use one of these methods:

**Method 1: Use RunPod's file browser (easiest)**
1. In the RunPod pod page, click **Connect**
2. Click **Start Jupyter Notebook** → **Connect to Jupyter Notebook**
3. Navigate to the project folder in Jupyter's file browser
4. Find `benchmark/benchmark_results.png` and download it

**Method 2: SCP from terminal (if using SSH)**
```bash
# Run this on YOUR LOCAL machine (not the RunPod machine)
scp -P <port> root@<ip>:/workspace/Fused-4-bit-Dequantize-Linear-CUDA-Kernel/benchmark/benchmark_results.png ./
```

**Method 3: Push results to GitHub**
```bash
# On the RunPod machine
cd /workspace/Fused-4-bit-Dequantize-Linear-CUDA-Kernel

# Save benchmark output to a text file
python benchmark/run_benchmark.py > benchmark/results.txt 2>&1

# Commit and push
git add benchmark/benchmark_results.png benchmark/results.txt
git commit -m "Add benchmark results from RunPod (RTX A4000)"
git push origin main
```

> For git push, you'll need to configure credentials:
> ```bash
> git config user.email "your@email.com"
> git config user.name "Your Name"
> ```
> And either use a personal access token or SSH key for authentication.

---

## 10. Stop Your Pod (Important!)

**You are charged by the minute while the pod is running.** When you're done:

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods)
2. Click the **Stop** button (square icon) on your pod
3. If you're completely done, click **Terminate** (trash icon) to delete it

> **Stop** pauses billing but keeps your data. **Terminate** deletes everything.
> Since we don't need persistent data, **Terminate** is fine.

---

## 11. Troubleshooting

### "No module named torch"

The RunPod template didn't include PyTorch. Install it manually:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Build fails with "nvcc not found"

CUDA toolkit isn't in the PATH. Try:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip install -e .
```

### Build fails with "unsupported GNU version"

The CUDA toolkit version doesn't match the system GCC. Try:
```bash
# Check versions
nvcc --version
gcc --version

# If GCC is too new, install an older one
apt-get update && apt-get install -y gcc-11 g++-11
export CC=gcc-11 CXX=g++-11
pip install -e .
```

### CUDA kernel test fails with large error

If `test_cuda_large_dims` fails with error > 1e-2:
- This can happen with certain GPU architectures due to floating-point differences
- Try increasing the tolerance in the test, or verify the error is small relative to output magnitude

### "CUDA out of memory"

Your GPU doesn't have enough VRAM. Solutions:
- Use a smaller test size in the benchmark
- Restart the pod (clears GPU memory)
- Choose a GPU with more VRAM

### Permission denied when pushing to GitHub

Set up authentication:
```bash
# Using personal access token (recommended)
git remote set-url origin https://<YOUR_TOKEN>@github.com/samy19980109/Fused-4-bit-Dequantize-Linear-CUDA-Kernel.git
git push origin main
```

---

## Quick Reference: All Commands

```bash
# ── On RunPod machine ──

# 1. Verify GPU
nvidia-smi

# 2. Clone project
cd /workspace
git clone https://github.com/samy19980109/Fused-4-bit-Dequantize-Linear-CUDA-Kernel.git
cd Fused-4-bit-Dequantize-Linear-CUDA-Kernel

# 3. Build CUDA extension
pip install -e .

# 4. Verify build
python -c "import fused_quant_linear_cuda; print('OK')"

# 5. Run tests
pytest tests/ -v

# 6. Run benchmarks
pip install matplotlib
python benchmark/run_benchmark.py

# 7. (Optional) Save results to file
python benchmark/run_benchmark.py > benchmark/results.txt 2>&1

# 8. STOP YOUR POD when done!
```
