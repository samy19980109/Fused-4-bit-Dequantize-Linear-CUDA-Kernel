/*
 * Fused 4-bit Dequantize + Linear CUDA Kernel
 *
 * ============================================================================
 * CUDA PROGRAMMING PRIMER (for those new to GPU programming)
 * ============================================================================
 *
 * A GPU has thousands of small cores that run the same code on different data.
 * You write a "kernel" function that runs on EACH core (called a "thread").
 *
 * Threads are organized into groups called "blocks" (up to 1024 threads each).
 * Blocks are organized into a "grid". You choose the grid/block sizes when
 * launching the kernel.
 *
 *   grid  = how many blocks to launch (can be 1D, 2D, or 3D)
 *   block = how many threads per block
 *
 * Each thread knows its position via built-in variables:
 *   threadIdx.x  — thread's index within its block (0 to blockDim.x - 1)
 *   blockIdx.x   — which block this thread is in (0 to gridDim.x - 1)
 *   blockIdx.y   — second dimension of the grid (we use this for batch index)
 *
 * === MEMORY HIERARCHY (from slowest to fastest) ===
 *
 *   Global Memory  (~800 GB/s) — Main GPU RAM. All threads can access it.
 *                                This is where our input/weight tensors live.
 *
 *   Shared Memory  (~10 TB/s)  — Small fast memory shared by threads in one block.
 *                                We load input vector tiles here so all 256 threads
 *                                can reuse the data without hitting global memory.
 *                                Declared with __shared__ keyword.
 *
 *   Registers      (fastest)   — Private to each thread. Our accumulator 'sum'
 *                                and scale/zp values live here.
 *
 * === WHAT THIS KERNEL DOES ===
 *
 * Computes: output[b][j] = sum_over_i( dequant(packed_weights[j][i]) * input[b][i] )
 *
 * For each output element, one thread:
 *   1. Loops over the input dimension in tiles
 *   2. Loads input tiles into shared memory (all threads cooperate)
 *   3. Loads packed weight bytes, extracts two 4-bit values per byte
 *   4. Dequantizes: w_float = (w_int - zero_point) * scale
 *   5. Multiplies with input and accumulates the dot product
 *
 * === KEY OPTIMIZATIONS ===
 *
 * 1. Shared Memory Caching: Instead of each thread reading the input vector
 *    from slow global memory, we load it once into fast shared memory.
 *    256 threads share one copy = 256x fewer global memory reads for input.
 *
 * 2. Vectorized Loads (uint4): Instead of loading weight bytes one at a time,
 *    we load 16 bytes at once using uint4. This gives us 32 weight values
 *    per load instruction, much more efficient than 32 separate byte loads.
 *
 * 3. Fused Multiply-Add (__fmaf_rn): Does (a * b + c) in a single instruction
 *    instead of separate multiply and add. Faster and more precise.
 *
 * 4. Loop Unrolling (#pragma unroll): Tells the compiler to unroll loops,
 *    eliminating loop overhead (counter increment, branch) at the cost of
 *    larger compiled code.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* === TUNING CONSTANTS === */

// BLOCK_SIZE: Number of threads per block.
// 256 is a common choice that gives good GPU "occupancy" (keeping cores busy).
// Each thread computes one output value, so one block handles 256 output neurons.
constexpr int BLOCK_SIZE = 256;

// TILE_SIZE: How many input elements we load into shared memory at once.
// 512 floats = 512 * 4 bytes = 2048 bytes of shared memory per block.
// GPUs typically have 48-164 KB of shared memory per block, so this is fine.
// We process TILE_SIZE/2 = 256 packed weight bytes per tile.
constexpr int TILE_SIZE = 512;


/*
 * The GPU kernel function. This code runs on every thread simultaneously.
 *
 * __global__ means this function is called from the CPU but runs on the GPU.
 * __restrict__ is a hint telling the compiler that pointers don't overlap,
 * enabling more aggressive optimization.
 */
__global__ void quantized_linear_optimized_kernel(
    const float* __restrict__ input,            // [batch, input_dim]
    const uint8_t* __restrict__ packed_weights,  // [output_dim, packed_dim]
    const float* __restrict__ scales,            // [output_dim]
    const float* __restrict__ zero_points,       // [output_dim]
    float* __restrict__ output,                  // [batch, output_dim]
    int input_dim,    // number of input features (e.g., 4096)
    int output_dim,   // number of output features (e.g., 11008)
    int packed_dim    // = input_dim / 2 (each byte holds 2 weights)
) {
    // --- Figure out which output element this thread is responsible for ---
    //
    // We launched the grid as: blocks = (ceil(output_dim/256), batch_size)
    //   blockIdx.y = which batch element (0, 1, 2, ...)
    //   blockIdx.x = which chunk of 256 output neurons
    //   threadIdx.x = which of the 256 threads in this block (0-255)
    //
    // So this thread computes: output[b][out_idx]
    int b = blockIdx.y;                          // batch index
    int out_base = blockIdx.x * BLOCK_SIZE;      // first output neuron for this block
    int tid = threadIdx.x;                       // thread's local index (0-255)
    int out_idx = out_base + tid;                // this thread's output neuron index

    // --- Shared memory declaration ---
    // __shared__ means this memory is shared between all 256 threads in this block.
    // It's much faster than global memory (~10x), so we use it to cache the input
    // vector so all threads can read it without hammering global memory.
    __shared__ float s_input[TILE_SIZE];

    // --- Load this thread's per-row quantization parameters into registers ---
    // Registers are the fastest memory (private to each thread).
    // Each output row has its own scale and zero_point.
    float scale = 0.0f, zp = 0.0f;
    if (out_idx < output_dim) {
        scale = scales[out_idx];
        zp = zero_points[out_idx];
    }

    // This is where we accumulate the dot product. It stays in a register
    // throughout the entire computation (very fast, no memory traffic).
    float sum = 0.0f;

    // Pointer to this batch element's input vector in global memory
    const float* inp = input + b * input_dim;

    // Pointer to this output neuron's weight row in the packed weight matrix.
    // Each row has packed_dim bytes (= input_dim/2 packed weight pairs).
    const uint8_t* w_row = (out_idx < output_dim)
                           ? packed_weights + out_idx * packed_dim
                           : nullptr;

    // --- TILED PROCESSING ---
    // The input vector might be very long (e.g., 4096 elements).
    // We can't fit it all in shared memory at once, so we process it in tiles.
    // Each tile loads TILE_SIZE (512) input floats into shared memory.
    int num_tiles = (input_dim + TILE_SIZE - 1) / TILE_SIZE;  // ceiling division

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_SIZE;       // where this tile starts in the input
        int tile_len = min(TILE_SIZE, input_dim - tile_start);  // handle last partial tile

        // --- COOPERATIVE LOADING INTO SHARED MEMORY ---
        // All 256 threads in this block work together to load the tile.
        // Thread 0 loads elements 0, 256, 512, ...
        // Thread 1 loads elements 1, 257, 513, ...
        // etc.
        // This is much faster than having one thread load everything.
        for (int i = tid; i < tile_len; i += BLOCK_SIZE) {
            s_input[i] = inp[tile_start + i];
        }

        // __syncthreads() is a barrier: ALL threads in this block must reach
        // this point before ANY thread continues past it. This ensures the
        // shared memory is fully loaded before we start reading from it.
        __syncthreads();

        // --- PROCESS THIS TILE'S WEIGHTS ---
        if (out_idx < output_dim) {
            // Calculate where this tile's weights start in the packed byte array.
            // Since 2 input elements are packed per byte: packed_offset = input_offset / 2
            int packed_tile_start = tile_start / 2;
            int packed_tile_len = tile_len / 2;  // number of packed bytes in this tile

            // === VECTORIZED WEIGHT LOADING ===
            //
            // Instead of loading weight bytes one at a time (slow), we use uint4
            // which loads 16 bytes (128 bits) in a single memory transaction.
            //
            // uint4 is a CUDA built-in type with four 32-bit fields: .x, .y, .z, .w
            // So one uint4 load gives us 16 bytes = 16 packed pairs = 32 weight values.
            //
            // This is like reading a book one page at a time vs one word at a time.
            int vec_iters = packed_tile_len / 16;    // how many 16-byte chunks we can do
            int vec_remainder = packed_tile_len % 16; // leftover bytes after vectorized loads

            // Reinterpret the byte pointer as a uint4 pointer for 16-byte loads.
            // This tells the GPU to use wide memory transactions.
            const uint4* w_vec = reinterpret_cast<const uint4*>(w_row + packed_tile_start);

            for (int v = 0; v < vec_iters; v++) {
                // Load 16 bytes in one shot! This is the key optimization.
                uint4 pack16 = w_vec[v];

                // uint4 has 4 fields, each holding 4 bytes (a 32-bit unsigned int):
                //   pack16.x = bytes 0-3   (8 weight values)
                //   pack16.y = bytes 4-7   (8 weight values)
                //   pack16.z = bytes 8-11  (8 weight values)
                //   pack16.w = bytes 12-15 (8 weight values)
                // Total: 32 weight values per uint4 load
                uint32_t chunks[4] = {pack16.x, pack16.y, pack16.z, pack16.w};

                // Each uint4 covers 32 input floats (16 bytes × 2 values/byte)
                int base_input_idx = v * 32;

                // #pragma unroll tells the compiler to unroll this loop at compile time.
                // Instead of: for c=0..3 { body }, it generates: body(0); body(1); body(2); body(3);
                // This eliminates loop overhead (counter, comparison, branch).
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    // 'word' is one 32-bit chunk = 4 packed bytes = 8 weight values
                    uint32_t word = chunks[c];

                    #pragma unroll
                    for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
                        // Extract one byte from the 32-bit word.
                        // Shift right by (byte_idx * 8) bits, then mask off the low 8 bits.
                        // byte_idx=0: bits 0-7,  byte_idx=1: bits 8-15,
                        // byte_idx=2: bits 16-23, byte_idx=3: bits 24-31
                        uint8_t packed_byte = (word >> (byte_idx * 8)) & 0xFF;

                        // Unpack the two 4-bit values from this byte:
                        //   Low nibble (bits 0-3) = even-indexed weight
                        //   High nibble (bits 4-7) = odd-indexed weight
                        float w0 = (float)(packed_byte & 0x0F);  // AND with 00001111
                        float w1 = (float)(packed_byte >> 4);     // shift right 4 bits

                        // Which input elements do these weights multiply with?
                        // Each chunk (c) covers 8 inputs, each byte covers 2 inputs.
                        int inp_idx = base_input_idx + c * 8 + byte_idx * 2;

                        // === DEQUANTIZE + MULTIPLY-ACCUMULATE ===
                        //
                        // __fmaf_rn(a, b, c) computes (a * b) + c in one instruction.
                        // "_rn" means "round to nearest" (standard rounding mode).
                        //
                        // Step 1: Dequantize the weight
                        //   dq0 = (w0 - zero_point) * scale + 0.0
                        //       = (w0 - zp) * scale
                        // Step 2: Multiply with input and add to running sum
                        //   sum = dq0 * s_input[inp_idx] + sum
                        float dq0 = __fmaf_rn(w0 - zp, scale, 0.0f);  // dequantize
                        float dq1 = __fmaf_rn(w1 - zp, scale, 0.0f);  // dequantize

                        sum = __fmaf_rn(dq0, s_input[inp_idx], sum);      // accumulate
                        sum = __fmaf_rn(dq1, s_input[inp_idx + 1], sum);  // accumulate
                    }
                }
            }

            // --- HANDLE REMAINDER ---
            // If packed_tile_len isn't a multiple of 16, process leftover bytes
            // one at a time using the same dequantize + accumulate logic.
            int rem_start = vec_iters * 16;
            for (int i = 0; i < vec_remainder; i++) {
                uint8_t packed_byte = w_row[packed_tile_start + rem_start + i];
                float w0 = (float)(packed_byte & 0x0F);
                float w1 = (float)(packed_byte >> 4);

                int inp_idx = (rem_start + i) * 2;
                float dq0 = __fmaf_rn(w0 - zp, scale, 0.0f);
                float dq1 = __fmaf_rn(w1 - zp, scale, 0.0f);

                sum = __fmaf_rn(dq0, s_input[inp_idx], sum);
                sum = __fmaf_rn(dq1, s_input[inp_idx + 1], sum);
            }
        }

        // Wait for all threads to finish with this tile's shared memory
        // before loading the next tile (otherwise we'd overwrite data
        // that some threads are still reading).
        __syncthreads();
    }

    // --- WRITE THE FINAL RESULT ---
    // After processing all tiles, 'sum' contains the complete dot product
    // for output[b][out_idx]. Write it to global memory.
    if (out_idx < output_dim) {
        output[b * output_dim + out_idx] = sum;
    }
}


/*
 * C++ wrapper function called from Python (via pybind11).
 *
 * This function:
 *   1. Validates inputs (correct types, shapes, device)
 *   2. Allocates the output tensor
 *   3. Configures and launches the CUDA kernel
 *   4. Returns the output tensor to Python
 *
 * It does NOT run on the GPU — it runs on the CPU and tells the GPU what to do.
 */
torch::Tensor quantized_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points
) {
    // Handle 1D input: if the user passes a single vector [input_dim],
    // temporarily add a batch dimension [1, input_dim] so the kernel code
    // can always assume 2D input. We'll squeeze it back at the end.
    bool squeeze = false;
    if (input.dim() == 1) {
        input = input.unsqueeze(0);  // [input_dim] → [1, input_dim]
        squeeze = true;
    }

    // --- INPUT VALIDATION ---
    // TORCH_CHECK is like assert but gives a clear error message.
    // All tensors must be on the GPU (CUDA), since our kernel runs on the GPU.
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(zero_points.is_cuda(), "zero_points must be a CUDA tensor");

    // Tensors must be contiguous in memory (no gaps or strides).
    // This is needed because our kernel accesses memory with pointer arithmetic,
    // which only works correctly for contiguous data layouts.
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(packed_weights.is_contiguous(), "packed_weights must be contiguous");

    // Check data types match what the kernel expects
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8, "packed_weights must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(zero_points.dtype() == torch::kFloat32, "zero_points must be float32");

    // Extract dimensions
    int batch_size = input.size(0);            // number of input vectors
    int input_dim = input.size(1);             // length of each input vector
    int output_dim = packed_weights.size(0);   // number of output neurons
    int packed_dim = packed_weights.size(1);   // = input_dim / 2

    TORCH_CHECK(packed_dim == input_dim / 2,
                "packed_weights dim 1 must be input_dim / 2");

    // Allocate the output tensor on the same device as the input
    auto output = torch::empty({batch_size, output_dim}, input.options());

    // --- CONFIGURE THE KERNEL LAUNCH ---
    //
    // dim3 is a CUDA type for specifying 1D/2D/3D dimensions.
    //
    // threads: 256 threads per block (each computes one output element).
    //
    // blocks: 2D grid where:
    //   x-dimension = ceil(output_dim / 256) — enough blocks to cover all output neurons
    //   y-dimension = batch_size — one "row" of blocks per batch element
    //
    // Example for output_dim=11008, batch_size=4:
    //   blocks = (44, 4) → 176 blocks total, each with 256 threads = 45056 threads
    //   (44 * 256 = 11264 ≥ 11008, so we have enough threads for all output neurons)
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((output_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);

    // Launch the kernel!
    // The <<<blocks, threads>>> syntax is CUDA's way of specifying the grid configuration.
    // This schedules the kernel to run on the GPU asynchronously (CPU doesn't wait).
    quantized_linear_optimized_kernel<<<blocks, threads>>>(
        // .data_ptr<T>() extracts a raw C pointer from a PyTorch tensor.
        // The kernel needs raw pointers because it's C/CUDA code, not Python.
        input.data_ptr<float>(),
        packed_weights.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zero_points.data_ptr<float>(),
        output.data_ptr<float>(),
        input_dim,
        output_dim,
        packed_dim
    );

    // If the input was originally 1D, remove the batch dimension we added
    if (squeeze) {
        output = output.squeeze(0);  // [1, output_dim] → [output_dim]
    }

    return output;
}
