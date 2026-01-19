//
// GEMM (General Matrix Multiplication) - Metal Compute Shader
// Optimized for Apple Silicon M4 Pro with simdgroup_matrix
//
// Computes C = alpha * A @ B + beta * C
// Supports FP16 for 2x throughput on M4 Pro tensor cores
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Tile sizes optimized for M4 Pro L1 cache (128KB) and threadgroup memory (16KB)
constant uint TILE_M = 64;
constant uint TILE_N = 64;
constant uint TILE_K = 32;

// SIMD group matrix dimensions (8x8 for half precision)
constant uint SIMD_M = 8;
constant uint SIMD_N = 8;
constant uint SIMD_K = 8;

// GEMM parameters structure (matches Rust GemmParams)
struct GemmParams {
    uint m;      // Rows of A and C
    uint n;      // Columns of B and C
    uint k;      // Columns of A, rows of B
    uint lda;    // Leading dimension of A
    uint ldb;    // Leading dimension of B
    uint ldc;    // Leading dimension of C
    float alpha; // Scale factor for A @ B
    float beta;  // Scale factor for C
};

// FP16 GEMM using simdgroup_matrix (M4 Pro tensor cores)
// Grid: (tiles_n, tiles_m, 1)
// Threadgroup: (TILE_M, TILE_N/8, 1)
kernel void gemm_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Tile coordinates
    uint tile_m = gid.y;
    uint tile_n = gid.x;

    // Global row/col this thread is responsible for
    uint row = tile_m * TILE_M + tid.y;
    uint col = tile_n * TILE_N + tid.x * 8 + simd_lane % 8;

    // Bounds check
    if (row >= params.m || col >= params.n) {
        return;
    }

    // Shared memory for tiled multiplication
    threadgroup half shared_a[TILE_M][TILE_K];
    threadgroup half shared_b[TILE_K][TILE_N];

    // Accumulator fragments (simdgroup_matrix for 8x8 multiplication)
    simdgroup_half8x8 c_frag;
    c_frag = simdgroup_half8x8(0.0h);

    // Number of K tiles
    uint num_k_tiles = (params.k + TILE_K - 1) / TILE_K;

    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        uint k_start = k_tile * TILE_K;

        // Cooperative loading of A tile
        for (uint i = tid.y; i < TILE_M; i += TILE_M / 8) {
            for (uint j = tid.x; j < TILE_K; j += TILE_N / 8) {
                uint a_row = tile_m * TILE_M + i;
                uint a_col = k_start + j;
                if (a_row < params.m && a_col < params.k) {
                    shared_a[i][j] = A[a_row * params.lda + a_col];
                } else {
                    shared_a[i][j] = 0.0h;
                }
            }
        }

        // Cooperative loading of B tile
        for (uint i = tid.y; i < TILE_K; i += TILE_M / 8) {
            for (uint j = tid.x; j < TILE_N; j += TILE_N / 8) {
                uint b_row = k_start + i;
                uint b_col = tile_n * TILE_N + j;
                if (b_row < params.k && b_col < params.n) {
                    shared_b[i][j] = B[b_row * params.ldb + b_col];
                } else {
                    shared_b[i][j] = 0.0h;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute using simdgroup_matrix multiply-accumulate
        for (uint k = 0; k < TILE_K; k += SIMD_K) {
            simdgroup_half8x8 a_frag;
            simdgroup_half8x8 b_frag;

            // Load A fragment (8x8 block)
            simdgroup_load(a_frag, &shared_a[tid.y * 8][k], TILE_K);

            // Load B fragment (8x8 block)
            simdgroup_load(b_frag, &shared_b[k][tid.x * 8], TILE_N);

            // Multiply-accumulate
            simdgroup_multiply_accumulate(c_frag, a_frag, b_frag, c_frag);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result with alpha/beta scaling
    half alpha_h = half(params.alpha);
    half beta_h = half(params.beta);

    // Write back 8x8 result tile
    for (uint i = 0; i < 8; i++) {
        for (uint j = 0; j < 8; j++) {
            uint out_row = tile_m * TILE_M + tid.y * 8 + i;
            uint out_col = tile_n * TILE_N + tid.x * 8 + j;

            if (out_row < params.m && out_col < params.n) {
                uint out_idx = out_row * params.ldc + out_col;
                half old_val = beta_h != 0.0h ? C[out_idx] : 0.0h;
                C[out_idx] = alpha_h * c_frag[i][j] + beta_h * old_val;
            }
        }
    }
}

// FP32 GEMM (fallback for accuracy-critical operations)
kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Calculate tile position
    uint tile_m = gid.y / 16;
    uint tile_n = gid.x / 16;

    uint local_row = tid.y;
    uint local_col = tid.x;

    uint row = tile_m * 16 + local_row;
    uint col = tile_n * 16 + local_col;

    if (row >= params.m || col >= params.n) {
        return;
    }

    // Shared memory tiles
    threadgroup float shared_a[16][32];
    threadgroup float shared_b[32][16];

    float sum = 0.0f;

    // Process K in tiles
    uint num_k_tiles = (params.k + 31) / 32;

    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        uint k_start = k_tile * 32;

        // Load A tile (16 rows, 32 cols)
        for (uint j = local_col; j < 32; j += 16) {
            uint a_col = k_start + j;
            if (a_col < params.k) {
                shared_a[local_row][j] = A[row * params.lda + a_col];
            } else {
                shared_a[local_row][j] = 0.0f;
            }
        }

        // Load B tile (32 rows, 16 cols)
        for (uint i = local_row; i < 32; i += 16) {
            uint b_row = k_start + i;
            if (b_row < params.k) {
                shared_b[i][local_col] = B[b_row * params.ldb + col];
            } else {
                shared_b[i][local_col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        #pragma unroll
        for (uint k = 0; k < 32; k++) {
            sum += shared_a[local_row][k] * shared_b[k][local_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store with alpha/beta scaling
    uint out_idx = row * params.ldc + col;
    float old_val = params.beta != 0.0f ? C[out_idx] : 0.0f;
    C[out_idx] = params.alpha * sum + params.beta * old_val;
}

// Batched GEMM for attention score computation
kernel void batched_gemm_f32(
    device const float* A [[buffer(0)]],  // [batch, m, k]
    device const float* B [[buffer(1)]],  // [batch, k, n]
    device float* C [[buffer(2)]],        // [batch, m, n]
    constant uint4& dims [[buffer(3)]],   // (m, n, k, batch)
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    uint m = dims.x;
    uint n = dims.y;
    uint k = dims.z;
    uint num_batches = dims.w;

    if (batch >= num_batches || row >= m || col >= n) {
        return;
    }

    // Compute offset for this batch
    uint a_offset = batch * m * k;
    uint b_offset = batch * k * n;
    uint c_offset = batch * m * n;

    // Compute dot product
    float sum = 0.0f;
    for (uint i = 0; i < k; i++) {
        sum += A[a_offset + row * k + i] * B[b_offset + i * n + col];
    }

    C[c_offset + row * n + col] = sum;
}

// Vector-matrix multiplication (for single-token generation)
kernel void gemv_f32(
    device const float* x [[buffer(0)]],      // [k]
    device const float* W [[buffer(1)]],      // [n, k]
    device float* y [[buffer(2)]],            // [n]
    constant uint2& dims [[buffer(3)]],       // (n, k)
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint n = dims.x;
    uint k = dims.y;

    if (gid >= n) {
        return;
    }

    // Each thread computes one output element
    float sum = 0.0f;

    #pragma unroll 4
    for (uint i = 0; i < k; i++) {
        sum += x[i] * W[gid * k + i];
    }

    y[gid] = sum;
}

// Element-wise operations
kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < len) {
        c[gid] = a[gid] + b[gid];
    }
}

kernel void elementwise_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < len) {
        c[gid] = a[gid] * b[gid];
    }
}

// SiLU activation: x * sigmoid(x)
kernel void silu(
    device float* x [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < len) {
        float val = x[gid];
        x[gid] = val / (1.0f + exp(-val));
    }
}

// Fused SiLU + multiply (for MLP)
kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < len) {
        float g = gate[gid];
        float silu_g = g / (1.0f + exp(-g));
        out[gid] = silu_g * up[gid];
    }
}
