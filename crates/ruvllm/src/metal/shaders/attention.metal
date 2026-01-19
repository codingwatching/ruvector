//
// Flash Attention 2 - Metal Compute Shader
// Optimized for Apple Silicon M4 Pro
//
// Memory-efficient attention using tiled computation with O(N) memory complexity.
// Uses online softmax for numerical stability.
//

#include <metal_stdlib>
using namespace metal;

// Constants optimized for M4 Pro (16KB threadgroup memory)
constant uint TILE_SIZE = 64;
constant uint HEAD_DIM_MAX = 128;
constant uint WARP_SIZE = 32;

// Attention parameters structure (matches Rust AttentionParams)
struct AttentionParams {
    uint num_heads;      // Number of query heads
    uint num_kv_heads;   // Number of key-value heads
    uint head_dim;       // Dimension per head
    uint seq_len;        // Query sequence length
    uint kv_len;         // Key-value sequence length
    float scale;         // Softmax scale (1/sqrt(head_dim))
    uint causal;         // Whether to apply causal mask
    uint _padding;       // Alignment padding
};

// Online softmax state
struct SoftmaxState {
    float max_val;
    float sum_exp;
};

// Update online softmax state
inline SoftmaxState update_softmax(SoftmaxState state, float new_val) {
    SoftmaxState new_state;
    if (new_val > state.max_val) {
        float exp_diff = exp(state.max_val - new_val);
        new_state.sum_exp = state.sum_exp * exp_diff + 1.0f;
        new_state.max_val = new_val;
    } else {
        new_state.sum_exp = state.sum_exp + exp(new_val - state.max_val);
        new_state.max_val = state.max_val;
    }
    return new_state;
}

// Flash Attention kernel
// Computes: output = softmax(Q @ K^T / scale) @ V
//
// Grid: (head_dim, num_heads, seq_len)
// Threadgroup: (head_dim, 1, 1)
kernel void flash_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Thread indices
    uint d = tid.x;           // Position within head dimension
    uint head = gid.y;        // Query head index
    uint seq_pos = gid.z;     // Query sequence position

    // Bounds check
    if (d >= params.head_dim || head >= params.num_heads || seq_pos >= params.seq_len) {
        return;
    }

    // GQA: map query head to KV head
    uint kv_head = head / (params.num_heads / params.num_kv_heads);

    // Shared memory for tiled computation
    threadgroup float shared_k[TILE_SIZE][HEAD_DIM_MAX];
    threadgroup float shared_v[TILE_SIZE][HEAD_DIM_MAX];
    threadgroup float shared_scores[TILE_SIZE];

    // Query offset: [seq_pos, head, d]
    uint q_offset = (seq_pos * params.num_heads + head) * params.head_dim + d;
    float q_val = query[q_offset];

    // Initialize online softmax and output accumulator
    SoftmaxState softmax_state = {-INFINITY, 0.0f};
    float output_acc = 0.0f;
    float prev_scale = 0.0f;

    // Number of tiles
    uint num_tiles = (params.kv_len + TILE_SIZE - 1) / TILE_SIZE;

    // Process KV in tiles
    for (uint tile = 0; tile < num_tiles; tile++) {
        uint tile_start = tile * TILE_SIZE;
        uint tile_end = min(tile_start + TILE_SIZE, params.kv_len);
        uint tile_len = tile_end - tile_start;

        // Cooperative load of K and V into shared memory
        for (uint t = 0; t < tile_len; t++) {
            uint kv_pos = tile_start + t;
            uint kv_offset = (kv_pos * params.num_kv_heads + kv_head) * params.head_dim + d;

            shared_k[t][d] = key[kv_offset];
            shared_v[t][d] = value[kv_offset];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention scores for this tile
        for (uint t = 0; t < tile_len; t++) {
            uint kv_pos = tile_start + t;

            // Apply causal mask
            if (params.causal && kv_pos > seq_pos) {
                continue;
            }

            // Compute Q.K^T with parallel reduction
            float dot = 0.0f;
            for (uint i = 0; i < params.head_dim; i++) {
                // Each thread computes partial dot product
                if (d == 0) {
                    dot += query[(seq_pos * params.num_heads + head) * params.head_dim + i] *
                           shared_k[t][i];
                }
            }

            // Only thread 0 updates softmax
            if (d == 0) {
                float score = dot * params.scale;

                // Update online softmax
                SoftmaxState new_state = update_softmax(softmax_state, score);

                // Rescale previous output if max changed
                if (new_state.max_val != softmax_state.max_val) {
                    float rescale = exp(softmax_state.max_val - new_state.max_val);
                    output_acc *= rescale;
                }

                // Compute attention weight
                float weight = exp(score - new_state.max_val);

                softmax_state = new_state;
                shared_scores[t] = weight;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate weighted values
        for (uint t = 0; t < tile_len; t++) {
            uint kv_pos = tile_start + t;

            if (params.causal && kv_pos > seq_pos) {
                continue;
            }

            output_acc += shared_scores[t] * shared_v[t][d];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by sum of exponentials
    if (softmax_state.sum_exp > 0.0f) {
        output_acc /= softmax_state.sum_exp;
    }

    // Write output: [seq_pos, head, d]
    uint out_offset = (seq_pos * params.num_heads + head) * params.head_dim + d;
    output[out_offset] = output_acc;
}

// Optimized Flash Attention with simdgroup operations
// Uses simd_sum for efficient reductions
kernel void flash_attention_simd(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint head = gid.y;
    uint seq_pos = gid.z;

    if (head >= params.num_heads || seq_pos >= params.seq_len) {
        return;
    }

    uint kv_head = head / (params.num_heads / params.num_kv_heads);

    // Each simd group processes part of the head dimension
    uint d_start = simd_group * WARP_SIZE;
    uint d = d_start + simd_lane;

    if (d >= params.head_dim) {
        return;
    }

    // Load query value for this dimension
    uint q_offset = (seq_pos * params.num_heads + head) * params.head_dim + d;
    float q_val = query[q_offset];

    // Online softmax state (per simd group)
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_val = 0.0f;

    // Process each KV position
    for (uint kv_pos = 0; kv_pos < params.kv_len; kv_pos++) {
        // Causal mask
        if (params.causal && kv_pos > seq_pos) {
            continue;
        }

        // Load K and V for this position
        uint kv_offset = (kv_pos * params.num_kv_heads + kv_head) * params.head_dim + d;
        float k_val = key[kv_offset];
        float v_val = value[kv_offset];

        // Compute dot product within simd group
        float partial_dot = q_val * k_val;
        float dot = simd_sum(partial_dot);

        // Scale
        float score = dot * params.scale;

        // Online softmax update
        if (score > max_score) {
            float exp_diff = exp(max_score - score);
            sum_exp = sum_exp * exp_diff + 1.0f;
            output_val *= exp_diff;
            max_score = score;
        } else {
            sum_exp += exp(score - max_score);
        }

        // Accumulate weighted value
        float weight = exp(score - max_score);
        output_val += weight * v_val;
    }

    // Normalize
    if (sum_exp > 0.0f) {
        output_val /= sum_exp;
    }

    // Write output
    uint out_offset = (seq_pos * params.num_heads + head) * params.head_dim + d;
    output[out_offset] = output_val;
}

// Softmax kernel (standalone for when needed separately)
kernel void softmax(
    device float* x [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    // Find max (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = tid; i < len; i += threads_per_group) {
        local_max = max(local_max, x[i]);
    }
    shared_max[tid] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_max[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < len; i += threads_per_group) {
        float exp_val = exp(x[i] - max_val);
        x[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum_val = shared_sum[0];

    // Normalize
    for (uint i = tid; i < len; i += threads_per_group) {
        x[i] /= sum_val;
    }
}

// Causal mask application
kernel void apply_causal_mask(
    device float* scores [[buffer(0)]],
    constant uint& seq_len [[buffer(1)]],
    constant uint& kv_len [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_pos = gid.y;
    uint k_pos = gid.x;

    if (q_pos >= seq_len || k_pos >= kv_len) {
        return;
    }

    if (k_pos > q_pos) {
        scores[q_pos * kv_len + k_pos] = -INFINITY;
    }
}
