//
// Normalization Kernels - Metal Compute Shader
// Optimized for Apple Silicon M4 Pro
//
// Includes:
// - RMSNorm (Root Mean Square Layer Normalization)
// - LayerNorm (Layer Normalization)
// - BatchNorm (Batch Normalization)
//

#include <metal_stdlib>
using namespace metal;

// Normalization parameters structure (matches Rust NormParams)
struct NormParams {
    uint hidden_size;       // Hidden dimension
    float eps;              // Epsilon for numerical stability
    uint elements_per_thread;  // Elements per thread for distribution
    uint _padding;          // Alignment padding
};

// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
// This is the normalization used in LLaMA, Mistral, etc.
//
// Grid: (hidden_size, batch_size, 1)
// Threadgroup: (min(hidden_size, 1024), 1, 1)
kernel void rms_norm(
    device float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant NormParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.y;
    uint hidden_size = params.hidden_size;
    float eps = params.eps;

    // Offset to this batch element
    uint offset = batch_idx * hidden_size;

    // Shared memory for parallel reduction
    threadgroup float shared_sum[1024];

    // Step 1: Compute sum of squares (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = x[offset + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within threadgroup
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < threads_per_group) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute RMS
    float rms = sqrt(shared_sum[0] / float(hidden_size) + eps);
    float inv_rms = 1.0f / rms;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Normalize and apply weight
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        x[offset + i] = x[offset + i] * inv_rms * weight[i];
    }
}

// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
// Standard layer normalization with optional bias
kernel void layer_norm(
    device float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],  // Can be nullptr (all zeros)
    constant NormParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.y;
    uint hidden_size = params.hidden_size;
    float eps = params.eps;

    uint offset = batch_idx * hidden_size;

    threadgroup float shared_sum[1024];
    threadgroup float shared_sum_sq[1024];

    // Step 1: Compute mean and variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = x[offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < threads_per_group) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(hidden_size);
    float var = shared_sum_sq[0] / float(hidden_size) - mean * mean;
    float inv_std = rsqrt(var + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Normalize, scale, and shift
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float normalized = (x[offset + i] - mean) * inv_std;
        float bias_val = bias ? bias[i] : 0.0f;
        x[offset + i] = normalized * weight[i] + bias_val;
    }
}

// RMSNorm with fused residual addition
// Computes: output = RMSNorm(x + residual) * weight
// And also stores the updated residual
kernel void rms_norm_residual(
    device float* x [[buffer(0)]],           // Input (will be modified in-place)
    device float* residual [[buffer(1)]],    // Residual (read and updated)
    device const float* weight [[buffer(2)]],
    constant NormParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.y;
    uint hidden_size = params.hidden_size;
    float eps = params.eps;

    uint offset = batch_idx * hidden_size;

    threadgroup float shared_sum[1024];

    // Step 1: Add residual and compute sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = x[offset + i] + residual[offset + i];
        // Store the sum back to residual for next layer
        residual[offset + i] = val;
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < threads_per_group) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared_sum[0] / float(hidden_size) + eps);
    float inv_rms = 1.0f / rms;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Normalize and apply weight
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        x[offset + i] = residual[offset + i] * inv_rms * weight[i];
    }
}

// FP16 RMSNorm for efficiency
kernel void rms_norm_f16(
    device half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    constant NormParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.y;
    uint hidden_size = params.hidden_size;
    half eps = half(params.eps);

    uint offset = batch_idx * hidden_size;

    threadgroup float shared_sum[1024];  // Use float for reduction accuracy

    // Compute sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        float val = float(x[offset + i]);
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < threads_per_group) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    half inv_rms = half(rsqrt(shared_sum[0] / float(hidden_size) + float(eps)));

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize and apply weight
    for (uint i = tid; i < hidden_size; i += threads_per_group) {
        x[offset + i] = x[offset + i] * inv_rms * weight[i];
    }
}

// Group RMSNorm (for channel-first tensors)
// Normalizes over groups of channels
kernel void group_rms_norm(
    device float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& num_groups [[buffer(2)]],
    constant uint& channels_per_group [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.z;
    uint group_idx = gid.y;
    uint spatial_idx = gid.x;

    uint channels = num_groups * channels_per_group;
    uint group_offset = group_idx * channels_per_group;

    threadgroup float shared_sum[256];

    // Compute sum of squares for this group
    float local_sum = 0.0f;
    for (uint c = tid; c < channels_per_group; c += threads_per_group) {
        uint idx = batch_idx * channels + group_offset + c;
        float val = x[idx];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(channels_per_group) + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize
    for (uint c = tid; c < channels_per_group; c += threads_per_group) {
        uint idx = batch_idx * channels + group_offset + c;
        x[idx] = x[idx] * inv_rms * weight[group_offset + c];
    }
}
