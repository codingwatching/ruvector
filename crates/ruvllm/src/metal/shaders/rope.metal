//
// Rotary Position Embeddings (RoPE) - Metal Compute Shader
// Optimized for Apple Silicon M4 Pro
//
// Applies rotary embeddings to query and key tensors for position encoding.
// Used in LLaMA, Mistral, and other modern transformer architectures.
//

#include <metal_stdlib>
using namespace metal;

// RoPE parameters structure (matches Rust RopeParams)
struct RopeParams {
    uint head_dim;      // Head dimension (must be even)
    uint num_heads;     // Number of heads
    uint position;      // Current position
    float theta_base;   // Base for frequency calculation (default 10000)
};

// Apply RoPE to a tensor
// Input shape: [batch, num_heads, head_dim]
//
// RoPE applies rotation:
//   x[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
//   x[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
//
// where theta = position * (theta_base ^ (-2i / head_dim))
//
// Grid: (head_dim, num_heads, batch)
// Threadgroup: (head_dim, 1, 1)
kernel void apply_rope(
    device float* x [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],  // Precomputed cos values
    device const float* sin_table [[buffer(2)]],  // Precomputed sin values
    constant RopeParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint d = gid.x;           // Position in head dimension
    uint head = gid.y;        // Head index
    uint batch = gid.z;       // Batch index

    uint head_dim = params.head_dim;
    uint num_heads = params.num_heads;

    // Only process pairs (even indices)
    if (d >= head_dim / 2) {
        return;
    }

    // Offset into the tensor
    uint offset = (batch * num_heads + head) * head_dim;

    // Get the pair of values
    float x0 = x[offset + 2 * d];
    float x1 = x[offset + 2 * d + 1];

    // Get precomputed cos/sin
    float cos_val = cos_table[d];
    float sin_val = sin_table[d];

    // Apply rotation
    x[offset + 2 * d] = x0 * cos_val - x1 * sin_val;
    x[offset + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
}

// Apply RoPE with inline frequency computation (no precomputed tables)
kernel void apply_rope_inline(
    device float* x [[buffer(0)]],
    constant RopeParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint d = gid.x;
    uint head = gid.y;
    uint batch = gid.z;

    uint head_dim = params.head_dim;
    uint num_heads = params.num_heads;
    uint position = params.position;
    float theta_base = params.theta_base;

    if (d >= head_dim / 2) {
        return;
    }

    uint offset = (batch * num_heads + head) * head_dim;

    // Compute frequency for this dimension
    float freq = 1.0f / pow(theta_base, float(2 * d) / float(head_dim));
    float angle = float(position) * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = x[offset + 2 * d];
    float x1 = x[offset + 2 * d + 1];

    x[offset + 2 * d] = x0 * cos_val - x1 * sin_val;
    x[offset + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
}

// Apply RoPE to multiple positions at once (for parallel token processing)
kernel void apply_rope_batched(
    device float* x [[buffer(0)]],                // [batch, seq_len, num_heads, head_dim]
    device const uint* positions [[buffer(1)]],   // [batch, seq_len] positions
    constant uint& num_heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant float& theta_base [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    uint d = gid.x;
    uint head = gid.y;
    uint seq_batch = gid.z;

    uint batch = seq_batch / seq_len;
    uint seq_pos = seq_batch % seq_len;

    if (d >= head_dim / 2) {
        return;
    }

    // Get the position for this token
    uint position = positions[batch * seq_len + seq_pos];

    // Compute offset
    uint offset = ((batch * seq_len + seq_pos) * num_heads + head) * head_dim;

    // Compute frequency
    float freq = 1.0f / pow(theta_base, float(2 * d) / float(head_dim));
    float angle = float(position) * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = x[offset + 2 * d];
    float x1 = x[offset + 2 * d + 1];

    x[offset + 2 * d] = x0 * cos_val - x1 * sin_val;
    x[offset + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
}

// FP16 RoPE for efficiency
kernel void apply_rope_f16(
    device half* x [[buffer(0)]],
    device const half* cos_table [[buffer(1)]],
    device const half* sin_table [[buffer(2)]],
    constant RopeParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint head = gid.y;
    uint batch = gid.z;

    if (d >= params.head_dim / 2) {
        return;
    }

    uint offset = (batch * params.num_heads + head) * params.head_dim;

    half x0 = x[offset + 2 * d];
    half x1 = x[offset + 2 * d + 1];

    half cos_val = cos_table[d];
    half sin_val = sin_table[d];

    x[offset + 2 * d] = x0 * cos_val - x1 * sin_val;
    x[offset + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
}

// Precompute RoPE cos/sin tables
kernel void precompute_rope_tables(
    device float* cos_table [[buffer(0)]],  // [max_seq_len, head_dim/2]
    device float* sin_table [[buffer(1)]],  // [max_seq_len, head_dim/2]
    constant uint& head_dim [[buffer(2)]],
    constant uint& max_seq_len [[buffer(3)]],
    constant float& theta_base [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pos = gid.y;
    uint d = gid.x;

    if (pos >= max_seq_len || d >= head_dim / 2) {
        return;
    }

    float freq = 1.0f / pow(theta_base, float(2 * d) / float(head_dim));
    float angle = float(pos) * freq;

    uint idx = pos * (head_dim / 2) + d;
    cos_table[idx] = cos(angle);
    sin_table[idx] = sin(angle);
}

// ALiBi (Attention with Linear Biases) - alternative to RoPE
// Adds linear bias based on position difference
kernel void apply_alibi(
    device float* attn_scores [[buffer(0)]],  // [batch, num_heads, seq_len, kv_len]
    constant uint& seq_len [[buffer(1)]],
    constant uint& kv_len [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint q_pos = gid.y;
    uint k_pos = gid.x;
    uint batch_head = gid.z;

    uint batch = batch_head / num_heads;
    uint head = batch_head % num_heads;

    if (q_pos >= seq_len || k_pos >= kv_len) {
        return;
    }

    // Compute ALiBi slope for this head
    // Slopes are typically: 2^(-8/num_heads), 2^(-16/num_heads), ...
    float slope = pow(2.0f, -8.0f * float(head + 1) / float(num_heads));

    // Compute position difference
    int pos_diff = int(q_pos) - int(k_pos);

    // Apply bias (negative for future positions in causal attention)
    float bias = slope * float(pos_diff);

    uint idx = ((batch * num_heads + head) * seq_len + q_pos) * kv_len + k_pos;
    attn_scores[idx] += bias;
}

// YaRN (Yet another RoPE extension) for extended context
// Supports position interpolation and NTK-aware scaling
struct YaRNParams {
    uint head_dim;
    uint num_heads;
    uint position;
    float theta_base;
    float scale;           // Position scale factor
    float attn_scale;      // Attention scale factor
    float beta_fast;       // High-frequency extrapolation factor
    float beta_slow;       // Low-frequency interpolation factor
    uint original_max_len; // Original training context length
};

kernel void apply_rope_yarn(
    device float* x [[buffer(0)]],
    constant YaRNParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint head = gid.y;
    uint batch = gid.z;

    if (d >= params.head_dim / 2) {
        return;
    }

    uint offset = (batch * params.num_heads + head) * params.head_dim;

    // YaRN frequency scaling
    float freq_base = 1.0f / pow(params.theta_base, float(2 * d) / float(params.head_dim));

    // Compute wavelength
    float wavelength = 2.0f * M_PI_F / freq_base;

    // Compute ramp function (linear interpolation between slow and fast)
    float low = float(params.original_max_len) / params.beta_fast;
    float high = float(params.original_max_len) / params.beta_slow;

    float ramp = 0.0f;
    if (wavelength < low) {
        ramp = 0.0f;  // High frequency: extrapolate
    } else if (wavelength > high) {
        ramp = 1.0f;  // Low frequency: interpolate
    } else {
        ramp = (wavelength - low) / (high - low);  // In between
    }

    // Scale frequency
    float freq = freq_base * (1.0f - ramp + ramp / params.scale);
    float angle = float(params.position) * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    float x0 = x[offset + 2 * d];
    float x1 = x[offset + 2 * d + 1];

    x[offset + 2 * d] = x0 * cos_val - x1 * sin_val;
    x[offset + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
}
