//! High-level Metal operations
//!
//! Provides convenient wrappers around Metal compute operations.

use super::{MetalContext, MetalConfig, AttentionParams, GemmParams, NormParams, RopeParams};
use crate::error::{Result, RuvLLMError};
use crate::kernels::AttentionConfig;

/// Batch matrix multiplication with Metal
///
/// Computes batched C = A @ B for multiple matrices.
pub fn batched_gemm_metal(
    ctx: &MetalContext,
    a: &[f32],
    b: &[f32],
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if a.len() != batch_size * m * k {
        return Err(RuvLLMError::InvalidOperation(format!(
            "Batched GEMM A size mismatch: {} != {}",
            a.len(),
            batch_size * m * k
        )));
    }
    if b.len() != batch_size * k * n {
        return Err(RuvLLMError::InvalidOperation(format!(
            "Batched GEMM B size mismatch: {} != {}",
            b.len(),
            batch_size * k * n
        )));
    }

    let mut results = Vec::with_capacity(batch_size * m * n);

    // Process each batch
    for batch in 0..batch_size {
        let a_start = batch * m * k;
        let a_end = a_start + m * k;
        let b_start = batch * k * n;
        let b_end = b_start + k * n;

        let c = ctx.gemm_f32(&a[a_start..a_end], &b[b_start..b_end], m, n, k)?;
        results.extend_from_slice(&c);
    }

    Ok(results)
}

/// Fused attention operation
///
/// Computes attention with fused softmax for efficiency.
pub fn fused_attention_metal(
    ctx: &MetalContext,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    // Validate inputs
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;

    if query.len() % q_size != 0 {
        return Err(RuvLLMError::InvalidOperation(format!(
            "Query size {} not divisible by head size {}",
            query.len(),
            q_size
        )));
    }

    ctx.flash_attention(query, key, value, config)
}

/// Layer normalization with Metal
pub fn layer_norm_metal(
    ctx: &MetalContext,
    x: &mut [f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<()> {
    // RMSNorm as base
    ctx.rms_norm(x, weight, eps)?;

    // Apply bias if provided
    if let Some(bias) = bias {
        for (xi, &bi) in x.iter_mut().zip(bias.iter()) {
            *xi += bi;
        }
    }

    Ok(())
}

/// Fused MLP operation
///
/// Computes: output = down_proj(silu(gate_proj(x)) * up_proj(x))
pub fn fused_mlp_metal(
    ctx: &MetalContext,
    x: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<Vec<f32>> {
    let batch_size = x.len() / hidden_size;

    // Gate projection: x @ gate_weight^T
    let gate = ctx.gemm_f32(x, gate_weight, batch_size, intermediate_size, hidden_size)?;

    // Up projection: x @ up_weight^T
    let up = ctx.gemm_f32(x, up_weight, batch_size, intermediate_size, hidden_size)?;

    // SiLU and multiply
    let mut hidden: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu = g / (1.0 + (-g).exp());
            silu * u
        })
        .collect();

    // Down projection: hidden @ down_weight^T
    ctx.gemm_f32(&hidden, down_weight, batch_size, hidden_size, intermediate_size)
}

/// Convert FP32 to FP16
pub fn fp32_to_fp16(data: &[f32]) -> Vec<half::f16> {
    data.iter().map(|&x| half::f16::from_f32(x)).collect()
}

/// Convert FP16 to FP32
pub fn fp16_to_fp32(data: &[half::f16]) -> Vec<f32> {
    data.iter().map(|x| x.to_f32()).collect()
}

/// Quantize to INT8 with scale
pub fn quantize_int8(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = max_abs / 127.0;
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| (x * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (quantized, scale)
}

/// Dequantize from INT8
pub fn dequantize_int8(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| x as f32 * scale).collect()
}

/// Memory-efficient attention with chunking
///
/// Processes attention in chunks to reduce peak memory usage.
pub fn chunked_attention_metal(
    ctx: &MetalContext,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    config: &AttentionConfig,
    chunk_size: usize,
) -> Result<Vec<f32>> {
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let seq_len = query.len() / q_size;
    let kv_len = key.len() / kv_size;

    if seq_len <= chunk_size {
        // No chunking needed
        return ctx.flash_attention(query, key, value, config);
    }

    let mut output = vec![0.0f32; query.len()];

    // Process in chunks
    for chunk_start in (0..seq_len).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(seq_len);
        let chunk_len = chunk_end - chunk_start;

        let q_start = chunk_start * q_size;
        let q_end = chunk_end * q_size;
        let chunk_query = &query[q_start..q_end];

        let chunk_config = AttentionConfig {
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            max_seq_len: chunk_len,
            causal: config.causal,
            scale: config.scale,
        };

        let chunk_output = ctx.flash_attention(chunk_query, key, value, &chunk_config)?;

        output[q_start..q_end].copy_from_slice(&chunk_output);
    }

    Ok(output)
}

/// Speculative decoding helper
///
/// Verifies draft tokens against target model.
pub fn verify_speculative_tokens(
    draft_logits: &[f32],
    target_logits: &[f32],
    vocab_size: usize,
    num_draft_tokens: usize,
) -> (usize, Vec<usize>) {
    let mut accepted = Vec::with_capacity(num_draft_tokens);

    for i in 0..num_draft_tokens {
        let draft_start = i * vocab_size;
        let target_start = i * vocab_size;

        // Find argmax for both
        let draft_token = draft_logits[draft_start..draft_start + vocab_size]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let target_token = target_logits[target_start..target_start + vocab_size]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if draft_token == target_token {
            accepted.push(draft_token);
        } else {
            // First mismatch - accept target token and stop
            accepted.push(target_token);
            break;
        }
    }

    (accepted.len(), accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_conversion() {
        let data = vec![1.0f32, 2.0, -3.0, 0.5];
        let fp16 = fp32_to_fp16(&data);
        let back = fp16_to_fp32(&fp16);

        for (orig, converted) in data.iter().zip(back.iter()) {
            assert!((orig - converted).abs() < 0.01);
        }
    }

    #[test]
    fn test_int8_quantization() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let (quantized, scale) = quantize_int8(&data);
        let dequantized = dequantize_int8(&quantized, scale);

        for (orig, converted) in data.iter().zip(dequantized.iter()) {
            assert!((orig - converted).abs() < 0.02);
        }
    }

    #[test]
    fn test_speculative_verification() {
        let vocab_size = 10;
        let num_tokens = 3;

        // Draft: tokens 5, 3, 7
        let mut draft_logits = vec![0.0f32; vocab_size * num_tokens];
        draft_logits[5] = 10.0;
        draft_logits[vocab_size + 3] = 10.0;
        draft_logits[2 * vocab_size + 7] = 10.0;

        // Target: tokens 5, 3, 2 (mismatch at position 2)
        let mut target_logits = vec![0.0f32; vocab_size * num_tokens];
        target_logits[5] = 10.0;
        target_logits[vocab_size + 3] = 10.0;
        target_logits[2 * vocab_size + 2] = 10.0;

        let (num_accepted, tokens) = verify_speculative_tokens(
            &draft_logits,
            &target_logits,
            vocab_size,
            num_tokens,
        );

        assert_eq!(num_accepted, 3); // 2 accepted + 1 target correction
        assert_eq!(tokens, vec![5, 3, 2]);
    }
}
