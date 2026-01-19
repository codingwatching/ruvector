//! Integration tests for NEON-optimized kernels
//!
//! Tests attention, RoPE, normalization, and matrix multiplication kernels
//! comparing NEON implementations to scalar reference implementations.

use ruvllm_integration::kernels::{
    flash_attention_neon, grouped_query_attention_neon, multi_query_attention_neon,
    paged_attention_neon, PagedKvCache,
    gemm_neon, gemv_neon, batched_gemm_neon,
    layer_norm_neon, rms_norm_neon,
    apply_rope_neon, precompute_rope_tables, RopeConfig,
    AttentionConfig,
};
use ruvllm_integration::kernels::rope::{
    apply_inverse_rope_neon, apply_rope_with_tables, precompute_rope_tables_with_config, RopeTables,
};
use ruvllm_integration::kernels::norm::{batched_layer_norm_neon, batched_rms_norm_neon, compute_rms};
use ruvllm_integration::kernels::matmul::gemm_nt_neon;

// ========== Attention Tests ==========

#[test]
fn test_attention_matches_reference() {
    let head_dim = 64;
    let kv_len = 8;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let query: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
    let key: Vec<f32> = (0..kv_len * head_dim).map(|i| (i as f32) * 0.01).collect();
    let value: Vec<f32> = (0..kv_len * head_dim).map(|i| (i as f32) * 0.02).collect();

    // NEON implementation
    let output_neon = flash_attention_neon(&query, &key, &value, scale, false);

    // Reference scalar implementation
    let output_ref = attention_scalar_reference(&query, &key, &value, head_dim, kv_len, scale);

    assert_eq!(output_neon.len(), output_ref.len());
    for (neon_val, ref_val) in output_neon.iter().zip(output_ref.iter()) {
        assert!(
            (neon_val - ref_val).abs() < 1e-3,
            "Attention mismatch: {} vs {}",
            neon_val,
            ref_val
        );
    }
}

/// Scalar reference implementation for attention
fn attention_scalar_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Vec<f32> {
    // Compute attention scores
    let mut scores = Vec::with_capacity(kv_len);
    for t in 0..kv_len {
        let k_offset = t * head_dim;
        let score: f32 = query
            .iter()
            .zip(&key[k_offset..k_offset + head_dim])
            .map(|(q, k)| q * k * scale)
            .sum();
        scores.push(score);
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

    // Weighted sum of values
    let mut output = vec![0.0; head_dim];
    for (t, weight) in attn_weights.iter().enumerate() {
        let v_offset = t * head_dim;
        for (i, v) in value[v_offset..v_offset + head_dim].iter().enumerate() {
            output[i] += weight * v;
        }
    }

    output
}

#[test]
fn test_attention_with_various_lengths() {
    let head_dims = [16, 32, 64, 128];
    let kv_lengths = [1, 4, 8, 16, 32];

    for head_dim in head_dims {
        for kv_len in kv_lengths {
            let scale = 1.0 / (head_dim as f32).sqrt();

            let query: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
            let key: Vec<f32> = (0..kv_len * head_dim).map(|i| (i as f32) * 0.01).collect();
            let value: Vec<f32> = (0..kv_len * head_dim).map(|i| (i as f32) * 0.02).collect();

            let output = flash_attention_neon(&query, &key, &value, scale, false);

            assert_eq!(output.len(), head_dim, "head_dim={}, kv_len={}", head_dim, kv_len);
            assert!(
                output.iter().all(|&v| v.is_finite()),
                "Non-finite attention output for head_dim={}, kv_len={}",
                head_dim,
                kv_len
            );
        }
    }
}

#[test]
fn test_gqa_attention() {
    let config = AttentionConfig {
        num_heads: 8,
        num_kv_heads: 2, // GQA: 4 query heads share 1 KV head
        head_dim: 32,
        causal: false,
        ..Default::default()
    };

    let queries: Vec<f32> = (0..config.num_heads * config.head_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let kv_len = 4;
    let keys: Vec<f32> = (0..kv_len * config.num_kv_heads * config.head_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let values: Vec<f32> = (0..kv_len * config.num_kv_heads * config.head_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let output = grouped_query_attention_neon(&queries, &keys, &values, &config);

    assert_eq!(output.len(), config.num_heads * config.head_dim);
    assert!(output.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_mqa_attention() {
    let config = AttentionConfig {
        num_heads: 8,
        num_kv_heads: 1, // MQA: all query heads share 1 KV head
        head_dim: 32,
        causal: false,
        ..Default::default()
    };

    let queries: Vec<f32> = (0..config.num_heads * config.head_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let kv_len = 4;
    let keys: Vec<f32> = (0..kv_len * config.head_dim).map(|i| (i as f32) * 0.01).collect();
    let values: Vec<f32> = (0..kv_len * config.head_dim).map(|i| (i as f32) * 0.02).collect();

    let output = multi_query_attention_neon(&queries, &keys, &values, &config);

    assert_eq!(output.len(), config.num_heads * config.head_dim);
    assert!(output.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_paged_kv_cache() {
    let mut cache = PagedKvCache::new(16, 2, 32);

    // Add tokens
    for _ in 0..10 {
        let keys = vec![1.0; 2 * 32];
        let values = vec![2.0; 2 * 32];
        cache.append(&keys, &values);
    }

    assert_eq!(cache.num_tokens, 10);

    // Retrieve
    let all_keys = cache.get_keys();
    let all_values = cache.get_values();

    assert_eq!(all_keys.len(), 10 * 2 * 32);
    assert_eq!(all_values.len(), 10 * 2 * 32);
}

#[test]
fn test_paged_attention() {
    let mut cache = PagedKvCache::new(16, 1, 32);

    for _ in 0..8 {
        let keys: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.2).collect();
        cache.append(&keys, &values);
    }

    let query: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();
    let scale = 1.0 / 32.0f32.sqrt();

    let output = paged_attention_neon(&query, &cache, &[], scale);

    assert_eq!(output.len(), 32);
    assert!(output.iter().all(|&v| v.is_finite()));
}

// ========== RoPE Tests ==========

#[test]
fn test_rope_correctness() {
    let head_dim = 16;
    let base = 10000.0;

    // Position 0 should be identity (cos=1, sin=0)
    let mut x_pos0: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let original = x_pos0.clone();

    apply_rope_neon(&mut x_pos0, &[0], head_dim, base);

    for (orig, rotated) in original.iter().zip(x_pos0.iter()) {
        assert!(
            (orig - rotated).abs() < 1e-5,
            "Position 0 should be identity: {} vs {}",
            orig,
            rotated
        );
    }
}

#[test]
fn test_rope_rotation_at_nonzero_position() {
    let head_dim = 8;
    let base = 10000.0;

    let mut x: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let original = x.clone();

    apply_rope_neon(&mut x, &[1], head_dim, base);

    // At non-zero position, values should change
    assert!(
        x.iter().zip(original.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "RoPE should rotate at non-zero position"
    );
}

#[test]
fn test_rope_inverse_roundtrip() {
    let head_dim = 16;
    let base = 10000.0;

    let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let original = x.clone();

    // Apply RoPE then inverse
    apply_rope_neon(&mut x, &[5], head_dim, base);
    apply_inverse_rope_neon(&mut x, &[5], head_dim, base);

    for (orig, recovered) in original.iter().zip(x.iter()) {
        assert!(
            (orig - recovered).abs() < 1e-4,
            "Inverse RoPE should recover original: {} vs {}",
            orig,
            recovered
        );
    }
}

#[test]
fn test_rope_precomputed_tables() {
    let config = RopeConfig {
        head_dim: 32,
        max_seq_len: 64,
        base: 10000.0,
        ..Default::default()
    };

    let tables = precompute_rope_tables_with_config(&config);

    // Verify dimensions
    assert_eq!(tables.half_dim, 16);
    assert_eq!(tables.max_seq_len, 64);

    // Position 0 should have cos=1, sin=0
    let (cos0, sin0) = tables.get(0);
    for &c in cos0 {
        assert!((c - 1.0).abs() < 1e-5, "cos at pos 0 should be 1");
    }
    for &s in sin0 {
        assert!(s.abs() < 1e-5, "sin at pos 0 should be 0");
    }
}

#[test]
fn test_rope_tables_match_direct_computation() {
    let config = RopeConfig {
        head_dim: 16,
        max_seq_len: 32,
        base: 10000.0,
        ..Default::default()
    };

    let tables = precompute_rope_tables_with_config(&config);

    let mut x_direct: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let mut x_tables = x_direct.clone();

    // Apply with direct computation
    apply_rope_neon(&mut x_direct, &[7], config.head_dim, config.base);

    // Apply with tables
    apply_rope_with_tables(&mut x_tables, &[7], &tables);

    for (direct, table) in x_direct.iter().zip(x_tables.iter()) {
        assert!(
            (direct - table).abs() < 1e-4,
            "Table-based RoPE should match direct: {} vs {}",
            direct,
            table
        );
    }
}

#[test]
fn test_rope_multiple_tokens() {
    let head_dim = 8;
    let base = 10000.0;

    let mut x: Vec<f32> = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // Token 0
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // Token 1
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // Token 2
    ];
    let positions = vec![0, 1, 2];

    apply_rope_neon(&mut x, &positions, head_dim, base);

    // Token 0 should be unchanged
    assert!((x[0] - 1.0).abs() < 1e-5);
    assert!(x[1].abs() < 1e-5);

    // Tokens 1 and 2 should be rotated
    assert!(x.iter().skip(8).any(|&v| (v - 1.0).abs() > 1e-5 || v.abs() > 1e-5));
}

#[test]
fn test_rope_llama_config() {
    let config = RopeConfig::llama2(128, 4096);
    assert_eq!(config.base, 10000.0);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.max_seq_len, 4096);
}

#[test]
fn test_rope_llama3_config() {
    let config = RopeConfig::llama3(128, 8192);
    assert_eq!(config.base, 500000.0); // Higher base for longer context
    assert_eq!(config.head_dim, 128);
}

// ========== Normalization Tests ==========

#[test]
fn test_rms_norm_numerical_stability() {
    // Test with very small values
    let mut x_small: Vec<f32> = vec![1e-6, 1e-6, 1e-6, 1e-6];
    let weight = vec![1.0; 4];
    rms_norm_neon(&mut x_small, &weight, 1e-6);
    assert!(x_small.iter().all(|&v| v.is_finite()));

    // Test with zeros
    let mut x_zero: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
    rms_norm_neon(&mut x_zero, &weight, 1e-6);
    assert!(x_zero.iter().all(|&v| v.is_finite()));

    // Test with large values
    let mut x_large: Vec<f32> = vec![1e6, 1e6, 1e6, 1e6];
    rms_norm_neon(&mut x_large, &weight, 1e-6);
    assert!(x_large.iter().all(|&v| v.is_finite()));

    // Test with mixed signs
    let mut x_mixed: Vec<f32> = vec![-1.0, 1.0, -1.0, 1.0];
    rms_norm_neon(&mut x_mixed, &weight, 1e-6);
    assert!(x_mixed.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_rms_norm_matches_reference() {
    let dim = 64;
    let mut x_neon: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let mut x_ref = x_neon.clone();
    let weight: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let eps = 1e-6;

    // NEON implementation
    rms_norm_neon(&mut x_neon, &weight, eps);

    // Reference implementation
    rms_norm_scalar_reference(&mut x_ref, &weight, eps);

    for i in 0..dim {
        assert!(
            (x_neon[i] - x_ref[i]).abs() < 1e-4,
            "RMSNorm mismatch at {}: {} vs {}",
            i,
            x_neon[i],
            x_ref[i]
        );
    }
}

fn rms_norm_scalar_reference(x: &mut [f32], weight: &[f32], eps: f32) {
    let len = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    for (i, w) in weight.iter().enumerate() {
        x[i] = x[i] * inv_rms * w;
    }
}

#[test]
fn test_layer_norm_mean_and_variance() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0; 8];
    let bias = vec![0.0; 8];
    let eps = 1e-6;

    layer_norm_neon(&mut x, &weight, &bias, eps);

    // After LayerNorm, mean should be ~0
    let mean: f32 = x.iter().sum::<f32>() / 8.0;
    assert!(mean.abs() < 1e-4, "Mean should be ~0, got {}", mean);

    // Variance should be ~1
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
    assert!((var - 1.0).abs() < 1e-4, "Variance should be ~1, got {}", var);
}

#[test]
fn test_layer_norm_with_bias() {
    let mut x = vec![0.0, 0.0, 0.0, 0.0];
    let weight = vec![1.0; 4];
    let bias = vec![5.0; 4];
    let eps = 1e-6;

    layer_norm_neon(&mut x, &weight, &bias, eps);

    // With zero input, output should be bias
    for v in &x {
        assert!((v - 5.0).abs() < 1e-4, "Expected ~5.0, got {}", v);
    }
}

#[test]
fn test_batched_rms_norm() {
    let batch_size = 4;
    let dim = 32;
    let mut x: Vec<f32> = (0..batch_size * dim).map(|i| (i as f32) * 0.1).collect();
    let weight = vec![1.0; dim];

    batched_rms_norm_neon(&mut x, &weight, batch_size, dim, 1e-6);

    assert!(x.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_batched_layer_norm() {
    let batch_size = 4;
    let dim = 32;
    let mut x: Vec<f32> = (0..batch_size * dim).map(|i| (i as f32) * 0.1).collect();
    let weight = vec![1.0; dim];
    let bias = vec![0.0; dim];

    batched_layer_norm_neon(&mut x, &weight, &bias, batch_size, dim, 1e-6);

    // Check each batch vector
    for b in 0..batch_size {
        let offset = b * dim;
        let slice = &x[offset..offset + dim];
        let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
        assert!(mean.abs() < 1e-4, "Batch {} mean should be ~0, got {}", b, mean);
    }
}

#[test]
fn test_compute_rms() {
    let x = vec![3.0, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5) ~ 3.536
    let rms = compute_rms(&x);
    assert!((rms - 3.5355).abs() < 0.01, "RMS should be ~3.536, got {}", rms);
}

// ========== Matmul Tests ==========

#[test]
fn test_matmul_accuracy() {
    // 4x4 * 4x4 = 4x4
    let a = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    let b = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]; // Identity
    let mut c = vec![0.0; 16];

    gemm_neon(&a, &b, &mut c, 4, 4, 4);

    // A * I = A
    for (i, (a_val, c_val)) in a.iter().zip(c.iter()).enumerate() {
        assert!(
            (a_val - c_val).abs() < 1e-5,
            "Identity multiplication failed at {}: {} vs {}",
            i,
            a_val,
            c_val
        );
    }
}

#[test]
fn test_gemv_accuracy() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let x = vec![1.0, 2.0, 3.0]; // 3
    let mut y = vec![0.0; 2];

    gemv_neon(&a, &x, &mut y, 2, 3);

    // y[0] = 1*1 + 2*2 + 3*3 = 14
    // y[1] = 4*1 + 5*2 + 6*3 = 32
    assert!((y[0] - 14.0).abs() < 1e-5);
    assert!((y[1] - 32.0).abs() < 1e-5);
}

#[test]
fn test_gemm_matches_reference() {
    let m = 16;
    let k = 32;
    let n = 16;

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
    let mut c_neon = vec![0.0; m * n];
    let mut c_ref = vec![0.0; m * n];

    // NEON
    gemm_neon(&a, &b, &mut c_neon, m, k, n);

    // Reference
    gemm_scalar_reference(&a, &b, &mut c_ref, m, k, n);

    for i in 0..(m * n) {
        assert!(
            (c_neon[i] - c_ref[i]).abs() < 0.1,
            "GEMM mismatch at {}: {} vs {}",
            i,
            c_neon[i],
            c_ref[i]
        );
    }
}

fn gemm_scalar_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[test]
fn test_gemm_nt() {
    // Test A * B^T
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_t = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // B^T: 2x3
    let mut c = vec![0.0; 4];

    gemm_nt_neon(&a, &b_t, &mut c, 2, 3, 2);

    // c[0,0] = 1*1 + 2*3 + 3*5 = 22
    // c[0,1] = 1*2 + 2*4 + 3*6 = 28
    assert!((c[0] - 22.0).abs() < 1e-4, "c[0,0] = {}", c[0]);
    assert!((c[1] - 28.0).abs() < 1e-4, "c[0,1] = {}", c[1]);
}

#[test]
fn test_batched_gemm() {
    let batch = 4;
    let m = 8;
    let k = 16;
    let n = 8;

    let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.01).collect();
    let mut c = vec![0.0; batch * m * n];

    batched_gemm_neon(&a, &b, &mut c, batch, m, k, n);

    assert!(c.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_matmul_edge_cases() {
    // 1x1 matrix
    let a = vec![3.0];
    let b = vec![4.0];
    let mut c = vec![0.0];
    gemm_neon(&a, &b, &mut c, 1, 1, 1);
    assert!((c[0] - 12.0).abs() < 1e-5);

    // Rectangular matrices
    let a2 = vec![1.0, 2.0, 3.0]; // 1x3
    let b2 = vec![1.0, 2.0, 3.0]; // 3x1
    let mut c2 = vec![0.0];
    gemm_neon(&a2, &b2, &mut c2, 1, 3, 1);
    assert!((c2[0] - 14.0).abs() < 1e-5); // 1*1 + 2*2 + 3*3 = 14
}

// ========== AttentionConfig Tests ==========

#[test]
fn test_attention_config_default() {
    let config = AttentionConfig::default();
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert!(config.causal);
}

#[test]
fn test_attention_config_effective_scale() {
    let config = AttentionConfig {
        head_dim: 64,
        scale: 0.0, // Should be computed
        ..Default::default()
    };

    let expected_scale = 1.0 / (64.0f32).sqrt();
    assert!((config.effective_scale() - expected_scale).abs() < 1e-6);
}

#[test]
fn test_attention_config_gqa_ratio() {
    let config = AttentionConfig {
        num_heads: 8,
        num_kv_heads: 2,
        ..Default::default()
    };

    assert_eq!(config.gqa_ratio(), 4);
}
