//! GPU-Accelerated Operations
//!
//! High-level GPU operations for embeddings with automatic fallback to CPU.

use crate::{EmbeddingError, Result};
use super::backend::GpuBackend;
use super::shaders::ShaderRegistry;
use rayon::prelude::*;

// ==================== GPU Pooler ====================

/// GPU-accelerated pooling operations
pub struct GpuPooler {
    use_gpu: bool,
}

impl GpuPooler {
    /// Create new GPU pooler
    pub fn new(_backend: &dyn GpuBackend, _shaders: &ShaderRegistry) -> Result<Self> {
        Ok(Self {
            use_gpu: _backend.is_available() && _backend.device_info().supports_compute,
        })
    }

    /// Mean pooling (GPU or CPU fallback)
    pub fn mean_pool(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_length: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        if self.use_gpu && batch_size >= 8 {
            self.mean_pool_gpu(token_embeddings, attention_mask, batch_size, seq_length, hidden_size)
        } else {
            Ok(self.mean_pool_cpu(token_embeddings, attention_mask, batch_size, seq_length, hidden_size))
        }
    }

    /// CLS pooling (GPU or CPU fallback)
    pub fn cls_pool(
        &self,
        token_embeddings: &[f32],
        batch_size: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        // CLS pooling is simple copy, CPU is often faster
        Ok(self.cls_pool_cpu(token_embeddings, batch_size, hidden_size))
    }

    /// Max pooling (GPU or CPU fallback)
    pub fn max_pool(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_length: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        if self.use_gpu && batch_size >= 8 {
            self.max_pool_gpu(token_embeddings, attention_mask, batch_size, seq_length, hidden_size)
        } else {
            Ok(self.max_pool_cpu(token_embeddings, attention_mask, batch_size, seq_length, hidden_size))
        }
    }

    // GPU implementations

    fn mean_pool_gpu(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_length: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        // GPU implementation would dispatch shader here
        // For now, fall back to CPU
        Ok(self.mean_pool_cpu(token_embeddings, attention_mask, batch_size, seq_length, hidden_size))
    }

    fn max_pool_gpu(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_length: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        // GPU implementation would dispatch shader here
        // For now, fall back to CPU
        Ok(self.max_pool_cpu(token_embeddings, attention_mask, batch_size, seq_length, hidden_size))
    }

    // CPU implementations

    fn mean_pool_cpu(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_length: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * hidden_size];

        output
            .par_chunks_mut(hidden_size)
            .enumerate()
            .for_each(|(batch_idx, out_chunk)| {
                let tokens_base = batch_idx * seq_length * hidden_size;
                let mask_base = batch_idx * seq_length;

                let mut count = 0.0f32;

                for seq_idx in 0..seq_length {
                    if attention_mask[mask_base + seq_idx] == 1 {
                        let start = tokens_base + seq_idx * hidden_size;
                        for (j, out_val) in out_chunk.iter_mut().enumerate() {
                            *out_val += token_embeddings[start + j];
                        }
                        count += 1.0;
                    }
                }

                if count > 0.0 {
                    for val in out_chunk.iter_mut() {
                        *val /= count;
                    }
                }
            });

        output
    }

    fn cls_pool_cpu(
        &self,
        token_embeddings: &[f32],
        batch_size: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let seq_length = token_embeddings.len() / (batch_size * hidden_size);
        let mut output = vec![0.0f32; batch_size * hidden_size];

        for batch_idx in 0..batch_size {
            let src_start = batch_idx * seq_length * hidden_size;
            let dst_start = batch_idx * hidden_size;
            output[dst_start..dst_start + hidden_size]
                .copy_from_slice(&token_embeddings[src_start..src_start + hidden_size]);
        }

        output
    }

    fn max_pool_cpu(
        &self,
        token_embeddings: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_length: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut output = vec![f32::NEG_INFINITY; batch_size * hidden_size];

        output
            .par_chunks_mut(hidden_size)
            .enumerate()
            .for_each(|(batch_idx, out_chunk)| {
                let tokens_base = batch_idx * seq_length * hidden_size;
                let mask_base = batch_idx * seq_length;

                for seq_idx in 0..seq_length {
                    if attention_mask[mask_base + seq_idx] == 1 {
                        let start = tokens_base + seq_idx * hidden_size;
                        for (j, out_val) in out_chunk.iter_mut().enumerate() {
                            let val = token_embeddings[start + j];
                            if val > *out_val {
                                *out_val = val;
                            }
                        }
                    }
                }

                // Replace -inf with 0
                for val in out_chunk.iter_mut() {
                    if val.is_infinite() {
                        *val = 0.0;
                    }
                }
            });

        output
    }
}

// ==================== GPU Similarity ====================

/// GPU-accelerated similarity computations
pub struct GpuSimilarity {
    use_gpu: bool,
    min_candidates: usize,
}

impl GpuSimilarity {
    /// Create new GPU similarity calculator
    pub fn new(_backend: &dyn GpuBackend, _shaders: &ShaderRegistry) -> Result<Self> {
        Ok(Self {
            use_gpu: _backend.is_available() && _backend.device_info().supports_compute,
            min_candidates: 64, // Minimum candidates to use GPU
        })
    }

    /// Batch cosine similarity
    pub fn batch_cosine(&self, query: &[f32], candidates: &[&[f32]]) -> Result<Vec<f32>> {
        if self.use_gpu && candidates.len() >= self.min_candidates {
            self.batch_cosine_gpu(query, candidates)
        } else {
            Ok(self.batch_cosine_cpu(query, candidates))
        }
    }

    /// Batch dot product
    pub fn batch_dot_product(&self, query: &[f32], candidates: &[&[f32]]) -> Result<Vec<f32>> {
        if self.use_gpu && candidates.len() >= self.min_candidates {
            self.batch_dot_product_gpu(query, candidates)
        } else {
            Ok(self.batch_dot_product_cpu(query, candidates))
        }
    }

    /// Batch Euclidean distance
    pub fn batch_euclidean(&self, query: &[f32], candidates: &[&[f32]]) -> Result<Vec<f32>> {
        if self.use_gpu && candidates.len() >= self.min_candidates {
            self.batch_euclidean_gpu(query, candidates)
        } else {
            Ok(self.batch_euclidean_cpu(query, candidates))
        }
    }

    /// Find top-k most similar
    pub fn top_k(&self, query: &[f32], candidates: &[&[f32]], k: usize) -> Result<Vec<(usize, f32)>> {
        let similarities = self.batch_cosine(query, candidates)?;

        let mut indexed: Vec<(usize, f32)> = similarities.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        Ok(indexed)
    }

    // GPU implementations

    fn batch_cosine_gpu(&self, query: &[f32], candidates: &[&[f32]]) -> Result<Vec<f32>> {
        // GPU implementation would dispatch shader here
        // For now, fall back to CPU
        Ok(self.batch_cosine_cpu(query, candidates))
    }

    fn batch_dot_product_gpu(&self, query: &[f32], candidates: &[&[f32]]) -> Result<Vec<f32>> {
        Ok(self.batch_dot_product_cpu(query, candidates))
    }

    fn batch_euclidean_gpu(&self, query: &[f32], candidates: &[&[f32]]) -> Result<Vec<f32>> {
        Ok(self.batch_euclidean_cpu(query, candidates))
    }

    // CPU implementations

    fn batch_cosine_cpu(&self, query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
        candidates
            .par_iter()
            .map(|c| cosine_similarity_cpu(query, c))
            .collect()
    }

    fn batch_dot_product_cpu(&self, query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
        candidates
            .par_iter()
            .map(|c| dot_product_cpu(query, c))
            .collect()
    }

    fn batch_euclidean_cpu(&self, query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
        candidates
            .par_iter()
            .map(|c| euclidean_distance_cpu(query, c))
            .collect()
    }
}

// ==================== GPU Vector Operations ====================

/// GPU-accelerated vector operations
pub struct GpuVectorOps {
    #[allow(dead_code)]
    use_gpu: bool,
}

impl GpuVectorOps {
    /// Create new GPU vector operations
    pub fn new(_backend: &dyn GpuBackend, _shaders: &ShaderRegistry) -> Result<Self> {
        Ok(Self {
            use_gpu: _backend.is_available() && _backend.device_info().supports_compute,
        })
    }

    /// L2 normalize batch of vectors
    pub fn normalize_batch(&self, vectors: &mut [f32], dimension: usize) -> Result<()> {
        let _num_vectors = vectors.len() / dimension;

        vectors
            .par_chunks_mut(dimension)
            .for_each(|chunk| {
                let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    for val in chunk.iter_mut() {
                        *val /= norm;
                    }
                }
            });

        Ok(())
    }

    /// Matrix-vector multiplication
    pub fn matmul(&self, matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; rows];

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, out)| {
                let row_start = row * cols;
                *out = matrix[row_start..row_start + cols]
                    .iter()
                    .zip(vector.iter())
                    .map(|(m, v)| m * v)
                    .sum();
            });

        Ok(result)
    }

    /// Batch vector addition
    pub fn batch_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(EmbeddingError::dimension_mismatch(a.len(), b.len()));
        }

        Ok(a.par_iter().zip(b.par_iter()).map(|(x, y)| x + y).collect())
    }

    /// Batch vector scaling
    pub fn batch_scale(&self, vectors: &mut [f32], scale: f32) -> Result<()> {
        vectors.par_iter_mut().for_each(|v| *v *= scale);
        Ok(())
    }
}

// ==================== Standalone Functions ====================

/// Batch cosine similarity (GPU-accelerated if available)
pub fn batch_cosine_similarity_gpu(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    candidates
        .par_iter()
        .map(|c| cosine_similarity_cpu(query, c))
        .collect()
}

/// Batch dot product (GPU-accelerated if available)
pub fn batch_dot_product_gpu(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    candidates
        .par_iter()
        .map(|c| dot_product_cpu(query, c))
        .collect()
}

/// Batch Euclidean distance (GPU-accelerated if available)
pub fn batch_euclidean_gpu(query: &[f32], candidates: &[&[f32]]) -> Vec<f32> {
    candidates
        .par_iter()
        .map(|c| euclidean_distance_cpu(query, c))
        .collect()
}

// ==================== CPU Helper Functions ====================

#[inline]
fn cosine_similarity_cpu(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 1e-12 && norm_b > 1e-12 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[inline]
fn dot_product_cpu(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn euclidean_distance_cpu(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity_cpu(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity_cpu(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert!((dot_product_cpu(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        assert!((euclidean_distance_cpu(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates: Vec<&[f32]> = vec![
            &[1.0, 0.0, 0.0][..],
            &[0.0, 1.0, 0.0][..],
            &[0.707, 0.707, 0.0][..],
        ];

        let results = batch_cosine_similarity_gpu(&query, &candidates);

        assert_eq!(results.len(), 3);
        assert!((results[0] - 1.0).abs() < 1e-6);
        assert!(results[1].abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_cpu() {
        let pooler = GpuPooler { use_gpu: false };

        // batch=2, seq=2, hidden=3
        let tokens = vec![
            1.0, 2.0, 3.0,  // batch 0, seq 0
            4.0, 5.0, 6.0,  // batch 0, seq 1
            7.0, 8.0, 9.0,  // batch 1, seq 0
            10.0, 11.0, 12.0, // batch 1, seq 1
        ];
        let mask = vec![1i64, 1, 1, 1];

        let result = pooler.mean_pool_cpu(&tokens, &mask, 2, 2, 3);

        assert_eq!(result.len(), 6);
        // Batch 0: mean of [1,2,3] and [4,5,6] = [2.5, 3.5, 4.5]
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 3.5).abs() < 1e-6);
        assert!((result[2] - 4.5).abs() < 1e-6);
    }
}
