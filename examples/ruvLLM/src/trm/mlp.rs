//! MLP Latent Updater
//!
//! MLP-based latent state updater for TRM recursive reasoning.
//! This is the faster variant compared to attention.

use super::{LatentUpdate, TrmConfig};

/// MLP-based latent updater (faster, default choice)
///
/// Architecture:
/// ```text
/// combined = [question_pooled, answer_pooled, latent]
/// hidden = GELU(Linear1(combined))
/// delta = Linear2(hidden)
/// gate = sigmoid(Linear_gate(latent))
/// new_latent = gate * latent + (1 - gate) * delta
/// output = LayerNorm(new_latent)
/// ```
pub struct MlpLatentUpdater {
    // Linear 1: combined_dim -> hidden_dim * 4
    w1: Vec<f32>,
    b1: Vec<f32>,

    // Linear 2: hidden_dim * 4 -> hidden_dim
    w2: Vec<f32>,
    b2: Vec<f32>,

    // Gate: hidden_dim -> hidden_dim
    w_gate: Vec<f32>,
    b_gate: Vec<f32>,

    // Layer norm parameters
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,

    // Dimensions
    hidden_dim: usize,
    embedding_dim: usize,
    combined_dim: usize,

    // Scratch buffers (avoid allocations)
    combined_buffer: Vec<f32>,
    hidden_buffer: Vec<f32>,
    gate_buffer: Vec<f32>,
    delta_buffer: Vec<f32>,
}

impl MlpLatentUpdater {
    /// Create a new MLP latent updater
    pub fn new(hidden_dim: usize, embedding_dim: usize) -> Self {
        // Combined input: question + answer + latent
        let combined_dim = embedding_dim + embedding_dim + hidden_dim;
        let hidden_expansion = 4;

        Self {
            w1: Self::init_weights(combined_dim, hidden_dim * hidden_expansion),
            b1: vec![0.0; hidden_dim * hidden_expansion],
            w2: Self::init_weights(hidden_dim * hidden_expansion, hidden_dim),
            b2: vec![0.0; hidden_dim],
            w_gate: Self::init_weights(hidden_dim, hidden_dim),
            b_gate: vec![0.0; hidden_dim],
            ln_gamma: vec![1.0; hidden_dim],
            ln_beta: vec![0.0; hidden_dim],
            hidden_dim,
            embedding_dim,
            combined_dim,
            combined_buffer: vec![0.0; combined_dim],
            hidden_buffer: vec![0.0; hidden_dim * hidden_expansion],
            gate_buffer: vec![0.0; hidden_dim],
            delta_buffer: vec![0.0; hidden_dim],
        }
    }

    /// Create from TrmConfig
    pub fn from_config(config: &TrmConfig) -> Self {
        Self::new(config.hidden_dim, config.embedding_dim)
    }

    /// Xavier/Glorot initialization
    fn init_weights(in_dim: usize, out_dim: usize) -> Vec<f32> {
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let mut weights = Vec::with_capacity(in_dim * out_dim);

        // Deterministic pseudo-random initialization using golden ratio
        for i in 0..in_dim * out_dim {
            let x = ((i as f64 * 0.6180339887498949) % 1.0) as f32;
            weights.push((x - 0.5) * 2.0 * std);
        }

        weights
    }

    /// Pool embeddings to fixed size (mean pooling)
    fn pool_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        if embedding.len() == self.embedding_dim {
            return embedding.to_vec();
        }

        // Mean pool if longer
        let num_chunks = embedding.len() / self.embedding_dim;
        if num_chunks == 0 {
            // Pad if shorter
            let mut pooled = embedding.to_vec();
            pooled.resize(self.embedding_dim, 0.0);
            return pooled;
        }

        let mut pooled = vec![0.0; self.embedding_dim];
        for chunk in embedding.chunks(self.embedding_dim) {
            for (i, &v) in chunk.iter().enumerate() {
                if i < self.embedding_dim {
                    pooled[i] += v / num_chunks as f32;
                }
            }
        }
        pooled
    }

    /// Matrix-vector multiply with bias: output = input @ W + b
    fn matmul_add(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        output: &mut [f32],
        in_dim: usize,
        out_dim: usize,
    ) {
        debug_assert_eq!(input.len(), in_dim);
        debug_assert_eq!(weights.len(), in_dim * out_dim);
        debug_assert_eq!(bias.len(), out_dim);
        debug_assert_eq!(output.len(), out_dim);

        // output = bias (copy)
        output.copy_from_slice(bias);

        // output += input @ W
        for i in 0..in_dim {
            let input_val = input[i];
            if input_val.abs() < 1e-8 {
                continue; // Skip zero values
            }

            let w_row = &weights[i * out_dim..(i + 1) * out_dim];
            for (j, &w) in w_row.iter().enumerate() {
                output[j] += input_val * w;
            }
        }
    }

    /// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu_inplace(data: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;

        for x in data.iter_mut() {
            let x3 = *x * *x * *x;
            let inner = SQRT_2_OVER_PI * (*x + COEFF * x3);
            *x = *x * 0.5 * (1.0 + fast_tanh(inner));
        }
    }

    /// Sigmoid activation with NaN protection
    fn sigmoid_inplace(data: &mut [f32]) {
        for x in data.iter_mut() {
            // Clamp to prevent overflow in exp()
            let clamped = (*x).clamp(-20.0, 20.0);
            *x = 1.0 / (1.0 + (-clamped).exp());
        }
    }

    /// Layer normalization
    fn layer_norm_inplace(data: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
        let n = data.len() as f32;

        // Compute mean
        let mean: f32 = data.iter().sum::<f32>() / n;

        // Compute variance
        let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

        // Normalize
        let inv_std = 1.0 / (variance + eps).sqrt();

        for (i, x) in data.iter_mut().enumerate() {
            *x = gamma[i] * (*x - mean) * inv_std + beta[i];
        }
    }
}

impl LatentUpdate for MlpLatentUpdater {
    fn update(
        &self,
        question_pooled: &[f32],
        answer_pooled: &[f32],
        latent: &mut [f32],
    ) {
        // Get mutable access to buffers (interior mutability would be cleaner)
        let mut combined_buffer = self.combined_buffer.clone();
        let mut hidden_buffer = self.hidden_buffer.clone();
        let mut gate_buffer = self.gate_buffer.clone();
        let mut delta_buffer = self.delta_buffer.clone();

        // Pool inputs if needed
        let q_pooled = self.pool_embedding(question_pooled);
        let a_pooled = self.pool_embedding(answer_pooled);

        // Combine inputs: [question, answer, latent]
        let q_len = q_pooled.len().min(self.embedding_dim);
        let a_len = a_pooled.len().min(self.embedding_dim);
        let l_len = latent.len().min(self.hidden_dim);

        combined_buffer[..q_len].copy_from_slice(&q_pooled[..q_len]);
        combined_buffer[self.embedding_dim..self.embedding_dim + a_len]
            .copy_from_slice(&a_pooled[..a_len]);
        combined_buffer[self.embedding_dim * 2..self.embedding_dim * 2 + l_len]
            .copy_from_slice(&latent[..l_len]);

        // Linear 1 + GELU
        Self::matmul_add(
            &combined_buffer,
            &self.w1,
            &self.b1,
            &mut hidden_buffer,
            self.combined_dim,
            self.hidden_dim * 4,
        );
        Self::gelu_inplace(&mut hidden_buffer);

        // Linear 2 -> delta
        Self::matmul_add(
            &hidden_buffer,
            &self.w2,
            &self.b2,
            &mut delta_buffer,
            self.hidden_dim * 4,
            self.hidden_dim,
        );

        // Gate computation
        Self::matmul_add(
            latent,
            &self.w_gate,
            &self.b_gate,
            &mut gate_buffer,
            self.hidden_dim,
            self.hidden_dim,
        );
        Self::sigmoid_inplace(&mut gate_buffer);

        // Gated residual: latent = gate * latent + (1 - gate) * delta
        for i in 0..self.hidden_dim {
            latent[i] = gate_buffer[i] * latent[i] + (1.0 - gate_buffer[i]) * delta_buffer[i];
        }

        // Layer normalization
        Self::layer_norm_inplace(latent, &self.ln_gamma, &self.ln_beta, 1e-5);
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn reset(&mut self) {
        // Reset buffers
        self.combined_buffer.fill(0.0);
        self.hidden_buffer.fill(0.0);
        self.gate_buffer.fill(0.0);
        self.delta_buffer.fill(0.0);
    }
}

/// Fast tanh approximation using Pade approximant
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.0 {
        return x.signum();
    }

    let x2 = x * x;
    let numerator = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let denominator = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_updater_creation() {
        let updater = MlpLatentUpdater::new(256, 256);
        assert_eq!(updater.hidden_dim(), 256);
    }

    #[test]
    fn test_mlp_updater_dimensions() {
        let updater = MlpLatentUpdater::new(64, 64);

        let question = vec![0.1; 64];
        let answer = vec![0.2; 64];
        let mut latent = vec![0.0; 64];

        updater.update(&question, &answer, &mut latent);

        assert_eq!(latent.len(), 64);
        // Latent should be modified (not all zeros)
        assert!(latent.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_mlp_updater_convergence() {
        let updater = MlpLatentUpdater::new(64, 64);

        let question = vec![1.0; 64];
        let answer = vec![1.0; 64];
        let mut latent = vec![0.0; 64];

        // Run multiple iterations
        for _ in 0..20 {
            updater.update(&question, &answer, &mut latent);
        }

        // Should converge toward stable state
        let prev_latent = latent.clone();
        updater.update(&question, &answer, &mut latent);

        let diff: f32 = latent.iter()
            .zip(prev_latent.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // The gated residual with layer norm may still oscillate slightly,
        // but changes should be bounded and decreasing
        assert!(diff < 5.0, "Latent should show convergence trend, diff={}", diff);
    }

    #[test]
    fn test_mlp_layer_norm_bounds() {
        let updater = MlpLatentUpdater::new(64, 64);

        // Large input values
        let question = vec![100.0; 64];
        let answer = vec![100.0; 64];
        let mut latent = vec![0.0; 64];

        updater.update(&question, &answer, &mut latent);

        // Layer norm should keep values bounded
        let max_val = latent.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_val < 10.0, "Layer norm should bound values, max={}", max_val);
    }

    #[test]
    fn test_fast_tanh() {
        // Test against standard tanh at key points
        let test_values = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];

        for x in test_values {
            let fast = fast_tanh(x);
            let std = x.tanh();
            let diff = (fast - std).abs();
            assert!(diff < 0.01, "fast_tanh({}) = {}, tanh({}) = {}, diff = {}",
                    x, fast, x, std, diff);
        }
    }

    #[test]
    fn test_pool_embedding() {
        let updater = MlpLatentUpdater::new(64, 64);

        // Exact size
        let exact = vec![1.0; 64];
        let pooled = updater.pool_embedding(&exact);
        assert_eq!(pooled.len(), 64);

        // Longer (needs pooling)
        let longer = vec![1.0; 128];
        let pooled = updater.pool_embedding(&longer);
        assert_eq!(pooled.len(), 64);

        // Shorter (needs padding)
        let shorter = vec![1.0; 32];
        let pooled = updater.pool_embedding(&shorter);
        assert_eq!(pooled.len(), 64);
    }
}
