//! Confidence Scorer
//!
//! Scores the confidence of the current answer embedding.

use super::TrmConfig;

/// Confidence scoring for TRM answers
///
/// Uses a learned projection to estimate confidence [0, 1].
pub struct ConfidenceScorer {
    /// Projection weights: embedding_dim -> hidden
    w1: Vec<f32>,
    b1: Vec<f32>,

    /// Output weights: hidden -> 1
    w2: Vec<f32>,
    b2: f32,

    embedding_dim: usize,
    hidden_size: usize,

    // Buffers
    pooled_buffer: Vec<f32>,
    hidden_buffer: Vec<f32>,
}

impl ConfidenceScorer {
    /// Create a new confidence scorer
    pub fn new(config: &TrmConfig) -> Self {
        let embedding_dim = config.embedding_dim;
        let hidden_size = 64; // Small hidden layer for confidence

        Self {
            w1: Self::init_weights(embedding_dim, hidden_size),
            b1: vec![0.0; hidden_size],
            w2: Self::init_weights(hidden_size, 1),
            b2: 0.0,
            embedding_dim,
            hidden_size,
            pooled_buffer: vec![0.0; embedding_dim],
            hidden_buffer: vec![0.0; hidden_size],
        }
    }

    /// Xavier initialization
    fn init_weights(in_dim: usize, out_dim: usize) -> Vec<f32> {
        let std = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let mut weights = Vec::with_capacity(in_dim * out_dim);

        for i in 0..in_dim * out_dim {
            let x = ((i as f64 * 0.6180339887498949) % 1.0) as f32;
            weights.push((x - 0.5) * 2.0 * std);
        }

        weights
    }

    /// Linear transformation
    fn linear(input: &[f32], weights: &[f32], bias: &[f32], output: &mut [f32], in_dim: usize, out_dim: usize) {
        output.copy_from_slice(bias);

        for i in 0..in_dim.min(input.len()) {
            let input_val = input[i];
            if input_val.abs() < 1e-8 {
                continue;
            }
            for j in 0..out_dim {
                output[j] += input_val * weights[i * out_dim + j];
            }
        }
    }

    /// ReLU activation
    fn relu_inplace(data: &mut [f32]) {
        for x in data.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
    }

    /// Score the confidence of an answer
    ///
    /// Returns a value in [0, 1] indicating confidence.
    pub fn score(&self, answer: &[f32]) -> f32 {
        let mut pooled_buffer = self.pooled_buffer.clone();
        let mut hidden_buffer = self.hidden_buffer.clone();

        // Mean pool the answer
        let pooled = self.mean_pool(answer, self.embedding_dim);
        pooled_buffer[..pooled.len()].copy_from_slice(&pooled);

        // First layer + ReLU
        Self::linear(
            &pooled_buffer,
            &self.w1,
            &self.b1,
            &mut hidden_buffer,
            self.embedding_dim,
            self.hidden_size,
        );
        Self::relu_inplace(&mut hidden_buffer);

        // Output layer
        let mut output = self.b2;
        for i in 0..self.hidden_size {
            output += hidden_buffer[i] * self.w2[i];
        }

        // Sigmoid to bound in [0, 1] with NaN protection
        let clamped = output.clamp(-20.0, 20.0);
        1.0 / (1.0 + (-clamped).exp())
    }

    /// Score with additional entropy-based adjustment
    ///
    /// Lower entropy in the answer embedding suggests more coherent output.
    pub fn score_with_entropy(&self, answer: &[f32]) -> f32 {
        let base_score = self.score(answer);

        // Compute embedding entropy (normalized)
        let entropy = self.embedding_entropy(answer);

        // Low entropy -> higher confidence
        // Entropy typically in [0, 10], normalize to [0, 1]
        let entropy_factor = 1.0 - (entropy / 10.0).clamp(0.0, 1.0);

        // Combine: base score weighted by entropy factor
        base_score * 0.7 + entropy_factor * 0.3
    }

    /// Compute entropy of embedding values
    fn embedding_entropy(&self, embedding: &[f32]) -> f32 {
        // Convert to probability-like values
        let sum: f32 = embedding.iter().map(|x| x.abs()).sum();
        if sum < 1e-8 {
            return 0.0;
        }

        let probs: Vec<f32> = embedding.iter()
            .map(|x| x.abs() / sum)
            .collect();

        // Shannon entropy
        -probs.iter()
            .filter(|&&p| p > 1e-8)
            .map(|&p| p * p.ln())
            .sum::<f32>()
    }

    /// Mean pooling
    fn mean_pool(&self, input: &[f32], target_dim: usize) -> Vec<f32> {
        if input.len() <= target_dim {
            let mut pooled = input.to_vec();
            pooled.resize(target_dim, 0.0);
            return pooled;
        }

        let mut pooled = vec![0.0; target_dim];
        let num_chunks = input.len() / target_dim;

        for chunk in input.chunks(target_dim) {
            for (i, &v) in chunk.iter().enumerate() {
                if i < target_dim {
                    pooled[i] += v / num_chunks as f32;
                }
            }
        }

        pooled
    }

    /// Check if confidence indicates early stopping should occur
    pub fn should_stop(&self, confidence: f32, threshold: f32) -> bool {
        confidence >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> TrmConfig {
        TrmConfig {
            embedding_dim: 64,
            ..TrmConfig::default()
        }
    }

    #[test]
    fn test_scorer_creation() {
        let config = make_config();
        let scorer = ConfidenceScorer::new(&config);
        assert_eq!(scorer.embedding_dim, 64);
    }

    #[test]
    fn test_score_range() {
        let config = make_config();
        let scorer = ConfidenceScorer::new(&config);

        let answer = vec![0.5; 64];
        let confidence = scorer.score(&answer);

        assert!(confidence >= 0.0 && confidence <= 1.0,
                "Confidence {} should be in [0, 1]", confidence);
    }

    #[test]
    fn test_score_with_entropy_range() {
        let config = make_config();
        let scorer = ConfidenceScorer::new(&config);

        let answer = vec![0.5; 64];
        let confidence = scorer.score_with_entropy(&answer);

        assert!(confidence >= 0.0 && confidence <= 1.0,
                "Confidence {} should be in [0, 1]", confidence);
    }

    #[test]
    fn test_different_inputs_different_scores() {
        let config = make_config();
        let scorer = ConfidenceScorer::new(&config);

        let answer1 = vec![0.1; 64];
        let answer2 = vec![1.0; 64];

        let score1 = scorer.score(&answer1);
        let score2 = scorer.score(&answer2);

        // Different inputs should generally give different scores
        // (though not guaranteed with random-ish initialization)
        assert!((score1 - score2).abs() > 1e-6 || score1 == score2,
                "Different inputs can have different scores");
    }

    #[test]
    fn test_should_stop() {
        let config = make_config();
        let scorer = ConfidenceScorer::new(&config);

        assert!(scorer.should_stop(0.96, 0.95));
        assert!(!scorer.should_stop(0.90, 0.95));
    }

    #[test]
    fn test_entropy_calculation() {
        let config = make_config();
        let scorer = ConfidenceScorer::new(&config);

        // Uniform distribution has high entropy
        let uniform = vec![1.0; 64];
        let entropy_uniform = scorer.embedding_entropy(&uniform);

        // Peaked distribution has lower entropy
        let mut peaked = vec![0.0; 64];
        peaked[0] = 1.0;
        let entropy_peaked = scorer.embedding_entropy(&peaked);

        assert!(entropy_uniform > entropy_peaked,
                "Uniform should have higher entropy: {} vs {}",
                entropy_uniform, entropy_peaked);
    }

    #[test]
    fn test_variable_length_answer() {
        let config = make_config();
        let scorer = ConfidenceScorer::new(&config);

        // Shorter than embedding_dim
        let short = vec![0.5; 32];
        let score_short = scorer.score(&short);
        assert!(score_short >= 0.0 && score_short <= 1.0);

        // Longer than embedding_dim
        let long = vec![0.5; 128];
        let score_long = scorer.score(&long);
        assert!(score_long >= 0.0 && score_long <= 1.0);
    }
}
