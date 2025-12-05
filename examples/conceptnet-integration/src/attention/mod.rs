//! Attention Mechanisms for Commonsense Reasoning
//!
//! Specialized attention mechanisms for knowledge graph reasoning,
//! including relation-aware, hierarchical, and causal attention.
//!
//! ## Features
//! - Relation-typed attention weighting
//! - Hierarchical attention for taxonomies
//! - Causal attention for event chains
//! - Cross-lingual attention for multilingual ConceptNet
//! - Sparse attention for large graphs

use crate::api::RelationType;
use std::collections::HashMap;

/// Configuration for commonsense attention
#[derive(Debug, Clone)]
pub struct CommonsenseAttentionConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Use relation-specific projections
    pub relation_aware: bool,
    /// Temperature for softmax
    pub temperature: f32,
    /// Sparse attention threshold
    pub sparse_threshold: f32,
}

impl Default for CommonsenseAttentionConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_heads: 8,
            dropout: 0.1,
            relation_aware: true,
            temperature: 1.0,
            sparse_threshold: 0.01,
        }
    }
}

/// Relation-aware attention for knowledge graph reasoning
pub struct RelationAttention {
    config: CommonsenseAttentionConfig,
    relation_projections: HashMap<RelationType, Vec<f32>>,
    query_projection: Vec<f32>,
    key_projection: Vec<f32>,
    value_projection: Vec<f32>,
    output_projection: Vec<f32>,
}

impl RelationAttention {
    /// Create new relation attention
    pub fn new(config: CommonsenseAttentionConfig) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let dim = config.hidden_dim;
        let scale = (2.0 / (dim * 2) as f32).sqrt();

        let mut relation_projections = HashMap::new();
        for rel in Self::all_relations() {
            let proj: Vec<f32> = (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect();
            relation_projections.insert(rel, proj);
        }

        Self {
            query_projection: (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect(),
            key_projection: (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect(),
            value_projection: (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect(),
            output_projection: (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect(),
            config,
            relation_projections,
        }
    }

    /// Compute attention over neighbors with relation awareness
    pub fn forward(
        &self,
        query: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        relations: &[RelationType],
    ) -> Vec<f32> {
        let dim = self.config.hidden_dim;
        let head_dim = dim / self.config.num_heads;

        // Project query
        let q = self.linear_transform(query, &self.query_projection, dim);

        // Compute attention scores for each neighbor
        let mut scores: Vec<f32> = keys
            .iter()
            .zip(relations.iter())
            .map(|(key, rel)| {
                let k = self.linear_transform(key, &self.key_projection, dim);

                // Apply relation-specific transformation
                let k_rel = if self.config.relation_aware {
                    if let Some(rel_proj) = self.relation_projections.get(rel) {
                        self.linear_transform(&k, rel_proj, dim)
                    } else {
                        k
                    }
                } else {
                    k
                };

                // Scaled dot-product attention
                let score: f32 = q.iter().zip(k_rel.iter()).map(|(a, b)| a * b).sum();
                score / (head_dim as f32).sqrt() / self.config.temperature
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let attention_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();

        // Weighted sum of values
        let mut output = vec![0.0; dim];
        for (i, (value, weight)) in values.iter().zip(attention_weights.iter()).enumerate() {
            let v = self.linear_transform(value, &self.value_projection, dim);

            // Apply relation-aware value transformation
            let v_rel = if self.config.relation_aware && i < relations.len() {
                if let Some(rel_proj) = self.relation_projections.get(&relations[i]) {
                    self.linear_transform(&v, rel_proj, dim)
                } else {
                    v
                }
            } else {
                v
            };

            for (j, val) in v_rel.iter().enumerate() {
                output[j] += val * weight;
            }
        }

        // Output projection
        self.linear_transform(&output, &self.output_projection, dim)
    }

    /// Get attention weights for interpretability
    pub fn get_attention_weights(
        &self,
        query: &[f32],
        keys: &[Vec<f32>],
        relations: &[RelationType],
    ) -> Vec<f32> {
        let dim = self.config.hidden_dim;
        let head_dim = dim / self.config.num_heads;

        let q = self.linear_transform(query, &self.query_projection, dim);

        let mut scores: Vec<f32> = keys
            .iter()
            .zip(relations.iter())
            .map(|(key, rel)| {
                let k = self.linear_transform(key, &self.key_projection, dim);
                let k_rel = if self.config.relation_aware {
                    if let Some(rel_proj) = self.relation_projections.get(rel) {
                        self.linear_transform(&k, rel_proj, dim)
                    } else {
                        k
                    }
                } else {
                    k
                };

                let score: f32 = q.iter().zip(k_rel.iter()).map(|(a, b)| a * b).sum();
                score / (head_dim as f32).sqrt()
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        exp_scores.iter().map(|e| e / sum).collect()
    }

    fn linear_transform(&self, input: &[f32], weights: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len();
        let mut output = vec![0.0; out_dim];

        for i in 0..out_dim {
            for j in 0..in_dim.min(out_dim) {
                if i * in_dim + j < weights.len() {
                    output[i] += input[j] * weights[i * in_dim + j];
                }
            }
        }

        output
    }

    fn all_relations() -> Vec<RelationType> {
        vec![
            RelationType::IsA,
            RelationType::PartOf,
            RelationType::HasA,
            RelationType::UsedFor,
            RelationType::CapableOf,
            RelationType::AtLocation,
            RelationType::Causes,
            RelationType::HasPrerequisite,
            RelationType::HasProperty,
            RelationType::RelatedTo,
            RelationType::SimilarTo,
            RelationType::Synonym,
            RelationType::Antonym,
        ]
    }
}

/// Hierarchical attention for taxonomic reasoning
pub struct HierarchicalAttention {
    config: CommonsenseAttentionConfig,
    level_projections: Vec<Vec<f32>>,
    max_levels: usize,
}

impl HierarchicalAttention {
    /// Create hierarchical attention
    pub fn new(config: CommonsenseAttentionConfig, max_levels: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let dim = config.hidden_dim;
        let scale = (2.0 / (dim * 2) as f32).sqrt();

        let level_projections: Vec<Vec<f32>> = (0..max_levels)
            .map(|_| (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();

        Self {
            config,
            level_projections,
            max_levels,
        }
    }

    /// Compute hierarchical attention with level weighting
    pub fn forward(
        &self,
        query: &[f32],
        hierarchies: &[Vec<(Vec<f32>, usize)>], // (embedding, level) per path
    ) -> Vec<f32> {
        let dim = self.config.hidden_dim;
        let mut output = vec![0.0; dim];
        let mut total_weight = 0.0;

        for path in hierarchies {
            for (embedding, level) in path {
                if *level >= self.max_levels {
                    continue;
                }

                // Level decay: closer levels have higher weight
                let level_weight = 1.0 / (1.0 + *level as f32);

                // Similarity with query
                let sim = self.cosine_similarity(query, embedding);

                let weight = level_weight * sim.max(0.0);
                total_weight += weight;

                // Apply level-specific projection
                let projected = self.linear_transform(
                    embedding,
                    &self.level_projections[*level],
                    dim,
                );

                for (i, val) in projected.iter().enumerate() {
                    output[i] += val * weight;
                }
            }
        }

        if total_weight > 0.0 {
            for val in &mut output {
                *val /= total_weight;
            }
        }

        output
    }

    fn linear_transform(&self, input: &[f32], weights: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len();
        let mut output = vec![0.0; out_dim];

        for i in 0..out_dim {
            for j in 0..in_dim.min(out_dim) {
                if i * in_dim + j < weights.len() {
                    output[i] += input[j] * weights[i * in_dim + j];
                }
            }
        }

        output
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Causal attention for event chain reasoning
pub struct CausalAttention {
    config: CommonsenseAttentionConfig,
    temporal_decay: f32,
    forward_projection: Vec<f32>,
    backward_projection: Vec<f32>,
}

impl CausalAttention {
    /// Create causal attention
    pub fn new(config: CommonsenseAttentionConfig, temporal_decay: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let dim = config.hidden_dim;
        let scale = (2.0 / (dim * 2) as f32).sqrt();

        Self {
            forward_projection: (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect(),
            backward_projection: (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect(),
            config,
            temporal_decay,
        }
    }

    /// Compute causal attention over event sequence
    pub fn forward(
        &self,
        query: &[f32],
        events: &[(Vec<f32>, i32)], // (embedding, temporal_offset) - negative = past, positive = future
    ) -> Vec<f32> {
        let dim = self.config.hidden_dim;
        let mut output = vec![0.0; dim];
        let mut total_weight = 0.0;

        for (embedding, offset) in events {
            // Temporal weighting: decay with distance, but allow both past and future
            let temporal_weight = (-(*offset as f32).abs() * self.temporal_decay).exp();

            // Use different projection for past vs future
            let projected = if *offset <= 0 {
                self.linear_transform(embedding, &self.backward_projection, dim)
            } else {
                self.linear_transform(embedding, &self.forward_projection, dim)
            };

            // Attention with query
            let sim = self.cosine_similarity(query, &projected);
            let weight = temporal_weight * sim.max(0.0);
            total_weight += weight;

            for (i, val) in projected.iter().enumerate() {
                output[i] += val * weight;
            }
        }

        if total_weight > 0.0 {
            for val in &mut output {
                *val /= total_weight;
            }
        }

        output
    }

    fn linear_transform(&self, input: &[f32], weights: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len();
        let mut output = vec![0.0; out_dim];

        for i in 0..out_dim {
            for j in 0..in_dim.min(out_dim) {
                if i * in_dim + j < weights.len() {
                    output[i] += input[j] * weights[i * in_dim + j];
                }
            }
        }

        output
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Cross-lingual attention for multilingual reasoning
pub struct CrossLingualAttention {
    config: CommonsenseAttentionConfig,
    language_projections: HashMap<String, Vec<f32>>,
    alignment_projection: Vec<f32>,
}

impl CrossLingualAttention {
    /// Create cross-lingual attention
    pub fn new(config: CommonsenseAttentionConfig, languages: &[&str]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let dim = config.hidden_dim;
        let scale = (2.0 / (dim * 2) as f32).sqrt();

        let mut language_projections = HashMap::new();
        for lang in languages {
            let proj: Vec<f32> = (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect();
            language_projections.insert(lang.to_string(), proj);
        }

        Self {
            alignment_projection: (0..dim * dim).map(|_| rng.gen_range(-scale..scale)).collect(),
            config,
            language_projections,
        }
    }

    /// Compute cross-lingual attention
    pub fn forward(
        &self,
        query: &[f32],
        query_language: &str,
        candidates: &[(Vec<f32>, String)], // (embedding, language)
    ) -> Vec<f32> {
        let dim = self.config.hidden_dim;

        // Project query to shared space
        let query_proj = if let Some(lang_proj) = self.language_projections.get(query_language) {
            self.linear_transform(query, lang_proj, dim)
        } else {
            query.to_vec()
        };

        let aligned_query = self.linear_transform(&query_proj, &self.alignment_projection, dim);

        // Compute attention over candidates
        let mut scores: Vec<f32> = candidates
            .iter()
            .map(|(emb, lang)| {
                let emb_proj = if let Some(lang_proj) = self.language_projections.get(lang) {
                    self.linear_transform(emb, lang_proj, dim)
                } else {
                    emb.clone()
                };

                let aligned_emb = self.linear_transform(&emb_proj, &self.alignment_projection, dim);
                self.cosine_similarity(&aligned_query, &aligned_emb)
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();

        // Weighted sum
        let mut output = vec![0.0; dim];
        for ((emb, lang), weight) in candidates.iter().zip(weights.iter()) {
            let emb_proj = if let Some(lang_proj) = self.language_projections.get(lang) {
                self.linear_transform(emb, lang_proj, dim)
            } else {
                emb.clone()
            };

            for (i, val) in emb_proj.iter().enumerate() {
                if i < dim {
                    output[i] += val * weight;
                }
            }
        }

        output
    }

    fn linear_transform(&self, input: &[f32], weights: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len();
        let mut output = vec![0.0; out_dim];

        for i in 0..out_dim {
            for j in 0..in_dim.min(out_dim) {
                if i * in_dim + j < weights.len() {
                    output[i] += input[j] * weights[i * in_dim + j];
                }
            }
        }

        output
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relation_attention() {
        let config = CommonsenseAttentionConfig {
            hidden_dim: 64,
            num_heads: 4,
            ..Default::default()
        };

        let attention = RelationAttention::new(config);

        let query = vec![0.1; 64];
        let keys = vec![vec![0.2; 64], vec![0.3; 64]];
        let values = vec![vec![0.4; 64], vec![0.5; 64]];
        let relations = vec![RelationType::IsA, RelationType::HasA];

        let output = attention.forward(&query, &keys, &values, &relations);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_hierarchical_attention() {
        let config = CommonsenseAttentionConfig {
            hidden_dim: 64,
            ..Default::default()
        };

        let attention = HierarchicalAttention::new(config, 5);

        let query = vec![0.1; 64];
        let hierarchies = vec![vec![
            (vec![0.2; 64], 0),
            (vec![0.3; 64], 1),
            (vec![0.4; 64], 2),
        ]];

        let output = attention.forward(&query, &hierarchies);
        assert_eq!(output.len(), 64);
    }
}
