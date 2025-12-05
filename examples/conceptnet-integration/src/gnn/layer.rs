//! GNN Layer Implementations for Knowledge Graphs
//!
//! Specialized layers for learning over ConceptNet's structure.

use crate::api::RelationType;
use std::collections::HashMap;
use rand::Rng;

/// Configuration for CommonsenseGNN
#[derive(Debug, Clone)]
pub struct GNNConfig {
    /// Input embedding dimension
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Number of GNN layers
    pub num_layers: usize,
    /// Use relation-aware attention
    pub relation_aware: bool,
    /// Residual connections
    pub residual: bool,
    /// Layer normalization
    pub layer_norm: bool,
}

impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 300,  // Numberbatch dimension
            hidden_dim: 256,
            output_dim: 128,
            num_heads: 4,
            dropout: 0.1,
            num_layers: 2,
            relation_aware: true,
            residual: true,
            layer_norm: true,
        }
    }
}

/// Main GNN model for commonsense reasoning
pub struct CommonsenseGNN {
    config: GNNConfig,
    layers: Vec<RelationAwareLayer>,
    relation_embeddings: HashMap<RelationType, Vec<f32>>,
    output_projection: Vec<f32>, // hidden_dim * output_dim
}

impl CommonsenseGNN {
    /// Create a new CommonsenseGNN
    pub fn new(config: GNNConfig) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize layers
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut in_dim = config.input_dim;

        for i in 0..config.num_layers {
            let out_dim = if i == config.num_layers - 1 {
                config.hidden_dim
            } else {
                config.hidden_dim
            };

            layers.push(RelationAwareLayer::new(
                in_dim,
                out_dim,
                config.num_heads,
                config.dropout,
            ));

            in_dim = out_dim;
        }

        // Initialize relation embeddings (one per relation type)
        let mut relation_embeddings = HashMap::new();
        for rel in Self::all_relations() {
            let emb: Vec<f32> = (0..config.hidden_dim)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect();
            relation_embeddings.insert(rel, emb);
        }

        // Output projection
        let output_projection: Vec<f32> = (0..config.hidden_dim * config.output_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self {
            config,
            layers,
            relation_embeddings,
            output_projection,
        }
    }

    /// Forward pass through the GNN
    pub fn forward(
        &self,
        node_embeddings: &[Vec<f32>],
        adjacency: &[(usize, usize, RelationType, f32)], // (src, dst, relation, weight)
    ) -> Vec<Vec<f32>> {
        let mut current = node_embeddings.to_vec();

        // Process through each layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut next = vec![vec![0.0; self.config.hidden_dim]; current.len()];

            // Aggregate messages for each node
            for &(src, dst, ref rel, weight) in adjacency {
                if src >= current.len() || dst >= current.len() {
                    continue;
                }

                let rel_emb = self.relation_embeddings.get(rel).unwrap();
                let msg = layer.compute_message(&current[src], rel_emb, weight);

                // Aggregate into destination
                for (i, val) in msg.iter().enumerate() {
                    if i < next[dst].len() {
                        next[dst][i] += val;
                    }
                }
            }

            // Apply transformation and activation
            for (i, node_hidden) in next.iter_mut().enumerate() {
                *node_hidden = layer.update(&current[i], node_hidden);

                // Residual connection
                if self.config.residual && layer_idx > 0 && current[i].len() == node_hidden.len() {
                    for (j, val) in current[i].iter().enumerate() {
                        node_hidden[j] += val;
                    }
                }

                // Layer normalization
                if self.config.layer_norm {
                    Self::layer_norm(node_hidden);
                }
            }

            current = next;
        }

        // Project to output dimension
        current
            .iter()
            .map(|h| self.project_output(h))
            .collect()
    }

    /// Compute similarity between two node embeddings
    pub fn compute_similarity(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
        Self::cosine_similarity(emb1, emb2)
    }

    /// Predict if an edge should exist
    pub fn predict_link(
        &self,
        src_emb: &[f32],
        dst_emb: &[f32],
        relation: &RelationType,
    ) -> f32 {
        let rel_emb = self.relation_embeddings.get(relation).unwrap();

        // TransE-style scoring: score = -||h + r - t||
        let mut score = 0.0;
        for i in 0..src_emb.len().min(dst_emb.len()).min(rel_emb.len()) {
            let diff = src_emb[i] + rel_emb[i] - dst_emb[i];
            score += diff * diff;
        }

        // Convert to probability
        1.0 / (1.0 + score.sqrt())
    }

    /// Get relation embedding
    pub fn get_relation_embedding(&self, relation: &RelationType) -> Option<&Vec<f32>> {
        self.relation_embeddings.get(relation)
    }

    // Helper methods

    fn project_output(&self, hidden: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.config.output_dim];

        for i in 0..self.config.output_dim {
            for j in 0..hidden.len().min(self.config.hidden_dim) {
                output[i] += hidden[j] * self.output_projection[i * self.config.hidden_dim + j];
            }
        }

        output
    }

    fn layer_norm(x: &mut [f32]) {
        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (var + 1e-6).sqrt();

        for v in x.iter_mut() {
            *v = (*v - mean) / std;
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
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
            RelationType::InstanceOf,
            RelationType::MadeOf,
            RelationType::MotivatedByGoal,
            RelationType::HasSubevent,
        ]
    }
}

/// Relation-aware message passing layer
pub struct RelationAwareLayer {
    input_dim: usize,
    output_dim: usize,
    num_heads: usize,
    dropout: f32,
    // Weights
    query_weights: Vec<f32>,   // input_dim * output_dim
    key_weights: Vec<f32>,     // input_dim * output_dim
    value_weights: Vec<f32>,   // input_dim * output_dim
    relation_weights: Vec<f32>, // output_dim * output_dim
    update_weights: Vec<f32>,  // 2*output_dim * output_dim
}

impl RelationAwareLayer {
    /// Create a new relation-aware layer
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize, dropout: f32) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();

        Self {
            input_dim,
            output_dim,
            num_heads,
            dropout,
            query_weights: (0..input_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            key_weights: (0..input_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            value_weights: (0..input_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            relation_weights: (0..output_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            update_weights: (0..2 * output_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
        }
    }

    /// Compute message from source to destination
    pub fn compute_message(&self, src: &[f32], relation: &[f32], weight: f32) -> Vec<f32> {
        // Transform source
        let transformed = self.linear_transform(src, &self.value_weights, self.output_dim);

        // Apply relation transformation
        let mut message = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            for j in 0..self.output_dim.min(transformed.len()) {
                message[i] += transformed[j] * self.relation_weights[i * self.output_dim + j];
            }
            // Add relation embedding
            if i < relation.len() {
                message[i] += relation[i] * 0.1;
            }
            // Scale by edge weight
            message[i] *= weight;
        }

        message
    }

    /// Update node representation
    pub fn update(&self, node: &[f32], aggregated: &[f32]) -> Vec<f32> {
        // Concatenate node representation with aggregated messages
        let mut combined = vec![0.0; self.output_dim * 2];
        for i in 0..self.output_dim.min(node.len()) {
            combined[i] = node[i];
        }
        for i in 0..self.output_dim.min(aggregated.len()) {
            combined[self.output_dim + i] = aggregated[i];
        }

        // Transform
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            for j in 0..combined.len() {
                output[i] += combined[j] * self.update_weights[i * 2 * self.output_dim + j];
            }
            // ReLU activation
            output[i] = output[i].max(0.0);
        }

        output
    }

    fn linear_transform(&self, input: &[f32], weights: &[f32], out_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0; out_dim];
        for i in 0..out_dim {
            for j in 0..input.len().min(self.input_dim) {
                output[i] += input[j] * weights[i * self.input_dim + j];
            }
        }
        output
    }
}

/// Standard message passing layer (relation-agnostic)
pub struct MessagePassingLayer {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl MessagePassingLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();

        Self {
            input_dim,
            output_dim,
            weights: (0..input_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            bias: (0..output_dim).map(|_| 0.0).collect(),
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            output[i] = self.bias[i];
            for j in 0..x.len().min(self.input_dim) {
                output[i] += x[j] * self.weights[i * self.input_dim + j];
            }
            output[i] = output[i].max(0.0); // ReLU
        }
        output
    }
}

/// Knowledge Graph Convolution layer (CompGCN-style)
pub struct KnowledgeGraphConv {
    input_dim: usize,
    output_dim: usize,
    composition_op: CompositionOp,
    weights_subject: Vec<f32>,
    weights_object: Vec<f32>,
    weights_self: Vec<f32>,
}

/// Composition operation for combining entity and relation embeddings
#[derive(Debug, Clone, Copy)]
pub enum CompositionOp {
    Subtraction,
    Multiplication,
    CircularCorrelation,
}

impl KnowledgeGraphConv {
    pub fn new(input_dim: usize, output_dim: usize, composition_op: CompositionOp) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();

        Self {
            input_dim,
            output_dim,
            composition_op,
            weights_subject: (0..input_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            weights_object: (0..input_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
            weights_self: (0..input_dim * output_dim)
                .map(|_| rng.gen_range(-scale..scale))
                .collect(),
        }
    }

    /// Compose entity and relation embeddings
    pub fn compose(&self, entity: &[f32], relation: &[f32]) -> Vec<f32> {
        match self.composition_op {
            CompositionOp::Subtraction => {
                entity
                    .iter()
                    .zip(relation.iter())
                    .map(|(e, r)| e - r)
                    .collect()
            }
            CompositionOp::Multiplication => {
                entity
                    .iter()
                    .zip(relation.iter())
                    .map(|(e, r)| e * r)
                    .collect()
            }
            CompositionOp::CircularCorrelation => {
                // Simplified circular correlation
                let n = entity.len().min(relation.len());
                let mut result = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        result[i] += entity[j] * relation[(i + j) % n];
                    }
                }
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_forward() {
        let config = GNNConfig {
            input_dim: 64,
            hidden_dim: 32,
            output_dim: 16,
            num_heads: 2,
            num_layers: 2,
            ..Default::default()
        };

        let gnn = CommonsenseGNN::new(config);

        // Create some node embeddings
        let embeddings = vec![
            vec![0.1; 64],
            vec![0.2; 64],
            vec![0.3; 64],
        ];

        // Create adjacency list
        let adjacency = vec![
            (0, 1, RelationType::IsA, 1.0),
            (1, 2, RelationType::HasA, 0.8),
            (0, 2, RelationType::RelatedTo, 0.5),
        ];

        let outputs = gnn.forward(&embeddings, &adjacency);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 16);
    }

    #[test]
    fn test_link_prediction() {
        let config = GNNConfig::default();
        let gnn = CommonsenseGNN::new(config);

        let src = vec![0.5; 300];
        let dst = vec![0.5; 300];

        let score = gnn.predict_link(&src, &dst, &RelationType::IsA);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
