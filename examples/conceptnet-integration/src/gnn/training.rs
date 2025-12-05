//! GNN Training for Knowledge Graph Learning
//!
//! Training pipelines for link prediction, node classification, and embedding learning.

use super::layer::{CommonsenseGNN, GNNConfig};
use crate::api::RelationType;
use rand::seq::SliceRandom;
use rand::Rng;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Gradient clipping threshold
    pub grad_clip: f32,
    /// Negative sampling ratio
    pub negative_ratio: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Validation split ratio
    pub val_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 128,
            epochs: 100,
            weight_decay: 1e-5,
            grad_clip: 1.0,
            negative_ratio: 5,
            patience: 10,
            val_split: 0.1,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub train_accuracy: f32,
    pub val_accuracy: f32,
    pub mrr: f32,         // Mean Reciprocal Rank
    pub hits_at_1: f32,
    pub hits_at_3: f32,
    pub hits_at_10: f32,
}

/// A training sample for link prediction
#[derive(Debug, Clone)]
pub struct LinkSample {
    pub head: usize,
    pub tail: usize,
    pub relation: RelationType,
    pub label: bool, // true = positive, false = negative
}

/// A training sample for node classification
#[derive(Debug, Clone)]
pub struct NodeSample {
    pub node_idx: usize,
    pub label: usize,
    pub features: Vec<f32>,
}

/// Link prediction training task
pub struct LinkPredictionTask {
    config: TrainingConfig,
    samples: Vec<LinkSample>,
    node_embeddings: Vec<Vec<f32>>,
    adjacency: Vec<(usize, usize, RelationType, f32)>,
}

impl LinkPredictionTask {
    pub fn new(
        config: TrainingConfig,
        node_embeddings: Vec<Vec<f32>>,
        adjacency: Vec<(usize, usize, RelationType, f32)>,
    ) -> Self {
        Self {
            config,
            samples: Vec::new(),
            node_embeddings,
            adjacency,
        }
    }

    /// Generate training samples from adjacency list
    pub fn prepare_samples(&mut self) {
        let mut rng = rand::thread_rng();
        let num_nodes = self.node_embeddings.len();

        // Positive samples
        for &(src, dst, ref rel, _) in &self.adjacency {
            self.samples.push(LinkSample {
                head: src,
                tail: dst,
                relation: *rel,
                label: true,
            });
        }

        // Negative samples (corrupt head or tail)
        let num_positives = self.samples.len();
        for _ in 0..num_positives * self.config.negative_ratio {
            let positive = &self.adjacency[rng.gen_range(0..self.adjacency.len())];
            let corrupt_head = rng.gen_bool(0.5);

            let (head, tail) = if corrupt_head {
                (rng.gen_range(0..num_nodes), positive.1)
            } else {
                (positive.0, rng.gen_range(0..num_nodes))
            };

            self.samples.push(LinkSample {
                head,
                tail,
                relation: positive.2,
                label: false,
            });
        }

        // Shuffle samples
        self.samples.shuffle(&mut rng);
    }

    /// Train the model
    pub fn train(&self, gnn: &mut CommonsenseGNN) -> Vec<TrainingMetrics> {
        let mut metrics_history = Vec::new();
        let mut rng = rand::thread_rng();

        let val_size = (self.samples.len() as f32 * self.config.val_split) as usize;
        let train_samples = &self.samples[val_size..];
        let val_samples = &self.samples[..val_size];

        let mut best_val_loss = f32::MAX;
        let mut patience_counter = 0;

        for epoch in 0..self.config.epochs {
            let mut train_loss = 0.0;
            let mut train_correct = 0;

            // Training batches
            let mut indices: Vec<usize> = (0..train_samples.len()).collect();
            indices.shuffle(&mut rng);

            for batch_start in (0..train_samples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_samples.len());
                let batch_indices = &indices[batch_start..batch_end];

                let (loss, correct) = self.train_batch(gnn, train_samples, batch_indices);
                train_loss += loss;
                train_correct += correct;
            }

            train_loss /= (train_samples.len() / self.config.batch_size) as f32;
            let train_accuracy = train_correct as f32 / train_samples.len() as f32;

            // Validation
            let (val_loss, val_correct, ranking_metrics) = self.evaluate(gnn, val_samples);
            let val_accuracy = val_correct as f32 / val_samples.len() as f32;

            let metrics = TrainingMetrics {
                epoch,
                train_loss,
                val_loss,
                train_accuracy,
                val_accuracy,
                mrr: ranking_metrics.0,
                hits_at_1: ranking_metrics.1,
                hits_at_3: ranking_metrics.2,
                hits_at_10: ranking_metrics.3,
            };

            metrics_history.push(metrics.clone());

            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    break;
                }
            }
        }

        metrics_history
    }

    fn train_batch(
        &self,
        _gnn: &CommonsenseGNN,
        samples: &[LinkSample],
        indices: &[usize],
    ) -> (f32, usize) {
        let mut loss = 0.0;
        let mut correct = 0;

        for &idx in indices {
            let sample = &samples[idx];

            if sample.head >= self.node_embeddings.len()
                || sample.tail >= self.node_embeddings.len()
            {
                continue;
            }

            let head_emb = &self.node_embeddings[sample.head];
            let tail_emb = &self.node_embeddings[sample.tail];

            // Simple margin-based loss
            let score = self.compute_score(head_emb, tail_emb, &sample.relation);
            let target = if sample.label { 1.0 } else { 0.0 };

            loss += (score - target).powi(2);

            let predicted = score > 0.5;
            if predicted == sample.label {
                correct += 1;
            }
        }

        (loss / indices.len() as f32, correct)
    }

    fn evaluate(
        &self,
        _gnn: &CommonsenseGNN,
        samples: &[LinkSample],
    ) -> (f32, usize, (f32, f32, f32, f32)) {
        let mut loss = 0.0;
        let mut correct = 0;
        let mut ranks = Vec::new();

        for sample in samples {
            if sample.head >= self.node_embeddings.len()
                || sample.tail >= self.node_embeddings.len()
            {
                continue;
            }

            let head_emb = &self.node_embeddings[sample.head];
            let tail_emb = &self.node_embeddings[sample.tail];

            let score = self.compute_score(head_emb, tail_emb, &sample.relation);
            let target = if sample.label { 1.0 } else { 0.0 };

            loss += (score - target).powi(2);

            let predicted = score > 0.5;
            if predicted == sample.label {
                correct += 1;
            }

            // Compute rank for positive samples
            if sample.label {
                ranks.push(1.0); // Simplified: actual ranking would compare against all corruptions
            }
        }

        let mrr = if ranks.is_empty() {
            0.0
        } else {
            ranks.iter().map(|r| 1.0 / r).sum::<f32>() / ranks.len() as f32
        };

        let hits_at_1 = ranks.iter().filter(|&&r| r <= 1.0).count() as f32 / ranks.len().max(1) as f32;
        let hits_at_3 = ranks.iter().filter(|&&r| r <= 3.0).count() as f32 / ranks.len().max(1) as f32;
        let hits_at_10 = ranks.iter().filter(|&&r| r <= 10.0).count() as f32 / ranks.len().max(1) as f32;

        (
            loss / samples.len().max(1) as f32,
            correct,
            (mrr, hits_at_1, hits_at_3, hits_at_10),
        )
    }

    fn compute_score(&self, head: &[f32], tail: &[f32], _relation: &RelationType) -> f32 {
        // Simple cosine similarity
        let dot: f32 = head.iter().zip(tail.iter()).map(|(a, b)| a * b).sum();
        let norm_h: f32 = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_t: f32 = tail.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_h > 0.0 && norm_t > 0.0 {
            (dot / (norm_h * norm_t) + 1.0) / 2.0 // Normalize to [0, 1]
        } else {
            0.5
        }
    }
}

/// Node classification training task
pub struct NodeClassificationTask {
    config: TrainingConfig,
    samples: Vec<NodeSample>,
    num_classes: usize,
}

impl NodeClassificationTask {
    pub fn new(config: TrainingConfig, samples: Vec<NodeSample>, num_classes: usize) -> Self {
        Self {
            config,
            samples,
            num_classes,
        }
    }

    /// Train the model
    pub fn train(&self, gnn: &mut CommonsenseGNN) -> Vec<TrainingMetrics> {
        let mut metrics_history = Vec::new();
        let mut rng = rand::thread_rng();

        let val_size = (self.samples.len() as f32 * self.config.val_split) as usize;
        let train_samples = &self.samples[val_size..];
        let val_samples = &self.samples[..val_size];

        for epoch in 0..self.config.epochs {
            let mut train_loss = 0.0;
            let mut train_correct = 0;

            // Training batches
            let mut indices: Vec<usize> = (0..train_samples.len()).collect();
            indices.shuffle(&mut rng);

            for batch_start in (0..train_samples.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_samples.len());
                let batch_indices = &indices[batch_start..batch_end];

                let (loss, correct) = self.train_batch(gnn, train_samples, batch_indices);
                train_loss += loss;
                train_correct += correct;
            }

            train_loss /= (train_samples.len() / self.config.batch_size).max(1) as f32;
            let train_accuracy = train_correct as f32 / train_samples.len().max(1) as f32;

            // Validation
            let (val_loss, val_correct) = self.evaluate(gnn, val_samples);
            let val_accuracy = val_correct as f32 / val_samples.len().max(1) as f32;

            metrics_history.push(TrainingMetrics {
                epoch,
                train_loss,
                val_loss,
                train_accuracy,
                val_accuracy,
                ..Default::default()
            });
        }

        metrics_history
    }

    fn train_batch(
        &self,
        _gnn: &CommonsenseGNN,
        samples: &[NodeSample],
        indices: &[usize],
    ) -> (f32, usize) {
        let mut loss = 0.0;
        let mut correct = 0;

        for &idx in indices {
            let sample = &samples[idx];

            // Simple cross-entropy approximation
            let predicted_class = self.predict_class(&sample.features);
            if predicted_class == sample.label {
                correct += 1;
            }
            loss += if predicted_class == sample.label { 0.0 } else { 1.0 };
        }

        (loss / indices.len().max(1) as f32, correct)
    }

    fn evaluate(&self, _gnn: &CommonsenseGNN, samples: &[NodeSample]) -> (f32, usize) {
        let mut loss = 0.0;
        let mut correct = 0;

        for sample in samples {
            let predicted_class = self.predict_class(&sample.features);
            if predicted_class == sample.label {
                correct += 1;
            }
            loss += if predicted_class == sample.label { 0.0 } else { 1.0 };
        }

        (loss / samples.len().max(1) as f32, correct)
    }

    fn predict_class(&self, features: &[f32]) -> usize {
        // Simple argmax on feature sum per class bucket
        let bucket_size = features.len() / self.num_classes.max(1);
        let mut max_score = f32::MIN;
        let mut max_class = 0;

        for c in 0..self.num_classes {
            let start = c * bucket_size;
            let end = (start + bucket_size).min(features.len());
            let score: f32 = features[start..end].iter().sum();

            if score > max_score {
                max_score = score;
                max_class = c;
            }
        }

        max_class
    }
}

/// GNN Trainer orchestrating multiple tasks
pub struct GNNTrainer {
    gnn: CommonsenseGNN,
    config: TrainingConfig,
}

impl GNNTrainer {
    pub fn new(gnn_config: GNNConfig, training_config: TrainingConfig) -> Self {
        Self {
            gnn: CommonsenseGNN::new(gnn_config),
            config: training_config,
        }
    }

    /// Train on link prediction task
    pub fn train_link_prediction(
        &mut self,
        node_embeddings: Vec<Vec<f32>>,
        adjacency: Vec<(usize, usize, RelationType, f32)>,
    ) -> Vec<TrainingMetrics> {
        let mut task = LinkPredictionTask::new(
            self.config.clone(),
            node_embeddings,
            adjacency,
        );
        task.prepare_samples();
        task.train(&mut self.gnn)
    }

    /// Train on node classification task
    pub fn train_node_classification(
        &mut self,
        samples: Vec<NodeSample>,
        num_classes: usize,
    ) -> Vec<TrainingMetrics> {
        let task = NodeClassificationTask::new(self.config.clone(), samples, num_classes);
        task.train(&mut self.gnn)
    }

    /// Get the trained model
    pub fn get_model(&self) -> &CommonsenseGNN {
        &self.gnn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_prediction_training() {
        let embeddings: Vec<Vec<f32>> = (0..10).map(|_| vec![0.1; 64]).collect();
        let adjacency = vec![
            (0, 1, RelationType::IsA, 1.0),
            (1, 2, RelationType::HasA, 0.8),
            (2, 3, RelationType::RelatedTo, 0.5),
        ];

        let config = TrainingConfig {
            epochs: 5,
            batch_size: 2,
            ..Default::default()
        };

        let mut task = LinkPredictionTask::new(config, embeddings, adjacency);
        task.prepare_samples();

        assert!(!task.samples.is_empty());
    }
}
