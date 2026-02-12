//! HNSW-backed vector storage.
//!
//! Provides in-memory vector storage with cosine similarity search
//! and metadata filtering. In production, this would wrap ruvector-core's
//! VectorDB with HNSW indexing for O(log n) search.

use crate::capture::CapturedFrame;
use crate::config::StorageConfig;
use crate::error::{OsPipeError, Result};
use crate::storage::embedding::cosine_similarity;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A vector embedding stored with its metadata.
#[derive(Debug, Clone)]
pub struct StoredEmbedding {
    /// Unique identifier matching the source frame.
    pub id: Uuid,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// JSON metadata about the source frame.
    pub metadata: serde_json::Value,
    /// When the source frame was captured.
    pub timestamp: DateTime<Utc>,
}

/// A search result returned from the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// ID of the matched embedding.
    pub id: Uuid,
    /// Cosine similarity score (higher is more similar).
    pub score: f32,
    /// Metadata of the matched embedding.
    pub metadata: serde_json::Value,
}

/// Filter criteria for narrowing search results.
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Filter by application name.
    pub app: Option<String>,
    /// Filter by start time (inclusive).
    pub time_start: Option<DateTime<Utc>>,
    /// Filter by end time (inclusive).
    pub time_end: Option<DateTime<Utc>>,
    /// Filter by content type (e.g., "ocr", "transcription", "ui_event").
    pub content_type: Option<String>,
    /// Filter by monitor index.
    pub monitor: Option<u32>,
}

/// In-memory vector store with brute-force cosine similarity search.
///
/// This is a development implementation. Production deployments would
/// use ruvector-core's HNSW index for approximate nearest neighbor
/// search at O(log n) complexity.
pub struct VectorStore {
    config: StorageConfig,
    embeddings: Vec<StoredEmbedding>,
    dimension: usize,
}

impl VectorStore {
    /// Create a new vector store with the given configuration.
    pub fn new(config: StorageConfig) -> Result<Self> {
        let dimension = config.embedding_dim;
        if dimension == 0 {
            return Err(OsPipeError::Storage(
                "embedding_dim must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            config,
            embeddings: Vec::new(),
            dimension,
        })
    }

    /// Insert a captured frame with its pre-computed embedding.
    pub fn insert(&mut self, frame: &CapturedFrame, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(OsPipeError::Storage(format!(
                "Expected embedding dimension {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        let metadata = serde_json::json!({
            "text": frame.text_content(),
            "content_type": frame.content_type(),
            "app_name": frame.metadata.app_name,
            "window_title": frame.metadata.window_title,
            "monitor_id": frame.metadata.monitor_id,
            "confidence": frame.metadata.confidence,
        });

        self.embeddings.push(StoredEmbedding {
            id: frame.id,
            vector: embedding.to_vec(),
            metadata,
            timestamp: frame.timestamp,
        });

        Ok(())
    }

    /// Search for the k most similar embeddings to the query vector.
    pub fn search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query_embedding.len() != self.dimension {
            return Err(OsPipeError::Search(format!(
                "Expected query dimension {}, got {}",
                self.dimension,
                query_embedding.len()
            )));
        }

        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, stored)| {
                let score = cosine_similarity(query_embedding, &stored.vector);
                (i, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(i, score)| {
                let stored = &self.embeddings[i];
                SearchResult {
                    id: stored.id,
                    score,
                    metadata: stored.metadata.clone(),
                }
            })
            .collect())
    }

    /// Search with metadata filtering applied before scoring.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(OsPipeError::Search(format!(
                "Expected query dimension {}, got {}",
                self.dimension,
                query.len()
            )));
        }

        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .filter(|(_, stored)| self.matches_filter(stored, filter))
            .map(|(i, stored)| {
                let score = cosine_similarity(query, &stored.vector);
                (i, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(i, score)| {
                let stored = &self.embeddings[i];
                SearchResult {
                    id: stored.id,
                    score,
                    metadata: stored.metadata.clone(),
                }
            })
            .collect())
    }

    /// Return the number of stored embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Return true if the store contains no embeddings.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Return the configured embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return a reference to the storage configuration.
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    /// Get a stored embedding by its ID.
    pub fn get(&self, id: &Uuid) -> Option<&StoredEmbedding> {
        self.embeddings.iter().find(|e| e.id == *id)
    }

    /// Check whether a stored embedding matches the given filter.
    fn matches_filter(&self, stored: &StoredEmbedding, filter: &SearchFilter) -> bool {
        if let Some(ref app) = filter.app {
            let stored_app = stored
                .metadata
                .get("app_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if stored_app != app {
                return false;
            }
        }

        if let Some(start) = filter.time_start {
            if stored.timestamp < start {
                return false;
            }
        }

        if let Some(end) = filter.time_end {
            if stored.timestamp > end {
                return false;
            }
        }

        if let Some(ref ct) = filter.content_type {
            let stored_ct = stored
                .metadata
                .get("content_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if stored_ct != ct {
                return false;
            }
        }

        if let Some(monitor) = filter.monitor {
            let stored_monitor = stored
                .metadata
                .get("monitor_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            if stored_monitor != Some(monitor) {
                return false;
            }
        }

        true
    }
}
