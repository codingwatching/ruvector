//! Vector storage and embedding engine.
//!
//! Provides HNSW-backed vector storage for captured frames with
//! cosine similarity search, metadata filtering, and a mock embedding
//! engine for development (production would use ONNX or API-based models).

pub mod embedding;
pub mod vector_store;

pub use embedding::EmbeddingEngine;
pub use vector_store::{SearchFilter, SearchResult, StoredEmbedding, VectorStore};
