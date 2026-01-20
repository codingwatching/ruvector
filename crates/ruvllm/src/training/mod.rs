//! # Training Module
//!
//! This module provides training data generation and fine-tuning utilities
//! for RuvLTRA models, including Claude Flow task datasets.

pub mod claude_dataset;

#[cfg(test)]
mod tests;

pub use claude_dataset::{
    ClaudeTaskDataset, ClaudeTaskExample, TaskCategory, TaskMetadata,
    ComplexityLevel, DomainType, DatasetConfig, AugmentationConfig,
    DatasetGenerator, DatasetStats,
};
