//! WASM Simulator backend
//!
//! Pure Rust implementation that runs in WASM environments.
//! Uses the same quantized kernels as native sim but optimized for WASM.

#![cfg(feature = "wasm")]

use std::collections::HashMap;
use std::sync::Arc;

use crate::artifact::ModelArtifact;
use crate::backend::{BackendStats, TransformerBackend};
use crate::error::{Error, Result};
use crate::gating::CoherenceGate;
use crate::types::{
    BackendKind, FixedShape, GateDecision, GateHint, InferenceRequest, InferenceResult, ModelId,
    QuantSpec, WitnessLog,
};

/// Loaded model for WASM simulation
struct WasmModel {
    /// Model artifact
    artifact: ModelArtifact,
    /// Prepacked embedding table
    embeddings: Vec<i8>,
    /// Number of layers
    num_layers: usize,
}

/// WASM simulator backend
pub struct WasmSimBackend {
    /// Loaded models
    models: HashMap<ModelId, WasmModel>,
    /// Coherence gate
    gate: Arc<dyn CoherenceGate>,
    /// Statistics
    stats: BackendStats,
}

impl WasmSimBackend {
    /// Create a new WASM simulator backend
    pub fn new(gate: Arc<dyn CoherenceGate>) -> Self {
        Self {
            models: HashMap::new(),
            gate,
            stats: BackendStats::default(),
        }
    }

    /// Run simplified inference for WASM
    fn run_inference(
        &self,
        model: &WasmModel,
        tokens: &[u16],
        gate_hint: &GateHint,
    ) -> (Vec<i16>, GateDecision) {
        let shape = &model.artifact.manifest.shape;

        // Check preflight
        let preflight = self.gate.preflight(gate_hint);
        if !preflight.did_run() {
            return (vec![0i16; shape.vocab as usize], preflight);
        }

        // Simple embedding lookup and output
        let vocab = shape.vocab as usize;
        let d_model = shape.d_model as usize;

        // Generate output logits (simplified for WASM)
        let mut logits = vec![0i16; vocab];

        // Use last token for prediction
        if let Some(&last_token) = tokens.last() {
            // Simple next-token prediction based on embedding similarity
            let token_idx = last_token as usize;
            let embed_offset = token_idx * d_model;

            if embed_offset + d_model <= model.embeddings.len() {
                // Compute dot products with all embeddings
                for v in 0..vocab.min(model.embeddings.len() / d_model) {
                    let v_offset = v * d_model;
                    let mut dot = 0i32;
                    for d in 0..d_model {
                        if embed_offset + d < model.embeddings.len()
                            && v_offset + d < model.embeddings.len()
                        {
                            dot += model.embeddings[embed_offset + d] as i32
                                * model.embeddings[v_offset + d] as i32;
                        }
                    }
                    logits[v] = (dot >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
            } else {
                // Fallback: use token index for deterministic output
                for v in 0..vocab {
                    logits[v] = ((v as i32 + token_idx as i32) % 1000 - 500) as i16;
                }
            }
        }

        // Check for early exit
        for layer in 0..model.num_layers as u8 {
            let signal = (layer as i16) * 100 + gate_hint.coherence_score_q / 10;
            if let Some(decision) = self.gate.checkpoint(layer, signal) {
                return (logits, decision);
            }
        }

        (logits, GateDecision::RanFull)
    }
}

impl TransformerBackend for WasmSimBackend {
    fn load(&self, artifact: &ModelArtifact) -> Result<ModelId> {
        // Note: WASM backend uses interior mutability pattern in practice
        // This is a simplified implementation
        Err(Error::FeatureNotAvailable(
            "Use WasmEngine for WASM loading".into(),
        ))
    }

    fn infer(&self, req: InferenceRequest) -> Result<InferenceResult> {
        let start = js_sys::Date::now();

        // Get model
        let model = self
            .models
            .get(&req.model)
            .ok_or_else(|| Error::ModelNotFound(req.model))?;

        // Run inference
        let (logits, gate_decision) = self.run_inference(model, req.tokens, &req.gate_hint);

        let latency_ns = ((js_sys::Date::now() - start) * 1_000_000.0) as u32;

        // Compute top-K
        let topk: Vec<(u16, i16)> = {
            let mut indexed: Vec<_> = logits.iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.cmp(a.1));
            indexed
                .into_iter()
                .take(16)
                .map(|(i, &v)| (i as u16, v))
                .collect()
        };

        let witness = WitnessLog::new(
            model.artifact.model_hash(),
            model.artifact.quant_hash(),
            BackendKind::WasmSim,
            0,
            latency_ns,
            gate_decision,
        );

        Ok(InferenceResult::new(logits, Some(topk), witness))
    }

    fn unload(&self, _model: ModelId) -> Result<()> {
        Err(Error::FeatureNotAvailable(
            "Use WasmEngine for WASM unloading".into(),
        ))
    }

    fn is_loaded(&self, model: ModelId) -> bool {
        self.models.contains_key(&model)
    }

    fn kind(&self) -> BackendKind {
        BackendKind::WasmSim
    }

    fn stats(&self) -> BackendStats {
        self.stats.clone()
    }
}
