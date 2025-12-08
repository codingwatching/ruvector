//! iOS & Browser Optimized WASM Vector Database
//!
//! A high-performance vector database designed for iOS and browser deployment.
//! Supports both WasmKit (Swift native) and wasm-bindgen (browser) targets.
//!
//! ## Features
//! - HNSW index for O(log n) approximate nearest neighbor search
//! - Scalar, Binary, and Product quantization for memory efficiency
//! - SIMD-optimized distance calculations (iOS 16.4+ / Safari 16.4+)
//! - Content embedding and recommendation engine
//! - Q-learning for adaptive recommendations
//! - Sub-100ms latency, <5MB binary target
//!
//! ## Build Targets
//! - Native (WasmKit): `cargo build --target wasm32-wasip1 --release`
//! - Browser: `cargo build --target wasm32-unknown-unknown --release --features browser`
//! - SIMD: Add `RUSTFLAGS="-C target-feature=+simd128"`

// Standard library for wasip1 target
use std::vec::Vec;
use core::slice;

// ============================================
// Core Modules
// ============================================

pub mod simd;
pub mod distance;
pub mod quantization;
pub mod hnsw;
pub mod ios_capabilities;
mod embeddings;
mod qlearning;
mod attention;

pub use simd::{dot_product, l2_distance, l2_norm, cosine_similarity, normalize, softmax};
pub use distance::{DistanceMetric, euclidean_distance, manhattan_distance};
pub use quantization::{ScalarQuantized, BinaryQuantized, ProductQuantized, PQCodebook};
pub use hnsw::{HnswIndex, HnswConfig};
pub use ios_capabilities::{RuntimeCapabilities, OptimizationTier, MemoryConfig, Capability};
pub use embeddings::{ContentEmbedder, ContentMetadata, VibeState};
pub use qlearning::{QLearner, UserInteraction, InteractionType};
pub use attention::{AttentionHead, MultiHeadAttention, AttentionRanker};

// ============================================
// Global State
// ============================================

static mut ENGINE: Option<RecommendationEngine> = None;
static mut MEMORY_POOL: Option<MemoryPool> = None;
static mut VECTOR_DB: Option<VectorDatabase> = None;

/// Memory pool for WASM linear memory communication
struct MemoryPool {
    buffer: Vec<u8>,
    offset: usize,
}

impl MemoryPool {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0u8; size],
            offset: 0,
        }
    }

    fn reset(&mut self) {
        self.offset = 0;
    }

    fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        if self.offset + size <= self.buffer.len() {
            let ptr = unsafe { self.buffer.as_mut_ptr().add(self.offset) };
            self.offset += size;
            Some(ptr)
        } else {
            None
        }
    }

    fn ptr(&self) -> *const u8 {
        self.buffer.as_ptr()
    }
}

// ============================================
// Vector Database (HNSW + Quantization)
// ============================================

/// Quantization mode for vectors
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum QuantizationMode {
    /// No quantization (full f32)
    None = 0,
    /// Scalar quantization (4x compression)
    Scalar = 1,
    /// Binary quantization (32x compression)
    Binary = 2,
}

/// Unified vector database combining HNSW with optional quantization
pub struct VectorDatabase {
    /// HNSW index for ANN search
    index: HnswIndex,
    /// Scalar quantized vectors (for memory-efficient storage)
    scalar_store: Vec<(u64, ScalarQuantized)>,
    /// Binary quantized vectors (for fast pre-filtering)
    binary_store: Vec<(u64, BinaryQuantized)>,
    /// Quantization mode
    quant_mode: QuantizationMode,
    /// Vector dimension
    dim: usize,
}

impl VectorDatabase {
    /// Create a new vector database
    pub fn new(dim: usize, metric: DistanceMetric, quant_mode: QuantizationMode) -> Self {
        let config = HnswConfig {
            m: 16,
            m_max_0: 32,
            ef_construction: 100,
            ef_search: 50,
            level_mult: 0.36,
        };

        Self {
            index: HnswIndex::new(dim, metric, config),
            scalar_store: Vec::new(),
            binary_store: Vec::new(),
            quant_mode,
            dim,
        }
    }

    /// Create with custom HNSW config
    pub fn with_config(
        dim: usize,
        metric: DistanceMetric,
        quant_mode: QuantizationMode,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let config = HnswConfig {
            m,
            m_max_0: m * 2,
            ef_construction,
            ef_search: 50,
            level_mult: 1.0 / (m as f32).ln(),
        };

        Self {
            index: HnswIndex::new(dim, metric, config),
            scalar_store: Vec::new(),
            binary_store: Vec::new(),
            quant_mode,
            dim,
        }
    }

    /// Insert a vector with optional quantization
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) -> bool {
        if vector.len() != self.dim {
            return false;
        }

        // Store quantized version based on mode
        match self.quant_mode {
            QuantizationMode::Scalar => {
                let sq = ScalarQuantized::quantize(&vector);
                self.scalar_store.push((id, sq));
            }
            QuantizationMode::Binary => {
                let bq = BinaryQuantized::quantize(&vector);
                self.binary_store.push((id, bq));
            }
            QuantizationMode::None => {}
        }

        // Always insert into HNSW for accurate search
        self.index.insert(id, vector)
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.index.search(query, k)
    }

    /// Search with custom ef parameter
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        self.index.search_with_ef(query, k, ef)
    }

    /// Fast pre-filter using binary quantization (if available)
    pub fn prefilter_binary(&self, query: &[f32], threshold: u32) -> Vec<u64> {
        if self.binary_store.is_empty() {
            return vec![];
        }

        let query_bq = BinaryQuantized::quantize(query);
        self.binary_store
            .iter()
            .filter(|(_, bq)| bq.distance(&query_bq) <= threshold)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get vector by ID (reconstructed if quantized)
    pub fn get(&self, id: u64) -> Option<Vec<f32>> {
        // Try HNSW first
        if let Some(v) = self.index.get(id) {
            return Some(v.to_vec());
        }

        // Try scalar store
        if let Some((_, sq)) = self.scalar_store.iter().find(|(i, _)| *i == id) {
            return Some(sq.reconstruct());
        }

        None
    }

    /// Get database size
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let hnsw_size = self.index.len() * self.dim * 4; // Approximate
        let scalar_size = self.scalar_store.iter()
            .map(|(_, sq)| sq.memory_size())
            .sum::<usize>();
        let binary_size = self.binary_store.iter()
            .map(|(_, bq)| bq.memory_size())
            .sum::<usize>();

        hnsw_size + scalar_size + binary_size
    }
}

// ============================================
// Recommendation Engine
// ============================================

/// Main recommendation engine combining all components
pub struct RecommendationEngine {
    embedder: ContentEmbedder,
    learner: QLearner,
    ranker: AttentionRanker,
    /// Content embeddings cache
    content_cache: Vec<(u64, Vec<f32>)>,
    /// User history (content IDs)
    history: Vec<u64>,
    /// Current vibe state embedding
    vibe_embedding: Vec<f32>,
}

impl RecommendationEngine {
    /// Create a new recommendation engine
    pub fn new(embedding_dim: usize, num_actions: usize) -> Self {
        let embedder = ContentEmbedder::new(embedding_dim);
        let learner = QLearner::new(num_actions);
        let ranker = AttentionRanker::new(embedding_dim, 4);

        Self {
            embedder,
            learner,
            ranker,
            content_cache: Vec::with_capacity(100),
            history: Vec::with_capacity(50),
            vibe_embedding: vec![0.0; embedding_dim],
        }
    }

    /// Embed content and cache the result
    pub fn embed_content(&mut self, content: &ContentMetadata) -> &[f32] {
        if let Some(pos) = self.content_cache.iter().position(|(id, _)| *id == content.id) {
            return &self.content_cache[pos].1;
        }

        let embedding = self.embedder.embed(content);

        if self.content_cache.len() >= 100 {
            self.content_cache.remove(0);
        }
        self.content_cache.push((content.id, embedding));

        &self.content_cache.last().unwrap().1
    }

    /// Update vibe state
    pub fn set_vibe(&mut self, vibe: &VibeState) {
        self.vibe_embedding = vibe.to_embedding(&self.embedder);
    }

    /// Get recommendations based on current vibe and history
    pub fn get_recommendations(&self, candidate_ids: &[u64], top_k: usize) -> Vec<(u64, f32)> {
        if candidate_ids.is_empty() {
            return Vec::new();
        }

        let action_ranks = self.learner.rank_actions(&self.vibe_embedding);

        let mut scored: Vec<(u64, f32)> = candidate_ids.iter()
            .enumerate()
            .map(|(i, &id)| {
                let q_rank = action_ranks.iter()
                    .position(|&a| a == i % self.learner.update_count().max(1) as usize)
                    .unwrap_or(action_ranks.len()) as f32;
                let q_score = 1.0 / (1.0 + q_rank);

                let recency_penalty = if self.history.contains(&id) { 0.5 } else { 1.0 };

                (id, q_score * recency_penalty)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
    }

    /// Record a user interaction for learning
    pub fn learn(&mut self, interaction: &UserInteraction) {
        if self.history.len() >= 50 {
            self.history.remove(0);
        }
        self.history.push(interaction.content_id);

        let action = (interaction.content_id % 100) as usize;
        self.learner.update(
            &self.vibe_embedding,
            action,
            interaction,
            &self.vibe_embedding,
        );
    }

    /// Serialize engine state
    pub fn save_state(&self) -> Vec<u8> {
        self.learner.serialize()
    }

    /// Load engine state
    pub fn load_state(&mut self, data: &[u8]) -> bool {
        if let Some(learner) = QLearner::deserialize(data) {
            self.learner = learner;
            true
        } else {
            false
        }
    }
}

// ============================================
// WASM Exports - Vector Database
// ============================================

/// Create a vector database
#[no_mangle]
pub extern "C" fn db_create(dim: u32, metric: u8, quant_mode: u8, m: u32, ef_construction: u32) -> i32 {
    unsafe {
        MEMORY_POOL = Some(MemoryPool::new(2 * 1024 * 1024)); // 2MB pool
        VECTOR_DB = Some(VectorDatabase::with_config(
            dim as usize,
            DistanceMetric::from_u8(metric),
            match quant_mode {
                1 => QuantizationMode::Scalar,
                2 => QuantizationMode::Binary,
                _ => QuantizationMode::None,
            },
            m as usize,
            ef_construction as usize,
        ));
    }
    0
}

/// Insert vector into database
#[no_mangle]
pub extern "C" fn db_insert(id: u64, vector_ptr: *const f32, len: u32) -> i32 {
    unsafe {
        if let Some(db) = VECTOR_DB.as_mut() {
            let vector = slice::from_raw_parts(vector_ptr, len as usize).to_vec();
            if db.insert(id, vector) { 0 } else { -1 }
        } else {
            -1
        }
    }
}

/// Search database for k nearest neighbors
#[no_mangle]
pub extern "C" fn db_search(
    query_ptr: *const f32,
    query_len: u32,
    k: u32,
    ef: u32,
    out_ids: *mut u64,
    out_distances: *mut f32,
) -> u32 {
    unsafe {
        if let Some(db) = VECTOR_DB.as_ref() {
            let query = slice::from_raw_parts(query_ptr, query_len as usize);
            let results = db.search_with_ef(query, k as usize, ef as usize);

            let ids = slice::from_raw_parts_mut(out_ids, results.len());
            let distances = slice::from_raw_parts_mut(out_distances, results.len());

            for (i, (id, dist)) in results.iter().enumerate() {
                ids[i] = *id;
                distances[i] = *dist;
            }

            results.len() as u32
        } else {
            0
        }
    }
}

/// Get database size
#[no_mangle]
pub extern "C" fn db_size() -> u32 {
    unsafe {
        VECTOR_DB.as_ref().map(|db| db.len() as u32).unwrap_or(0)
    }
}

/// Get estimated memory usage
#[no_mangle]
pub extern "C" fn db_memory_usage() -> u64 {
    unsafe {
        VECTOR_DB.as_ref().map(|db| db.memory_usage() as u64).unwrap_or(0)
    }
}

// ============================================
// WASM Exports - Recommendation Engine
// ============================================

/// Initialize the recommendation engine
#[no_mangle]
pub extern "C" fn rec_init(dim: u32, actions: u32) -> i32 {
    unsafe {
        if MEMORY_POOL.is_none() {
            MEMORY_POOL = Some(MemoryPool::new(1024 * 1024));
        }
        ENGINE = Some(RecommendationEngine::new(dim as usize, actions as usize));
    }
    0
}

/// Get pointer to the shared memory buffer
#[no_mangle]
pub extern "C" fn get_memory_ptr() -> *const u8 {
    unsafe {
        MEMORY_POOL.as_ref().map(|p| p.ptr()).unwrap_or(core::ptr::null())
    }
}

/// Allocate space in the shared memory buffer
#[no_mangle]
pub extern "C" fn mem_alloc(size: u32) -> *mut u8 {
    unsafe {
        MEMORY_POOL.as_mut()
            .and_then(|p| p.alloc(size as usize))
            .unwrap_or(core::ptr::null_mut())
    }
}

/// Reset the memory pool
#[no_mangle]
pub extern "C" fn reset_memory() {
    unsafe {
        if let Some(pool) = MEMORY_POOL.as_mut() {
            pool.reset();
        }
    }
}

/// Embed content and return pointer
#[no_mangle]
pub extern "C" fn rec_embed(
    content_id: u64,
    content_type: u8,
    duration_secs: u32,
    category_flags: u32,
    popularity: f32,
    recency: f32,
) -> *const f32 {
    unsafe {
        if let Some(engine) = ENGINE.as_mut() {
            let content = ContentMetadata {
                id: content_id,
                content_type,
                duration_secs,
                category_flags,
                popularity,
                recency,
            };

            let embedding = engine.embed_content(&content);
            embedding.as_ptr()
        } else {
            core::ptr::null()
        }
    }
}

/// Set the current vibe state
#[no_mangle]
pub extern "C" fn rec_set_vibe(
    energy: f32,
    mood: f32,
    focus: f32,
    time_context: f32,
    pref0: f32,
    pref1: f32,
    pref2: f32,
    pref3: f32,
) {
    unsafe {
        if let Some(engine) = ENGINE.as_mut() {
            let vibe = VibeState {
                energy,
                mood,
                focus,
                time_context,
                preferences: [pref0, pref1, pref2, pref3],
            };
            engine.set_vibe(&vibe);
        }
    }
}

/// Get recommendations
#[no_mangle]
pub extern "C" fn rec_get_recommendations(
    candidates_ptr: *const u64,
    candidates_len: u32,
    top_k: u32,
    out_ptr: *mut u8,
) -> u32 {
    unsafe {
        if let Some(engine) = ENGINE.as_ref() {
            let candidates = slice::from_raw_parts(candidates_ptr, candidates_len as usize);
            let recs = engine.get_recommendations(candidates, top_k as usize);

            let out = slice::from_raw_parts_mut(out_ptr, recs.len() * 12);
            for (i, (id, score)) in recs.iter().enumerate() {
                let offset = i * 12;
                out[offset..offset + 8].copy_from_slice(&id.to_le_bytes());
                out[offset + 8..offset + 12].copy_from_slice(&score.to_le_bytes());
            }

            recs.len() as u32
        } else {
            0
        }
    }
}

/// Record a user interaction
#[no_mangle]
pub extern "C" fn rec_learn(
    content_id: u64,
    interaction_type: u8,
    time_spent: f32,
    position: u8,
) {
    unsafe {
        if let Some(engine) = ENGINE.as_mut() {
            let interaction = UserInteraction {
                content_id,
                interaction: match interaction_type {
                    0 => InteractionType::View,
                    1 => InteractionType::Like,
                    2 => InteractionType::Share,
                    3 => InteractionType::Skip,
                    4 => InteractionType::Complete,
                    _ => InteractionType::Dismiss,
                },
                time_spent,
                position,
            };
            engine.learn(&interaction);
        }
    }
}

/// Save engine state
#[no_mangle]
pub extern "C" fn rec_save_state() -> u32 {
    unsafe {
        if let (Some(engine), Some(pool)) = (ENGINE.as_ref(), MEMORY_POOL.as_mut()) {
            let state = engine.save_state();
            let size = state.len();

            if let Some(ptr) = pool.alloc(size) {
                core::ptr::copy_nonoverlapping(state.as_ptr(), ptr, size);
                size as u32
            } else {
                0
            }
        } else {
            0
        }
    }
}

/// Load engine state
#[no_mangle]
pub extern "C" fn rec_load_state(ptr: *const u8, len: u32) -> i32 {
    unsafe {
        if let Some(engine) = ENGINE.as_mut() {
            let data = slice::from_raw_parts(ptr, len as usize);
            if engine.load_state(data) { 0 } else { -1 }
        } else {
            -1
        }
    }
}

/// Get exploration rate
#[no_mangle]
pub extern "C" fn rec_get_exploration_rate() -> f32 {
    unsafe {
        ENGINE.as_ref()
            .map(|e| e.learner.exploration_rate())
            .unwrap_or(0.0)
    }
}

/// Get update count
#[no_mangle]
pub extern "C" fn rec_get_update_count() -> u64 {
    unsafe {
        ENGINE.as_ref()
            .map(|e| e.learner.update_count())
            .unwrap_or(0)
    }
}

// ============================================
// Legacy API Compatibility
// ============================================

/// Initialize (legacy - use rec_init)
#[no_mangle]
pub extern "C" fn init(dim: u32, actions: u32) -> i32 {
    rec_init(dim, actions)
}

/// Embed content (legacy)
#[no_mangle]
pub extern "C" fn embed_content(
    content_id: u64,
    content_type: u8,
    duration_secs: u32,
    category_flags: u32,
    popularity: f32,
    recency: f32,
) -> *const f32 {
    rec_embed(content_id, content_type, duration_secs, category_flags, popularity, recency)
}

/// Set vibe (legacy)
#[no_mangle]
pub extern "C" fn set_vibe(
    energy: f32,
    mood: f32,
    focus: f32,
    time_context: f32,
    pref0: f32,
    pref1: f32,
    pref2: f32,
    pref3: f32,
) {
    rec_set_vibe(energy, mood, focus, time_context, pref0, pref1, pref2, pref3)
}

/// Get recommendations (legacy)
#[no_mangle]
pub extern "C" fn get_recommendations(
    candidates_ptr: *const u64,
    candidates_len: u32,
    top_k: u32,
    out_ptr: *mut u8,
) -> u32 {
    rec_get_recommendations(candidates_ptr, candidates_len, top_k, out_ptr)
}

/// Update learning (legacy)
#[no_mangle]
pub extern "C" fn update_learning(
    content_id: u64,
    interaction_type: u8,
    time_spent: f32,
    position: u8,
) {
    rec_learn(content_id, interaction_type, time_spent, position)
}

/// Save state (legacy)
#[no_mangle]
pub extern "C" fn save_state() -> u32 {
    rec_save_state()
}

/// Load state (legacy)
#[no_mangle]
pub extern "C" fn load_state(ptr: *const u8, len: u32) -> i32 {
    rec_load_state(ptr, len)
}

/// Get embedding dim
#[no_mangle]
pub extern "C" fn get_embedding_dim() -> u32 {
    unsafe {
        ENGINE.as_ref()
            .map(|e| e.embedder.dim() as u32)
            .unwrap_or(0)
    }
}

/// Get exploration rate (legacy)
#[no_mangle]
pub extern "C" fn get_exploration_rate() -> f32 {
    rec_get_exploration_rate()
}

/// Get update count (legacy)
#[no_mangle]
pub extern "C" fn get_update_count() -> u64 {
    rec_get_update_count()
}

/// Compute similarity
#[no_mangle]
pub extern "C" fn compute_similarity(id_a: u64, id_b: u64) -> f32 {
    unsafe {
        if let Some(engine) = ENGINE.as_ref() {
            let emb_a = engine.content_cache.iter()
                .find(|(id, _)| *id == id_a)
                .map(|(_, e)| e);
            let emb_b = engine.content_cache.iter()
                .find(|(id, _)| *id == id_b)
                .map(|(_, e)| e);

            if let (Some(a), Some(b)) = (emb_a, emb_b) {
                ContentEmbedder::similarity(a, b)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

// ============================================
// Browser Module (wasm-bindgen)
// ============================================

#[cfg(feature = "browser")]
pub mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;
    use serde::{Serialize, Deserialize};

    /// Browser-friendly vector database wrapper
    #[wasm_bindgen]
    pub struct WasmVectorDB {
        db: VectorDatabase,
    }

    #[wasm_bindgen]
    impl WasmVectorDB {
        /// Create a new vector database
        #[wasm_bindgen(constructor)]
        pub fn new(dim: u32, metric: u8, quant_mode: u8) -> Self {
            Self {
                db: VectorDatabase::new(
                    dim as usize,
                    DistanceMetric::from_u8(metric),
                    match quant_mode {
                        1 => QuantizationMode::Scalar,
                        2 => QuantizationMode::Binary,
                        _ => QuantizationMode::None,
                    },
                ),
            }
        }

        /// Insert a vector
        pub fn insert(&mut self, id: u64, vector: &[f32]) -> bool {
            self.db.insert(id, vector.to_vec())
        }

        /// Search for nearest neighbors (returns JSON)
        pub fn search(&self, query: &[f32], k: u32) -> String {
            let results = self.db.search(query, k as usize);
            serde_json::to_string(&results).unwrap_or_default()
        }

        /// Get database size
        pub fn size(&self) -> u32 {
            self.db.len() as u32
        }

        /// Get memory usage
        pub fn memory_usage(&self) -> u64 {
            self.db.memory_usage() as u64
        }
    }

    /// Browser-friendly recommendation engine wrapper
    #[wasm_bindgen]
    pub struct WasmRecommendationEngine {
        engine: RecommendationEngine,
    }

    #[wasm_bindgen]
    impl WasmRecommendationEngine {
        /// Create a new engine
        #[wasm_bindgen(constructor)]
        pub fn new(dim: u32, actions: u32) -> Self {
            Self {
                engine: RecommendationEngine::new(dim as usize, actions as usize),
            }
        }

        /// Set vibe state
        pub fn set_vibe(&mut self, energy: f32, mood: f32, focus: f32, time_context: f32) {
            let vibe = VibeState {
                energy,
                mood,
                focus,
                time_context,
                preferences: [0.25, 0.25, 0.25, 0.25],
            };
            self.engine.set_vibe(&vibe);
        }

        /// Get exploration rate
        pub fn exploration_rate(&self) -> f32 {
            self.engine.learner.exploration_rate()
        }

        /// Get update count
        pub fn update_count(&self) -> u64 {
            self.engine.learner.update_count()
        }
    }

    /// Get SIMD capability
    #[wasm_bindgen]
    pub fn has_simd_support() -> bool {
        simd::simd_available()
    }

    /// Get compile-time features
    #[wasm_bindgen]
    pub fn get_features() -> String {
        let features = ios_capabilities::compile_time_capabilities();
        format!("{{\"flags\":{},\"simd\":{}}}", features, simd::simd_available())
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_database() {
        let mut db = VectorDatabase::new(4, DistanceMetric::Euclidean, QuantizationMode::None);

        assert!(db.insert(1, vec![1.0, 0.0, 0.0, 0.0]));
        assert!(db.insert(2, vec![0.0, 1.0, 0.0, 0.0]));
        assert!(db.insert(3, vec![0.5, 0.5, 0.0, 0.0]));

        let results = db.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_database_with_quantization() {
        let mut db = VectorDatabase::new(4, DistanceMetric::Cosine, QuantizationMode::Scalar);

        for i in 0..10u64 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            db.insert(i, v);
        }

        assert_eq!(db.len(), 10);
        assert!(db.memory_usage() > 0);
    }

    #[test]
    fn test_engine_creation() {
        let engine = RecommendationEngine::new(64, 100);
        assert!(engine.content_cache.is_empty());
    }

    #[test]
    fn test_embed_and_cache() {
        let mut engine = RecommendationEngine::new(64, 100);
        let content = ContentMetadata {
            id: 1,
            content_type: 0,
            duration_secs: 120,
            category_flags: 0b1010,
            popularity: 0.8,
            recency: 0.9,
        };

        let emb1 = engine.embed_content(&content).to_vec();
        let emb2 = engine.embed_content(&content).to_vec();

        assert_eq!(emb1, emb2);
        assert_eq!(engine.content_cache.len(), 1);
    }

    #[test]
    fn test_recommendations() {
        let engine = RecommendationEngine::new(64, 100);
        let candidates: Vec<u64> = (1..=10).collect();

        let recs = engine.get_recommendations(&candidates, 5);
        assert!(recs.len() <= 5);
    }
}
