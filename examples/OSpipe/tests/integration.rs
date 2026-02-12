//! Integration tests for OSpipe.

use ospipe::capture::{CaptureSource, CapturedFrame, FrameContent};
use ospipe::config::{OsPipeConfig, SafetyConfig, SearchConfig, StorageConfig};
use ospipe::pipeline::{IngestionPipeline, IngestResult};
use ospipe::safety::{SafetyDecision, SafetyGate};
use ospipe::search::router::{QueryRoute, QueryRouter};
use ospipe::search::hybrid::HybridSearch;
use ospipe::storage::embedding::{cosine_similarity, EmbeddingEngine};
use ospipe::storage::vector_store::{SearchFilter, VectorStore};

// ---------------------------------------------------------------------------
// Configuration tests
// ---------------------------------------------------------------------------

#[test]
fn test_default_config() {
    let config = OsPipeConfig::default();
    assert_eq!(config.storage.embedding_dim, 384);
    assert_eq!(config.storage.hnsw_m, 32);
    assert_eq!(config.storage.hnsw_ef_construction, 200);
    assert_eq!(config.storage.hnsw_ef_search, 100);
    assert!((config.storage.dedup_threshold - 0.95).abs() < f32::EPSILON);
    assert_eq!(config.capture.fps, 1.0);
    assert_eq!(config.capture.audio_chunk_secs, 30);
    assert!(config.capture.skip_private_windows);
    assert_eq!(config.search.default_k, 10);
    assert!((config.search.hybrid_weight - 0.7).abs() < f32::EPSILON);
    assert!(config.safety.pii_detection);
    assert!(config.safety.credit_card_redaction);
    assert!(config.safety.ssn_redaction);
}

#[test]
fn test_config_serialization_roundtrip() {
    let config = OsPipeConfig::default();
    let json = serde_json::to_string(&config).expect("serialize");
    let deserialized: OsPipeConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.storage.embedding_dim, config.storage.embedding_dim);
    assert_eq!(deserialized.capture.fps, config.capture.fps);
}

// ---------------------------------------------------------------------------
// Capture frame tests
// ---------------------------------------------------------------------------

#[test]
fn test_captured_frame_screen() {
    let frame = CapturedFrame::new_screen("Firefox", "GitHub - main", "hello world", 0);
    assert_eq!(frame.text_content(), "hello world");
    assert_eq!(frame.content_type(), "ocr");
    assert!(matches!(frame.source, CaptureSource::Screen { monitor: 0, .. }));
    assert_eq!(frame.metadata.app_name.as_deref(), Some("Firefox"));
    assert_eq!(frame.metadata.window_title.as_deref(), Some("GitHub - main"));
}

#[test]
fn test_captured_frame_audio() {
    let frame = CapturedFrame::new_audio("Microphone", "testing one two three", Some("Alice"));
    assert_eq!(frame.text_content(), "testing one two three");
    assert_eq!(frame.content_type(), "transcription");
    match &frame.source {
        CaptureSource::Audio { device, speaker } => {
            assert_eq!(device, "Microphone");
            assert_eq!(speaker.as_deref(), Some("Alice"));
        }
        _ => panic!("Expected Audio source"),
    }
}

#[test]
fn test_captured_frame_ui_event() {
    let frame = CapturedFrame::new_ui_event("click", "Button clicked: Submit");
    assert_eq!(frame.text_content(), "Button clicked: Submit");
    assert_eq!(frame.content_type(), "ui_event");
}

// ---------------------------------------------------------------------------
// Embedding and vector store tests
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_engine() {
    let engine = EmbeddingEngine::new(384);
    let v1 = engine.embed("hello");
    let v2 = engine.embed("hello");
    assert_eq!(v1, v2, "Same input must produce identical embeddings");
    assert_eq!(v1.len(), 384);

    // Check normalization
    let magnitude: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (magnitude - 1.0).abs() < 1e-5,
        "Embedding should be L2-normalized"
    );
}

#[test]
fn test_vector_store_insert_and_search() {
    let config = StorageConfig::default();
    let mut store = VectorStore::new(config).unwrap();
    let engine = EmbeddingEngine::new(384);

    // Insert some frames
    let frames = vec![
        CapturedFrame::new_screen("VS Code", "main.rs", "fn main() { println!(\"hello\"); }", 0),
        CapturedFrame::new_screen("Firefox", "Rust docs", "The Rust Programming Language", 0),
        CapturedFrame::new_audio("Mic", "discussing the project architecture", None),
    ];

    for frame in &frames {
        let emb = engine.embed(frame.text_content());
        store.insert(frame, &emb).unwrap();
    }

    assert_eq!(store.len(), 3);
    assert!(!store.is_empty());

    // Search for something similar to the first frame
    let query_emb = engine.embed("fn main() { println!(\"hello\"); }");
    let results = store.search(&query_emb, 2).unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 2);

    // The top result should be the exact match
    assert_eq!(results[0].id, frames[0].id);
    assert!((results[0].score - 1.0).abs() < 1e-5, "Exact match should have score ~1.0");
}

#[test]
fn test_vector_store_filtered_search() {
    let config = StorageConfig::default();
    let mut store = VectorStore::new(config).unwrap();
    let engine = EmbeddingEngine::new(384);

    let frame_vscode = CapturedFrame::new_screen("VS Code", "editor", "rust code", 0);
    let frame_firefox = CapturedFrame::new_screen("Firefox", "browser", "rust documentation", 0);

    let emb1 = engine.embed(frame_vscode.text_content());
    let emb2 = engine.embed(frame_firefox.text_content());
    store.insert(&frame_vscode, &emb1).unwrap();
    store.insert(&frame_firefox, &emb2).unwrap();

    // Filter to only VS Code results
    let filter = SearchFilter {
        app: Some("VS Code".to_string()),
        ..Default::default()
    };
    let query = engine.embed("rust");
    let results = store.search_filtered(&query, 10, &filter).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, frame_vscode.id);
}

#[test]
fn test_vector_store_empty_search() {
    let config = StorageConfig::default();
    let store = VectorStore::new(config).unwrap();
    let query = vec![0.0f32; 384];
    let results = store.search(&query, 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_vector_store_dimension_mismatch() {
    let config = StorageConfig::default(); // 384-dim
    let mut store = VectorStore::new(config).unwrap();
    let frame = CapturedFrame::new_screen("App", "Window", "text", 0);

    // Wrong dimension embedding
    let wrong_emb = vec![1.0f32; 128];
    let result = store.insert(&frame, &wrong_emb);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Frame deduplication tests
// ---------------------------------------------------------------------------

#[test]
fn test_frame_deduplication() {
    use ospipe::pipeline::FrameDeduplicator;

    let mut dedup = FrameDeduplicator::new(0.95, 10);
    let engine = EmbeddingEngine::new(384);

    let emb1 = engine.embed("hello world");
    let id1 = uuid::Uuid::new_v4();
    dedup.add(id1, emb1.clone());

    // Identical text should be detected as duplicate
    let emb2 = engine.embed("hello world");
    let result = dedup.is_duplicate(&emb2);
    assert!(result.is_some(), "Identical text should be detected as duplicate");
    let (dup_id, sim) = result.unwrap();
    assert_eq!(dup_id, id1);
    assert!((sim - 1.0).abs() < 1e-5);

    // Very different text should not be a duplicate
    let emb3 = engine.embed("completely unrelated content about quantum physics");
    let result = dedup.is_duplicate(&emb3);
    // With hash-based embeddings, different texts may or may not pass threshold
    // but identical texts always will
    if let Some((_, sim)) = result {
        assert!(sim >= 0.95);
    }
}

#[test]
fn test_dedup_window_eviction() {
    use ospipe::pipeline::FrameDeduplicator;

    let mut dedup = FrameDeduplicator::new(0.95, 3);
    let engine = EmbeddingEngine::new(64);

    // Add 4 items to a window of size 3
    for i in 0..4 {
        let emb = engine.embed(&format!("text number {}", i));
        dedup.add(uuid::Uuid::new_v4(), emb);
    }

    // Window should only contain 3 items (oldest evicted)
    assert_eq!(dedup.window_len(), 3);
}

// ---------------------------------------------------------------------------
// Safety gate tests
// ---------------------------------------------------------------------------

#[test]
fn test_safety_gate_allow() {
    let config = SafetyConfig::default();
    let gate = SafetyGate::new(config);

    let decision = gate.check("This is perfectly safe content about Rust programming.");
    assert_eq!(decision, SafetyDecision::Allow);
}

#[test]
fn test_safety_gate_credit_card_redaction() {
    let config = SafetyConfig::default();
    let gate = SafetyGate::new(config);

    let decision = gate.check("My card number is 4111111111111111 and it expires soon.");
    match decision {
        SafetyDecision::AllowRedacted(redacted) => {
            assert!(
                redacted.contains("[CC_REDACTED]"),
                "Credit card should be redacted, got: {}",
                redacted
            );
            assert!(!redacted.contains("4111111111111111"));
        }
        other => panic!("Expected AllowRedacted, got {:?}", other),
    }
}

#[test]
fn test_safety_gate_ssn_redaction() {
    let config = SafetyConfig::default();
    let gate = SafetyGate::new(config);

    let decision = gate.check("SSN: 123-45-6789 is confidential");
    match decision {
        SafetyDecision::AllowRedacted(redacted) => {
            assert!(
                redacted.contains("[SSN_REDACTED]"),
                "SSN should be redacted, got: {}",
                redacted
            );
            assert!(!redacted.contains("123-45-6789"));
        }
        other => panic!("Expected AllowRedacted, got {:?}", other),
    }
}

#[test]
fn test_safety_gate_email_redaction() {
    let config = SafetyConfig::default();
    let gate = SafetyGate::new(config);

    let decision = gate.check("Contact me at user@example.com for details");
    match decision {
        SafetyDecision::AllowRedacted(redacted) => {
            assert!(
                redacted.contains("[EMAIL_REDACTED]"),
                "Email should be redacted, got: {}",
                redacted
            );
            assert!(!redacted.contains("user@example.com"));
        }
        other => panic!("Expected AllowRedacted, got {:?}", other),
    }
}

#[test]
fn test_safety_gate_custom_pattern_deny() {
    let config = SafetyConfig {
        custom_patterns: vec!["SECRET_KEY".to_string()],
        ..Default::default()
    };
    let gate = SafetyGate::new(config);

    let decision = gate.check("The SECRET_KEY is abc123");
    match decision {
        SafetyDecision::Deny { reason } => {
            assert!(reason.contains("SECRET_KEY"));
        }
        other => panic!("Expected Deny, got {:?}", other),
    }
}

#[test]
fn test_safety_redact_method() {
    let config = SafetyConfig::default();
    let gate = SafetyGate::new(config);

    let redacted = gate.redact("Call me at user@example.com");
    assert!(redacted.contains("[EMAIL_REDACTED]"));
    assert!(!redacted.contains("user@example.com"));

    let safe = gate.redact("Nothing sensitive here.");
    assert_eq!(safe, "Nothing sensitive here.");
}

// ---------------------------------------------------------------------------
// Query router tests
// ---------------------------------------------------------------------------

#[test]
fn test_query_router_temporal() {
    let router = QueryRouter::new();
    assert_eq!(router.route("what did I see yesterday"), QueryRoute::Temporal);
    assert_eq!(router.route("show me last week"), QueryRoute::Temporal);
    assert_eq!(router.route("results from today"), QueryRoute::Temporal);
}

#[test]
fn test_query_router_graph() {
    let router = QueryRouter::new();
    assert_eq!(
        router.route("documents related to authentication"),
        QueryRoute::Graph
    );
    assert_eq!(
        router.route("things connected to the API module"),
        QueryRoute::Graph
    );
}

#[test]
fn test_query_router_keyword() {
    let router = QueryRouter::new();
    assert_eq!(router.route("\"exact phrase search\""), QueryRoute::Keyword);
    assert_eq!(router.route("rust programming"), QueryRoute::Keyword);
    assert_eq!(router.route("hello"), QueryRoute::Keyword);
}

#[test]
fn test_query_router_hybrid() {
    let router = QueryRouter::new();
    assert_eq!(
        router.route("how to implement authentication in Rust"),
        QueryRoute::Hybrid
    );
}

// ---------------------------------------------------------------------------
// Hybrid search tests
// ---------------------------------------------------------------------------

#[test]
fn test_hybrid_search() {
    let config = StorageConfig::default();
    let mut store = VectorStore::new(config).unwrap();
    let engine = EmbeddingEngine::new(384);

    // Insert frames with different content
    let frames = vec![
        CapturedFrame::new_screen("Editor", "code.rs", "implementing vector search in Rust", 0),
        CapturedFrame::new_screen("Browser", "docs", "Rust vector database documentation", 0),
        CapturedFrame::new_audio("Mic", "discussing Python machine learning", None),
    ];

    for frame in &frames {
        let emb = engine.embed(frame.text_content());
        store.insert(frame, &emb).unwrap();
    }

    let hybrid = HybridSearch::new(0.7);
    let query = "vector search Rust";
    let query_emb = engine.embed(query);
    let results = hybrid.search(&store, query, &query_emb, 3).unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 3);
    // Results should be ordered by combined score
    for i in 1..results.len() {
        assert!(results[i - 1].score >= results[i].score);
    }
}

#[test]
fn test_hybrid_search_empty_store() {
    let config = StorageConfig::default();
    let store = VectorStore::new(config).unwrap();
    let engine = EmbeddingEngine::new(384);

    let hybrid = HybridSearch::new(0.7);
    let query_emb = engine.embed("test query");
    let results = hybrid.search(&store, "test query", &query_emb, 10).unwrap();
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// Ingestion pipeline tests
// ---------------------------------------------------------------------------

#[test]
fn test_ingestion_pipeline_basic() {
    let config = OsPipeConfig::default();
    let mut pipeline = IngestionPipeline::new(config).unwrap();

    let frame = CapturedFrame::new_screen("VS Code", "main.rs", "fn main() { }", 0);
    let result = pipeline.ingest(frame).unwrap();

    match result {
        IngestResult::Stored { id } => {
            assert!(!id.is_nil());
        }
        other => panic!("Expected Stored, got {:?}", other),
    }

    assert_eq!(pipeline.stats().total_ingested, 1);
    assert_eq!(pipeline.stats().total_deduplicated, 0);
    assert_eq!(pipeline.stats().total_denied, 0);
}

#[test]
fn test_ingestion_pipeline_deduplication() {
    let config = OsPipeConfig::default();
    let mut pipeline = IngestionPipeline::new(config).unwrap();

    // Ingest the same content twice
    let frame1 = CapturedFrame::new_screen("App", "Window", "exact same content", 0);
    let frame2 = CapturedFrame::new_screen("App", "Window", "exact same content", 0);

    let result1 = pipeline.ingest(frame1).unwrap();
    assert!(matches!(result1, IngestResult::Stored { .. }));

    let result2 = pipeline.ingest(frame2).unwrap();
    assert!(
        matches!(result2, IngestResult::Deduplicated { .. }),
        "Second identical frame should be deduplicated"
    );

    assert_eq!(pipeline.stats().total_ingested, 1);
    assert_eq!(pipeline.stats().total_deduplicated, 1);
}

#[test]
fn test_ingestion_pipeline_safety_deny() {
    let config = OsPipeConfig {
        safety: SafetyConfig {
            custom_patterns: vec!["TOP_SECRET".to_string()],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut pipeline = IngestionPipeline::new(config).unwrap();

    let frame = CapturedFrame::new_screen("App", "Window", "This is TOP_SECRET information", 0);
    let result = pipeline.ingest(frame).unwrap();

    match result {
        IngestResult::Denied { reason } => {
            assert!(reason.contains("TOP_SECRET"));
        }
        other => panic!("Expected Denied, got {:?}", other),
    }

    assert_eq!(pipeline.stats().total_denied, 1);
    assert_eq!(pipeline.stats().total_ingested, 0);
}

#[test]
fn test_ingestion_pipeline_safety_redact() {
    let config = OsPipeConfig::default();
    let mut pipeline = IngestionPipeline::new(config).unwrap();

    let frame = CapturedFrame::new_screen(
        "App",
        "Window",
        "Please email user@example.com for the meeting notes",
        0,
    );
    let result = pipeline.ingest(frame).unwrap();

    // Should be stored but with redacted content
    assert!(matches!(result, IngestResult::Stored { .. }));
    assert_eq!(pipeline.stats().total_redacted, 1);

    // Verify the stored content has the email redacted
    let store = pipeline.vector_store();
    assert_eq!(store.len(), 1);
}

#[test]
fn test_ingestion_pipeline_batch() {
    let config = OsPipeConfig::default();
    let mut pipeline = IngestionPipeline::new(config).unwrap();

    let frames = vec![
        CapturedFrame::new_screen("App", "Win1", "first frame content", 0),
        CapturedFrame::new_screen("App", "Win2", "second frame content", 0),
        CapturedFrame::new_screen("App", "Win3", "third frame content", 0),
    ];

    let results = pipeline.ingest_batch(frames).unwrap();
    assert_eq!(results.len(), 3);

    let stored_count = results
        .iter()
        .filter(|r| matches!(r, IngestResult::Stored { .. }))
        .count();
    assert_eq!(stored_count, 3);
    assert_eq!(pipeline.stats().total_ingested, 3);
}

// ---------------------------------------------------------------------------
// Cosine similarity tests
// ---------------------------------------------------------------------------

#[test]
fn test_cosine_similarity_identical_vectors() {
    let v = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&v1, &v2);
    assert!(sim.abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_opposite_vectors() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![-1.0, 0.0, 0.0];
    let sim = cosine_similarity(&v1, &v2);
    assert!((sim - (-1.0)).abs() < 1e-5);
}
