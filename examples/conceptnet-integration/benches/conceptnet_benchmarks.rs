//! Benchmarks for ConceptNet-RuVector integration

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use conceptnet_integration::graph::builder::{ConceptNetGraphBuilder, GraphBuildConfig};
use conceptnet_integration::gnn::layer::{CommonsenseGNN, GNNConfig};
use conceptnet_integration::attention::{RelationAttention, CommonsenseAttentionConfig};
use conceptnet_integration::numberbatch::MockNumberbatch;
use conceptnet_integration::api::RelationType;

fn bench_graph_shortest_path(c: &mut Criterion) {
    // Build a graph for benchmarking
    let mut builder = ConceptNetGraphBuilder::default_config();

    // Create chain: node_0 -> node_1 -> ... -> node_99
    for i in 0..99 {
        let edge = create_mock_edge(
            &format!("node_{}", i),
            &format!("node_{}", i + 1),
            "IsA",
            2.0,
        );
        builder.add_edge(&edge).unwrap();
    }

    c.bench_function("shortest_path_100_nodes", |b| {
        b.iter(|| {
            builder.shortest_path(
                black_box("/c/en/node_0"),
                black_box("/c/en/node_50"),
                black_box(100),
            )
        })
    });
}

fn bench_graph_neighbors(c: &mut Criterion) {
    let mut builder = ConceptNetGraphBuilder::default_config();

    // Create hub with many connections
    for i in 0..100 {
        let edge = create_mock_edge("hub", &format!("spoke_{}", i), "RelatedTo", 2.0);
        builder.add_edge(&edge).unwrap();
    }

    c.bench_function("get_neighbors_100_edges", |b| {
        b.iter(|| builder.get_neighbors(black_box("/c/en/hub")))
    });
}

fn bench_gnn_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("gnn_forward");

    for num_nodes in [10, 50, 100, 500].iter() {
        let config = GNNConfig {
            input_dim: 300,
            hidden_dim: 256,
            output_dim: 128,
            num_heads: 4,
            num_layers: 2,
            ..Default::default()
        };

        let gnn = CommonsenseGNN::new(config);

        let embeddings: Vec<Vec<f32>> = (0..*num_nodes)
            .map(|i| vec![(i as f32 * 0.01); 300])
            .collect();

        // Create sparse adjacency (each node connects to next)
        let adjacency: Vec<_> = (0..num_nodes - 1)
            .map(|i| (i, i + 1, RelationType::RelatedTo, 1.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            num_nodes,
            |b, _| {
                b.iter(|| gnn.forward(black_box(&embeddings), black_box(&adjacency)))
            },
        );
    }

    group.finish();
}

fn bench_gnn_link_prediction(c: &mut Criterion) {
    let config = GNNConfig::default();
    let gnn = CommonsenseGNN::new(config);

    let src = vec![0.5; 300];
    let dst = vec![0.6; 300];

    c.bench_function("link_prediction", |b| {
        b.iter(|| {
            gnn.predict_link(
                black_box(&src),
                black_box(&dst),
                black_box(&RelationType::IsA),
            )
        })
    });
}

fn bench_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_forward");

    for num_keys in [10, 50, 100, 500].iter() {
        let config = CommonsenseAttentionConfig {
            hidden_dim: 256,
            num_heads: 8,
            ..Default::default()
        };

        let attention = RelationAttention::new(config);

        let query = vec![0.1; 256];
        let keys: Vec<Vec<f32>> = (0..*num_keys)
            .map(|i| vec![(i as f32 * 0.01); 256])
            .collect();
        let values = keys.clone();
        let relations: Vec<RelationType> = (0..*num_keys)
            .map(|i| match i % 5 {
                0 => RelationType::IsA,
                1 => RelationType::HasA,
                2 => RelationType::UsedFor,
                3 => RelationType::RelatedTo,
                _ => RelationType::PartOf,
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("keys", num_keys),
            num_keys,
            |b, _| {
                b.iter(|| {
                    attention.forward(
                        black_box(&query),
                        black_box(&keys),
                        black_box(&values),
                        black_box(&relations),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_numberbatch_similarity(c: &mut Criterion) {
    let mock = MockNumberbatch::new(300);

    // Pre-generate embeddings
    let embeddings: Vec<Vec<f32>> = (0..1000)
        .map(|i| mock.get(&format!("/c/en/concept_{}", i)))
        .collect();

    c.bench_function("cosine_similarity_1000", |b| {
        b.iter(|| {
            let mut total = 0.0f32;
            for i in 0..100 {
                let sim = cosine_similarity(&embeddings[i], &embeddings[i + 1]);
                total += sim;
            }
            black_box(total)
        })
    });
}

fn bench_numberbatch_most_similar(c: &mut Criterion) {
    use conceptnet_integration::numberbatch::Numberbatch;

    let mut nb = Numberbatch::new(300);

    // Add many concepts
    for i in 0..1000 {
        nb.set(
            &format!("/c/en/concept_{}", i),
            (0..300).map(|j| ((i * j) as f32).sin()).collect(),
        )
        .unwrap();
    }

    c.bench_function("most_similar_k10", |b| {
        b.iter(|| nb.most_similar(black_box("/c/en/concept_0"), black_box(10)))
    });
}

// Helper functions

fn create_mock_edge(
    start: &str,
    end: &str,
    rel: &str,
    weight: f64,
) -> conceptnet_integration::api::Edge {
    use conceptnet_integration::api::{ConceptNode, Edge, Relation};

    Edge {
        id: format!("/a/[/r/{}/,/c/en/{}/,/c/en/{}/]", rel, start, end),
        start: ConceptNode {
            id: format!("/c/en/{}", start),
            label: Some(start.to_string()),
            language: Some("en".to_string()),
            term: Some(start.to_string()),
            sense_label: None,
        },
        end: ConceptNode {
            id: format!("/c/en/{}", end),
            label: Some(end.to_string()),
            language: Some("en".to_string()),
            term: Some(end.to_string()),
            sense_label: None,
        },
        rel: Relation {
            id: format!("/r/{}", rel),
            label: Some(rel.to_string()),
        },
        weight,
        surface_text: None,
        license: None,
        dataset: None,
        sources: vec![],
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

criterion_group!(
    benches,
    bench_graph_shortest_path,
    bench_graph_neighbors,
    bench_gnn_forward,
    bench_gnn_link_prediction,
    bench_attention_forward,
    bench_numberbatch_similarity,
    bench_numberbatch_most_similar,
);
criterion_main!(benches);
