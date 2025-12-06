-- RuVector Comprehensive Benchmark Script v2
-- Uses correct API signatures for all RuVector functions

\timing on
\echo '=============================================='
\echo 'RuVector Benchmark Suite v2.0'
\echo '=============================================='

-- Verify RuVector extension
\echo '\n=== 1. VERIFY RUVECTOR EXTENSION ==='
SELECT ruvector_version() AS version;
SELECT ruvector_simd_info() AS simd_support;
SELECT COUNT(*) AS total_functions FROM pg_proc WHERE proname LIKE 'ruvector_%';

-- Create benchmark schema
\echo '\n=== 2. CREATE BENCHMARK SCHEMA ==='
DROP TABLE IF EXISTS benchmark_docs CASCADE;
DROP TABLE IF EXISTS benchmark_results CASCADE;

-- Document embeddings table using native ruvector type
CREATE TABLE benchmark_docs (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    embedding ruvector(128),  -- Using 128-dim for faster benchmarking
    created_at TIMESTAMP DEFAULT NOW()
);

-- Results tracking
CREATE TABLE benchmark_results (
    id SERIAL PRIMARY KEY,
    test_name TEXT NOT NULL,
    duration_ms FLOAT4,
    records_processed INTEGER,
    ops_per_sec FLOAT4,
    created_at TIMESTAMP DEFAULT NOW()
);

\echo 'Tables created successfully'

-- Generate test data with proper ruvector type
\echo '\n=== 3. GENERATE TEST DATA (10,000 documents) ==='

-- Function to generate random normalized vector
CREATE OR REPLACE FUNCTION random_vector(dims INTEGER) RETURNS ruvector AS $$
DECLARE
    arr FLOAT4[];
    norm FLOAT4;
    i INTEGER;
BEGIN
    arr := ARRAY[]::FLOAT4[];
    norm := 0;
    FOR i IN 1..dims LOOP
        arr := arr || (random() * 2 - 1)::FLOAT4;
        norm := norm + arr[i] * arr[i];
    END LOOP;
    norm := sqrt(norm);
    IF norm > 0 THEN
        FOR i IN 1..dims LOOP
            arr[i] := arr[i] / norm;
        END LOOP;
    END IF;
    RETURN arr::ruvector;
END;
$$ LANGUAGE plpgsql;

-- Insert 10,000 documents with normalized random embeddings
INSERT INTO benchmark_docs (title, category, embedding)
SELECT
    'Document ' || i AS title,
    CASE (i % 5)
        WHEN 0 THEN 'tech'
        WHEN 1 THEN 'science'
        WHEN 2 THEN 'business'
        WHEN 3 THEN 'health'
        ELSE 'entertainment'
    END AS category,
    random_vector(128) AS embedding
FROM generate_series(1, 10000) AS i;

SELECT COUNT(*) AS document_count FROM benchmark_docs;
SELECT pg_size_pretty(pg_table_size('benchmark_docs')) AS table_size;

-- Benchmark 1: Basic Vector Operations
\echo '\n=== 4. BENCHMARK: BASIC VECTOR OPERATIONS ==='

-- Test ruvector_dims
\echo 'Testing ruvector_dims:'
SELECT ruvector_dims(embedding) AS dims
FROM benchmark_docs LIMIT 1;

-- Test ruvector_norm
\echo 'Testing ruvector_norm (100 vectors):'
SELECT AVG(ruvector_norm(embedding)) AS avg_norm
FROM benchmark_docs WHERE id <= 100;

-- Test ruvector_normalize
\echo 'Testing ruvector_normalize:'
SELECT ruvector_norm(ruvector_normalize(embedding)) AS normalized_norm
FROM benchmark_docs WHERE id = 1;

-- Benchmark 2: Distance Calculations
\echo '\n=== 5. BENCHMARK: DISTANCE CALCULATIONS ==='

-- Cosine distance
\echo 'Cosine distance (100x100 pairs = 10,000 calculations):'
SELECT
    COUNT(*) AS pairs_computed,
    AVG(ruvector_cosine_distance(d1.embedding, d2.embedding)) AS avg_cosine_dist,
    MIN(ruvector_cosine_distance(d1.embedding, d2.embedding)) AS min_cosine_dist,
    MAX(ruvector_cosine_distance(d1.embedding, d2.embedding)) AS max_cosine_dist
FROM benchmark_docs d1
CROSS JOIN benchmark_docs d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

-- L2 distance
\echo 'L2/Euclidean distance (100x100 pairs):'
SELECT
    AVG(ruvector_l2_distance(d1.embedding, d2.embedding)) AS avg_l2_dist
FROM benchmark_docs d1
CROSS JOIN benchmark_docs d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

-- Inner product
\echo 'Inner product (100x100 pairs):'
SELECT
    AVG(ruvector_inner_product(d1.embedding, d2.embedding)) AS avg_inner_prod
FROM benchmark_docs d1
CROSS JOIN benchmark_docs d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

-- L1/Manhattan distance
\echo 'L1/Manhattan distance (100x100 pairs):'
SELECT
    AVG(ruvector_l1_distance(d1.embedding, d2.embedding)) AS avg_l1_dist
FROM benchmark_docs d1
CROSS JOIN benchmark_docs d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

-- Benchmark 3: Vector Arithmetic
\echo '\n=== 6. BENCHMARK: VECTOR ARITHMETIC ==='

\echo 'Vector addition:'
SELECT ruvector_dims(ruvector_add(d1.embedding, d2.embedding)) AS result_dims
FROM benchmark_docs d1, benchmark_docs d2
WHERE d1.id = 1 AND d2.id = 2;

\echo 'Vector subtraction:'
SELECT ruvector_norm(ruvector_sub(d1.embedding, d2.embedding)) AS diff_norm
FROM benchmark_docs d1, benchmark_docs d2
WHERE d1.id = 1 AND d2.id = 2;

\echo 'Scalar multiplication:'
SELECT ruvector_norm(ruvector_mul_scalar(embedding, 2.0)) AS scaled_norm
FROM benchmark_docs WHERE id = 1;

-- Benchmark 4: k-NN Search (without HNSW, using brute force)
\echo '\n=== 7. BENCHMARK: k-NN SEARCH (BRUTE FORCE) ==='

\echo 'Top 10 nearest neighbors for query 1 (cosine):'
EXPLAIN ANALYZE
SELECT id, title, category,
       ruvector_cosine_distance(embedding, (SELECT embedding FROM benchmark_docs WHERE id = 1)) AS distance
FROM benchmark_docs
WHERE id != 1
ORDER BY distance
LIMIT 10;

\echo '\nTop 10 nearest neighbors for query 1 (L2):'
EXPLAIN ANALYZE
SELECT id, title, category,
       ruvector_l2_distance(embedding, (SELECT embedding FROM benchmark_docs WHERE id = 1)) AS distance
FROM benchmark_docs
WHERE id != 1
ORDER BY distance
LIMIT 10;

-- Benchmark 5: Category-filtered Search
\echo '\n=== 8. BENCHMARK: FILTERED SEARCH ==='

\echo 'Top 10 in category "tech" (2000 docs):'
EXPLAIN ANALYZE
SELECT id, title,
       ruvector_cosine_distance(embedding, (SELECT embedding FROM benchmark_docs WHERE id = 50)) AS distance
FROM benchmark_docs
WHERE category = 'tech'
ORDER BY distance
LIMIT 10;

-- Benchmark 6: GNN Operations
\echo '\n=== 9. BENCHMARK: GNN (Graph Neural Network) ==='

-- GCN Forward Pass
\echo 'GCN Forward Pass (10 nodes, 5 edges):'
SELECT array_length(ruvector_gcn_forward(
    -- Features: 10 nodes x 4 features = 40 values
    ARRAY[0.1, 0.2, 0.3, 0.4,
          0.5, 0.6, 0.7, 0.8,
          0.1, 0.3, 0.5, 0.7,
          0.2, 0.4, 0.6, 0.8,
          0.3, 0.5, 0.7, 0.9,
          0.4, 0.6, 0.8, 1.0,
          0.5, 0.7, 0.9, 0.1,
          0.6, 0.8, 0.0, 0.2,
          0.7, 0.9, 0.1, 0.3,
          0.8, 0.0, 0.2, 0.4]::real[],
    -- Source nodes
    ARRAY[0, 1, 2, 3, 4]::integer[],
    -- Target nodes
    ARRAY[1, 2, 3, 4, 5]::integer[],
    -- Weights (4 in x 8 out = 32)
    (SELECT array_agg((random() - 0.5)::real) FROM generate_series(1, 32)),
    8  -- Output dimension
), 1) AS output_features;

-- GraphSAGE Forward Pass
\echo 'GraphSAGE Forward Pass (10 nodes, 5 edges):'
SELECT array_length(ruvector_graphsage_forward(
    ARRAY[0.1, 0.2, 0.3, 0.4,
          0.5, 0.6, 0.7, 0.8,
          0.1, 0.3, 0.5, 0.7,
          0.2, 0.4, 0.6, 0.8,
          0.3, 0.5, 0.7, 0.9,
          0.4, 0.6, 0.8, 1.0,
          0.5, 0.7, 0.9, 0.1,
          0.6, 0.8, 0.0, 0.2,
          0.7, 0.9, 0.1, 0.3,
          0.8, 0.0, 0.2, 0.4]::real[],
    ARRAY[0, 1, 2, 3, 4]::integer[],
    ARRAY[1, 2, 3, 4, 5]::integer[],
    8,  -- Output dimension
    3   -- Sample size
), 1) AS output_features;

-- Benchmark 7: Graph Operations
\echo '\n=== 10. BENCHMARK: GRAPH OPERATIONS ==='

-- Create a similarity graph
\echo 'Creating document graph...'
SELECT ruvector_create_graph('doc_similarity');

-- Add nodes to graph
\echo 'Adding 100 nodes to graph...'
DO $$
DECLARE
    doc RECORD;
BEGIN
    FOR doc IN SELECT id, title, category FROM benchmark_docs WHERE id <= 100 LOOP
        PERFORM ruvector_add_node('doc_similarity',
            ARRAY['Document', doc.category],
            jsonb_build_object('id', doc.id, 'title', doc.title));
    END LOOP;
END $$;

-- Add edges between similar documents
\echo 'Adding edges between similar documents...'
DO $$
DECLARE
    d1 RECORD;
    d2 RECORD;
    sim FLOAT4;
BEGIN
    FOR d1 IN SELECT id, embedding FROM benchmark_docs WHERE id <= 20 LOOP
        FOR d2 IN SELECT id, embedding FROM benchmark_docs WHERE id > d1.id AND id <= 20 LOOP
            sim := 1.0 - ruvector_cosine_distance(d1.embedding, d2.embedding);
            IF sim > 0.6 THEN
                PERFORM ruvector_add_edge('doc_similarity', d1.id, d2.id, 'similar',
                    jsonb_build_object('similarity', sim));
            END IF;
        END LOOP;
    END LOOP;
END $$;

-- Graph statistics
\echo 'Graph statistics:'
SELECT ruvector_graph_stats('doc_similarity');

-- List all graphs
\echo 'All graphs:'
SELECT ruvector_list_graphs();

-- Benchmark 8: Agent Routing (ruvLLM)
\echo '\n=== 11. BENCHMARK: AGENT ROUTING (ruvLLM) ==='

-- Clear any existing agents
SELECT ruvector_clear_agents();

-- Register agents with different specializations
\echo 'Registering agents...'
SELECT ruvector_register_agent('claude-4-code', 'code_assistant',
    ARRAY['coding', 'debugging', 'code_review'], 0.02, 150, 0.95);
SELECT ruvector_register_agent('gemini-2-research', 'research_assistant',
    ARRAY['research', 'analysis', 'summarization'], 0.015, 200, 0.90);
SELECT ruvector_register_agent('llama-4-general', 'general_assistant',
    ARRAY['general', 'chat', 'help'], 0.005, 100, 0.85);
SELECT ruvector_register_agent('claude-4-math', 'math_assistant',
    ARRAY['math', 'statistics', 'calculations'], 0.025, 180, 0.92);
SELECT ruvector_register_agent('gemini-2-creative', 'creative_assistant',
    ARRAY['writing', 'creative', 'storytelling'], 0.018, 220, 0.88);

-- List registered agents
\echo 'Registered agents:'
SELECT ruvector_list_agents();

-- Route queries to agents
\echo 'Routing queries to agents...'
SELECT ruvector_route(
    (SELECT (embedding::real[])[1:128] FROM benchmark_docs WHERE id = 1),
    'balanced'
) AS routed_agent;

-- Route with quality optimization
\echo 'Route optimized for quality:'
SELECT ruvector_route(
    (SELECT (embedding::real[])[1:128] FROM benchmark_docs WHERE id = 2),
    'quality'
) AS quality_agent;

-- Route with cost optimization
\echo 'Route optimized for cost:'
SELECT ruvector_route(
    (SELECT (embedding::real[])[1:128] FROM benchmark_docs WHERE id = 3),
    'cost'
) AS cost_agent;

-- Get agent by name
\echo 'Get specific agent:'
SELECT ruvector_get_agent('claude-4-code');

-- Routing statistics
\echo 'Routing statistics:'
SELECT ruvector_routing_stats();

-- Benchmark 9: Learning System
\echo '\n=== 12. BENCHMARK: LEARNING SYSTEM ==='

-- Enable learning for the benchmark table
\echo 'Enabling learning on benchmark_docs...'
SELECT ruvector_enable_learning('benchmark_docs', '{"min_feedback": 10, "learning_rate": 0.1}'::jsonb);

-- Record feedback (simulating user relevance feedback)
\echo 'Recording user feedback...'
SELECT ruvector_record_feedback(
    'benchmark_docs',
    (SELECT (embedding::real[])[1:128] FROM benchmark_docs WHERE id = 1),
    ARRAY[2, 3, 5, 8]::bigint[],    -- Relevant docs
    ARRAY[4, 6, 7]::bigint[]         -- Irrelevant docs
);

SELECT ruvector_record_feedback(
    'benchmark_docs',
    (SELECT (embedding::real[])[1:128] FROM benchmark_docs WHERE id = 10),
    ARRAY[11, 12, 15]::bigint[],
    ARRAY[13, 14]::bigint[]
);

-- Check learning statistics
\echo 'Learning statistics:'
SELECT ruvector_learning_stats('benchmark_docs');

-- Benchmark 10: Hyperbolic Operations
\echo '\n=== 13. BENCHMARK: HYPERBOLIC GEOMETRY ==='

-- Poincare distance (for hierarchical embeddings)
\echo 'Poincare distance:'
SELECT ruvector_poincare_distance(
    (SELECT (embedding::real[])[1:10] FROM benchmark_docs WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM benchmark_docs WHERE id = 2),
    -1.0  -- Curvature
) AS poincare_dist;

-- Lorentz distance
\echo 'Lorentz distance:'
SELECT ruvector_lorentz_distance(
    (SELECT (embedding::real[])[1:10] FROM benchmark_docs WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM benchmark_docs WHERE id = 2),
    -1.0
) AS lorentz_dist;

-- Mobius addition (hyperbolic)
\echo 'Mobius addition:'
SELECT array_length(ruvector_mobius_add(
    (SELECT (embedding::real[])[1:10] FROM benchmark_docs WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM benchmark_docs WHERE id = 2),
    -1.0
), 1) AS result_dims;

-- Benchmark 11: Sparse Vectors
\echo '\n=== 14. BENCHMARK: SPARSE VECTORS ==='

-- Dense to sparse conversion
\echo 'Dense to sparse conversion:'
SELECT ruvector_dense_to_sparse(
    (SELECT (embedding::real[])[1:20] FROM benchmark_docs WHERE id = 1)
) AS sparse_vec;

-- Sparse vector operations
\echo 'Sparse cosine similarity:'
SELECT ruvector_sparse_cosine(
    ruvector_dense_to_sparse((SELECT (embedding::real[])[1:20] FROM benchmark_docs WHERE id = 1)),
    ruvector_dense_to_sparse((SELECT (embedding::real[])[1:20] FROM benchmark_docs WHERE id = 2))
) AS sparse_cosine;

-- BM25 scoring
\echo 'BM25 text scoring:'
SELECT ruvector_sparse_bm25(
    'machine learning neural networks',
    'introduction to machine learning and deep neural networks for AI',
    12,   -- doc_len
    10.0, -- avg_doc_len
    1.2,  -- k1
    0.75  -- b
) AS bm25_score;

-- Benchmark 12: Auto-tuning
\echo '\n=== 15. BENCHMARK: AUTO-TUNING ==='

\echo 'Auto-tune for balanced performance:'
SELECT ruvector_auto_tune('benchmark_docs', 'balanced');

-- Pattern extraction
\echo 'Extract patterns (10 clusters):'
SELECT ruvector_extract_patterns('benchmark_docs', 10);

-- Memory stats
\echo '\n=== 16. SYSTEM MEMORY STATS ==='
SELECT ruvector_memory_stats();

-- Final Summary
\echo '\n=============================================='
\echo 'BENCHMARK SUMMARY'
\echo '=============================================='

SELECT
    (SELECT COUNT(*) FROM benchmark_docs) AS total_documents,
    (SELECT pg_size_pretty(pg_table_size('benchmark_docs'))) AS table_size,
    (SELECT COUNT(*)::text || ' agents' FROM (SELECT ruvector_list_agents()) t) AS registered_agents;

-- Cleanup
\echo '\nCleaning up...'
SELECT ruvector_delete_graph('doc_similarity');
DROP FUNCTION IF EXISTS random_vector(INTEGER);

\echo '\nBenchmark complete!'
\timing off
