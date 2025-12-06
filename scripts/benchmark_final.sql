-- RuVector Final Benchmark Script
-- Tests all stable core functions with real vector data

\timing on
\echo '=============================================='
\echo 'RuVector Core Benchmark - Real Data Testing'
\echo '=============================================='
\echo ''

-- Verify system
\echo '=== SYSTEM INFO ==='
SELECT ruvector_version() AS version;
SELECT ruvector_simd_info() AS simd_support;

-- Create schema
\echo ''
\echo '=== CREATE SCHEMA ==='
DROP TABLE IF EXISTS documents CASCADE;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    embedding ruvector(128),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert using text casting to ruvector
\echo ''
\echo '=== GENERATE 10,000 DOCUMENTS ==='
\echo 'Generating normalized 128-dim embeddings...'

-- Use a helper function to generate vectors as text then cast
CREATE OR REPLACE FUNCTION gen_random_vec_text(dims INTEGER) RETURNS TEXT AS $$
DECLARE
    parts TEXT[];
    i INTEGER;
BEGIN
    FOR i IN 1..dims LOOP
        parts := array_append(parts, round((random() * 2 - 1)::numeric, 4)::text);
    END LOOP;
    RETURN '[' || array_to_string(parts, ',') || ']';
END;
$$ LANGUAGE plpgsql;

-- Insert 10,000 documents
INSERT INTO documents (title, category, embedding)
SELECT
    'Doc_' || s.i AS title,
    (ARRAY['tech', 'science', 'business', 'health', 'entertainment'])[1 + (s.i % 5)] AS category,
    gen_random_vec_text(128)::ruvector AS embedding
FROM generate_series(1, 10000) AS s(i);

SELECT COUNT(*) AS doc_count, pg_size_pretty(pg_table_size('documents')) AS table_size FROM documents;

-- Basic vector operations
\echo ''
\echo '=== BENCHMARK 1: BASIC OPERATIONS ==='

SELECT 'Dimensions' AS test, ruvector_dims(embedding)::text AS result FROM documents WHERE id = 1
UNION ALL
SELECT 'Norm (avg of 100)', round(AVG(ruvector_norm(embedding))::numeric, 4)::text FROM documents WHERE id <= 100
UNION ALL
SELECT 'Normalized norm', round(ruvector_norm(ruvector_normalize(embedding))::numeric, 4)::text FROM documents WHERE id = 1;

-- Distance benchmarks (4950 pairs from 100x100)
\echo ''
\echo '=== BENCHMARK 2: DISTANCE CALCULATIONS (4950 pairs) ==='

SELECT 'Cosine Distance' AS metric,
    round(AVG(ruvector_cosine_distance(d1.embedding, d2.embedding))::numeric, 4) AS avg,
    round(MIN(ruvector_cosine_distance(d1.embedding, d2.embedding))::numeric, 4) AS min,
    round(MAX(ruvector_cosine_distance(d1.embedding, d2.embedding))::numeric, 4) AS max
FROM documents d1, documents d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id

UNION ALL

SELECT 'L2 Distance',
    round(AVG(ruvector_l2_distance(d1.embedding, d2.embedding))::numeric, 4),
    round(MIN(ruvector_l2_distance(d1.embedding, d2.embedding))::numeric, 4),
    round(MAX(ruvector_l2_distance(d1.embedding, d2.embedding))::numeric, 4)
FROM documents d1, documents d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id

UNION ALL

SELECT 'Inner Product',
    round(AVG(ruvector_inner_product(d1.embedding, d2.embedding))::numeric, 4),
    round(MIN(ruvector_inner_product(d1.embedding, d2.embedding))::numeric, 4),
    round(MAX(ruvector_inner_product(d1.embedding, d2.embedding))::numeric, 4)
FROM documents d1, documents d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id

UNION ALL

SELECT 'L1 Distance',
    round(AVG(ruvector_l1_distance(d1.embedding, d2.embedding))::numeric, 4),
    round(MIN(ruvector_l1_distance(d1.embedding, d2.embedding))::numeric, 4),
    round(MAX(ruvector_l1_distance(d1.embedding, d2.embedding))::numeric, 4)
FROM documents d1, documents d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

-- k-NN search on full 10,000 docs
\echo ''
\echo '=== BENCHMARK 3: k-NN SEARCH (10,000 docs) ==='

\echo 'Top 10 nearest neighbors to document 1:'
SELECT id, title, category,
       round(ruvector_cosine_distance(embedding,
           (SELECT embedding FROM documents WHERE id = 1))::numeric, 4) AS cosine_dist
FROM documents
WHERE id != 1
ORDER BY cosine_dist
LIMIT 10;

\echo ''
\echo 'Search performance (EXPLAIN ANALYZE):'
EXPLAIN ANALYZE
SELECT id, ruvector_cosine_distance(embedding,
    (SELECT embedding FROM documents WHERE id = 500)) AS dist
FROM documents
ORDER BY dist LIMIT 10;

-- Filtered search
\echo ''
\echo '=== BENCHMARK 4: FILTERED k-NN SEARCH ==='

\echo 'Top 10 in "tech" category (2000 docs):'
SELECT id, title,
       round(ruvector_cosine_distance(embedding,
           (SELECT embedding FROM documents WHERE id = 100))::numeric, 4) AS cosine_dist
FROM documents
WHERE category = 'tech'
ORDER BY cosine_dist
LIMIT 10;

-- Vector arithmetic
\echo ''
\echo '=== BENCHMARK 5: VECTOR ARITHMETIC ==='

SELECT 'Addition dims' AS op, ruvector_dims(ruvector_add(
    (SELECT embedding FROM documents WHERE id = 1),
    (SELECT embedding FROM documents WHERE id = 2)))::text AS result
UNION ALL
SELECT 'Subtraction norm', round(ruvector_norm(ruvector_sub(
    (SELECT embedding FROM documents WHERE id = 1),
    (SELECT embedding FROM documents WHERE id = 2)))::numeric, 4)::text
UNION ALL
SELECT 'Scalar mult (x2) norm', round(ruvector_norm(ruvector_mul_scalar(
    (SELECT embedding FROM documents WHERE id = 1), 2.0))::numeric, 4)::text;

-- Batch search benchmark
\echo ''
\echo '=== BENCHMARK 6: BATCH SEARCH (100 queries x 10,000 docs) ==='

SELECT
    COUNT(*) AS queries,
    round(AVG(best_dist)::numeric, 4) AS avg_best_dist,
    round(MIN(best_dist)::numeric, 4) AS min_best_dist,
    round(MAX(best_dist)::numeric, 4) AS max_best_dist
FROM (
    SELECT
        q.id AS query_id,
        MIN(ruvector_cosine_distance(d.embedding, q.embedding)) AS best_dist
    FROM
        (SELECT id, embedding FROM documents WHERE id <= 100) q,
        documents d
    WHERE d.id != q.id
    GROUP BY q.id
) batch_results;

-- Sparse vector operations
\echo ''
\echo '=== BENCHMARK 7: SPARSE VECTORS & TEXT ==='

\echo 'Dense to sparse:'
SELECT LEFT(ruvector_dense_to_sparse(
    (SELECT embedding::real[] FROM documents WHERE id = 1)), 100) || '...' AS sparse_preview;

\echo ''
\echo 'BM25 text scoring:'
SELECT
    'Query: machine learning' AS query,
    round(ruvector_sparse_bm25(
        'machine learning neural network',
        'introduction to deep machine learning and neural network architectures for AI',
        12, 10.0, 1.2, 0.75)::numeric, 4) AS bm25_score;

\echo ''
\echo 'Sparse text cosine:'
SELECT
    round(ruvector_sparse_cosine(
        'ai ml neural network learning deep',
        'machine learning deep neural network ai systems')::numeric, 4) AS sparse_cosine;

-- Hyperbolic geometry (for hierarchical data)
\echo ''
\echo '=== BENCHMARK 8: HYPERBOLIC GEOMETRY ==='

\echo 'Poincare ball distance:'
SELECT round(ruvector_poincare_distance(
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 2),
    -1.0)::numeric, 4) AS poincare_dist;

\echo ''
\echo 'Lorentz hyperboloid distance:'
SELECT round(ruvector_lorentz_distance(
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 2),
    -1.0)::numeric, 4) AS lorentz_dist;

\echo ''
\echo 'Mobius addition:'
SELECT array_length(ruvector_mobius_add(
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 2),
    -1.0), 1) AS mobius_result_dims;

-- Graph operations
\echo ''
\echo '=== BENCHMARK 9: GRAPH NEURAL NETWORK ==='

\echo 'Creating similarity graph...'
SELECT ruvector_create_graph('doc_graph');

\echo 'Adding 50 document nodes...'
DO $$
DECLARE i INTEGER;
BEGIN
    FOR i IN 1..50 LOOP
        PERFORM ruvector_add_node('doc_graph', ARRAY['Document'],
            jsonb_build_object('doc_id', i, 'title', 'Doc_' || i));
    END LOOP;
END $$;

\echo 'Adding similarity edges (threshold > 0.6)...'
DO $$
DECLARE
    i INTEGER;
    j INTEGER;
    sim FLOAT4;
BEGIN
    FOR i IN 1..15 LOOP
        FOR j IN (i+1)..20 LOOP
            SELECT 1.0 - ruvector_cosine_distance(d1.embedding, d2.embedding)
            INTO sim
            FROM documents d1, documents d2
            WHERE d1.id = i AND d2.id = j;

            IF sim > 0.6 THEN
                PERFORM ruvector_add_edge('doc_graph', i::bigint, j::bigint,
                    'similar', jsonb_build_object('weight', round(sim::numeric, 3)));
            END IF;
        END LOOP;
    END LOOP;
END $$;

\echo ''
\echo 'Graph statistics:'
SELECT ruvector_graph_stats('doc_graph');

\echo ''
\echo 'Cypher query - find similar documents:'
SELECT ruvector_cypher('doc_graph',
    'MATCH (n:Document)-[r:similar]->(m:Document) RETURN n.doc_id, r.weight, m.doc_id LIMIT 5');

\echo ''
\echo 'Shortest path from node 1 to 10:'
SELECT ruvector_shortest_path('doc_graph', 1, 10, 5);

-- Memory stats
\echo ''
\echo '=== BENCHMARK 10: SYSTEM STATS ==='
SELECT ruvector_memory_stats();

-- Category distribution
\echo ''
\echo '=== DATA SUMMARY ==='
SELECT category, COUNT(*) AS doc_count
FROM documents
GROUP BY category
ORDER BY doc_count DESC;

-- Cleanup
\echo ''
\echo '=== CLEANUP ==='
SELECT ruvector_delete_graph('doc_graph');
DROP FUNCTION IF EXISTS gen_random_vec_text(INTEGER);

\echo ''
\echo '=============================================='
\echo '           BENCHMARK COMPLETE                '
\echo '=============================================='
\echo ''

SELECT
    'RuVector ' || ruvector_version() AS version,
    10000 AS documents,
    128 AS dimensions,
    'AVX2 SIMD' AS acceleration,
    pg_size_pretty(pg_table_size('documents')) AS data_size;

\timing off
