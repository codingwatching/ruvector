-- RuVector Core Benchmark Script
-- Tests stable core functions with real vector data

\timing on
\echo '=============================================='
\echo 'RuVector Core Benchmark - Real Data Testing'
\echo '=============================================='

-- Verify system
\echo '\n=== SYSTEM INFO ==='
SELECT ruvector_version() AS version;
SELECT ruvector_simd_info() AS simd_support;

-- Create schema
\echo '\n=== CREATE SCHEMA ==='
DROP TABLE IF EXISTS documents CASCADE;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    embedding ruvector(128),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert 10,000 documents using inline SQL (not PL/pgSQL function)
\echo '\n=== GENERATE 10,000 DOCUMENTS ==='
\echo 'This may take 30-60 seconds...'

-- Generate embeddings directly in SQL
INSERT INTO documents (title, category, embedding)
SELECT
    'Doc_' || s.i AS title,
    (ARRAY['tech', 'science', 'business', 'health', 'entertainment'])[1 + (s.i % 5)] AS category,
    -- Create embedding by normalizing random values
    (SELECT string_to_ruvector(
        '[' || array_to_string(
            (SELECT array_agg(round((random() * 2 - 1)::numeric, 4)::float4)
             FROM generate_series(1, 128)), ',') || ']'
    )) AS embedding
FROM generate_series(1, 10000) AS s(i);

SELECT COUNT(*) AS doc_count, pg_size_pretty(pg_table_size('documents')) AS size FROM documents;

-- Basic vector operations
\echo '\n=== BENCHMARK 1: BASIC OPERATIONS ==='

\echo 'Dimensions check:'
SELECT ruvector_dims(embedding) AS dims FROM documents LIMIT 1;

\echo 'Norm calculation (first 100):'
SELECT AVG(ruvector_norm(embedding)) AS avg_norm FROM documents WHERE id <= 100;

\echo 'Normalization check:'
SELECT ruvector_norm(ruvector_normalize(embedding)) AS norm_after_normalize
FROM documents WHERE id = 1;

-- Distance benchmarks
\echo '\n=== BENCHMARK 2: DISTANCE CALCULATIONS (10,000 pairs) ==='

\echo 'Cosine distance:'
SELECT
    COUNT(*) AS pairs,
    round(AVG(ruvector_cosine_distance(d1.embedding, d2.embedding))::numeric, 4) AS avg_cosine,
    round(MIN(ruvector_cosine_distance(d1.embedding, d2.embedding))::numeric, 4) AS min_cosine,
    round(MAX(ruvector_cosine_distance(d1.embedding, d2.embedding))::numeric, 4) AS max_cosine
FROM documents d1
CROSS JOIN documents d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

\echo 'L2 distance:'
SELECT
    round(AVG(ruvector_l2_distance(d1.embedding, d2.embedding))::numeric, 4) AS avg_l2,
    round(MIN(ruvector_l2_distance(d1.embedding, d2.embedding))::numeric, 4) AS min_l2,
    round(MAX(ruvector_l2_distance(d1.embedding, d2.embedding))::numeric, 4) AS max_l2
FROM documents d1
CROSS JOIN documents d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

\echo 'Inner product:'
SELECT
    round(AVG(ruvector_inner_product(d1.embedding, d2.embedding))::numeric, 4) AS avg_inner,
    round(MIN(ruvector_inner_product(d1.embedding, d2.embedding))::numeric, 4) AS min_inner,
    round(MAX(ruvector_inner_product(d1.embedding, d2.embedding))::numeric, 4) AS max_inner
FROM documents d1
CROSS JOIN documents d2
WHERE d1.id <= 100 AND d2.id <= 100 AND d1.id < d2.id;

-- k-NN search
\echo '\n=== BENCHMARK 3: k-NN SEARCH ==='

\echo 'Full scan search - Top 10 nearest to doc 1:'
SELECT id, title, category,
       round(ruvector_cosine_distance(embedding,
           (SELECT embedding FROM documents WHERE id = 1))::numeric, 4) AS distance
FROM documents
WHERE id != 1
ORDER BY distance
LIMIT 10;

\echo '\nExplain analyze (full 10k scan):'
EXPLAIN ANALYZE
SELECT id, ruvector_cosine_distance(embedding,
    (SELECT embedding FROM documents WHERE id = 500)) AS dist
FROM documents
ORDER BY dist LIMIT 10;

-- Filtered search
\echo '\n=== BENCHMARK 4: FILTERED SEARCH ==='

\echo 'Top 10 in tech category (2000 docs):'
SELECT id, title,
       round(ruvector_cosine_distance(embedding,
           (SELECT embedding FROM documents WHERE id = 100))::numeric, 4) AS distance
FROM documents
WHERE category = 'tech'
ORDER BY distance
LIMIT 10;

-- Vector arithmetic
\echo '\n=== BENCHMARK 5: VECTOR ARITHMETIC ==='

\echo 'Vector addition:'
SELECT ruvector_dims(ruvector_add(
    (SELECT embedding FROM documents WHERE id = 1),
    (SELECT embedding FROM documents WHERE id = 2)
)) AS result_dims;

\echo 'Vector difference norm:'
SELECT round(ruvector_norm(ruvector_sub(
    (SELECT embedding FROM documents WHERE id = 1),
    (SELECT embedding FROM documents WHERE id = 2)
))::numeric, 4) AS diff_norm;

\echo 'Scalar multiply:'
SELECT round(ruvector_norm(ruvector_mul_scalar(
    (SELECT embedding FROM documents WHERE id = 1), 2.0
))::numeric, 4) AS scaled_norm;

-- Batch search benchmark
\echo '\n=== BENCHMARK 6: BATCH SEARCH (100 queries) ==='

SELECT
    COUNT(*) AS total_queries,
    round(AVG(best_distance)::numeric, 4) AS avg_best_distance
FROM (
    SELECT
        query_id,
        MIN(ruvector_cosine_distance(d.embedding, q.embedding)) AS best_distance
    FROM
        (SELECT id AS query_id, embedding FROM documents WHERE id <= 100) q,
        documents d
    WHERE d.id != q.query_id
    GROUP BY query_id
) batch_results;

-- Sparse vector operations
\echo '\n=== BENCHMARK 7: SPARSE VECTORS ==='

\echo 'Dense to sparse conversion:'
SELECT ruvector_dense_to_sparse(
    (SELECT embedding::real[] FROM documents WHERE id = 1)
) AS sparse_rep;

\echo 'BM25 text scoring:'
SELECT
    round(ruvector_sparse_bm25(
        'machine learning neural network deep',
        'introduction to deep machine learning and neural network architectures',
        10, 8.5, 1.2, 0.75
    )::numeric, 4) AS bm25_score;

\echo 'Sparse text similarity:'
SELECT
    round(ruvector_sparse_cosine(
        'ai ml neural network learning',
        'machine learning deep neural network ai'
    )::numeric, 4) AS sparse_cosine;

-- Hyperbolic geometry
\echo '\n=== BENCHMARK 8: HYPERBOLIC GEOMETRY ==='

\echo 'Poincare distance (hierarchical embeddings):'
SELECT round(ruvector_poincare_distance(
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 2),
    -1.0
)::numeric, 4) AS poincare_dist;

\echo 'Mobius addition:'
SELECT array_length(ruvector_mobius_add(
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 1),
    (SELECT (embedding::real[])[1:10] FROM documents WHERE id = 2),
    -1.0
), 1) AS mobius_dims;

-- Graph operations
\echo '\n=== BENCHMARK 9: GRAPH OPERATIONS ==='

SELECT ruvector_create_graph('similarity_graph');

\echo 'Adding nodes to graph...'
DO $$
BEGIN
    FOR i IN 1..50 LOOP
        PERFORM ruvector_add_node('similarity_graph',
            ARRAY['Document'],
            jsonb_build_object('doc_id', i));
    END LOOP;
END $$;

\echo 'Adding similarity edges...'
DO $$
DECLARE
    sim FLOAT4;
BEGIN
    FOR i IN 1..10 LOOP
        FOR j IN (i+1)..15 LOOP
            -- Calculate similarity
            SELECT 1.0 - ruvector_cosine_distance(d1.embedding, d2.embedding)
            INTO sim
            FROM documents d1, documents d2
            WHERE d1.id = i AND d2.id = j;

            IF sim > 0.5 THEN
                PERFORM ruvector_add_edge('similarity_graph', i::bigint, j::bigint, 'similar',
                    jsonb_build_object('weight', sim));
            END IF;
        END LOOP;
    END LOOP;
END $$;

\echo 'Graph stats:'
SELECT ruvector_graph_stats('similarity_graph');

\echo 'Shortest path (1 to 5):'
SELECT ruvector_shortest_path('similarity_graph', 1, 5, 5);

-- Memory stats
\echo '\n=== BENCHMARK 10: SYSTEM STATS ==='
SELECT ruvector_memory_stats();

-- Category distribution
\echo '\n=== DATA DISTRIBUTION ==='
SELECT category, COUNT(*) AS count FROM documents GROUP BY category ORDER BY category;

-- Cleanup
\echo '\n=== CLEANUP ==='
SELECT ruvector_delete_graph('similarity_graph');

-- Final summary
\echo '\n=============================================='
\echo 'BENCHMARK COMPLETE'
\echo '=============================================='
SELECT
    'RuVector 0.2.5' AS version,
    10000 AS documents_tested,
    128 AS vector_dims,
    'AVX2 SIMD' AS acceleration,
    pg_size_pretty(pg_table_size('documents')) AS data_size;

\timing off
