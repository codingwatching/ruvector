-- RuVector Comprehensive Benchmark Script
-- Tests all major RuVector extension capabilities with real data

\timing on
\echo '=============================================='
\echo 'RuVector Benchmark Suite v1.0'
\echo '=============================================='

-- Verify RuVector extension
\echo '\n=== 1. VERIFY RUVECTOR EXTENSION ==='
SELECT ruvector_version() AS version;
SELECT COUNT(*) AS total_functions FROM pg_proc WHERE proname LIKE 'ruvector_%';

-- Create benchmark schema
\echo '\n=== 2. CREATE BENCHMARK SCHEMA ==='
DROP TABLE IF EXISTS benchmark_documents CASCADE;
DROP TABLE IF EXISTS benchmark_agents CASCADE;
DROP TABLE IF EXISTS benchmark_queries CASCADE;

-- Document embeddings table (1536-dim for OpenAI compatibility)
CREATE TABLE benchmark_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    embedding FLOAT4[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Agent definitions for routing
CREATE TABLE benchmark_agents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding FLOAT4[] NOT NULL,
    llm_provider TEXT NOT NULL
);

-- Query log for GNN learning
CREATE TABLE benchmark_queries (
    id SERIAL PRIMARY KEY,
    query_embedding FLOAT4[] NOT NULL,
    result_ids INTEGER[],
    user_feedback FLOAT4 DEFAULT 0.0,
    query_time_ms FLOAT4,
    created_at TIMESTAMP DEFAULT NOW()
);

\echo 'Tables created successfully'

-- Generate realistic embeddings using RuVector functions
\echo '\n=== 3. GENERATE TEST DATA (10,000 documents) ==='

-- Insert 10,000 documents with normalized random embeddings
-- Using categories: tech, science, business, health, entertainment
INSERT INTO benchmark_documents (title, category, embedding)
SELECT
    'Document ' || i AS title,
    CASE (i % 5)
        WHEN 0 THEN 'tech'
        WHEN 1 THEN 'science'
        WHEN 2 THEN 'business'
        WHEN 3 THEN 'health'
        ELSE 'entertainment'
    END AS category,
    -- Generate 1536-dim normalized random vectors
    (SELECT array_agg(
        -- Normalize: divide by sqrt(1536) ≈ 39.2
        (random() - 0.5) / 19.6
    ) FROM generate_series(1, 1536)) AS embedding
FROM generate_series(1, 10000) AS i;

SELECT COUNT(*) AS document_count FROM benchmark_documents;

-- Insert agents for routing benchmark
\echo '\n=== 4. CREATE AGENT DEFINITIONS ==='
INSERT INTO benchmark_agents (name, description, llm_provider, embedding)
VALUES
    ('CodeAssistant', 'Helps with programming and code review', 'Claude 4',
     (SELECT array_agg((random() - 0.5) / 19.6) FROM generate_series(1, 1536))),
    ('ResearchAgent', 'Analyzes scientific papers and data', 'Gemini 2.0',
     (SELECT array_agg((random() - 0.5) / 19.6) FROM generate_series(1, 1536))),
    ('BusinessAnalyst', 'Financial analysis and market research', 'Claude 4',
     (SELECT array_agg((random() - 0.5) / 19.6) FROM generate_series(1, 1536))),
    ('HealthAdvisor', 'Medical information and wellness tips', 'Llama 4',
     (SELECT array_agg((random() - 0.5) / 19.6) FROM generate_series(1, 1536))),
    ('CreativeWriter', 'Content creation and storytelling', 'Gemini 2.0',
     (SELECT array_agg((random() - 0.5) / 19.6) FROM generate_series(1, 1536)));

SELECT name, llm_provider FROM benchmark_agents;

-- Benchmark 1: Basic vector operations
\echo '\n=== 5. BENCHMARK: BASIC VECTOR OPERATIONS ==='

-- Test ruvector_dims
SELECT ruvector_dims(embedding::ruvector) AS dims
FROM benchmark_documents LIMIT 1;

-- Test cosine distance calculation (100 pairs)
\echo 'Cosine distance (100 pairs):'
SELECT AVG(ruvector_cosine_distance(
    d1.embedding::ruvector,
    d2.embedding::ruvector
)) AS avg_distance
FROM benchmark_documents d1
CROSS JOIN benchmark_documents d2
WHERE d1.id <= 10 AND d2.id BETWEEN 11 AND 20;

-- Benchmark 2: HNSW Index Creation
\echo '\n=== 6. BENCHMARK: HNSW INDEX CREATION ==='

-- Create HNSW index with custom parameters
\echo 'Creating HNSW index (M=16, ef_construction=200)...'
CREATE INDEX idx_documents_hnsw ON benchmark_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Check index size
SELECT pg_size_pretty(pg_relation_size('idx_documents_hnsw')) AS index_size;

-- Benchmark 3: Vector Search Performance
\echo '\n=== 7. BENCHMARK: VECTOR SEARCH (k-NN) ==='

-- Generate query vector
\echo 'Query 1: Top 10 nearest neighbors'
EXPLAIN ANALYZE
SELECT id, title, category,
       embedding <-> (SELECT embedding FROM benchmark_documents WHERE id = 1)::ruvector AS distance
FROM benchmark_documents
ORDER BY distance
LIMIT 10;

-- Benchmark 4: Category-filtered search
\echo '\nQuery 2: Top 10 in category "tech"'
EXPLAIN ANALYZE
SELECT id, title,
       embedding <-> (SELECT embedding FROM benchmark_documents WHERE id = 100)::ruvector AS distance
FROM benchmark_documents
WHERE category = 'tech'
ORDER BY distance
LIMIT 10;

-- Benchmark 5: Batch similarity search
\echo '\n=== 8. BENCHMARK: BATCH SIMILARITY SEARCH (100 queries) ==='

-- Run 100 similarity searches
SELECT COUNT(*) AS queries_executed,
       AVG(distance) AS avg_distance,
       MIN(distance) AS min_distance,
       MAX(distance) AS max_distance
FROM (
    SELECT query_id,
           (SELECT embedding <-> d.embedding::ruvector
            FROM benchmark_documents d
            ORDER BY embedding <-> d.embedding
            LIMIT 1) AS distance
    FROM (
        SELECT id AS query_id, embedding
        FROM benchmark_documents
        WHERE id <= 100
    ) queries
) results;

-- Benchmark 6: GNN Graph Operations
\echo '\n=== 9. BENCHMARK: GNN GRAPH OPERATIONS ==='

-- Create similarity graph
\echo 'Creating document similarity graph...'
SELECT ruvector_create_graph('doc_graph');

-- Add nodes (first 100 documents)
\echo 'Adding 100 nodes...'
SELECT COUNT(*) AS nodes_added FROM (
    SELECT ruvector_add_node('doc_graph', id::text, embedding::ruvector)
    FROM benchmark_documents
    WHERE id <= 100
) t;

-- Add edges based on similarity
\echo 'Adding edges (similarity > 0.8)...'
SELECT COUNT(*) AS edges_added FROM (
    SELECT ruvector_add_edge(
        'doc_graph',
        d1.id::text,
        d2.id::text,
        1.0 - ruvector_cosine_distance(d1.embedding::ruvector, d2.embedding::ruvector)
    )
    FROM benchmark_documents d1
    CROSS JOIN benchmark_documents d2
    WHERE d1.id < d2.id
      AND d1.id <= 20 AND d2.id <= 20
      AND ruvector_cosine_distance(d1.embedding::ruvector, d2.embedding::ruvector) < 0.2
) t;

-- Benchmark 7: Agent Routing
\echo '\n=== 10. BENCHMARK: AGENT ROUTING (ruvLLM) ==='

-- Route query to best agent
\echo 'Routing 10 queries to agents...'
SELECT
    q.id AS query_doc,
    a.name AS best_agent,
    a.llm_provider,
    ruvector_cosine_distance(q.embedding::ruvector, a.embedding::ruvector) AS distance
FROM benchmark_documents q
CROSS JOIN LATERAL (
    SELECT name, llm_provider, embedding
    FROM benchmark_agents
    ORDER BY q.embedding <-> embedding::ruvector
    LIMIT 1
) a
WHERE q.id <= 10;

-- Register agents with ruvLLM
\echo 'Registering agents with ruvLLM...'
SELECT ruvector_register_agent('CodeAssistant',
    (SELECT embedding FROM benchmark_agents WHERE name = 'CodeAssistant')::ruvector,
    '{"llm": "Claude 4", "specialization": "code"}'::jsonb);

SELECT ruvector_register_agent('ResearchAgent',
    (SELECT embedding FROM benchmark_agents WHERE name = 'ResearchAgent')::ruvector,
    '{"llm": "Gemini 2.0", "specialization": "research"}'::jsonb);

-- Route a query
\echo 'Test agent routing:'
SELECT ruvector_route(
    (SELECT embedding FROM benchmark_documents WHERE id = 1)::ruvector
) AS routed_agent;

-- Benchmark 8: GNN Learning
\echo '\n=== 11. BENCHMARK: GNN LEARNING ==='

-- Enable learning mode
SELECT ruvector_enable_learning(true);

-- Record positive feedback for search results
\echo 'Recording feedback for GNN training...'
SELECT ruvector_record_feedback(
    (SELECT embedding FROM benchmark_documents WHERE id = i)::ruvector,
    ARRAY[i+1, i+2, i+3]::integer[],
    0.9  -- High relevance score
)
FROM generate_series(1, 50) AS i;

-- Check learning stats
\echo 'Learning statistics:'
SELECT ruvector_learning_stats();

-- Benchmark 9: Memory and Performance Stats
\echo '\n=== 12. BENCHMARK: SYSTEM STATS ==='
SELECT ruvector_memory_stats();

-- Benchmark 10: Pattern Extraction
\echo '\n=== 13. BENCHMARK: PATTERN EXTRACTION ==='
\echo 'Extracting patterns from embeddings...'
SELECT ruvector_extract_patterns(embedding::ruvector) AS patterns
FROM benchmark_documents
WHERE id = 1;

-- Benchmark 11: Sparse Vector Operations
\echo '\n=== 14. BENCHMARK: SPARSE VECTORS ==='
\echo 'Dense to sparse conversion:'
SELECT ruvector_dense_to_sparse(embedding::ruvector, 0.01) AS sparse_dims
FROM benchmark_documents
WHERE id = 1;

-- Benchmark 12: Auto-tuning
\echo '\n=== 15. BENCHMARK: AUTO-TUNING ==='
SELECT ruvector_auto_tune();

-- Final Summary
\echo '\n=============================================='
\echo 'BENCHMARK SUMMARY'
\echo '=============================================='
SELECT
    (SELECT COUNT(*) FROM benchmark_documents) AS total_documents,
    (SELECT COUNT(*) FROM benchmark_agents) AS total_agents,
    (SELECT pg_size_pretty(pg_table_size('benchmark_documents'))) AS table_size,
    (SELECT pg_size_pretty(pg_relation_size('idx_documents_hnsw'))) AS index_size;

\echo '\nBenchmark complete!'
\timing off
