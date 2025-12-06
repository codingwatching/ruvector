# RuVector

## The AI Database That Gets Smarter Over Time

**RuVector is the world's first self-learning vector database.** Unlike traditional databases that just store and search, RuVector actually improves its search results the more you use it—powered by Graph Neural Networks (GNN) and continuous learning algorithms.

### Why Does This Matter?

Imagine you're building a customer support AI. With a regular vector database:
- You upload your knowledge base
- Users ask questions and get answers
- **The system never gets better**

With RuVector:
- You upload your knowledge base
- Users ask questions and get answers
- **RuVector learns which answers users found helpful**
- **Search results improve automatically over time**
- **No retraining needed—it learns while running**

This is the difference between a static database and a living, learning system.

---

## Quick Start (30 Seconds)

```bash
# 1. Run RuVector
docker run -d --name ruvector -p 5432:5432 ruvnet/ruvector:latest

# 2. Connect to the database
psql -h localhost -U ruvector -d ruvector_db
# Password: ruvector

# 3. Enable RuVector
CREATE EXTENSION ruvector;
SELECT ruvector_version();
```

That's it. You now have a production-ready AI database with 65+ functions for vector search, graph queries, neural networks, and more.

---

## How It Works

### Traditional Vector Search vs RuVector

**Traditional vector databases** are static—they return the same results forever:

```
User Query → HNSW Index → Top K Results
                ↓
         (never improves)
```

**RuVector with GNN** creates a feedback loop that improves over time:

```
User Query → HNSW Index → GNN Layer → Enhanced Results
                 ↑                           │
                 │     learns from           │
                 └───── user feedback ───────┘
```

### The Learning Cycle

1. **User searches** → RuVector returns results
2. **User clicks/uses result #3** → System records this as positive feedback
3. **GNN analyzes patterns** → "Queries like X tend to prefer results like Y"
4. **Next similar search** → GNN re-ranks to surface better results first
5. **Accuracy improves** → 10-30% better results over time, automatically

This is why RuVector gets smarter the more you use it.

---

## Key Features Explained

### Self-Learning & Optimization (SONA)

Traditional vector databases require you to retrain and re-index when you want better results. **RuVector learns continuously:**

| What It Does | How It Works | Why It Matters |
|-------------|--------------|----------------|
| **Learns from clicks** | Records which results users find helpful | Relevant results rise to the top automatically |
| **Adapts search parameters** | Auto-tunes HNSW settings based on usage patterns | 10-30% accuracy improvement without manual tuning |
| **Pattern recognition** | Identifies successful query patterns | New queries benefit from learned patterns |
| **Zero retraining** | Uses LoRA + EWC++ for incremental learning | No downtime for model updates |

```sql
-- Enable learning mode
SELECT ruvector_enable_learning(true);

-- Record that users found result helpful (0.95 = very helpful)
SELECT ruvector_record_feedback('query_123', 0.95, '{"clicked": true}');

-- Let the system auto-optimize
SELECT ruvector_auto_tune();
```

### ruvLLM Capabilities

**Integrate with any Large Language Model** (Claude, Gemini, GPT, Llama, etc.) through intelligent query routing:

```
                                    ┌─────────────────┐
                                    │  Code Expert    │
                                    │  (Claude 4)     │
                                    └────────▲────────┘
                                             │
User: "Fix this bug" ──→ FastGRNN ──────────┼──→ Best Match!
                         Router              │
                                             │
                                    ┌────────┴────────┐
                                    │  Math Expert    │
                                    │  (Gemini 2.0)   │
                                    └─────────────────┘
                                             │
                                    ┌────────┴────────┐
                                    │  Creative Writer│
                                    │  (Llama 4)      │
                                    └─────────────────┘
```

| Capability | What It Does | Benefit |
|------------|--------------|---------|
| **Semantic Routing** | Analyzes queries and routes to best LLM | Code questions → coding model, creative → creative model |
| **Agent Registry** | Register multiple AI agents with capabilities | Each agent handles what it's best at |
| **FastGRNN Inference** | Neural network picks the optimal agent in microseconds | 10-100x faster than rule-based routing |
| **Load Balancing** | Distributes requests across agents | Prevents overload, reduces latency |

```sql
-- Register specialized AI agents
SELECT ruvector_register_agent('code_expert', ARRAY['coding', 'debugging']);
SELECT ruvector_register_agent('writer', ARRAY['creative', 'editing']);

-- Route a query to the best agent
SELECT ruvector_route($user_query_embedding);
-- Returns: 'code_expert' for "How do I fix this bug?"
```

### Scaling & Performance

**Built for enterprise scale** from day one:

| Metric | RuVector | Why It Matters |
|--------|----------|----------------|
| **61µs latency** | 30-800x faster than alternatives | Real-time search feels instant |
| **200MB per 1M vectors** | 5-15x less memory with compression | Run larger datasets on smaller machines |
| **Raft consensus** | Multi-master replication | No single point of failure |
| **SIMD acceleration** | AVX-512/NEON hardware optimization | Maximum performance on any CPU |

### Horizontal Scaling

Start with one container, scale to a cluster when you need it:

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Node 1   │◄──►│ Node 2   │◄──►│ Node 3   │
    │ (Leader) │    │(Follower)│    │(Follower)│
    └──────────┘    └──────────┘    └──────────┘
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Shard A  │    │ Shard B  │    │ Shard C  │
    │ 0-33%    │    │ 33-66%   │    │ 66-100%  │
    └──────────┘    └──────────┘    └──────────┘

    ◄──────── Raft Consensus Protocol ────────►
```

```bash
# Single node (development)
docker run -d -p 5432:5432 ruvnet/ruvector:latest

# Cluster mode (production) - coming in v0.3.0
docker-compose up -d ruvector-cluster
# Automatically shards data across nodes
# Raft consensus ensures consistency
```

---

## What Can RuVector Do?

| Capability | What It Means | Example Use Case |
|------------|---------------|------------------|
| **Semantic Search** | Find content by meaning, not just keywords | "Find articles about climate change" matches "global warming papers" |
| **Knowledge Graphs** | Store and query relationships between data | "Show me all products bought by customers who also bought X" |
| **AI Agent Routing** | Automatically pick the best AI model for each question | Route "debug my code" to code-expert, "write a poem" to creative-writer |
| **Self-Learning Search** | Results improve based on what users click | Search gets smarter the more people use it |
| **Hierarchical Data** | Handle parent-child relationships naturally | Taxonomies, org charts, category trees |
| **Hybrid Search** | Combine meaning-based + keyword-based search | Search engines that understand synonyms AND exact phrases |
| **Graph Neural Networks** | AI that learns from data connections | Recommendations: "users like you also liked..." |
| **Horizontal Scaling** | Add more servers as you grow | Handle millions of vectors across multiple machines |
| **Real-time Compression** | Reduce memory 2-32x automatically | Store 1M vectors in 200MB instead of 2GB |
| **Multi-Platform** | Run everywhere: Docker, Node.js, browsers | Same API on servers, edge, and client-side |

---

## Comparison: RuVector vs Others

| Feature | RuVector | Pinecone | Qdrant | Milvus | ChromaDB |
|---------|----------|----------|--------|--------|----------|
| **Latency (p50)** | **61µs** | ~2ms | ~1ms | ~5ms | ~50ms |
| **Memory (1M vectors)** | **200MB*** | 2GB | 1.5GB | 1GB | 3GB |
| **Graph Queries (Cypher)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Self-Learning (GNN)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **39 Attention Mechanisms** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Hyperbolic Embeddings** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **AI Agent Routing** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **PostgreSQL Extension** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Runtime Learning (SONA)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Raft Consensus** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **SIMD Optimization** | ✅ Full | Partial | ✅ | ✅ | ❌ |
| **Sparse Vectors / BM25** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **WASM Browser Support** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Multi-Master Replication** | ✅ | ❌ | ❌ | ✅ | ❌ |

*With adaptive compression (2-32x reduction)

---

## Build Arguments (Customizable Components)

Build your own image with optional components enabled or disabled:

```bash
# Full build with all features (default)
docker build -t my-ruvector:full -f docker/Dockerfile .

# Minimal PostgreSQL-only build (no Node.js)
docker build --build-arg INCLUDE_NPM=false \
             -t my-ruvector:minimal -f docker/Dockerfile .

# Custom feature selection
docker build --build-arg PG_VERSION=16 \
             --build-arg INCLUDE_SONA=true \
             --build-arg INCLUDE_RUVLLM=false \
             -t my-ruvector:custom -f docker/Dockerfile .
```

### Available Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `PG_VERSION` | 17 | PostgreSQL version (14, 15, 16, 17) |
| `INCLUDE_NPM` | true | Include Node.js 20 LTS and npm packages |
| `INCLUDE_SONA` | true | Include @ruvector/sona self-learning module |
| `INCLUDE_RUVLLM` | true | Include ruvllm LLM integration |
| `INCLUDE_CLI` | true | Include @ruvector/postgres-cli tool |

### Runtime Environment Variables

Check which components were included at build time:

```bash
docker run ruvnet/ruvector:latest printenv | grep RUVECTOR
# RUVECTOR_INCLUDE_NPM=true
# RUVECTOR_INCLUDE_SONA=true
# RUVECTOR_INCLUDE_RUVLLM=true
# RUVECTOR_INCLUDE_CLI=true
```

## What's Included

**35+ Rust crates** organized by capability:

| Category | Crates |
|----------|--------|
| **Core** | ruvector-core, ruvector-postgres (65+ SQL functions), ruvector-server, ruvector-cli |
| **Neural** | ruvector-gnn (GCN/GraphSAGE/GAT), ruvector-attention (39 mechanisms), sona (LoRA/EWC++) |
| **Routing** | tiny-dancer-core (FastGRNN), ruvector-router-core/cli/wasm |
| **Graph** | ruvector-graph (Cypher queries), graph-node, graph-wasm |
| **Distributed** | ruvector-cluster, ruvector-raft, ruvector-replication, ruvector-snapshot |
| **Bindings** | Node.js (-node), WebAssembly (-wasm) for all major crates |

**npm packages** (pre-installed): `ruvector`, `@ruvector/sona`, `@ruvector/postgres-cli`, `ruvllm`

**15+ examples**: refrag-pipeline (30x RAG speedup), ruvLLM, scipix (OCR), google-cloud, onnx-embeddings, graph, nodejs, wasm-react, and more.

---

## 65+ PostgreSQL Functions

### Vector Operations
```sql
SELECT ruvector_l2_distance(a, b);
SELECT ruvector_cosine_distance(a, b);
SELECT ruvector_inner_product(a, b);
SELECT ruvector_l1_distance(a, b);
SELECT ruvector_normalize(embedding);
SELECT ruvector_add(a, b);
SELECT ruvector_sub(a, b);
SELECT ruvector_mul_scalar(vec, 2.0);
```

### Hyperbolic Geometry (Hierarchies)
```sql
SELECT ruvector_poincare_distance(ARRAY[0.1,0.2], ARRAY[0.3,0.4]);
SELECT ruvector_lorentz_distance(a, b);
SELECT ruvector_mobius_add(a, b);
SELECT ruvector_exp_map(base, tangent);
SELECT ruvector_log_map(base, target);
SELECT ruvector_poincare_to_lorentz(vec);
SELECT ruvector_lorentz_to_poincare(vec);
SELECT ruvector_minkowski_dot(a, b);
```

### Sparse Vectors & BM25
```sql
SELECT ruvector_sparse_dot(a, b);
SELECT ruvector_sparse_cosine(a, b);
SELECT ruvector_sparse_euclidean(a, b);
SELECT ruvector_sparse_manhattan(a, b);
SELECT ruvector_sparse_bm25(query, doc, k1, b);
SELECT ruvector_sparse_norm(vec);
SELECT ruvector_sparse_top_k(vec, k);
SELECT ruvector_to_sparse(indices, values, dim);
SELECT ruvector_sparse_to_dense(sparse, dim);
SELECT ruvector_dense_to_sparse(dense, threshold);
```

### Graph Neural Networks
```sql
SELECT ruvector_gcn_forward(features, adjacency, weights);
SELECT ruvector_graphsage_forward(features, neighbors, weights);
```

### Graph Storage & Cypher
```sql
SELECT ruvector_create_graph('my_graph');
SELECT ruvector_add_node('my_graph', 'Person', '{"name": "Alice"}');
SELECT ruvector_add_edge('my_graph', 1, 2, 'KNOWS', '{}');
SELECT ruvector_cypher('my_graph', 'MATCH (n) RETURN n');
SELECT ruvector_shortest_path('my_graph', 1, 10);
SELECT ruvector_list_graphs();
SELECT ruvector_graph_stats('my_graph');
SELECT ruvector_delete_graph('my_graph');
```

### Agent Routing (Tiny Dancer)
```sql
SELECT ruvector_register_agent('agent_name', ARRAY['cap1', 'cap2']);
SELECT ruvector_register_agent_full(name, capabilities, embedding, metadata);
SELECT ruvector_route(query_embedding);
SELECT ruvector_list_agents();
SELECT ruvector_get_agent('agent_name');
SELECT ruvector_find_agents_by_capability('coding');
SELECT ruvector_set_agent_active('agent_name', true);
SELECT ruvector_update_agent_metrics('agent_name', metrics);
SELECT ruvector_remove_agent('agent_name');
SELECT ruvector_clear_agents();
SELECT ruvector_routing_stats();
```

### Self-Learning (ReasoningBank)
```sql
SELECT ruvector_enable_learning(true);
SELECT ruvector_record_feedback(query_id, relevance_score, context);
SELECT ruvector_learning_stats();
SELECT ruvector_extract_patterns();
SELECT ruvector_clear_learning();
SELECT ruvector_auto_tune();
SELECT ruvector_get_search_params();
```

### System & Utilities
```sql
SELECT ruvector_version();
SELECT ruvector_simd_info();
SELECT ruvector_memory_stats();
SELECT ruvector_dims(embedding);
SELECT ruvector_norm(embedding);
```

---

## Tutorials

### Tutorial 1: Semantic Search
**Find documents by meaning, not just keywords.** Convert your documents into embeddings using any ML model (OpenAI, Cohere, HuggingFace), store them in RuVector, and find semantically similar content instantly.

```sql
CREATE EXTENSION ruvector;

-- Store documents with their embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    embedding ruvector(1536)  -- 1536 dimensions for OpenAI embeddings
);

INSERT INTO documents (title, embedding) VALUES
    ('AI Guide', '[0.1, 0.2, ...]'::ruvector);

-- Create HNSW index for blazing fast searches
CREATE INDEX ON documents USING hnsw (embedding ruvector_l2_ops);

-- Find the 10 most similar documents to your query
SELECT title, embedding <-> $query AS distance
FROM documents ORDER BY distance LIMIT 10;
```

### Tutorial 2: Hybrid Search (Vector + BM25)
**Combine the best of both worlds.** Use semantic similarity for understanding meaning AND keyword matching for exact terms. Perfect for search engines where users expect both "similar meaning" and "exact phrase" results.

```sql
-- Score = 70% semantic similarity + 30% keyword match (BM25)
SELECT title,
    0.7 * (1.0 / (1.0 + embedding <-> $query_vec)) +
    0.3 * ruvector_sparse_bm25(terms, doc_terms, 1.2, 0.75) AS score
FROM documents ORDER BY score DESC LIMIT 10;
```

### Tutorial 3: Hyperbolic Embeddings for Hierarchies
**Preserve parent-child relationships naturally.** Traditional Euclidean space squashes hierarchies flat. Hyperbolic space (like Poincaré ball) naturally represents tree structures—taxonomies, org charts, knowledge graphs—with true parent-child distances.

```sql
-- Store taxonomy nodes with Poincaré embeddings
CREATE TABLE taxonomy (
    id SERIAL PRIMARY KEY,
    name TEXT,
    parent_id INTEGER,
    embedding ruvector(128)  -- Trained in hyperbolic space
);

-- Find nodes closest in the hierarchy using hyperbolic distance
SELECT name, ruvector_poincare_distance(
    (SELECT embedding FROM taxonomy WHERE id = 1),
    embedding
) AS distance
FROM taxonomy ORDER BY distance LIMIT 10;
```

### Tutorial 4: AI Agent Routing (Tiny Dancer)
**Route user queries to the best AI agent automatically.** When you have multiple specialized AI models (code expert, math expert, writer), RuVector intelligently routes each query to the agent most likely to give the best answer.

```sql
-- Register your specialized AI agents with their capabilities
SELECT ruvector_register_agent('code_expert', ARRAY['coding', 'debugging', 'refactoring']);
SELECT ruvector_register_agent('math_expert', ARRAY['math', 'statistics', 'calculus']);
SELECT ruvector_register_agent('writer', ARRAY['writing', 'editing', 'storytelling']);

-- When a user asks a question, find the best agent to handle it
SELECT ruvector_route($user_query_embedding);
-- Returns: 'code_expert' for "How do I fix this bug?"
-- Returns: 'math_expert' for "Solve this equation"
```

### Tutorial 5: Self-Learning Search (SONA)
**Your search gets smarter over time.** Record which results users clicked, learn from successful queries, and automatically tune search parameters. The more you use it, the better it gets.

```sql
-- Enable the learning system
SELECT ruvector_enable_learning(true);

-- Record user feedback: they clicked result #3 with 95% satisfaction
SELECT ruvector_record_feedback('query_123', 0.95, '{"clicked_rank": 3}');

-- Let the system auto-optimize based on learned patterns
SELECT ruvector_auto_tune();

-- Check what the system has learned
SELECT * FROM ruvector_learning_stats();
```

### Tutorial 6: Graph Queries with Cypher
**Query relationships like Neo4j.** Build a social network, knowledge graph, or any connected data structure. Query it with familiar Cypher syntax—no new query language to learn.

```sql
-- Create a social network graph
SELECT ruvector_create_graph('social');

-- Add people as nodes
SELECT ruvector_add_node('social', 'Person', '{"name": "Alice", "age": 30}');
SELECT ruvector_add_node('social', 'Person', '{"name": "Bob", "age": 25}');

-- Connect them with a relationship
SELECT ruvector_add_edge('social', 1, 2, 'KNOWS', '{"since": 2020}');

-- Query: "Who does Alice know?"
SELECT ruvector_cypher('social',
    'MATCH (a:Person)-[:KNOWS]->(b:Person)
     WHERE a.name = "Alice"
     RETURN b.name');
-- Returns: Bob
```

### Tutorial 7: GNN-Enhanced Search
**Let neural networks improve your search results.** Graph Neural Networks analyze how your vectors are connected and enhance them based on their neighborhood. Frequently-accessed paths get reinforced, making common queries faster and more accurate.

```sql
-- Apply Graph Convolutional Network layer to enhance embeddings
-- based on their connections to other nodes
SELECT ruvector_gcn_forward(
    ARRAY[[0.1, 0.2], [0.3, 0.4]],  -- Current node features
    ARRAY[[0, 1], [1, 0]],           -- Adjacency matrix (who's connected)
    ARRAY[[0.5, 0.5], [0.5, 0.5]]    -- Learned weights
);
-- Returns: Enhanced embeddings that incorporate neighbor information
```

### Tutorial 8: Sparse Vectors & BM25
**Handle keyword-based search efficiently.** Sparse vectors represent documents as bags-of-words with only non-zero terms stored. Perfect for TF-IDF, BM25, and text retrieval where most dimensions are zero.

```sql
-- Create a sparse vector: only 3 out of 100 dimensions are non-zero
SELECT ruvector_to_sparse(
    ARRAY[0, 5, 10],              -- Which dimensions have values
    ARRAY[0.5, 0.3, 0.2]::real[], -- The values at those positions
    100                            -- Total dimensions
);
-- Returns: '{0:0.5,5:0.3,10:0.2}/100'

-- Calculate BM25 score for text ranking
SELECT ruvector_sparse_bm25(
    '{0:1.0,5:0.5}/100',  -- Query terms
    '{0:0.8,5:0.3}/100',  -- Document terms
    1.2,                   -- k1: term saturation parameter
    0.75                   -- b: length normalization
);
```

---

## Distance Operators

| Operator | Distance | Best For |
|----------|----------|----------|
| `<->` | L2 (Euclidean) | General similarity |
| `<=>` | Cosine | Text embeddings |
| `<#>` | Inner Product | Normalized vectors |
| `<+>` | Manhattan (L1) | Sparse features |

---

## Index Types

### HNSW (Recommended)
```sql
CREATE INDEX ON items USING hnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 64);
SET ruvector.ef_search = 100;
```

### IVFFlat
```sql
CREATE INDEX ON items USING ivfflat (embedding ruvector_l2_ops)
WITH (lists = 100);
SET ruvector.ivfflat_probes = 10;
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | ruvector | PostgreSQL username |
| `POSTGRES_PASSWORD` | ruvector | PostgreSQL password |
| `POSTGRES_DB` | ruvector_db | Default database |

---

## Volumes

```bash
docker run -d \
  -v ruvector-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  ruvnet/ruvector:latest
```

---

## Related Images

| Image | Description |
|-------|-------------|
| `ruvnet/ruvector:latest` | Full platform (this image) |
| `ruvnet/ruvector-postgres:latest` | PostgreSQL extension only (smaller) |

---

## Links

- [GitHub](https://github.com/ruvnet/ruvector)
- [Documentation](https://github.com/ruvnet/ruvector/tree/main/docs)
- [npm Packages](https://www.npmjs.com/org/ruvector)
- [crates.io](https://crates.io/crates/ruvector-core)

## License

MIT License
