# Self-Learning Postgres DB - Vector Database for AI Agents

A distributed vector database that **truly learns**. Store embeddings, query with semantic search, and let the index improve itself through TRM (Tiny Recursive Models), SONA (Self-Optimizing Neural Architecture), and Graph Neural Networks.

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-blue)](https://apify.com/ruv/self-learning-postgres-db)
[![PostgreSQL 17](https://img.shields.io/badge/PostgreSQL-17.7-blue)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.1-green)](https://github.com/ruvnet/ruvector)

## Key AI Features

| Feature | Description |
|---------|-------------|
| **TRM** | 7M parameter recursive reasoning (83% on GSM8K) |
| **SONA** | 3-tier learning (Instant/Background/Deep) |
| **EWC++** | Anti-forgetting protection (λ=2000) |
| **GNN** | Graph Neural Network index optimization |
| **Trajectory Tracking** | Learn from query patterns |

---

## Features

**30+ Operations** for complete vector database management:

- **Semantic Search** - Find documents by meaning, not just keywords
- **Batch Operations** - Insert and search thousands of documents efficiently
- **Hybrid Search** - Combine vector similarity with keyword matching
- **RAG Support** - Built-in Retrieval-Augmented Generation queries
- **Self-Learning** - GNN training for index optimization
- **Clustering** - K-means document clustering
- **Deduplication** - Find and remove duplicate content
- **Export/Import** - JSON and CSV data migration

**Zero Setup Required:**
- Embedded PostgreSQL with ruvector extension
- Local AI embeddings (no OpenAI API key needed)
- Automatic table and index creation

---

## Quick Start (30 Seconds)

### Full Demo

```json
{
  "action": "full_workflow",
  "query": "How does machine learning work?",
  "documents": [
    {"content": "Machine learning is AI that learns patterns from data.", "metadata": {"category": "AI"}},
    {"content": "PostgreSQL is a powerful relational database.", "metadata": {"category": "Database"}},
    {"content": "Neural networks consist of layers of nodes.", "metadata": {"category": "AI"}},
    {"content": "Vector databases store embeddings for similarity search.", "metadata": {"category": "Database"}}
  ]
}
```

**Result:** Documents ranked by semantic relevance to your query.

---

## All 38 Actions

### Document Operations
| Action | Description |
|--------|-------------|
| `insert` | Add documents with auto-generated embeddings |
| `batch_insert` | Efficiently insert large document sets |
| `get` | Retrieve single document by ID |
| `list` | List documents with filtering |
| `update` | Modify existing document content/metadata |
| `delete` | Remove documents by ID, IDs, or filter |
| `upsert` | Insert or update (smart merge) |

### Search Operations
| Action | Description |
|--------|-------------|
| `search` | Semantic similarity search |
| `batch_search` | Multiple queries in one call |
| `hybrid_search` | Vector + BM25 keyword combined |
| `multi_query_search` | Aggregate results from multiple queries |
| `mmr_search` | Maximal Marginal Relevance (diverse results) |
| `graph_search` | Graph-based similarity traversal |
| `range_search` | All results within distance threshold |

### Table Operations
| Action | Description |
|--------|-------------|
| `create_table` | Create new vector collection |
| `drop_table` | Delete collection |
| `list_tables` | Show all vector tables |
| `table_stats` | Collection statistics and metrics |
| `create_index` | Add HNSW or IVFFlat index |
| `reindex` | Rebuild indexes |

### Self-Learning / GNN / SONA
| Action | Description |
|--------|-------------|
| `train_gnn` | Train Graph Neural Network on data |
| `optimize_index` | Auto-tune HNSW parameters |
| `analyze_patterns` | Analyze data distribution |
| `sona_learn` | Trigger TRM/SONA background learning cycle |
| `sona_status` | Check SONA learning status and capabilities |

### Clustering & Deduplication
| Action | Description |
|--------|-------------|
| `cluster` | K-means document clustering |
| `find_duplicates` | Detect similar document pairs |
| `deduplicate` | Remove duplicate documents |

### Data Operations
| Action | Description |
|--------|-------------|
| `export` | Export to JSON or CSV |
| `import` | Import from JSON data |

### AI / RAG
| Action | Description |
|--------|-------------|
| `rag_query` | Build RAG context from search results |
| `summarize` | Document statistics and previews |

### Utility
| Action | Description |
|--------|-------------|
| `ping` | Test database connection |
| `version` | Get version and feature info |
| `embedding_models` | List available models |
| `generate_embedding` | Create embeddings without storing |
| `similarity` | Compare similarity of two texts |

---

## Use Cases

### 1. AI Agent Memory

```json
{
  "action": "insert",
  "tableName": "agent_memory",
  "documents": [
    {"content": "User prefers dark mode", "metadata": {"user_id": "123", "type": "preference"}},
    {"content": "User asked about Python tutorials", "metadata": {"user_id": "123", "type": "history"}}
  ]
}
```

Retrieve memories:
```json
{
  "action": "search",
  "tableName": "agent_memory",
  "query": "What does this user like?",
  "filter": "metadata->>'user_id' = '123'"
}
```

### 2. RAG Pipeline

```json
{
  "action": "rag_query",
  "query": "How do I return a product?",
  "topK": 5,
  "ragMaxTokens": 2000
}
```

Returns context ready to feed to your LLM.

### 3. Batch Document Processing

```json
{
  "action": "batch_insert",
  "batchSize": 100,
  "documents": [
    // ... thousands of documents
  ]
}
```

### 4. Find & Remove Duplicates

```json
{
  "action": "find_duplicates",
  "similarityThreshold": 0.95
}
```

Then:
```json
{
  "action": "deduplicate",
  "similarityThreshold": 0.95
}
```

### 5. Document Clustering

```json
{
  "action": "cluster",
  "numClusters": 10,
  "clusteringAlgorithm": "kmeans"
}
```

### 6. Index Optimization

```json
{
  "action": "optimize_index",
  "enableLearning": true
}
```

### 7. SONA Self-Learning

Check learning status:
```json
{
  "action": "sona_status"
}
```

Trigger learning cycle:
```json
{
  "action": "sona_learn",
  "ewcLambda": 2000,
  "patternThreshold": 0.7
}
```

---

## Parameters Reference

### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | `search` | Operation to perform |
| `connectionString` | string | embedded | PostgreSQL URL for persistence |
| `tableName` | string | `documents` | Table/collection name |

### Search Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | - | Natural language search query |
| `queryVector` | array | - | Pre-computed embedding vector |
| `topK` | integer | 10 | Number of results |
| `distanceMetric` | string | `cosine` | cosine, l2, inner_product, manhattan |
| `filter` | string | - | SQL WHERE clause |
| `minScore` | number | 0 | Minimum similarity score (0-1) |
| `maxDistance` | number | - | Maximum distance threshold |

### Embedding Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embeddingModel` | string | `all-MiniLM-L6-v2` | AI embedding model |
| `generateEmbeddings` | boolean | true | Auto-generate embeddings |
| `dimensions` | integer | 384 | Vector dimensions |

### Index Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indexType` | string | `hnsw` | hnsw, ivfflat, none |
| `hnswM` | integer | 16 | HNSW max connections |
| `hnswEfConstruction` | integer | 64 | HNSW build quality |
| `hnswEfSearch` | integer | 100 | HNSW search quality |
| `ivfLists` | integer | 100 | IVFFlat partitions |

### GNN Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enableLearning` | boolean | false | Enable self-learning |
| `learningRate` | number | 0.01 | GNN learning rate |
| `gnnLayers` | integer | 2 | GNN layer count |
| `trainEpochs` | integer | 10 | Training epochs |

### SONA / TRM Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sonaEnabled` | boolean | true | Enable TRM/SONA self-learning |
| `ewcLambda` | number | 2000 | EWC++ anti-forgetting strength |
| `patternThreshold` | number | 0.7 | Pattern recognition confidence |
| `maxTrajectories` | integer | 100 | Max trajectory steps to track |
| `sonaLearningTiers` | array | ["instant", "background"] | Learning tiers to enable |

### Clustering Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numClusters` | integer | 10 | K-means clusters |
| `similarityThreshold` | number | 0.95 | Duplicate detection threshold |

---

## Embedding Models

| Model | Dimensions | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Prototyping |
| `bge-small-en-v1.5` | 384 | Fast | Excellent | Production |
| `bge-base-en-v1.5` | 768 | Medium | Better | High accuracy |
| `nomic-embed-text-v1` | 768 | Medium | Good | Long documents (8K) |
| `gte-small` | 384 | Fast | Good | General use |
| `e5-small-v2` | 384 | Fast | Good | Multilingual |

---

## Persistent Storage

### Hybrid Persistence Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Actor Run                            │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Key-Value    │───▶│ Embedded     │───▶│ Key-Value │ │
│  │ Store (load) │    │ PostgreSQL   │    │ (save)    │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│       START              WORK               END         │
└─────────────────────────────────────────────────────────┘
```

**Flow:**
1. **On Start** → Load documents from Key-Value Store into embedded PostgreSQL
2. **During Run** → Full vector search capabilities (HNSW, cosine, etc.)
3. **On End** → Export documents back to Key-Value Store

### Storage Options Comparison

| Feature | External PostgreSQL | Apify Key-Value Store |
|---------|---------------------|----------------------|
| Setup required | Yes | No |
| Cost | Separate billing | Included in Apify |
| Max size | Unlimited | ~9GB per store |
| Cold start | Fast | Slower (load data) |
| Best for | Large/production | Small-medium datasets |

### External PostgreSQL

For persistent storage with external database:

```json
{
  "connectionString": "postgresql://user:password@host:5432/database",
  "action": "search",
  "query": "Your query"
}
```

**Supported:**
- PostgreSQL 14+ with ruvector extension
- PostgreSQL with pgvector (compatibility mode)
- Supabase, Neon, AWS RDS, etc.

---

## API Integration

### Python
```python
from apify_client import ApifyClient

client = ApifyClient("your-api-token")
run = client.actor("ruv/self-learning-postgres-db").call(run_input={
    "action": "search",
    "query": "machine learning basics",
    "topK": 5
})
results = client.dataset(run["defaultDatasetId"]).list_items().items
```

### JavaScript
```javascript
import { ApifyClient } from 'apify-client';

const client = new ApifyClient({ token: 'your-api-token' });
const run = await client.actor('ruv/self-learning-postgres-db').call({
    action: 'search',
    query: 'machine learning basics',
    topK: 5
});
const results = await client.dataset(run.defaultDatasetId).listItems();
```

### cURL
```bash
curl -X POST "https://api.apify.com/v2/acts/ruv~self-learning-postgres-db/runs" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "search",
    "query": "machine learning",
    "topK": 10
  }'
```

---

## Performance

Built on PostgreSQL 17.7 with AVX-512 SIMD acceleration:

| Dataset Size | Search Time | Accuracy |
|--------------|-------------|----------|
| 10,000 docs | ~0.3ms | 99%+ |
| 100,000 docs | ~0.5ms | 99%+ |
| 1,000,000 docs | ~1.2ms | 98%+ |

---

## Pricing (Apify Pay-per-event)

### Core Events
| Event | Price | Description |
|-------|-------|-------------|
| Actor Start | $0.001 | Per GB memory used |
| Document Insert | $0.001 | Per document stored |
| Vector Search | $0.001 | Per search query |
| Result | $0.0005 | Per result returned |

### Advanced Operations
| Event | Price | Description |
|-------|-------|-------------|
| Batch Operation | $0.002 | Per batch insert/search |
| RAG Query | $0.002 | Per RAG context build |
| GNN Training | $0.01 | Per training session |
| Clustering | $0.005 | Per cluster operation |
| Deduplication | $0.003 | Per dedupe run |
| Data Export | $0.002 | Per export |
| Data Import | $0.002 | Per import |
| Table Operation | $0.001 | Create/drop table |
| Index Operation | $0.002 | Create/optimize index |
| Similarity Check | $0.001 | Per comparison |
| Embedding Generation | $0.001 | Per embedding |

**Volume Discounts:**
- Bronze: -14% off results
- Silver: -26% off results
- Gold: -40% off results

---

## Development

### Local Testing

```bash
# Start ruvector-postgres
docker run -d --name ruvector-pg -e POSTGRES_PASSWORD=secret -p 5432:5432 ruvnet/ruvector-postgres:latest

# Run tests
DATABASE_URL="postgresql://postgres:secret@localhost:5432/postgres" npm test
```

### Deployment

```bash
# Set your API token in root .env
echo "APIFY_API_TOKEN=your_token" >> ../../../.env

# Deploy
npm run deploy
```

---

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Apify Store](https://apify.com/ruv/self-learning-postgres-db)
- [Docker Image](https://hub.docker.com/r/ruvnet/ruvector-postgres)
- [RuVector Documentation](https://github.com/ruvnet/ruvector/tree/main/crates/ruvector-postgres)

---

## Support

- [Open an Issue](https://github.com/ruvnet/ruvector/issues)
- [Apify Community](https://discord.gg/apify)

---

**Built with RuVector** - High-performance vector search with TRM/SONA self-learning for the AI era.
