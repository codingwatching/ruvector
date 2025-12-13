# @ruvector/rvlite

[![npm version](https://img.shields.io/npm/v/@ruvector/rvlite.svg)](https://www.npmjs.com/package/@ruvector/rvlite)
[![npm downloads](https://img.shields.io/npm/dm/@ruvector/rvlite.svg)](https://www.npmjs.com/package/@ruvector/rvlite)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/ruvnet/ruvector)
[![WASM](https://img.shields.io/badge/WebAssembly-Powered-654FF0.svg)](https://webassembly.org/)

**A standalone vector database with SQL, SPARQL, Cypher, and GNN support - powered by RuVector WASM.**

RvLite runs entirely in your browser with no server required. It combines vector similarity search, graph neural networks, and three powerful query languages in a single ~850KB WASM bundle. Includes an interactive dashboard with supply chain simulation demos.

## Highlights

- **100% Client-Side** - No server, no backend, runs entirely in browser via WebAssembly
- **Three Query Languages** - SQL, SPARQL, and Cypher in one database
- **Graph Neural Networks** - Train GNN models and generate embeddings in-browser
- **Vector Search** - Fast similarity search with cosine, euclidean, dot product metrics
- **Interactive Demos** - Supply chain simulation, weather disruption modeling, semantic search
- **Built-in Dashboard** - Full-featured web UI with live query execution
- **Browser Persistence** - IndexedDB storage survives page refreshes
- **Tiny Footprint** - ~850KB WASM bundle (gzipped ~300KB)

## Installation

```bash
npm install @ruvector/rvlite
```

## Quick Start

```bash
# Start the dashboard (no install required)
npx @ruvector/rvlite@latest serve

# Custom port
npx @ruvector/rvlite@latest serve --port 8080

# Interactive REPL
npx @ruvector/rvlite@latest repl
```

Then open **http://localhost:3000** in your browser.

---

## Dashboard

RvLite includes a full-featured web dashboard for interactive database exploration, query execution, and AI-powered simulations.

### Dashboard Overview

The dashboard is a React-based single-page application that provides a complete interface for working with RvLite's vector database, RDF triple store, property graph, and neural network capabilities.

```
┌─────────────────────────────────────────────────────────────────┐
│  RvLite Dashboard                                    [Save][Load]│
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │    SQL      │ │   SPARQL    │ │   Cypher    │ │    GNN    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Query Editor                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ SELECT id, vector <-> '[0.1, 0.2, ...]' AS distance        ││
│  │ FROM vectors ORDER BY distance LIMIT 10                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                              [▶ Execute Query]  │
├─────────────────────────────────────────────────────────────────┤
│  Results                              │  Simulation Controls    │
│  ┌────────────────────────────────────┼─────────────────────────┤
│  │ [                                  │  ▶ Start Simulation     │
│  │   { "id": "vec_001", ... },        │  ⏸ Pause                │
│  │   { "id": "vec_002", ... }         │  🌧️ Trigger Weather     │
│  │ ]                                  │  Speed: [====----]      │
│  └────────────────────────────────────┴─────────────────────────┤
│  Vectors: 1,234 │ Triples: 567 │ Nodes: 89 │ GNN: Trained ✓    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Interactive Demos & Simulations

### 🏭 Supply Chain Simulation

A complete supply chain network simulation demonstrating all RvLite capabilities working together:

- **Graph-based Supply Network** - Models suppliers, warehouses, retailers as Cypher nodes
- **Route Optimization** - Finds alternate delivery routes when paths are blocked
- **Inventory Management** - Tracks products with perishability, shelf life, demand volatility
- **Real-time Visualization** - Watch the supply chain respond to changes

```cypher
-- Supply chain graph structure
CREATE (supplier:Supplier {name: 'Farm A', location: 'California'})
CREATE (warehouse:Warehouse {name: 'Distribution Center', capacity: 10000})
CREATE (store:Store {name: 'Grocery Store #1', region: 'Northeast'})
CREATE (supplier)-[:SUPPLIES {product: 'produce', leadTime: 2}]->(warehouse)
CREATE (warehouse)-[:DELIVERS {route: 'I-95', distance: 500}]->(store)
```

### 🌦️ Weather Disruption Demo

AI-powered weather event simulation showing how RvLite handles real-world disruptions:

- **Weather Monitoring** - Simulates weather conditions across delivery regions
- **Disruption Prediction** - Predicts potential supply chain impacts
- **AI Remediation** - Automatically finds alternate routes and pre-positions inventory
- **Multi-Feature Integration** - Combines SQL, SPARQL, Cypher, and GNN in one workflow

**How it works:**
1. Weather event triggers (winter storm, hurricane, flooding)
2. Affected routes identified via Cypher graph queries
3. Vector search finds similar historical disruptions
4. GNN predicts optimal remediation strategies
5. Supply chain automatically reroutes

### 🔍 Semantic Search Demo

Demonstrates vector similarity search with real-world examples:

- **Document Search** - Find semantically similar documents
- **Product Recommendations** - Similar items based on embeddings
- **Query Patterns** - Learn from successful search patterns

### 🧠 GNN Training Demo

Interactive graph neural network capabilities:

- **Pattern Learning** - Train GNN on graph structure
- **Embedding Generation** - Generate vector embeddings from graph nodes
- **Accuracy Metrics** - Real-time training progress and accuracy display

```javascript
// GNN workflow in dashboard
1. Execute queries → Build patterns
2. Train GNN on patterns
3. Get embeddings for any node
4. Use embeddings for similarity search
```

---

## Dashboard Features

### 🔍 Multi-Language Query Editor

Switch seamlessly between query languages:

| Language | Use Case | Example |
|----------|----------|---------|
| **SQL** | Vector similarity search, metadata filtering | `SELECT * FROM vectors WHERE category = 'tech'` |
| **SPARQL** | RDF triple queries, semantic relationships | `SELECT ?s ?p ?o WHERE { ?s ?p ?o }` |
| **Cypher** | Graph traversal, supply chain modeling | `MATCH (n)-[:SUPPLIES]->(m) RETURN n, m` |

### 📊 Vector Operations Panel

- **Insert Vectors** - Add embeddings with metadata via UI form
- **Batch Import** - Paste JSON arrays of vectors
- **Search Interface** - Enter query vectors and k value
- **Results Visualization** - See distances and metadata in formatted JSON

### 🔗 RDF Triple Store

- **Add Triples** - Subject, Predicate, Object input form
- **Triple Browser** - View all stored triples
- **SPARQL Console** - Execute SPARQL queries with syntax highlighting
- **Statistics** - Live triple count

### 🕸️ Property Graph (Cypher)

- **Node Creator** - Create labeled nodes with properties
- **Relationship Builder** - Connect nodes with typed edges
- **Supply Chain Templates** - Pre-built supply network structures
- **Graph Stats** - Node and relationship counts

### 🧠 Graph Neural Network (GNN)

- **Train GNN** - Learn patterns from graph structure
- **Get Embeddings** - Generate vector representations of nodes
- **Pattern Recognition** - Identify similar structures
- **Accuracy Tracking** - Monitor training progress

### 💾 Persistence Controls

- **Save to IndexedDB** - One-click database persistence
- **Load from Storage** - Restore previous session
- **Clear Storage** - Reset persisted data
- **Export JSON** - Download database as JSON file
- **Import JSON** - Upload and restore from JSON

### 📈 Real-Time Statistics

The dashboard footer displays live statistics:

- Total vectors stored
- Number of RDF triples
- Graph node count
- Relationship count
- GNN training status
- Current distance metric

### 🎮 Simulation Controls

For interactive demos:

- **Start/Pause** - Control simulation playback
- **Speed Slider** - Adjust simulation speed
- **Trigger Events** - Manually trigger weather disruptions
- **Reset** - Return to initial state

---

## Pre-Built Datasets

The dashboard includes ready-to-use demo scenarios:

| Dataset | Description | Features Demonstrated |
|---------|-------------|----------------------|
| **Supply Chain** | Grocery supply network | Cypher, GNN, Route optimization |
| **Weather Response** | Disruption remediation | All features integrated |
| **Semantic Search** | Document similarity | SQL, Vector search |
| **Complete Demo** | All capabilities | Full feature showcase |

---

## JavaScript API

### Browser/JavaScript Usage

```javascript
import init, { RvLite, RvLiteConfig } from '@ruvector/rvlite';

// Initialize WASM
await init();

// Create database (384 dimensions for all-MiniLM-L6-v2 embeddings)
const config = new RvLiteConfig(384);
const db = new RvLite(config);

// Insert vectors with metadata
const embedding = new Float32Array(384).fill(0.1);
const id = db.insert(embedding, { text: "Hello world", category: "greeting" });

// Search similar vectors
const query = new Float32Array(384).fill(0.1);
const results = db.search(query, 5);
console.log(results);
// [{ id: "...", distance: 0.0, metadata: { text: "Hello world", ... } }]
```

---

## Query Languages

### SQL - Vector Search

```sql
-- Create table with vector column
CREATE TABLE documents (id TEXT PRIMARY KEY, vector VECTOR(384))

-- Insert with vector literal
INSERT INTO documents (id, vector) VALUES ('doc1', '[0.1, 0.2, ...]')

-- Vector similarity search (k-NN)
SELECT id, vector <-> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY distance
LIMIT 10

-- Filter by metadata
SELECT * FROM documents WHERE category = 'tech'
```

### SPARQL - RDF Triples

```javascript
// Add semantic triples
db.add_triple('<http://ex.org/doc1>', '<http://ex.org/hasTitle>', '"Hello World"');
db.add_triple('<http://ex.org/doc1>', '<http://ex.org/hasAuthor>', '<http://ex.org/alice>');
```

```sparql
-- Query all triples
SELECT ?s ?p ?o WHERE { ?s ?p ?o }

-- Filter by predicate
SELECT ?doc ?title WHERE {
  ?doc <http://ex.org/hasTitle> ?title
}

-- ASK queries
ASK { <http://ex.org/doc1> ?p ?o }
```

### Cypher - Property Graph

```cypher
-- Create supply chain nodes
CREATE (s:Supplier {name: 'Farm A', region: 'West'})
CREATE (w:Warehouse {name: 'DC-1', capacity: 5000})
CREATE (r:Retailer {name: 'Store #1', demand: 100})

-- Create relationships
CREATE (s)-[:SUPPLIES {product: 'produce', cost: 10}]->(w)
CREATE (w)-[:DELIVERS {route: 'Route-A', distance: 50}]->(r)

-- Find supply routes
MATCH (s:Supplier)-[:SUPPLIES]->(w)-[:DELIVERS]->(r)
RETURN s.name, w.name, r.name

-- Find alternate routes when disrupted
MATCH (s:Supplier)-[*1..3]->(r:Retailer)
WHERE NOT (s)-[:SUPPLIES {blocked: true}]->()
RETURN path
```

---

## Persistence

RvLite uses IndexedDB for browser-based persistence:

```javascript
// Initialize storage (required before save/load)
await db.init_storage();

// Save current state
await db.save();

// Load from IndexedDB
const loaded = await RvLite.load(config);

// Check if saved state exists
const hasSaved = await RvLite.has_saved_state();

// Clear saved state
await RvLite.clear_storage();

// Check storage availability
const available = RvLite.is_storage_available();
```

---

## CLI Commands

```bash
# Start dashboard server
npx @ruvector/rvlite serve [--port <port>]

# Interactive REPL (experimental, full features in browser)
npx @ruvector/rvlite repl

# Show version
npx @ruvector/rvlite --version

# Help
npx @ruvector/rvlite --help
```

---

## API Reference

### RvLiteConfig

```typescript
class RvLiteConfig {
  constructor(dimensions: number);
  get_dimensions(): number;
  get_distance_metric(): string;
  with_distance_metric(metric: string): RvLiteConfig;
}
```

**Distance Metrics**: `cosine` (default), `euclidean`, `dotproduct`, `manhattan`

### RvLite

```typescript
class RvLite {
  constructor(config: RvLiteConfig);
  static default(): RvLite;  // 384 dimensions, cosine

  // Vector Operations
  insert(vector: Float32Array, metadata?: object): string;
  insert_with_id(id: string, vector: Float32Array, metadata?: object): void;
  search(query: Float32Array, k: number): SearchResult[];
  search_with_filter(query: Float32Array, k: number, filter: object): SearchResult[];
  get(id: string): VectorEntry | null;
  delete(id: string): boolean;
  len(): number;
  is_empty(): boolean;

  // Query Languages
  sql(query: string): QueryResult;
  sparql(query: string): QueryResult;
  cypher(query: string): QueryResult;

  // RDF Triples
  add_triple(subject: string, predicate: string, object: string): void;
  triple_count(): number;
  clear_triples(): void;

  // Cypher Graph
  cypher_clear(): void;
  cypher_stats(): object;

  // Persistence (IndexedDB)
  init_storage(): Promise<void>;
  save(): Promise<void>;
  static load(config: RvLiteConfig): Promise<RvLite>;
  static has_saved_state(): Promise<boolean>;
  static clear_storage(): Promise<void>;
  static is_storage_available(): boolean;

  // Export/Import
  export_json(): object;
  import_json(json: object): void;
  get_config(): object;
  get_version(): string;
  get_features(): object;
  is_ready(): boolean;
}
```

---

## Performance

| Metric | Value |
|--------|-------|
| WASM Bundle | ~850KB |
| Gzipped | ~300KB |
| Search (10K vectors) | <1ms |
| Insert | <0.1ms |
| GNN Training (100 patterns) | <500ms |
| Memory (10K vectors, 384d) | ~15MB |

---

## Use Cases

- **Supply Chain Optimization** - Model and optimize delivery networks
- **Disruption Planning** - Simulate weather events and plan responses
- **Semantic Search** - Find similar documents, images, or products
- **RAG Applications** - Retrieval-augmented generation with local embeddings
- **Knowledge Graphs** - Store and query RDF triples with SPARQL
- **Recommendation Systems** - Content-based filtering with vector similarity
- **Offline-First Apps** - Full database in browser with IndexedDB persistence
- **Prototyping** - Quick vector search experiments without infrastructure
- **Education** - Learn SQL, SPARQL, Cypher, and GNN in an interactive environment

---

## Browser Compatibility

| Browser | Minimum Version |
|---------|-----------------|
| Chrome | 89+ |
| Firefox | 89+ |
| Safari | 15+ |
| Edge | 89+ |

Requires WebAssembly and IndexedDB support.

---

## Related Packages

- [`@ruvector/core`](https://www.npmjs.com/package/@ruvector/core) - Native Node.js bindings
- [`ruvector`](https://www.npmjs.com/package/ruvector) - Full RuVector package

## License

MIT OR Apache-2.0

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Documentation](https://github.com/ruvnet/ruvector/tree/main/crates/rvlite)
- [Issues](https://github.com/ruvnet/ruvector/issues)
