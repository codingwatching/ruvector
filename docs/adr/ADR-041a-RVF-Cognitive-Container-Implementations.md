# ADR-041a: RVF Cognitive Container Implementations

**Status**: Proposed
**Date**: 2026-02-20
**Parent**: ADR-041 (RVF Cognitive Containers)

---

## Overview

Detailed implementation specifications for each of the 15 RVF cognitive containers identified in ADR-041. Each container defines its MCP tools, HTTP/WS API, port assignments, security tiers, and access control matrix.

---

## Security Tiers (All Containers)

Every RVF container supports 5 access levels, configurable via `SECURITY_SEG` metadata:

| Level | Name | Auth | Network | Witness | Use Case |
|-------|------|------|---------|---------|----------|
| **L0** | Public | None | Open | Read-only | Public demos, read-only exploration |
| **L1** | Token | API key header | CORS-restricted | Append-only | SaaS integrations, dev teams |
| **L2** | Signed | JWT RS256 + RBAC | mTLS optional | Signed (Ed25519) | Multi-tenant production |
| **L3** | Attested | JWT + TEE attestation | mTLS required | Signed + measurement | Regulated industries (HIPAA, SOC2) |
| **L4** | Sealed | ZK proofs + TEE | Air-gapped / VPN | Signed + ZK commitments | Military, national security |

### Auth Middleware Stack (per request)

```
Request → Rate Limiter → CORS → Auth (L0-L4) → AIDefence → Input Validation → Handler → Witness Log → Response
```

### Common Configuration

```json
{
  "security": {
    "level": "L2",
    "auth": {
      "type": "jwt",
      "issuer": "https://auth.ruvector.io",
      "audience": "rvf-container",
      "algorithms": ["RS256", "EdDSA"],
      "jwks_uri": "https://auth.ruvector.io/.well-known/jwks.json"
    },
    "rbac": {
      "roles": ["reader", "writer", "admin", "auditor"],
      "default_role": "reader"
    },
    "rate_limit": {
      "requests_per_minute": 600,
      "requests_per_hour": 10000,
      "burst": 50
    },
    "cors": {
      "origins": ["https://*.ruvector.io"],
      "methods": ["GET", "POST", "DELETE"],
      "max_age": 3600
    },
    "tls": {
      "min_version": "1.3",
      "mtls": false,
      "cert_path": "/etc/rvf/tls/cert.pem",
      "key_path": "/etc/rvf/tls/key.pem"
    },
    "aidefence": {
      "enabled": true,
      "detect_injection": true,
      "detect_pii": true,
      "block_threshold": "medium"
    },
    "witness": {
      "sign_all": true,
      "algorithm": "ed25519",
      "chain_verification": true
    }
  }
}
```

---

## Port Allocation Plan

| Container | HTTP API | MCP (STDIO) | MCP (SSE) | WebSocket | TCP Wire |
|-----------|----------|-------------|-----------|-----------|----------|
| ruvllm | 8100 | stdin | 8101 | 8102 | 8103 |
| sona | 8110 | stdin | 8111 | 8112 | — |
| graph-node | 8120 | stdin | 8121 | 8122 | 8123 |
| ruvbot | 8130 | stdin | 8131 | 8132 | — |
| rvlite | 8140 | stdin | 8141 | 8142 | 8143 |
| rvf-solver | 8150 | stdin | 8151 | 8152 | — |
| router | 8160 | stdin | 8161 | 8162 | — |
| rudag | 8170 | stdin | 8171 | 8172 | — |
| rvdna | 8180 | stdin | 8181 | 8182 | — |
| ruqu-wasm | 8190 | stdin | 8191 | 8192 | — |
| agentic-synth | 8200 | stdin | 8201 | 8202 | — |
| raft | 8210 | stdin | 8211 | 8212 | 8213 |
| cognitum-gate | 8220 | stdin | 8221 | — | — |
| rvf-mcp-server | 8230 | stdin | 8231 | — | — |
| ospipe | 8240 | stdin | 8241 | 8242 | — |

---

## Container 1: `ruvllm.rvf` — LLM Inference Container

### MCP Tools (14)

```yaml
tools:
  # --- Model Management ---
  - name: ruvllm_list_models
    description: List available LLM models with capabilities and pricing
    input: {}
    roles: [reader, writer, admin]

  - name: ruvllm_load_model
    description: Load a model into the inference engine with optional LoRA adapter
    input:
      model_id: { type: string, required: true }       # e.g. "google/gemini-2.5-pro"
      lora_adapter: { type: string }                    # Path to LoRA weights in DELTA_SEG
      quantization: { type: string, enum: [none, int8, int4] }
    roles: [admin]

  - name: ruvllm_unload_model
    description: Unload a model to free memory
    input:
      model_id: { type: string, required: true }
    roles: [admin]

  # --- Inference ---
  - name: ruvllm_complete
    description: Generate a completion with automatic model routing
    input:
      prompt: { type: string, required: true }
      model: { type: string }                           # Override auto-routing
      max_tokens: { type: integer, default: 1024 }
      temperature: { type: number, default: 0.7 }
      stream: { type: boolean, default: false }
      system: { type: string }                          # System prompt
    roles: [reader, writer, admin]

  - name: ruvllm_embed
    description: Generate embeddings for text using WASM SIMD engine
    input:
      text: { type: string, required: true }
      model: { type: string, default: "local-wasm" }
      dimensions: { type: integer, default: 384 }
    roles: [reader, writer, admin]

  - name: ruvllm_embed_batch
    description: Batch embedding generation (up to 100 texts)
    input:
      texts: { type: array, items: string, required: true, maxItems: 100 }
      model: { type: string, default: "local-wasm" }
    roles: [writer, admin]

  # --- Routing & Policy ---
  - name: ruvllm_route
    description: Get routing recommendation without executing (dry-run)
    input:
      prompt: { type: string, required: true }
      budget: { type: string, enum: [cheap, balanced, quality] }
    roles: [reader, writer, admin]

  - name: ruvllm_set_policy
    description: Update routing policy (tier thresholds, cost limits, model preferences)
    input:
      policy: { type: object, required: true }
    roles: [admin]

  - name: ruvllm_get_policy
    description: Get current routing policy configuration
    input: {}
    roles: [reader, writer, admin]

  # --- Learning ---
  - name: ruvllm_feedback
    description: Submit quality feedback for a completion (drives SONA learning)
    input:
      completion_id: { type: string, required: true }
      score: { type: number, min: 0, max: 1 }
      feedback: { type: string }
    roles: [writer, admin]

  - name: ruvllm_export_lora
    description: Export current LoRA adapter weights as DELTA_SEG
    input:
      output_path: { type: string }
    roles: [admin]

  # --- Monitoring ---
  - name: ruvllm_status
    description: Get inference engine status (loaded models, cache, latency stats)
    input: {}
    roles: [reader, writer, admin]

  - name: ruvllm_cost_report
    description: Get cost breakdown by model, tier, and time period
    input:
      period: { type: string, enum: [hour, day, week, month], default: day }
    roles: [admin, auditor]

  - name: ruvllm_witness_log
    description: Get cryptographically signed inference audit trail
    input:
      limit: { type: integer, default: 100 }
      since: { type: string, format: iso8601 }
    roles: [auditor, admin]
```

### HTTP API

```
POST   /v1/completions              # OpenAI-compatible completion
POST   /v1/embeddings               # OpenAI-compatible embedding
POST   /v1/chat/completions         # OpenAI-compatible chat
GET    /v1/models                   # List models
GET    /v1/status                   # Engine status
GET    /v1/health                   # Health check
POST   /v1/feedback                 # Quality feedback
GET    /v1/cost                     # Cost report
GET    /v1/witness                  # Audit log
WS     /v1/stream                   # Streaming completions
```

### Access Control Matrix

| Operation | L0 Public | L1 Token | L2 Signed | L3 Attested | L4 Sealed |
|-----------|-----------|----------|-----------|-------------|-----------|
| List models | R | R | R | R | R |
| Complete | — | R (rate-limited) | RW | RW | RW |
| Embed | R (10/min) | R | RW | RW | RW |
| Set policy | — | — | Admin | Admin | Admin |
| Export LoRA | — | — | — | Admin | Admin |
| Witness log | — | — | Auditor | Auditor | Auditor |
| Load model | — | — | Admin | Admin+TEE | Admin+ZK |

---

## Container 2: `sona.rvf` — Self-Learning Container

### MCP Tools (12)

```yaml
tools:
  # --- Trajectory Management ---
  - name: sona_begin_trajectory
    description: Start a new learning trajectory (reasoning session)
    input:
      task_id: { type: string, required: true }
      context: { type: object }
    roles: [writer, admin]

  - name: sona_record_step
    description: Record a step in the current trajectory
    input:
      trajectory_id: { type: string, required: true }
      action: { type: string, required: true }
      observation: { type: string }
      reward: { type: number, min: -1, max: 1 }
    roles: [writer, admin]

  - name: sona_end_trajectory
    description: Finalize trajectory and trigger pattern extraction
    input:
      trajectory_id: { type: string, required: true }
      outcome: { type: string, enum: [success, failure, partial] }
      final_score: { type: number, min: 0, max: 1 }
    roles: [writer, admin]

  # --- Pattern Store ---
  - name: sona_search_patterns
    description: Search learned patterns by semantic similarity
    input:
      query: { type: string, required: true }
      k: { type: integer, default: 10 }
      min_confidence: { type: number, default: 0.5 }
    roles: [reader, writer, admin]

  - name: sona_store_pattern
    description: Manually store a pattern (bypass trajectory extraction)
    input:
      pattern: { type: string, required: true }
      category: { type: string }
      embedding: { type: array, items: number }
      confidence: { type: number, default: 0.8 }
    roles: [writer, admin]

  # --- Adaptation ---
  - name: sona_apply_lora
    description: Apply LoRA adapter to modify behavior
    input:
      adapter_id: { type: string, required: true }
      rank: { type: integer, default: 4 }
      alpha: { type: number, default: 1.0 }
    roles: [admin]

  - name: sona_consolidate
    description: Run EWC++ consolidation to prevent catastrophic forgetting
    input:
      lambda: { type: number, default: 0.4 }
    roles: [admin]

  - name: sona_transfer
    description: Export transfer prior for cross-domain deployment
    input:
      target_domain: { type: string, required: true }
    roles: [admin]

  - name: sona_import_prior
    description: Import transfer prior from another SONA instance
    input:
      prior_data: { type: string, format: base64 }
    roles: [admin]

  # --- Monitoring ---
  - name: sona_learning_status
    description: Get learning metrics (trajectories, patterns, adapters)
    input: {}
    roles: [reader, writer, admin]

  - name: sona_replay
    description: Replay a past trajectory for analysis or retraining
    input:
      trajectory_id: { type: string, required: true }
    roles: [reader, admin]

  - name: sona_witness_chain
    description: Get signed learning audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

### HTTP API

```
POST   /v1/trajectories             # Begin trajectory
PUT    /v1/trajectories/:id/step    # Record step
PUT    /v1/trajectories/:id/end     # End trajectory
GET    /v1/trajectories/:id         # Get trajectory
POST   /v1/patterns/search          # Search patterns
POST   /v1/patterns                 # Store pattern
POST   /v1/adapt/lora               # Apply LoRA
POST   /v1/adapt/consolidate        # EWC++ consolidation
POST   /v1/transfer/export          # Export prior
POST   /v1/transfer/import          # Import prior
GET    /v1/status                   # Learning status
GET    /v1/health                   # Health check
GET    /v1/witness                  # Audit trail
```

---

## Container 3: `graph.rvgraph` — Knowledge Graph Container

### MCP Tools (12)

```yaml
tools:
  - name: graph_cypher
    description: Execute a Cypher query against the knowledge graph
    input:
      query: { type: string, required: true }           # e.g. "MATCH (n:Person)-[:KNOWS]->(m) RETURN n,m"
      params: { type: object }                           # Query parameters
      timeout_ms: { type: integer, default: 5000 }
    roles: [reader, writer, admin]

  - name: graph_create_node
    description: Create a node with labels and properties
    input:
      labels: { type: array, items: string, required: true }
      properties: { type: object, required: true }
    roles: [writer, admin]

  - name: graph_create_edge
    description: Create a relationship between two nodes
    input:
      from_id: { type: string, required: true }
      to_id: { type: string, required: true }
      type: { type: string, required: true }
      properties: { type: object }
    roles: [writer, admin]

  - name: graph_semantic_search
    description: Search nodes by embedding similarity (vector + graph hybrid)
    input:
      query: { type: string, required: true }
      k: { type: integer, default: 10 }
      labels: { type: array, items: string }             # Filter by label
      hops: { type: integer, default: 0 }                # Expand N hops from results
    roles: [reader, writer, admin]

  - name: graph_shortest_path
    description: Find shortest path between two nodes
    input:
      from_id: { type: string, required: true }
      to_id: { type: string, required: true }
      max_depth: { type: integer, default: 10 }
      edge_types: { type: array, items: string }
    roles: [reader, writer, admin]

  - name: graph_subgraph
    description: Extract a subgraph around a node
    input:
      node_id: { type: string, required: true }
      depth: { type: integer, default: 2 }
      max_nodes: { type: integer, default: 100 }
    roles: [reader, writer, admin]

  - name: graph_snapshot
    description: Create a COW snapshot of the current graph state
    input:
      label: { type: string }
    roles: [admin]

  - name: graph_restore_snapshot
    description: Restore graph to a previous snapshot
    input:
      snapshot_id: { type: string, required: true }
    roles: [admin]

  - name: graph_diff
    description: Compare two snapshots and return delta
    input:
      snapshot_a: { type: string, required: true }
      snapshot_b: { type: string, required: true }
    roles: [reader, admin]

  - name: graph_schema
    description: Get or set the graph schema (labels, relationship types, constraints)
    input:
      action: { type: string, enum: [get, set], default: get }
      schema: { type: object }
    roles: [reader, admin]

  - name: graph_status
    description: Get graph statistics (nodes, edges, labels, memory)
    input: {}
    roles: [reader, writer, admin]

  - name: graph_witness
    description: Get graph mutation audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

### HTTP API

```
POST   /v1/cypher                   # Execute Cypher query
POST   /v1/nodes                    # Create node
POST   /v1/edges                    # Create edge
DELETE /v1/nodes/:id                # Delete node
DELETE /v1/edges/:id                # Delete edge
POST   /v1/search                   # Semantic search
GET    /v1/path/:from/:to           # Shortest path
GET    /v1/subgraph/:id             # Extract subgraph
POST   /v1/snapshots                # Create snapshot
POST   /v1/snapshots/:id/restore    # Restore snapshot
GET    /v1/snapshots/:a/diff/:b     # Diff snapshots
GET    /v1/schema                   # Get schema
GET    /v1/status                   # Graph stats
GET    /v1/health                   # Health check
WS     /v1/live                     # Live graph events
```

---

## Container 4: `ruvbot.rvf` — AI Assistant Container (Enhanced)

### MCP Tools (16)

```yaml
tools:
  # --- Session Management ---
  - name: ruvbot_create_session
    description: Create a new chat session with an agent
    input:
      agent_id: { type: string, default: "default-agent" }
      user_id: { type: string }
      platform: { type: string, enum: [api, slack, discord, telegram, cli] }
    roles: [writer, admin]

  - name: ruvbot_chat
    description: Send a message and get a response (with AIDefence protection)
    input:
      session_id: { type: string, required: true }
      message: { type: string, required: true }
      attachments: { type: array }
    roles: [writer, admin]

  - name: ruvbot_list_sessions
    description: List active sessions
    input:
      user_id: { type: string }
      limit: { type: integer, default: 20 }
    roles: [reader, admin]

  # --- Agent Management ---
  - name: ruvbot_spawn_agent
    description: Spawn a new agent with custom configuration
    input:
      name: { type: string, required: true }
      system_prompt: { type: string }
      model: { type: string }
      skills: { type: array, items: string }
    roles: [admin]

  - name: ruvbot_list_agents
    description: List all active agents
    input: {}
    roles: [reader, admin]

  - name: ruvbot_stop_agent
    description: Stop a running agent
    input:
      agent_id: { type: string, required: true }
    roles: [admin]

  # --- Memory ---
  - name: ruvbot_memory_store
    description: Store a memory with embedding for semantic retrieval
    input:
      content: { type: string, required: true }
      tags: { type: array, items: string }
      importance: { type: number, min: 0, max: 1, default: 0.5 }
      namespace: { type: string, default: "default" }
    roles: [writer, admin]

  - name: ruvbot_memory_search
    description: Semantic search across stored memories
    input:
      query: { type: string, required: true }
      k: { type: integer, default: 10 }
      namespace: { type: string }
      min_score: { type: number, default: 0.5 }
    roles: [reader, writer, admin]

  # --- Skills ---
  - name: ruvbot_list_skills
    description: List available skills (built-in + plugins)
    input: {}
    roles: [reader, admin]

  - name: ruvbot_invoke_skill
    description: Invoke a skill directly
    input:
      skill: { type: string, required: true }
      params: { type: object }
    roles: [writer, admin]

  # --- Security ---
  - name: ruvbot_security_scan
    description: Scan text for threats (prompt injection, PII, jailbreak)
    input:
      text: { type: string, required: true }
      strict: { type: boolean, default: false }
    roles: [reader, writer, admin]

  - name: ruvbot_security_audit
    description: Get security audit log
    input:
      limit: { type: integer, default: 50 }
      threats_only: { type: boolean, default: false }
    roles: [auditor, admin]

  # --- Templates ---
  - name: ruvbot_deploy_template
    description: Deploy a pre-built agent template
    input:
      template: { type: string, required: true, enum: [code-reviewer, doc-generator, test-generator, hive-mind, research-swarm] }
      options: { type: object }
    roles: [admin]

  # --- Monitoring ---
  - name: ruvbot_status
    description: Get bot status (agents, sessions, memory, uptime)
    input: {}
    roles: [reader, writer, admin]

  - name: ruvbot_models
    description: List available LLM models with pricing
    input: {}
    roles: [reader, writer, admin]

  - name: ruvbot_witness
    description: Get signed witness chain for all operations
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

---

## Container 5: `rvlite.rvf` — Edge Vector Database

### MCP Tools (10)

```yaml
tools:
  - name: rvlite_create
    description: Create a new lightweight vector store
    input:
      path: { type: string, required: true }
      dimensions: { type: integer, required: true }
      engine: { type: string, enum: [hnsw, flat, ivf], default: hnsw }
    roles: [admin]

  - name: rvlite_sql
    description: Execute a SQL query against the vector store
    input:
      store_id: { type: string, required: true }
      query: { type: string, required: true }            # SQL with vector extensions
    roles: [reader, writer, admin]

  - name: rvlite_sparql
    description: Execute a SPARQL query for RDF data
    input:
      store_id: { type: string, required: true }
      query: { type: string, required: true }
    roles: [reader, writer, admin]

  - name: rvlite_cypher
    description: Execute a Cypher query for graph data
    input:
      store_id: { type: string, required: true }
      query: { type: string, required: true }
    roles: [reader, writer, admin]

  - name: rvlite_ingest
    description: Insert vectors with metadata
    input:
      store_id: { type: string, required: true }
      entries: { type: array, required: true }
    roles: [writer, admin]

  - name: rvlite_search
    description: k-NN vector similarity search
    input:
      store_id: { type: string, required: true }
      vector: { type: array, items: number, required: true }
      k: { type: integer, default: 10 }
      filter: { type: object }
    roles: [reader, writer, admin]

  - name: rvlite_delete
    description: Delete vectors by ID or filter
    input:
      store_id: { type: string, required: true }
      ids: { type: array, items: string }
      filter: { type: object }
    roles: [writer, admin]

  - name: rvlite_compact
    description: Compact store and rebuild index
    input:
      store_id: { type: string, required: true }
    roles: [admin]

  - name: rvlite_status
    description: Get store statistics
    input:
      store_id: { type: string, required: true }
    roles: [reader, writer, admin]

  - name: rvlite_export
    description: Export store as portable .rvf file
    input:
      store_id: { type: string, required: true }
      output: { type: string, required: true }
      include_index: { type: boolean, default: true }
    roles: [admin]
```

---

## Container 6: `solver.rvf` — Temporal Reasoning Engine

### MCP Tools (10)

```yaml
tools:
  - name: solver_decide
    description: Make an optimal decision using Thompson Sampling
    input:
      context: { type: object, required: true }          # Current state
      actions: { type: array, items: string, required: true }
      objective: { type: string, enum: [maximize, minimize, balance] }
    roles: [writer, admin]

  - name: solver_update
    description: Update solver with outcome of a previous decision
    input:
      decision_id: { type: string, required: true }
      reward: { type: number, required: true }
      context: { type: object }
    roles: [writer, admin]

  - name: solver_predict
    description: Predict best action without committing
    input:
      context: { type: object, required: true }
      actions: { type: array, items: string, required: true }
      simulations: { type: integer, default: 1000 }
    roles: [reader, writer, admin]

  - name: solver_counterexample
    description: Submit a counterexample for policy refinement
    input:
      context: { type: object, required: true }
      expected_action: { type: string, required: true }
      actual_action: { type: string, required: true }
      reason: { type: string }
    roles: [writer, admin]

  - name: solver_get_policy
    description: Get current policy kernel state
    input: {}
    roles: [reader, writer, admin]

  - name: solver_set_policy
    description: Update policy kernel configuration
    input:
      policy: { type: object, required: true }
    roles: [admin]

  - name: solver_cost_curve
    description: Get convergence history and acceleration metrics
    input:
      window: { type: integer, default: 100 }
    roles: [reader, admin]

  - name: solver_transfer_export
    description: Export transfer prior for another domain
    input:
      target_domain: { type: string }
    roles: [admin]

  - name: solver_status
    description: Get solver state (decisions, convergence, policy version)
    input: {}
    roles: [reader, writer, admin]

  - name: solver_witness
    description: Get signed decision audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

---

## Container 7: `router.rvf` — Agent Routing Container

### MCP Tools (11)

```yaml
tools:
  - name: router_classify
    description: Classify intent and route to optimal agent/model
    input:
      text: { type: string, required: true }
      candidates: { type: array, items: string }          # Available agents
      budget: { type: string, enum: [tier1, tier2, tier3, auto], default: auto }
    roles: [reader, writer, admin]

  - name: router_register_route
    description: Register a new intent route
    input:
      intent: { type: string, required: true }
      handler: { type: string, required: true }           # Agent/model ID
      examples: { type: array, items: string, minItems: 3 }
      priority: { type: integer, default: 0 }
    roles: [admin]

  - name: router_remove_route
    description: Remove an intent route
    input:
      route_id: { type: string, required: true }
    roles: [admin]

  - name: router_list_routes
    description: List all registered routes with hit counts
    input: {}
    roles: [reader, admin]

  - name: router_hot_routes
    description: Get most-used routes from HOT_SEG
    input:
      limit: { type: integer, default: 20 }
    roles: [reader, admin]

  - name: router_circuit_status
    description: Get circuit breaker status for all routes
    input: {}
    roles: [reader, admin]

  - name: router_circuit_reset
    description: Reset a tripped circuit breaker
    input:
      route_id: { type: string, required: true }
    roles: [admin]

  - name: router_benchmark
    description: Run classification benchmark on test set
    input:
      test_set: { type: array, items: { text: string, expected: string } }
    roles: [admin]

  - name: router_ebpf_status
    description: Get eBPF fast-path acceleration metrics
    input: {}
    roles: [admin]

  - name: router_status
    description: Get router statistics (routes, hit rates, latency percentiles)
    input: {}
    roles: [reader, writer, admin]

  - name: router_witness
    description: Get routing decision audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

---

## Container 8: `rudag.rvf` — Workflow Orchestration Container

### MCP Tools (12)

```yaml
tools:
  - name: dag_create_workflow
    description: Create a new DAG workflow
    input:
      name: { type: string, required: true }
      tasks: { type: array, items: { id: string, deps: array, handler: string } }
    roles: [writer, admin]

  - name: dag_add_task
    description: Add a task to an existing workflow
    input:
      workflow_id: { type: string, required: true }
      task_id: { type: string, required: true }
      dependencies: { type: array, items: string }
      handler: { type: string, required: true }
      params: { type: object }
    roles: [writer, admin]

  - name: dag_execute
    description: Execute a workflow (topological order, parallel where possible)
    input:
      workflow_id: { type: string, required: true }
      dry_run: { type: boolean, default: false }
    roles: [writer, admin]

  - name: dag_task_status
    description: Get task execution status
    input:
      workflow_id: { type: string, required: true }
      task_id: { type: string }
    roles: [reader, writer, admin]

  - name: dag_critical_path
    description: Calculate critical path through the workflow
    input:
      workflow_id: { type: string, required: true }
    roles: [reader, admin]

  - name: dag_checkpoint
    description: Create a COW checkpoint of workflow state
    input:
      workflow_id: { type: string, required: true }
      label: { type: string }
    roles: [admin]

  - name: dag_rollback
    description: Rollback workflow to a checkpoint
    input:
      workflow_id: { type: string, required: true }
      checkpoint_id: { type: string, required: true }
    roles: [admin]

  - name: dag_list_workflows
    description: List all workflows
    input: {}
    roles: [reader, writer, admin]

  - name: dag_similar_tasks
    description: Find semantically similar tasks across workflows
    input:
      description: { type: string, required: true }
      k: { type: integer, default: 5 }
    roles: [reader, admin]

  - name: dag_sla_check
    description: Check if workflow meets SLA targets
    input:
      workflow_id: { type: string, required: true }
    roles: [reader, admin]

  - name: dag_status
    description: Get orchestrator statistics
    input: {}
    roles: [reader, writer, admin]

  - name: dag_witness
    description: Get workflow execution audit trail
    input:
      workflow_id: { type: string }
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

---

## Container 9: `rvdna.rvdna` — Genomic Analysis Container

### MCP Tools (11)

```yaml
tools:
  - name: rvdna_analyze_genome
    description: Run full genomic analysis pipeline (23andMe format)
    input:
      file_path: { type: string, required: true }
      panels: { type: array, items: string, enum: [pharmacogenomics, health, ancestry] }
    roles: [writer, admin]

  - name: rvdna_variant_lookup
    description: Look up a specific genetic variant (rsID)
    input:
      rs_id: { type: string, required: true }              # e.g. "rs1801133"
      genome_id: { type: string }
    roles: [reader, writer, admin]

  - name: rvdna_pharmacogenomics
    description: Get drug metabolism predictions (CYP2D6, CYP2C19)
    input:
      genome_id: { type: string, required: true }
      drugs: { type: array, items: string }
    roles: [reader, writer, admin]

  - name: rvdna_health_report
    description: Generate health variant report (17 variants)
    input:
      genome_id: { type: string, required: true }
    roles: [reader, writer, admin]

  - name: rvdna_similarity_search
    description: Find similar genomic profiles by k-mer embeddings
    input:
      genome_id: { type: string, required: true }
      k: { type: integer, default: 10 }
      metric: { type: string, enum: [cosine, l2], default: cosine }
    roles: [reader, admin]

  - name: rvdna_kmer_embed
    description: Generate k-mer embedding vector for a DNA sequence
    input:
      sequence: { type: string, required: true }
      k: { type: integer, default: 6 }
    roles: [reader, writer, admin]

  - name: rvdna_annotate
    description: Annotate variants with gene names, consequences, frequencies
    input:
      variants: { type: array, items: { chrom: string, pos: integer, ref: string, alt: string } }
    roles: [reader, writer, admin]

  - name: rvdna_export
    description: Export analysis results as .rvdna file with witness chain
    input:
      genome_id: { type: string, required: true }
      output: { type: string }
      encrypt: { type: boolean, default: true }
    roles: [admin]

  - name: rvdna_consent_grant
    description: Grant data access consent (records in WITNESS_SEG)
    input:
      genome_id: { type: string, required: true }
      grantee: { type: string, required: true }
      scope: { type: string, enum: [read, analyze, share], required: true }
      expires: { type: string, format: iso8601 }
    roles: [admin]

  - name: rvdna_status
    description: Get analysis pipeline status
    input: {}
    roles: [reader, writer, admin]

  - name: rvdna_witness
    description: Get genomic analysis audit trail (HIPAA-grade)
    input:
      genome_id: { type: string }
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

### Security: L3 Minimum (HIPAA)

Genomic data containers enforce L3 (Attested) as minimum:
- All data encrypted at rest (AES-256-GCM via CRYPTO_SEG)
- Consent chain in WITNESS_SEG (every access logged)
- PII detection blocks raw genomic data in prompts
- TEE attestation for computation proofs
- Data export requires explicit consent grant

---

## Container 10: `ruqu.rvf` — Quantum Simulation Container

### MCP Tools (10)

```yaml
tools:
  - name: ruqu_create_circuit
    description: Create a quantum circuit
    input:
      qubits: { type: integer, required: true, max: 25 }
      name: { type: string }
    roles: [writer, admin]

  - name: ruqu_add_gate
    description: Add a quantum gate to a circuit
    input:
      circuit_id: { type: string, required: true }
      gate: { type: string, required: true, enum: [H, X, Y, Z, CNOT, CZ, RX, RY, RZ, T, S, SWAP, TOFFOLI] }
      target: { type: integer, required: true }
      control: { type: integer }
      angle: { type: number }
    roles: [writer, admin]

  - name: ruqu_simulate
    description: Run quantum circuit simulation
    input:
      circuit_id: { type: string, required: true }
      shots: { type: integer, default: 1024 }
      noise_model: { type: string, enum: [none, depolarizing, amplitude_damping] }
    roles: [writer, admin]

  - name: ruqu_vqe
    description: Run Variational Quantum Eigensolver
    input:
      hamiltonian: { type: object, required: true }
      ansatz: { type: string, enum: [uccsd, hardware_efficient, qaoa] }
      max_iterations: { type: integer, default: 100 }
      optimizer: { type: string, enum: [cobyla, spsa, adam], default: cobyla }
    roles: [writer, admin]

  - name: ruqu_grover
    description: Run Grover's search algorithm
    input:
      oracle: { type: string, required: true }
      qubits: { type: integer, required: true }
    roles: [writer, admin]

  - name: ruqu_error_correct
    description: Apply surface code error correction
    input:
      circuit_id: { type: string, required: true }
      code_distance: { type: integer, default: 3 }
    roles: [admin]

  - name: ruqu_statevector
    description: Get full statevector (exponential in qubits)
    input:
      circuit_id: { type: string, required: true }
    roles: [reader, admin]

  - name: ruqu_expectation
    description: Calculate expectation value of an observable
    input:
      circuit_id: { type: string, required: true }
      observable: { type: object, required: true }
    roles: [reader, writer, admin]

  - name: ruqu_status
    description: Get simulator status (circuits, memory, convergence)
    input: {}
    roles: [reader, writer, admin]

  - name: ruqu_witness
    description: Get quantum computation audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

---

## Container 11: `synth.rvf` — Synthetic Data Container

### MCP Tools (8)

```yaml
tools:
  - name: synth_generate
    description: Generate synthetic data with provenance tracking
    input:
      schema: { type: object, required: true }
      count: { type: integer, required: true }
      model: { type: string, default: "gemini-2.5-pro" }
      seed: { type: integer }
      embed: { type: boolean, default: true }
    roles: [writer, admin]

  - name: synth_generate_from_template
    description: Generate data using a DSPy template
    input:
      template_id: { type: string, required: true }
      variables: { type: object }
      count: { type: integer, default: 10 }
    roles: [writer, admin]

  - name: synth_search
    description: Search generated data by similarity
    input:
      query: { type: string, required: true }
      k: { type: integer, default: 10 }
      dataset: { type: string }
    roles: [reader, writer, admin]

  - name: synth_validate
    description: Validate generated data against schema
    input:
      dataset_id: { type: string, required: true }
    roles: [reader, admin]

  - name: synth_provenance
    description: Get full generation provenance (model, prompt, seed, timestamp)
    input:
      item_id: { type: string, required: true }
    roles: [reader, auditor, admin]

  - name: synth_export
    description: Export dataset as .rvf with witness chain
    input:
      dataset_id: { type: string, required: true }
      format: { type: string, enum: [rvf, jsonl, parquet], default: rvf }
    roles: [admin]

  - name: synth_status
    description: Get generator status
    input: {}
    roles: [reader, writer, admin]

  - name: synth_witness
    description: Get data generation audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

---

## Container 12: `raft.rvf` — Distributed Consensus Container

### MCP Tools (10)

```yaml
tools:
  - name: raft_cluster_status
    description: Get cluster status (leader, term, members, health)
    input: {}
    roles: [reader, admin]

  - name: raft_propose
    description: Propose a state change to the cluster
    input:
      command: { type: object, required: true }
      timeout_ms: { type: integer, default: 5000 }
    roles: [writer, admin]

  - name: raft_read
    description: Consistent read from the replicated state
    input:
      key: { type: string, required: true }
      linearizable: { type: boolean, default: true }
    roles: [reader, writer, admin]

  - name: raft_add_member
    description: Add a new node to the cluster
    input:
      node_id: { type: string, required: true }
      address: { type: string, required: true }
    roles: [admin]

  - name: raft_remove_member
    description: Remove a node from the cluster
    input:
      node_id: { type: string, required: true }
    roles: [admin]

  - name: raft_snapshot
    description: Trigger a snapshot of the current state
    input: {}
    roles: [admin]

  - name: raft_log_entries
    description: Get recent Raft log entries
    input:
      from_index: { type: integer }
      limit: { type: integer, default: 100 }
    roles: [admin, auditor]

  - name: raft_transfer_leader
    description: Transfer leadership to another node
    input:
      target_node: { type: string, required: true }
    roles: [admin]

  - name: raft_status
    description: Get detailed Raft state (term, commit index, applied index)
    input: {}
    roles: [reader, admin]

  - name: raft_witness
    description: Get consensus decision audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

---

## Container 13: `safety-gate.rvf` — AI Safety Container

### MCP Tools (8)

```yaml
tools:
  - name: gate_permit
    description: Request permission for an AI action (permit/defer/deny)
    input:
      action_id: { type: string, required: true }
      action_type: { type: string, required: true }
      target: { type: object }
      context: { type: object }
      urgency: { type: string, enum: [low, medium, high, critical] }
    roles: [writer, admin]

  - name: gate_scan
    description: Scan text for threats without taking action
    input:
      text: { type: string, required: true }
      checks: { type: array, items: string, enum: [injection, jailbreak, pii, code, exfiltration] }
    roles: [reader, writer, admin]

  - name: gate_set_policy
    description: Update safety policy (thresholds, allowed actions, blocked patterns)
    input:
      policy: { type: object, required: true }
    roles: [admin]

  - name: gate_get_policy
    description: Get current safety policy
    input: {}
    roles: [reader, admin]

  - name: gate_receipt
    description: Get a witness receipt for a past decision
    input:
      sequence: { type: integer, required: true }
    roles: [auditor, admin]

  - name: gate_replay
    description: Deterministically replay a past decision for audit
    input:
      sequence: { type: integer, required: true }
      verify_chain: { type: boolean, default: true }
    roles: [auditor, admin]

  - name: gate_status
    description: Get gate statistics (permits, denies, defers, latency)
    input: {}
    roles: [reader, admin]

  - name: gate_witness
    description: Get full decision audit chain (Ed25519 signed)
    input:
      limit: { type: integer, default: 100 }
      decision: { type: string, enum: [permit, defer, deny] }
    roles: [auditor, admin]
```

---

## Container 14: `mcp-server.rvf` — MCP Protocol Adapter

### MCP Tools (10)

Inherits all 10 tools from existing `@ruvector/rvf-mcp-server`:
`rvf_create_store`, `rvf_open_store`, `rvf_close_store`, `rvf_ingest`, `rvf_query`, `rvf_delete`, `rvf_delete_filter`, `rvf_compact`, `rvf_status`, `rvf_list_stores`

**Enhancement**: Adds 4 new tools:

```yaml
  - name: rvf_embed_and_ingest
    description: Embed text and ingest in one call (WASM embedder)
    input:
      store_id: { type: string, required: true }
      texts: { type: array, items: string, required: true }
      metadata: { type: array, items: object }
    roles: [writer, admin]

  - name: rvf_semantic_search
    description: Natural language search (embed query + k-NN)
    input:
      store_id: { type: string, required: true }
      query: { type: string, required: true }
      k: { type: integer, default: 10 }
    roles: [reader, writer, admin]

  - name: rvf_witness_chain
    description: Get store witness audit trail
    input:
      store_id: { type: string, required: true }
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]

  - name: rvf_sign_store
    description: Cryptographically sign the store manifest
    input:
      store_id: { type: string, required: true }
      key_path: { type: string }
    roles: [admin]
```

---

## Container 15: `ospipe.rvf` — Personal Memory Container

### MCP Tools (8)

```yaml
tools:
  - name: ospipe_capture
    description: Capture and index screen content
    input:
      source: { type: string, enum: [screen, window, clipboard] }
      interval_ms: { type: integer, default: 5000 }
    roles: [writer, admin]

  - name: ospipe_search
    description: Search captured memories by semantic similarity
    input:
      query: { type: string, required: true }
      k: { type: integer, default: 10 }
      time_range: { type: object, properties: { from: string, to: string } }
    roles: [reader, writer, admin]

  - name: ospipe_recall
    description: Recall what was on screen at a specific time
    input:
      timestamp: { type: string, format: iso8601, required: true }
      context_window: { type: integer, default: 60 }     # seconds
    roles: [reader, writer, admin]

  - name: ospipe_forget
    description: Delete specific memories (GDPR right to erasure)
    input:
      memory_ids: { type: array, items: string }
      time_range: { type: object }
      pattern: { type: string }                           # Regex pattern to match & delete
    roles: [admin]

  - name: ospipe_export
    description: Export memories as .rvf file
    input:
      output: { type: string, required: true }
      time_range: { type: object }
      encrypt: { type: boolean, default: true }
    roles: [admin]

  - name: ospipe_privacy_scan
    description: Scan memories for PII and sensitive content
    input:
      auto_redact: { type: boolean, default: false }
    roles: [admin]

  - name: ospipe_status
    description: Get capture status (memories, index size, last capture)
    input: {}
    roles: [reader, admin]

  - name: ospipe_witness
    description: Get memory access audit trail
    input:
      limit: { type: integer, default: 50 }
    roles: [auditor, admin]
```

### Security: L2 Minimum (Personal Data)

Personal memory containers enforce L2 as minimum:
- All captures encrypted at rest
- PII auto-detection on ingest
- GDPR-compliant deletion (`ospipe_forget`)
- Local-only by default (no network egress without explicit config)
- Witness chain logs every access

---

## Cross-Container Communication Protocol

Containers communicate via **Transfer Prior Exchange**:

```
Container A (SONA)                    Container B (RuvLLM)
     │                                      │
     ├── sona_transfer_export ──────────────►│
     │   (TRANSFERPRIOR_SEG)                 │
     │                                       ├── ruvllm_feedback
     │                                       │   (updates routing)
     │◄──────────────── ruvllm_cost_report ──┤
     │                                       │
```

### Inter-Container MCP Discovery

Each container exposes a `_meta_capabilities` resource:

```json
{
  "uri": "rvf://capabilities",
  "name": "Container Capabilities",
  "mimeType": "application/json",
  "content": {
    "container": "ruvllm.rvf",
    "version": "2.3.0",
    "security_level": "L2",
    "tools": ["ruvllm_complete", "ruvllm_embed", "..."],
    "accepts_transfer_priors": true,
    "exports_transfer_priors": true,
    "witness_signed": true
  }
}
```

---

## Implementation Checklist

For each container, implementation requires:

1. [ ] `rvf.manifest.json` — Segment assembly config
2. [ ] `src/mcp/server.ts` — MCP tool definitions + handlers
3. [ ] `src/mcp/transports.ts` — STDIO + SSE transport
4. [ ] `src/api/routes.ts` — HTTP REST endpoints
5. [ ] `src/api/ws.ts` — WebSocket live events
6. [ ] `src/security/auth.ts` — L0-L4 auth middleware
7. [ ] `src/security/rbac.ts` — Role-based access control
8. [ ] `src/security/witness.ts` — Ed25519 witness chain
9. [ ] `src/security/aidefence.ts` — Input threat detection
10. [ ] `scripts/build-rvf.js` — Container builder
11. [ ] `scripts/run-rvf.js` — Container runner
12. [ ] `tests/` — MCP tool tests + security tests
