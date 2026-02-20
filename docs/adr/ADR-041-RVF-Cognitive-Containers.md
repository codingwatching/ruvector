# ADR-041: RVF Cognitive Containers — Packaging the RuVector Ecosystem

**Status**: Proposed
**Date**: 2026-02-20
**Author**: RuVector Team
**Supersedes**: None
**Related**: ADR-030 (RVF Computational Container), ADR-031 (COW Branching), ADR-033 (Quality Envelope), ADR-034 (QR Cognitive Seed), ADR-039 (AGI Integration), ADR-040 (Causal Atlas)

---

## Context

RuvBot (v0.3.1) demonstrated that a complete AI assistant — including a real Linux 6.6.80 microkernel, Node.js runtime bundle, agent profiles, witness chains, and cryptographic provenance — can ship as a single 3.7 MB `.rvf` file that boots on QEMU, Firecracker, or Cloud Hypervisor in under 1 second.

The RVF format provides 25 segment types, 12+ specialized headers, 5 domain profiles, and capabilities spanning from eBPF kernel fast-paths to WASM browser runtimes to TEE attestation. The RuVector monorepo contains 63 npm packages, of which **26 are viable candidates** for RVF cognitive container packaging.

This ADR defines which packages should be packaged as RVF cognitive containers, what segments each would use, and the implementation roadmap.

---

## Decision

Package 15 npm packages as self-contained RVF cognitive containers across 4 tiers, each leveraging different RVF segment capabilities to create portable, bootable, verifiable AI units.

---

## RVF Capabilities Inventory

### Segment Types Available (25)

| Range | Segments | Purpose |
|-------|----------|---------|
| 0x01-0x09 | Vec, Index, Overlay, Journal, Manifest, Quant, Meta, Hot, Sketch | Core vector storage |
| 0x0A-0x0D | Witness, Profile, Crypto, MetaIdx | Security & provenance |
| 0x0E-0x11 | Kernel, Ebpf, Wasm, Dashboard | Executable runtimes |
| 0x20-0x23 | CowMap, Refcount, Membership, Delta | Copy-on-write & branching |
| 0x30-0x32 | TransferPrior, PolicyKernel, CostCurve | Self-learning & governance |

### Header Formats

| Header | Size | Magic | Purpose |
|--------|------|-------|---------|
| SegmentHeader | 64 B | `RVFS` | Universal segment envelope |
| KernelHeader | 128 B | `RVKN` | Microkernel boot image |
| WasmHeader | 64 B | `RVWM` | WASM bytecode execution |
| EbpfHeader | 64 B | `RVBP` | eBPF kernel acceleration |
| DashboardHeader | 64 B | `RVDB` | Web UI bundle |
| WitnessHeader | 64 B | `RVWW` | Computation proofs |
| AgiContainerHeader | 64 B | `RVAG` | AGI orchestration |
| CowMapHeader | 64 B | `RVCM` | Copy-on-write snapshots |
| MembershipHeader | 96 B | `RVMB` | Vector visibility filter |
| DeltaHeader | 64 B | `RVDL` | Incremental updates |
| SeedHeader | 64 B | `RVQS` | QR cognitive seed |

### Domain Profiles

| Profile | Extension | Magic | Domain |
|---------|-----------|-------|--------|
| Generic | `.rvf` | — | Universal |
| Rvdna | `.rvdna` | `RDNA` | Genomics |
| RvText | `.rvtext` | `RTXT` | Language/NLP |
| RvGraph | `.rvgraph` | `RGRH` | Graph/Network |
| RvVision | `.rvvis` | `RVIS` | Vision/Imagery |

---

## Tier 1: Full Cognitive Containers (Boot + Runtime + Learning)

These packages become fully self-contained bootable units with microkernel, WASM runtime, embedded vectors, witness chains, and optional dashboard.

### 1.1 `@ruvector/ruvllm` — LLM Inference Container

**Package**: LLM orchestration with SONA adaptive learning, FastGRNN routing, 12+ model support
**File**: `ruvllm.rvf` (~8 MB estimated)

| Segment | Contents | Purpose |
|---------|----------|---------|
| `KERNEL_SEG` (0x0E) | Linux 6.6 bzImage | Self-boot for bare-metal LLM serving |
| `WASM_SEG` (0x10) | RuvLLM WASM runtime | Browser/edge inference fallback |
| `VEC_SEG` (0x01) | Pre-trained embedding cache | Fast cold-start for common queries |
| `INDEX_SEG` (0x02) | HNSW routing index | Model selection by query similarity |
| `PROFILE_SEG` (0x0B) | Model registry + LoRA adapters | Available models + fine-tuning state |
| `POLICYKERNEL_SEG` (0x31) | Routing policy (Tier 1/2/3) | Cost/latency/quality trade-off rules |
| `COSTCURVE_SEG` (0x32) | Historical latency/cost data | Self-optimization acceleration |
| `TRANSFERPRIOR_SEG` (0x30) | Cross-domain posteriors | Transfer learning between deployments |
| `WITNESS_SEG` (0x0A) | Inference provenance chain | Audit trail for every prediction |
| `DASHBOARD_SEG` (0x11) | Three.js model dashboard | Visualize routing, latency, cost |
| `META_SEG` (0x07) | Package metadata | Version, build hash, capabilities |
| `MANIFEST_SEG` (0x05) | Segment directory | File identity + lineage |

**Why RVF**: LLM serving requires model routing state, embedding caches, and cost optimization. A single `.rvf` file boots a self-learning LLM server with zero external dependencies.

### 1.2 `@ruvector/sona` — Self-Learning Container

**Package**: Runtime-adaptive learning (LoRA, EWC++, ReasoningBank)
**File**: `sona.rvf` (~5 MB estimated)

| Segment | Contents |
|---------|----------|
| `KERNEL_SEG` | Minimal Linux for bare-metal learning loops |
| `WASM_SEG` | SONA WASM runtime for browser/edge |
| `VEC_SEG` | Experience replay buffer (episodic memory) |
| `INDEX_SEG` | HNSW index over reasoning trajectories |
| `TRANSFERPRIOR_SEG` | Compressed posterior summaries from prior sessions |
| `POLICYKERNEL_SEG` | Learning rate schedules, EWC penalties |
| `DELTA_SEG` (0x23) | LoRA weight deltas (sparse row patches) |
| `QUANT_SEG` (0x06) | Quantized weight codebooks (4x-32x compression) |
| `WITNESS_SEG` | Learning trajectory audit trail |
| `CRYPTO_SEG` (0x0C) | Model signing keys |
| `META_SEG` | SONA version, training epoch, convergence metrics |

**Why RVF**: SONA's learning state (LoRA adapters, EWC fisher matrices, experience buffers) is naturally expressed as vector segments with delta patches. RVF's COW branching enables "what-if" learning experiments without data loss.

### 1.3 `@ruvector/graph-node` — Knowledge Graph Container

**Package**: Neo4j-compatible hypergraph database with Cypher queries
**File**: `graph.rvgraph` (~4 MB estimated, uses `.rvgraph` domain profile)

| Segment | Contents |
|---------|----------|
| `KERNEL_SEG` | Linux for server-mode graph queries |
| `WASM_SEG` | Graph-WASM for browser-native Cypher |
| `VEC_SEG` | Node/edge embedding vectors |
| `INDEX_SEG` | HNSW over graph embeddings |
| `OVERLAY_SEG` (0x03) | Graph partition overlays |
| `JOURNAL_SEG` (0x04) | Graph mutation log (add/delete nodes/edges) |
| `META_SEG` | Schema definitions, property types |
| `METAIDX_SEG` (0x0D) | Inverted index for property-filtered graph queries |
| `COWMAP_SEG` (0x20) | Copy-on-write graph snapshots |
| `REFCOUNT_SEG` (0x21) | Cluster reference counts for GC |
| `DASHBOARD_SEG` | Three.js graph visualizer |
| `WITNESS_SEG` | Graph mutation provenance |

**Why RVF**: Knowledge graphs need both vector similarity (for semantic queries) and structural indexes (for Cypher). RVF's overlay + journal + COW segments provide versioned graph state with zero-copy branching — perfect for "what if we add this edge?" experimentation.

### 1.4 `ruvbot` — AI Assistant Container

**Package**: Already shipped as v0.3.1 (3.7 MB)
**File**: `ruvbot.rvf` — **Reference implementation**

Already includes: KERNEL, WASM, META, PROFILE, WITNESS, MANIFEST segments.

**Enhancement path**: Add VEC_SEG (agent memory), INDEX_SEG (conversation search), DASHBOARD_SEG (admin UI), POLICYKERNEL_SEG (routing policy), TRANSFERPRIOR_SEG (cross-session learning).

---

## Tier 2: Runtime Containers (WASM + Vectors + Learning)

These packages don't need a kernel but benefit from WASM runtime, embedded vectors, and self-learning segments.

### 2.1 `rvlite` — Edge Vector Database

**File**: `rvlite.rvf` (~2 MB)

| Segment | Contents |
|---------|----------|
| `WASM_SEG` | RVLite WASM engine (SQL + SPARQL + Cypher) |
| `VEC_SEG` | Pre-seeded vector data |
| `INDEX_SEG` | HNSW index |
| `QUANT_SEG` | Quantization codebooks for memory-constrained devices |
| `META_SEG` | Schema + query templates |
| `MEMBERSHIP_SEG` (0x22) | Vector visibility filter for multi-tenant isolation |
| `WITNESS_SEG` | Data provenance chain |

**Why RVF**: Edge devices need a single file that contains the database engine, pre-seeded data, and index — no installation, no dependencies. The MEMBERSHIP_SEG enables per-tenant vector isolation on shared edge hardware.

### 2.2 `@ruvector/rvf-solver` — Temporal Reasoning Engine

**File**: `solver.rvf` (~1.5 MB)

| Segment | Contents |
|---------|----------|
| `WASM_SEG` | Solver WASM (Thompson Sampling + PolicyKernel) |
| `POLICYKERNEL_SEG` | Pre-trained policy weights |
| `COSTCURVE_SEG` | Historical convergence data |
| `TRANSFERPRIOR_SEG` | Cross-domain priors for new tasks |
| `VEC_SEG` | Counterexample embeddings |
| `INDEX_SEG` | HNSW over counterexample space |
| `WITNESS_SEG` | Decision audit trail |

**Why RVF**: The solver's policy state (Thompson posteriors, cost curves, transfer priors) maps directly to RVF's learning segments (0x30-0x32). A single file carries the solver's complete "experience" across deployments.

### 2.3 `@ruvector/router` + `@ruvector/tiny-dancer` — Agent Routing Container

**File**: `router.rvf` (~3 MB)

| Segment | Contents |
|---------|----------|
| `WASM_SEG` | Router + TinyDancer WASM runtime |
| `VEC_SEG` | Intent embedding vectors |
| `INDEX_SEG` | HNSW intent routing index |
| `EBPF_SEG` (0x0F) | eBPF XDP program for kernel-level packet routing |
| `POLICYKERNEL_SEG` | Circuit breaker thresholds + routing rules |
| `SKETCH_SEG` (0x09) | Access frequency sketches for hot-path optimization |
| `HOT_SEG` (0x08) | Temperature-promoted high-frequency routes |
| `META_SEG` | Route definitions, model mappings |

**Why RVF**: Agent routing benefits from eBPF acceleration (sub-microsecond packet classification), hot/cold data tiering (frequently-used routes stay in HOT_SEG), and access sketches (SKETCH_SEG) for adaptive optimization.

### 2.4 `@ruvector/rudag` — Workflow Orchestration Container

**File**: `rudag.rvf` (~2 MB)

| Segment | Contents |
|---------|----------|
| `WASM_SEG` | RuDAG WASM engine (topological sort, critical path) |
| `VEC_SEG` | Task embeddings for similarity-based scheduling |
| `INDEX_SEG` | HNSW over task vectors |
| `OVERLAY_SEG` | DAG partition overlays |
| `JOURNAL_SEG` | Task mutation log (add/complete/fail) |
| `COWMAP_SEG` | Workflow snapshots for rollback |
| `WITNESS_SEG` | Execution trace with timestamps |
| `META_SEG` | Workflow templates, SLA definitions |

**Why RVF**: Workflow state (task graph, execution history, rollback points) is naturally a DAG with vector-indexed task similarity. COW snapshots enable "replay from checkpoint" without losing history.

### 2.5 `@ruvector/rvdna` — Genomic Analysis Container

**File**: `rvdna.rvdna` (~6 MB, uses `.rvdna` domain profile)

| Segment | Contents |
|---------|----------|
| `WASM_SEG` | RVDNA analysis pipeline |
| `VEC_SEG` | Reference genome k-mer embeddings |
| `INDEX_SEG` | HNSW over genomic vectors |
| `PROFILE_SEG` | RVDNA domain profile (codons, motifs) |
| `QUANT_SEG` | Compressed reference genome codebooks |
| `META_SEG` | Gene annotations, variant databases |
| `METAIDX_SEG` | Inverted index for gene/variant filtering |
| `WITNESS_SEG` | Analysis provenance (critical for clinical use) |
| `CRYPTO_SEG` | Patient data encryption keys |

**Why RVF**: Genomic analysis requires carrying reference data, analysis pipelines, and audit trails together. Clinical genomics demands cryptographic provenance (WITNESS_SEG + CRYPTO_SEG) for regulatory compliance. The `.rvdna` profile enables domain-specific optimizations (codon-aware quantization, motif indexing).

### 2.6 `@ruvector/ruqu-wasm` — Quantum Simulation Container

**File**: `ruqu.rvf` (~3 MB)

| Segment | Contents |
|---------|----------|
| `WASM_SEG` | Quantum simulator (25-qubit, VQE, Grover, QAOA) |
| `VEC_SEG` | Pre-computed quantum state vectors |
| `INDEX_SEG` | HNSW over quantum states for variational search |
| `POLICYKERNEL_SEG` | Variational optimizer configuration |
| `COSTCURVE_SEG` | VQE convergence history |
| `DASHBOARD_SEG` | Bloch sphere + circuit visualizer |
| `WITNESS_SEG` | Quantum computation trace |

**Why RVF**: Quantum variational algorithms (VQE, QAOA) iterate over parameterized circuits — the cost curve and policy kernel segments track convergence state, enabling "resume from best parameters" across sessions.

---

## Tier 3: Data Containers (Vectors + Index + Provenance)

These packages primarily carry data with embedded indexes and provenance, without runtime execution.

### 3.1 `@ruvector/agentic-synth` — Synthetic Data Container

**File**: `synth-{domain}.rvf` (~variable)

| Segment | Contents |
|---------|----------|
| `VEC_SEG` | Generated synthetic embeddings |
| `INDEX_SEG` | HNSW over synthetic data |
| `META_SEG` | Generation parameters, DSPy templates |
| `JOURNAL_SEG` | Generation mutation log |
| `WITNESS_SEG` | Provenance chain (which model, which prompt, which seed) |
| `TRANSFERPRIOR_SEG` | Distribution statistics for transfer learning |

**Why RVF**: Synthetic data needs provenance tracking (which model generated it, with what parameters). The WITNESS_SEG creates an audit trail from prompt to embedding, critical for training data governance.

### 3.2 `@ruvector/replication` + `@ruvector/raft` — Distributed State Container

**File**: `cluster-{node}.rvf` (~variable)

| Segment | Contents |
|---------|----------|
| `VEC_SEG` | Replicated vector state |
| `INDEX_SEG` | Local HNSW partition |
| `OVERLAY_SEG` | Partition assignment overlays |
| `DELTA_SEG` | Incremental replication deltas |
| `JOURNAL_SEG` | Raft log entries |
| `MEMBERSHIP_SEG` | Cluster membership filter |
| `COWMAP_SEG` | Snapshot for consistent reads |
| `WITNESS_SEG` | Consensus decision audit |
| `CRYPTO_SEG` | Cluster authentication keys |

**Why RVF**: Distributed consensus state (Raft log, membership, partition maps) maps perfectly to RVF's journal + overlay + membership segments. Each node's state is a single portable file that can be snapshotted (COW), replicated (DELTA), and verified (WITNESS).

### 3.3 `@cognitum/gate` — AI Safety Container

**File**: `safety-gate.rvf` (~1 MB)

| Segment | Contents |
|---------|----------|
| `WASM_SEG` | Safety gate WASM (permit/defer/deny) |
| `VEC_SEG` | Threat pattern embeddings |
| `INDEX_SEG` | HNSW over threat patterns |
| `POLICYKERNEL_SEG` | Safety policy rules |
| `WITNESS_SEG` | Decision audit (every permit/deny logged) |
| `META_SEG` | Threat taxonomy, severity levels |

**Why RVF**: AI safety gates must be tamper-evident (WITNESS_SEG), policy-governed (POLICYKERNEL_SEG), and self-contained (no external dependencies that could be compromised). A single signed `.rvf` is the ideal distribution format.

---

## Tier 4: Seed Containers (QR-Scannable Cognitive Seeds)

These leverage ADR-034's QR Cognitive Seed format (max 2,953 bytes) for ultra-portable bootstrapping.

### 4.1 `@ruvector/rvf-mcp-server` — MCP Protocol Seed

**File**: QR code encoding a `SeedHeader` (64 B) + WASM microkernel (~2 KB)

The QR seed contains:
- `SEED_HAS_MICROKERNEL` flag + 2 KB WASM query engine
- `SEED_HAS_DOWNLOAD` flag + download manifest for full MCP server
- `SEED_STREAM_UPGRADE` flag for progressive enhancement

**Use case**: Scan a QR code at a conference → instant MCP server that progressively downloads its full capability set.

### 4.2 `@ruvector/ospipe` — Personal Memory Seed

**File**: QR code encoding a seed with download manifest

Scans to bootstrap a personal AI memory system that connects to Screenpipe, downloads its WASM runtime, and begins indexing screen content.

---

## Segment Usage Matrix

| Package | Vec | Idx | Ovl | Jnl | Man | Qnt | Meta | Hot | Skt | Wit | Pro | Cry | MIdx | Ker | Ebpf | Wasm | Dash | Cow | Ref | Mem | Del | TPr | Pol | Cost |
|---------|-----|-----|-----|-----|-----|-----|------|-----|-----|-----|-----|-----|------|-----|------|------|------|-----|-----|-----|-----|-----|-----|------|
| ruvllm | X | X | | | X | | X | | | X | X | | | X | | X | X | | | | | X | X | X |
| sona | X | X | | | X | X | X | | | X | | X | | X | | X | | | | | X | X | X |
| graph-node | X | X | X | X | X | | X | | | X | | | X | X | | X | X | X | X | | | | | |
| ruvbot | | | | | X | | X | | | X | X | | | X | | X | | | | | | | | |
| rvlite | X | X | | | X | X | X | | | X | | | | | | X | | | | X | | | | |
| rvf-solver | X | X | | | X | | X | | | X | | | | | | X | | | | | | X | X | X |
| router | X | X | | | X | | X | X | X | | | | | | X | X | | | | | | | X | |
| rudag | X | X | X | X | X | | X | | | X | | | | | | X | | X | | | | | | |
| rvdna | X | X | | | X | X | X | | | X | X | X | X | | | X | | | | | | | | |
| ruqu-wasm | X | X | | | X | | | | | X | | | | | | X | X | | | | | | X | X |
| agentic-synth | X | X | | X | X | | X | | | X | | | | | | | | | | | | X | | |
| raft | X | X | X | X | X | | | | | X | | X | | | | | | X | | X | X | | | |
| cognitum/gate | X | X | | | X | | X | | | X | | | | | | X | | | | | | | X | |

**Legend**: Vec=0x01, Idx=0x02, Ovl=0x03, Jnl=0x04, Man=0x05, Qnt=0x06, Meta=0x07, Hot=0x08, Skt=0x09, Wit=0x0A, Pro=0x0B, Cry=0x0C, MIdx=0x0D, Ker=0x0E, Ebpf=0x0F, Wasm=0x10, Dash=0x11, Cow=0x20, Ref=0x21, Mem=0x22, Del=0x23, TPr=0x30, Pol=0x31, Cost=0x32

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

1. **Generic `build-rvf` CLI** — Extend `ruvbot/scripts/build-rvf.js` into a reusable tool that accepts a package manifest and produces an `.rvf` file
2. **Generic `run-rvf` CLI** — Extend `ruvbot/scripts/run-rvf.js` to handle all segment types (not just KERNEL+WASM)
3. **npm `rvf-pack` package** — `npx rvf-pack build` reads `rvf.manifest.json`, assembles segments
4. **Rust `rvf-pack` crate** — Parallel tool for Rust packages

### Phase 2: Tier 1 Containers (Weeks 3-5)

1. `ruvllm.rvf` — LLM inference with embedded routing index + dashboard
2. `sona.rvf` — Self-learning with LoRA delta segments + experience replay
3. `graph.rvgraph` — Knowledge graph with COW snapshots + Cypher WASM

### Phase 3: Tier 2 Containers (Weeks 6-8)

1. `rvlite.rvf` — Edge database with quantized vectors
2. `solver.rvf` — Temporal reasoning with policy kernel
3. `router.rvf` — Agent routing with eBPF fast path
4. `rvdna.rvdna` — Genomic analysis with domain profile
5. `rudag.rvf` — Workflow orchestration with COW rollback

### Phase 4: Tier 3-4 Containers (Weeks 9-10)

1. Synthetic data, replication, safety gate containers
2. QR cognitive seeds for MCP and OSpipe
3. Cross-container transfer learning via TRANSFERPRIOR_SEG

### Phase 5: Ecosystem (Weeks 11-12)

1. **RVF Registry** — npm-like registry for `.rvf` files with content-addressed storage
2. **Container composition** — Merge multiple `.rvf` files into a composite container
3. **Progressive loading** — QR seed → WASM microkernel → full container download
4. **Inter-container communication** — RVF-to-RVF protocol via shared TRANSFERPRIOR_SEG

---

## Packaging Format

Each RVF cognitive container follows this structure:

```
rvf.manifest.json
{
  "name": "@ruvector/ruvllm",
  "version": "2.3.0",
  "profile": "generic",         // or "rvdna", "rvtext", "rvgraph", "rvvis"
  "output": "ruvllm.rvf",
  "segments": {
    "kernel": {
      "type": "kernel",
      "source": "kernel/bzImage",
      "arch": "x86_64",
      "compression": "gzip"
    },
    "wasm": {
      "type": "wasm",
      "source": "dist/ruvllm.wasm",
      "role": "combined",
      "target": "wasi-p2"
    },
    "vectors": {
      "type": "vec",
      "source": "data/embeddings.bin",
      "dimensions": 384,
      "count": 10000
    },
    "index": {
      "type": "index",
      "source": "data/hnsw.bin"
    },
    "dashboard": {
      "type": "dashboard",
      "source": "dashboard/dist/",
      "framework": "threejs",
      "entry": "index.html"
    },
    "policy": {
      "type": "policy_kernel",
      "source": "config/routing-policy.json"
    }
  },
  "signing": {
    "algorithm": "ed25519",
    "key": "keys/signing.key"
  }
}
```

Build command: `npx rvf-pack build` or `cargo rvf-pack build`

---

## Size Budget

| Container | Kernel | WASM | Vectors | Index | Other | Total |
|-----------|--------|------|---------|-------|-------|-------|
| ruvllm.rvf | 1.6 MB | 2 MB | 3 MB | 1 MB | 0.5 MB | ~8 MB |
| sona.rvf | 1.6 MB | 1 MB | 1.5 MB | 0.5 MB | 0.5 MB | ~5 MB |
| graph.rvgraph | 1.6 MB | 1.5 MB | 0.5 MB | 0.3 MB | 0.3 MB | ~4 MB |
| ruvbot.rvf | 1.6 MB | 2.2 MB | — | — | 2 KB | ~3.7 MB |
| rvlite.rvf | — | 1 MB | 0.5 MB | 0.3 MB | 0.2 MB | ~2 MB |
| solver.rvf | — | 0.8 MB | 0.3 MB | 0.2 MB | 0.2 MB | ~1.5 MB |
| router.rvf | — | 1.5 MB | 0.5 MB | 0.3 MB | 0.7 MB | ~3 MB |
| rvdna.rvdna | — | 1.5 MB | 3 MB | 1 MB | 0.5 MB | ~6 MB |
| ruqu.rvf | — | 2 MB | 0.5 MB | 0.3 MB | 0.2 MB | ~3 MB |
| rudag.rvf | — | 1 MB | 0.3 MB | 0.2 MB | 0.5 MB | ~2 MB |
| safety-gate.rvf | — | 0.5 MB | 0.3 MB | 0.2 MB | 0.1 MB | ~1 MB |

All containers target **< 10 MB** for fast distribution via npm, QR, or direct download.

---

## Packages NOT Recommended for RVF

| Package | Reason |
|---------|--------|
| Platform binaries (13 packages) | Architecture-specific, better as npm optionalDependencies |
| `@ruvector/cli` | CLI wrapper, not a standalone cognitive unit |
| `@ruvector/burst-scaling` | Infrastructure tool, not a portable AI unit |
| `@ruvector/scipix` | External API client, needs network |
| `@ruvector/graph-data-generator` | Data generation utility, not a container |
| `@ruvector/postgres-cli` | PostgreSQL integration, needs external DB |
| `@ruvector/agentic-synth-examples` | Examples/tutorials, not deployable |

---

## Consequences

### Positive
- **Single-file distribution**: Every AI capability ships as one portable file
- **Verifiable provenance**: Witness chains prove what's inside each container
- **Self-learning persistence**: Learning state (LoRA, EWC, policies) survives across deployments
- **Multi-runtime**: Same container runs on bare metal (kernel), browser (WASM), or edge (lite)
- **Composable**: Containers can share TRANSFERPRIOR_SEG for cross-domain learning

### Negative
- **Build complexity**: Each container needs a custom build pipeline
- **Size overhead**: Kernel adds 1.6 MB to every Tier 1 container
- **Testing surface**: 15 containers x 3 runtimes = 45 test configurations

### Mitigations
- Generic `rvf-pack` tool amortizes build complexity across all packages
- Kernel is optional — Tier 2-4 containers skip it, saving 1.6 MB
- CI matrix tests each container on its target runtimes only

---

## References

- ADR-030: RVF Computational Container Format
- ADR-031: COW Branching and Snapshot Segments
- ADR-033: Quality Envelope and Retrieval Guarantees
- ADR-034: QR Cognitive Seed Format
- ADR-039: AGI Container Integration
- ADR-040: Causal Atlas RVF Runtime
- RuvBot v0.3.1: Reference RVF implementation (3.7 MB, boots Linux 6.6.80)
