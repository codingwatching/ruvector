# ADR-003: Dynamic Cut Sparsifier with Low Recourse

**Status**: Proposed
**Date**: 2026-02-23
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-23 | ruv.io | Initial draft: 5-layer architecture with forest packing backbone |

---

## Plain Language Summary

**What is it?**

A fully dynamic cut sparsifier that maintains an explicit sparse graph H approximating all cuts of an evolving undirected capacitated graph G. It is designed for low recourse (few edge changes in H per update to G) and deterministic CPU execution.

**Why does it matter?**

The existing sparsifier in `ruvector-mincut/src/sparsify/` uses Benczur-Karger randomized sampling and Nagamochi-Ibaraki certificates. These are static: they rebuild from scratch or make ad hoc incremental updates without structural guarantees on recourse. The existing j-tree sparsifier in `ruvector-mincut/src/jtree/sparsifier.rs` focuses on vertex-split tolerance for the hierarchical decomposition.

This ADR introduces a **forest-packing-based dynamic sparsifier** that:

- Maintains k spanning forests whose union forms a cut approximation backbone
- Uses bounded replacement scanning to keep worst-case recourse low per update
- Produces an explicit H that downstream systems (coherence gate, mincut gated transformer, boundary sensors) can query directly
- Logs every structural change as a witness record for auditing

**How it relates to ADR-001 and ADR-002:**

| ADR | Role | Relationship |
|-----|------|-------------|
| **ADR-001** | Anytime-valid coherence gate | Consumes H for fast boundary signal computation instead of full G |
| **ADR-002** | j-Tree hierarchical decomposition | Uses the sparsifier as a backbone at Level 0; vertex-split tolerance feeds the hierarchy |
| **ADR-003** (this) | Dynamic cut sparsifier engine | Produces and maintains H with bounded recourse |

Think of ADR-003 as the engine that keeps the "radar screen" (H) updated, while ADR-001 reads the radar for safety decisions and ADR-002 builds coarser views on top of it.

---

## Context

### Current State

The codebase has three sparsification paths:

| Path | Location | Approach | Limitation |
|------|----------|----------|------------|
| Benczur-Karger | `sparsify/mod.rs` | Randomized edge sampling by strength | Static rebuild; no recourse bound; nondeterministic without seed pinning |
| Nagamochi-Ibaraki | `sparsify/mod.rs` | Deterministic k-certificate | Static; no dynamic update support |
| j-Tree sparsifier | `jtree/sparsifier.rs` | Forest packing with vertex-split tolerance | Optimized for hierarchy maintenance; not exposed as standalone cut approximation engine |

None of these maintain an **explicit H** with **bounded per-update recourse** suitable for always-on monitoring.

### The Need

The coherence gate (ADR-001) and mincut gated transformer run continuous cut queries. On a dense or rapidly evolving G:

1. Running exact min-cut on G per update is too expensive for continuous monitoring
2. Static sparsification requires periodic full rebuilds that spike latency
3. Ad hoc incremental updates (current `SparseGraph::insert_edge` / `delete_edge`) provide no structural guarantee on approximation quality or recourse

### Design Lineage

This design follows two lines of work:

1. **Forest packing for dynamic cut sparsification**: Maintaining k spanning forests where edge assignment follows a greedy packing rule. Low recourse arises from bounded replacement scanning when tree edges are deleted.

2. **Dynamic hierarchical j-tree decomposition** (arXiv:2601.09139): Vertex-split-tolerant cut sparsifiers with poly-logarithmic recourse, already partially implemented in the `jtree` module.

This ADR combines both ideas into a standalone engine with explicit witness logging.

---

## Decision

### Implement a 5-Layer Dynamic Cut Sparsifier in `ruvector-mincut`

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DYNAMIC CUT SPARSIFIER (ADR-003)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 5: Witness Logs & Policy Gates                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Compact witness record per update: epoch, deltas, recourse, metrics  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                        │
│  Layer 4: Coherence Integration Hooks                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Hook A: Boundary signal (local k-cut on H)                           │ │
│  │  Hook B: Mincut candidate extraction (upper bound coherence signal)   │ │
│  │  Hook C: Recourse & drift telemetry                                   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                        │
│  Layer 3: Explicit Sparsifier H                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Tree edges from all forests + sampled non-tree edges                 │ │
│  │  Deterministic SipHash scoring, incremental H maintenance             │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                        │
│  Layer 2: Dynamic Cut Backbone (Forest Packing)                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  k forests F₀..F_{k-1}, each a dynamic tree (Euler tour tree)        │ │
│  │  Greedy packing: edge enters lowest non-cycle forest                  │ │
│  │  Bounded replacement scanning on delete                               │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                        │
│  Layer 1: Graph Store & Epoching                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  GraphMap (EdgeKey → EdgeRecord), Adjacency lists, EpochClock         │ │
│  │  Stable ordering by EdgeKey, seeded determinism, ring buffer logging  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Layer 1: Graph Store and Epoching

**Data structures:**

- `VertexId`: `u32`
- `EdgeKey`: `(min(u,v), max(u,v))` — canonical ordering for determinism
- `EdgeRecord`: `{ u: u32, v: u32, cap: u64, alive: bool, last_epoch: u64 }`
- `GraphMap`: `HashMap<EdgeKey, EdgeRecord>`
- `Adjacency`: `Vec<SmallVec<EdgeKey>>` for degree walks
- `EpochClock`: monotonic `u64`

**Update API:**

- `apply_update(Update)` increments the epoch, writes to `GraphMap`, and pushes an `EdgeDelta` into a ring buffer for logging and recourse accounting.

**Determinism guarantees:**

- Stable ordering by `EdgeKey` — never iterate `HashMap` directly for decisions
- No random sampling unless seeded by `EpochClock` and a fixed `global_seed`
- All edge lists sorted by `EdgeKey` before structural operations

### Layer 2: Dynamic Cut Backbone via Forest Packing

This is the engine that maintains a small set of spanning forests whose union approximates cut structure.

**High-level idea:**

- Maintain `k` forests `F₀..F_{k-1}`
- Each forest is a dynamic tree supporting `link`, `cut`, and `connected` in O(log n) via the existing Euler tour tree implementation (`euler/mod.rs`)
- Edge assignment follows a greedy packing rule: an edge tries to enter the lowest-index forest where it does not create a cycle; otherwise it becomes a non-tree edge held in a level bucket

Low recourse comes from this structure: only a small number of edges move between tree and non-tree roles per update when replacement scanning is bounded.

**Dynamic tree choice:**

The existing `EulerTourTree` in `euler/mod.rs` provides O(log n) `link`, `cut`, and `connected` operations with treap-based balancing. The forest packing module will use this directly.

**Forest module interface:**

```rust
pub trait DynamicForest {
    fn connected(&self, u: u32, v: u32) -> bool;
    fn link(&mut self, u: u32, v: u32, e: EdgeKey);
    fn cut(&mut self, e: EdgeKey);
    fn find_replacement_edge(
        &self,
        cut_side_hint: u32,
        candidate_edges: &[EdgeKey],
        graph: &GraphStore,
    ) -> Option<EdgeKey>;
}
```

**Packing state:**

```rust
struct ForestPacking {
    forests: Vec<EulerTourTree>,      // k dynamic forests
    edge_level: HashMap<EdgeKey, u8>, // which forest index, or 255 for non-tree
    non_tree_buckets: Vec<Vec<EdgeKey>>, // per level
}
```

**Insert edge `e = (u, v)`:**

1. For `i` in `0..k`:
2. If `!forests[i].connected(u, v)` → link and set `edge_level[e] = i`, break
3. Else continue
4. If all levels create a cycle → push into highest `non_tree_bucket`

**Delete edge `e`:**

1. Look up `edge_level[e]`
2. If non-tree → remove from bucket
3. If tree edge → cut it, then scan candidate non-tree edges from same and higher levels for replacement
4. If replacement found → promote it into the tree; otherwise the forest stays split

**Bounded replacement scanning:**

- Cap the scan per update to `B` edges per level (configurable via `scan_budget_per_level`)
- If no replacement found within budget → mark the component as dirty and schedule a rebuild
- This bounded scan keeps worst-case latency stable on constrained devices while preserving amortized O(k log n) behavior

### Layer 3: Explicit Cut Sparsifier H

Build H as a sparse graph composed of:

- **All current tree edges** from all k forests
- **A bounded set of extra edges** sampled from non-tree buckets using deterministic hashing

**Edge weights:**

Store original capacities on edges included in H. For initial implementation, include edges with original weights (more edges = better fidelity). Level-scaled weights are a future optimization.

**Deterministic sampling rule (no nondeterminism):**

```rust
fn sample_score(global_seed: u64, epoch: u64, level: u8, edge: EdgeKey) -> u64 {
    siphash64(global_seed, epoch, level, edge.a, edge.b)
}

fn should_include(score: u64, threshold: u64) -> bool {
    score % threshold == 0
}
```

Where `threshold` depends on the level (configurable via `sample_mod_base` and `sample_level_scale`). This gives stable recourse because inclusion changes only when an edge moves levels or the epoch advances past a configurable window.

**H maintenance:**

- H is rebuilt incrementally, not from scratch
- On insert or delete, update only the edges that changed role or changed sample status
- Track `H_deltas` for witness logging

### Layer 4: Coherence Integration Hooks

Three hooks matching the existing coherence gate stack:

**Hook A — Boundary signal:**

- Compute `local_kcut` on H around recently touched vertices
- Emit boundary crossings as events for the coherence gate (ADR-001)

**Hook B — Mincut candidate extraction:**

- Run deterministic dynamic mincut on H (not G)
- Treat the result as an upper-bound coherence signal
- If cut value crosses a policy threshold → escalate to full graph or higher-fidelity sparsifier

**Hook C — Recourse and drift telemetry:**

- `recourse_edges_changed_in_H` per update
- `recourse_tree_swaps` per update
- `disagreement_score`: periodic comparison of mincut on H vs mincut on G for sampled epochs

### Layer 5: Witness Logs and Policy Gates

Every update produces a compact witness record:

```rust
pub struct WitnessRecord {
    pub epoch: u64,
    pub input_delta: Update,
    pub forest_ops: Vec<ForestOp>,       // link, cut, promote, demote
    pub h_edge_deltas: Vec<HEdgeDelta>,  // edges added/removed from H
    pub recourse: RecourseStats,
    pub cut_metrics: Option<CutMetrics>, // optional periodic cut check
}
```

This fits the "treat correctness as adversarially stressed" posture: we prove what changed, not that it is true.

---

## Module Structure

```
ruvector-mincut/src/
├── dyn_sparsifier/                # NEW: Dynamic Cut Sparsifier (ADR-003)
│   ├── mod.rs                     # Module exports, DynamicCutSparsifier public API
│   ├── graph_store.rs             # Layer 1: GraphMap, Adjacency, EpochClock
│   ├── forest.rs                  # Layer 2: DynamicForest trait, ForestPacking
│   ├── packing.rs                 # Layer 2: Greedy packing logic, replacement scanning
│   ├── sampler.rs                 # Layer 3: Deterministic SipHash sampling
│   ├── sparsifier.rs             # Layer 3: Explicit H construction and maintenance
│   ├── hooks.rs                   # Layer 4: Coherence integration hooks
│   ├── witness.rs                 # Layer 5: WitnessChain, WitnessRecord
│   └── metrics.rs                 # RecourseStats, performance counters
├── euler/                         # EXISTING: Euler tour tree (reused by Layer 2)
│   └── mod.rs
├── jtree/                         # EXISTING: j-Tree hierarchy (consumes Layer 3 output)
│   ├── sparsifier.rs             # Existing vertex-split-tolerant sparsifier
│   └── ...
├── sparsify/                      # EXISTING: Static sparsifiers (unchanged)
│   └── mod.rs
└── ...
```

---

## Key Types

```rust
/// Canonical edge key with deterministic ordering
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeKey {
    pub a: u32, // min(u, v)
    pub b: u32, // max(u, v)
}

/// Record for a single edge in the graph store
pub struct EdgeRecord {
    pub u: u32,
    pub v: u32,
    pub cap: u64,
    pub alive: bool,
    pub last_epoch: u64,
}

/// Graph update operations
pub enum Update {
    Insert { u: u32, v: u32, cap: u64 },
    Delete { u: u32, v: u32 },
}

/// Configuration for the dynamic cut sparsifier
pub struct SparsifierConfig {
    /// Number of spanning forests (higher = better approximation, more memory)
    pub k_forests: usize,
    /// Maximum edges scanned per level during replacement search
    pub scan_budget_per_level: usize,
    /// Base modulus for deterministic non-tree edge sampling
    pub sample_mod_base: u64,
    /// Scale factor for sampling threshold per level
    pub sample_level_scale: u64,
    /// Maximum edges rebuilt per component per epoch
    pub rebuild_component_budget: usize,
    /// Fixed seed for deterministic hashing
    pub global_seed: u64,
}
```

---

## Public API

```rust
pub struct DynamicCutSparsifier {
    cfg: SparsifierConfig,
    epoch: u64,
    graph: GraphStore,
    packing: ForestPacking,
    sampler: LevelSampler,
    h: SparseGraph,
    witness: WitnessChain,
}

impl DynamicCutSparsifier {
    /// Create a new sparsifier for a graph with n vertices
    pub fn new(n: usize, cfg: SparsifierConfig) -> Self;

    /// Apply an update and return the delta to H
    pub fn apply(&mut self, up: Update) -> SparsifierDelta;

    /// Export all edges currently in H
    pub fn export_edges(&self) -> Vec<(u32, u32, u64)>;

    /// Estimate cut value for a given vertex partition
    pub fn estimate_cut_value(&self, side: &[u32]) -> u64;

    /// Get recourse statistics for the last update
    pub fn recourse_stats(&self) -> RecourseStats;
}
```

---

## Detailed Update Algorithm

```
apply(update):
  1. epoch += 1
  2. graph.apply(update) → produces edge_key and delta flags
  3. packing.apply(update) → returns PackingDelta
     • list of tree edge changes
     • list of level changes
     • list of bucket changes
  4. sampler.update(epoch, PackingDelta) → returns SampleDelta
     • edges newly sampled in
     • edges sampled out
  5. sparsifier.merge(PackingDelta, SampleDelta) → updates H
  6. witness.append(epoch, update, deltas, metrics)
  7. return SparsifierDelta for downstream gates
```

**PackingDelta fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tree_added` | `Vec<EdgeKey>` | Edges newly added as tree edges |
| `tree_removed` | `Vec<EdgeKey>` | Edges removed from trees |
| `promoted` | `Vec<EdgeKey>` | Non-tree edges promoted to tree edges |
| `demoted` | `Vec<EdgeKey>` | Tree edges demoted to non-tree |
| `level_changed` | `Vec<(EdgeKey, u8, u8)>` | Edges that moved between levels |

**SampleDelta fields:**

| Field | Type | Description |
|-------|------|-------------|
| `add` | `Vec<EdgeKey>` | Non-tree edges newly sampled into H |
| `remove` | `Vec<EdgeKey>` | Non-tree edges removed from H sample |

**RecourseStats:**

| Field | Type | Description |
|-------|------|-------------|
| `h_edge_changes` | `usize` | Total edge changes in H this update |
| `forest_swaps` | `usize` | Tree/non-tree role swaps |
| `scan_steps` | `usize` | Total replacement scan steps taken |
| `rebuilds_triggered` | `usize` | Component rebuilds scheduled |

---

## Bounded Rebuild Strategy

When replacement scanning fails within the scan budget:

1. Mark the component ID as dirty
2. Schedule a component rebuild using deterministic BFS over adjacency restricted to alive edges
3. Rebuild only that component's forest assignments and buckets, capped by `rebuild_component_budget`
4. If the component is still too large, split the work over multiple epochs with a partial rebuild flag

This ensures the always-on monitoring loop does not spike latency.

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Graph map ops | O(1) average | HashMap insert/delete |
| Forest ops per update | O(k log n) | k forests, Euler tour tree operations |
| Replacement scans | O(k * B) | B = `scan_budget_per_level` |
| H updates | O(recourse) | Typically small; proportional to structural change |
| Memory | O(n + m + k * n) | n vertices, m edges in G, k forests |

---

## Integration with Existing Codebase

### Reuse of Euler Tour Tree

The `euler/mod.rs` implementation provides the dynamic tree operations needed by Layer 2. The `ForestPacking` module will instantiate k `EulerTourTree` instances, adapting the `NodeId = u64` interface to `u32` vertex IDs used by the graph store.

### Relationship to `jtree/sparsifier.rs`

The existing `DynamicCutSparsifier` in `jtree/sparsifier.rs` focuses on vertex-split tolerance for the j-tree hierarchy. The new `dyn_sparsifier` module is a **standalone engine** that can:

- Feed into the j-tree hierarchy as its Level 0 backbone
- Be used independently by the coherence gate for fast boundary queries
- Eventually replace the static `sparsify/mod.rs` for dynamic workloads

### Relationship to `sparsify/mod.rs`

The existing static sparsifiers remain unchanged. They serve use cases where a single-shot sparsification is sufficient (e.g., batch analysis). The new module handles the dynamic case.

---

## Failure Modes and Mitigations

### Mode 1: Recourse grows during adversarial churn

**Symptom**: `h_edge_changes` exceeds 50 per update consistently.

**Mitigations**:
- Increase `k_forests` to absorb more edge diversity
- Decrease sample churn by using epoch windows (only re-evaluate sampling every W epochs)
- Raise `scan_budget_per_level` slightly
- Enable component rebuild more aggressively

### Mode 2: Sparsifier fidelity degrades on dense components

**Symptom**: Disagreement score between mincut on H and mincut on G exceeds 5% for dense subgraphs.

**Mitigations**:
- Add a second sampling tier that includes more non-tree edges inside high-degree components (detected by the boundary sensor)
- Increase k specifically for components where degree > threshold

### Mode 3: Determinism breaks due to HashMap iteration order

**Symptom**: Different runs with same seed produce different H.

**Mitigation**: Never iterate `HashMap` directly for structural decisions. Always collect into `Vec<EdgeKey>` and sort before acting. This is enforced by code convention and tested in CI.

### Mode 4: Memory bloat on non-tree buckets

**Symptom**: Memory usage grows unbounded in `non_tree_buckets`.

**Mitigations**:
- Store `EdgeKey` in slab arrays with free lists
- Keep per-level compact vectors rebuilt periodically (every R epochs)
- Cap total non-tree edges stored per level

---

## Determinism Contract

All operations are deterministic given:

1. A fixed `global_seed`
2. A deterministic update stream (same sequence of `Insert`/`Delete` operations)

This is achieved by:

- Canonical `EdgeKey` ordering: `(min(u,v), max(u,v))`
- SipHash-based sampling seeded by `(global_seed, epoch, level, EdgeKey)`
- Sorted edge lists before any structural decision
- XorShift64 in the Euler tour tree seeded deterministically

---

## Acceptance Test

| Criterion | Target |
|-----------|--------|
| Graph size | n = 200,000 vertices, m = 2,000,000 edges |
| Update stream | 100,000 updates with fixed seed |
| Cut fidelity | Every 1,000 epochs, sample 200 random cuts; median relative error < 5% between G and H |
| Latency | p99 update time under device budget (target: < 1ms on x86, < 10ms on ARM) |
| Recourse | Average `h_edge_changes` < 50 per update |
| Determinism | Two runs with same seed produce identical H at every epoch |

**Benchmark output per epoch:**

```
epoch, update_type, scan_steps, forest_swaps, h_edge_changes, rebuilds_triggered, mincut_H
```

---

## Open Questions

### Q1: Cut preservation scope

Should H preserve cut values for **all** cuts, or only cuts above a policy threshold lambda (matching the existing caveat zone in the coherence gate)?

- **All cuts**: Simpler implementation, larger H, higher fidelity
- **Threshold lambda**: Smaller H, tunable, matches coherence gate's existing policy-driven filtering
- **Recommendation**: Start with all cuts for correctness validation, then add threshold filtering as an optimization

### Q2: Update stream characteristics

Is the update stream mostly **edge weight changes and toggles**, or does it include frequent **vertex splits and merges** (as the j-tree hierarchy focuses on)?

- **Weight changes + toggles**: Simpler forest maintenance, lower recourse
- **Vertex splits**: Requires the vertex-split tolerance from ADR-002's sparsifier design
- **Recommendation**: Implement edge insert/delete first; add vertex-split support by delegating to the existing `jtree/sparsifier.rs` machinery

---

## Future Extensions

### GPU/WASM Acceleration

If a GPU lane is needed later, dyGRASS-style localized random walks can identify spectrally critical edges for dynamic spectral sparsification. This could serve as a Tier 2 fidelity upgrade for attention health scoring in the mincut gated transformer.

### Directed Graphs

Recent work on fully dynamic directed cut and spectral sparsifiers exists, but the undirected case is the correct starting point for the mincut gate. Directed support can be added as a separate module when needed.

### Level-Scaled Weights

Instead of including edges with original capacities, scale weights by `2^level` to reduce H size while preserving cut approximation guarantees. This trades fidelity for memory and is appropriate once the base implementation is validated.

---

## References

- ADR-001: Anytime-Valid Coherence Gate
- ADR-002: Dynamic Hierarchical j-Tree Decomposition (arXiv:2601.09139)
- ADR-002-addendum-bmssp-integration
- arXiv:2512.13105 (El-Hayek/Henzinger/Li) — Exact deterministic dynamic min-cut
- arXiv:2601.09139 (Goranci/Henzinger/Kiss/Momeni/Zocklein, SODA 2026) — Dynamic hierarchical j-tree with vertex-split-tolerant sparsifier
- Benczur-Karger (1996) — Randomized cut sparsification by edge strengths
- Nagamochi-Ibaraki (1992) — Deterministic sparse k-certificates
