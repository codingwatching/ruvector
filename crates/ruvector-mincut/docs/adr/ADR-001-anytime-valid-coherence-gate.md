# ADR-001: Anytime-Valid Coherence Gate

**Status**: Proposed
**Date**: 2026-01-17
**Authors**: Research Team
**Deciders**: Architecture Review Board

## Plain Language Summary

**What is it?**

An Anytime-Valid Coherence Gate is a small control loop that decides, at any moment:

> "Is it safe to act right now, or should we pause or escalate?"

It does not try to be smart. It tries to be **safe**, **calm**, and **correct** about permission.

**Why "anytime-valid"?**

Because you can stop the computation at any time and still trust the decision.

Like a smoke detector:
- It can keep listening forever
- The moment it has enough evidence, it triggers
- If you stop listening early, whatever it already concluded is still valid

You are not waiting for a model to finish thinking. You are continuously monitoring stability.

**Why "coherence"?**

Coherence means: does the system's current state agree with itself?

In RuVector, coherence is measured from structure:
- RuVector holds relationships as vectors plus a graph
- Min-cut and boundary signals tell you when the graph is becoming fragile or splitting into conflicting regions
- If the system is splitting, you do not let it take big actions

**What it outputs:**

| Decision | Meaning |
|----------|---------|
| **Permit** | Stable enough, proceed |
| **Defer** | Uncertain, escalate to a stronger model or human |
| **Deny** | Unstable or policy-violating, block the action |

Every decision returns a short "receipt" explaining why.

**A concrete example:**

An agent wants to push a config change to a network device.
- If the dependency graph is stable and similar changes worked before → **Permit**
- If signals are weird (new dependencies, new actors, drift) → **Defer** and ask for confirmation
- If the change crosses a fragile boundary (touches a partition already unstable) → **Deny**

**Why it matters:**

It turns autonomy into something enterprises can trust because:
- Actions are bounded
- Uncertainty is handled explicitly
- You get an audit trail

*"Attention becomes a permission system, not a popularity contest"* — applied to whole-system actions instead of token attention.

---

## Context

The RuVector ecosystem requires a principled mechanism for controlling autonomous agent actions with:
- **Formal safety guarantees** under distribution shift
- **Computational efficiency** suitable for real-time enforcement
- **Auditable decision trails** with cryptographic receipts

Current approaches (threshold classifiers, rule-based systems, periodic audits) lack one or more of these properties. This ADR proposes the **Anytime-Valid Coherence Gate (AVCG)** - a 3-way algorithmic combination that converts coherence measurement into a deterministic control loop.

## Decision

We will implement an Anytime-Valid Coherence Gate that integrates three cutting-edge algorithmic components:

### 1. Dynamic Min-Cut with Witness Partitions

**Source**: El-Hayek, Henzinger, Li (arXiv:2512.13105, December 2025)

**Key Innovation**: Exact deterministic n^{o(1)} update time for cuts up to 2^{Θ(log^{3/4-c}n)}

**Integration**:
- Extends existing `SubpolynomialMinCut` in `ruvector-mincut/src/subpolynomial/mod.rs`
- Leverages existing `WitnessTree` for explicit partition certificates
- Uses deterministic `LocalKCut` for local cut verification

**Role in Gate**: Provides the **structural coherence signal** - identifies minimal intervention points in the agent action graph with explicit witness partitions showing which actions form the critical boundary to unsafe states.

### 2. Online Conformal Prediction with Shift-Awareness

**Sources**:
- Retrospective Adjustment (arXiv:2511.04275, November 2025)
- Conformal Optimistic Prediction (COP) (December 2025)
- CORE: RL-based Conformal Regression (October 2025)

**Key Innovation**: Distribution-free coverage guarantees that adapt to arbitrary distribution shift with faster recalibration via retrospective adjustment.

**Integration**:
- New module: `ruvector-mincut/src/conformal/` for prediction sets
- Interfaces with existing `GatePolicy` thresholds
- Wraps action outcome predictions with calibrated uncertainty

**Role in Gate**: Provides the **predictive uncertainty signal** - quantifies confidence in action outcomes, triggering DEFER when prediction sets are too large.

### 3. E-Values and E-Processes for Anytime-Valid Inference

**Sources**:
- Ramdas & Wang "Hypothesis Testing with E-values" (FnTStA 2025)
- ICML 2025 Tutorial on SAVI
- Sequential Randomization Tests (arXiv:2512.04366, December 2025)

**Key Innovation**: Evidence accumulation that remains valid at any stopping time, with multiplicative composition across experiments.

**Definition**: E-value e satisfies E[e] ≤ 1 under null hypothesis. E-processes are nonnegative supermartingales with E_0 = 1.

**Integration**:
- New module: `ruvector-mincut/src/eprocess/` for evidence tracking
- Integrates with existing `CutCertificate` for audit trails
- Enables anytime-valid stopping decisions

**Role in Gate**: Provides the **evidential validity signal** - accumulates statistical evidence for/against coherence with formal Type I error control at any stopping time.

## Gate Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ANYTIME-VALID COHERENCE GATE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│   │  DYNAMIC MIN-CUT │    │    CONFORMAL     │    │   E-PROCESS      │ │
│   │    (Structural)  │    │   (Predictive)   │    │  (Evidential)    │ │
│   │                  │    │                  │    │                  │ │
│   │  SubpolynomialMC │    │  ShiftAdaptive   │    │  CoherenceTest   │ │
│   │  WitnessTree     │───▶│  PredictionSet   │───▶│  EvidenceAccum   │ │
│   │  LocalKCut       │    │  COP/CORE        │    │  StoppingRule    │ │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘ │
│            │                       │                       │           │
│            ▼                       ▼                       ▼           │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    DECISION LOGIC                              │   │
│   │                                                                │   │
│   │   PERMIT: E_t > τ_permit ∧ action ∉ CriticalCut ∧ |C_t| small │   │
│   │   DEFER:  |C_t| large ∨ τ_deny < E_t < τ_permit               │   │
│   │   DENY:   E_t < τ_deny ∨ action ∈ WitnessPartition(unsafe)    │   │
│   │                                                                │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                               │                                        │
│                               ▼                                        │
│                    ┌─────────────────────┐                            │
│                    │   WITNESS RECEIPT   │                            │
│                    │  (cut + conf + e)   │                            │
│                    └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Integration with Existing Architecture

### Extension Points

| Component | Current Implementation | AVCG Extension |
|-----------|----------------------|----------------|
| `GatePacket` | λ as point estimate | Add `lambda_confidence_q15`, `e_value_log_q15` |
| `GateController` | Rule-based thresholds | Add `AnytimeGatePolicy` with adaptive thresholds |
| `WitnessTree` | Cut value only | Add `ConfidenceWitness` with staleness tracking |
| `CutCertificate` | Static verification | Add `EvidenceReceipt` with e-value trace |
| `TierDecision` | Fixed tiers | Add `required_confidence_for_tier` |

### New Modules

```
ruvector-mincut/
├── src/
│   ├── conformal/           # NEW: Online conformal prediction
│   │   ├── mod.rs
│   │   ├── prediction_set.rs
│   │   ├── cop.rs           # Conformal Optimistic Prediction
│   │   ├── retrospective.rs # Retrospective adjustment
│   │   └── core.rs          # RL-based conformal
│   ├── eprocess/            # NEW: E-value and e-process tracking
│   │   ├── mod.rs
│   │   ├── evalue.rs
│   │   ├── evidence_accum.rs
│   │   ├── stopping.rs
│   │   └── mixture.rs
│   ├── anytime_gate/        # NEW: Integrated gate controller
│   │   ├── mod.rs
│   │   ├── policy.rs
│   │   ├── decision.rs
│   │   └── receipt.rs
│   └── ...existing modules...
```

## Decision Rules

### Permit Conditions (all must hold)
1. E-process value E_t > τ_permit (sufficient evidence of coherence)
2. Action not in witness partition of critical cut
3. Conformal prediction set |C_t| < θ_confidence (confident prediction)

### Defer Conditions (any triggers)
1. Conformal prediction set |C_t| > θ_uncertainty (uncertain outcome)
2. E-process in indeterminate range: τ_deny < E_t < τ_permit
3. Deadline approaching without sufficient confidence

### Deny Conditions (any triggers)
1. E-process value E_t < τ_deny (strong evidence of incoherence)
2. Action in witness partition crossing to unsafe states
3. Structural impossibility via min-cut topology

## Threshold Configuration

| Threshold | Meaning | Recommended Default |
|-----------|---------|---------------------|
| τ_deny | E-process level indicating incoherence | 0.01 (1% false alarm) |
| τ_permit | E-process level indicating coherence | 100 (strong evidence) |
| θ_uncertainty | Conformal set size requiring deferral | Task-dependent |
| θ_confidence | Conformal set size for confident permit | Task-dependent |

## Witness Receipt Structure

```rust
pub struct WitnessReceipt {
    /// Timestamp of decision
    pub timestamp: u64,
    /// Action that was evaluated
    pub action_id: ActionId,
    /// Gate decision
    pub decision: GateDecision,

    // Structural witness (from min-cut)
    pub cut_value: f64,
    pub witness_partition: (Vec<VertexId>, Vec<VertexId>),
    pub critical_edges: Vec<EdgeId>,

    // Predictive witness (from conformal)
    pub prediction_set: ConformalSet,
    pub coverage_target: f32,
    pub shift_adaptation_rate: f32,

    // Evidential witness (from e-process)
    pub e_value: f64,
    pub e_process_cumulative: f64,
    pub stopping_valid: bool,

    // Cryptographic seal
    pub receipt_hash: [u8; 32],
}
```

## Security Hardening

### Threat Model

| Threat Actor | Capabilities | Target | Impact |
|--------------|--------------|--------|--------|
| **Malicious Agent** | Action injection, timing manipulation | Gate bypass | Unauthorized actions executed |
| **Network Adversary** | Message interception, replay | Receipt forgery | False audit trail |
| **Insider Threat** | Threshold modification, key access | Policy manipulation | Safety guarantees voided |
| **Byzantine Node** | Arbitrary behavior in distributed gate | Consensus corruption | Inconsistent decisions |

### Cryptographic Requirements

#### Receipt Signing (CRITICAL)

```rust
pub struct WitnessReceipt {
    // ... existing fields ...

    // Cryptographic seal (REQUIRED)
    pub receipt_hash: [u8; 32],         // Blake3 hash of serialized content
    pub signature: Ed25519Signature,     // REQUIRED, not optional
    pub signer_id: PublicKey,           // Identity of signing gate
    pub timestamp_proof: TimestampProof, // Prevents backdating
}

/// Timestamp proof prevents replay and backdating
pub struct TimestampProof {
    pub timestamp: u64,
    pub previous_receipt_hash: [u8; 32], // Chain linkage
    pub merkle_root: [u8; 32],           // Batch anchor
}

impl WitnessReceipt {
    /// Sign receipt - MUST be called before any external use
    pub fn sign(&mut self, key: &SigningKey) -> Result<(), CryptoError> {
        let content = self.serialize_without_signature();
        self.receipt_hash = blake3::hash(&content).into();
        self.signature = key.sign(&self.receipt_hash);
        Ok(())
    }

    /// Verify receipt integrity and authenticity
    pub fn verify(&self, trusted_keys: &KeyStore) -> Result<(), VerifyError> {
        // 1. Verify hash
        let expected_hash = blake3::hash(&self.serialize_without_signature());
        if self.receipt_hash != expected_hash.into() {
            return Err(VerifyError::HashMismatch);
        }

        // 2. Verify signature
        let public_key = trusted_keys.get(&self.signer_id)?;
        public_key.verify(&self.receipt_hash, &self.signature)?;

        // 3. Verify timestamp chain
        self.timestamp_proof.verify()?;

        Ok(())
    }
}
```

#### Key Management

| Key Type | Purpose | Rotation | Storage |
|----------|---------|----------|---------|
| Gate Signing Key | Sign receipts | 30 days | HSM or secure enclave |
| Receipt Verification Keys | Verify receipts | On rotation | Distributed key store |
| Threshold Keys | Multi-party signing | 90 days | Shamir secret sharing |

### Attack Mitigations

#### E-Value Manipulation Prevention

```rust
/// Bounds checking for e-value inputs
impl EValue {
    pub fn from_likelihood_ratio(
        likelihood_h1: f64,
        likelihood_h0: f64,
    ) -> Result<Self, EValueError> {
        // Prevent division by zero
        if likelihood_h0 <= f64::EPSILON {
            return Err(EValueError::InvalidDenominator);
        }

        let ratio = likelihood_h1 / likelihood_h0;

        // Bound extreme values to prevent overflow attacks
        let bounded = ratio.clamp(E_VALUE_MIN, E_VALUE_MAX);

        // Log if clamping occurred (potential attack indicator)
        if (bounded - ratio).abs() > f64::EPSILON {
            security_log!("E-value clamped: {} -> {}", ratio, bounded);
        }

        Ok(Self { value: bounded, ..Default::default() })
    }
}

const E_VALUE_MIN: f64 = 1e-10;
const E_VALUE_MAX: f64 = 1e10;
```

#### Race Condition Prevention

```rust
/// Atomic gate decision with sequence numbers
pub struct AtomicGateDecision {
    /// Monotonic sequence for ordering
    sequence: AtomicU64,
    /// Lock for decision atomicity
    decision_lock: RwLock<()>,
}

impl AtomicGateDecision {
    pub async fn evaluate(&self, action: &Action) -> GateResult {
        // Acquire exclusive lock for decision
        let _guard = self.decision_lock.write().await;

        // Get sequence number BEFORE evaluation
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);

        // Evaluate all three signals atomically
        let result = self.evaluate_internal(action, seq).await;

        // Sequence number in receipt ensures ordering
        result.with_sequence(seq)
    }
}
```

#### Replay Attack Prevention

```rust
/// Replay prevention via nonce tracking
pub struct ReplayGuard {
    /// Recent action hashes (bloom filter for efficiency)
    recent_actions: BloomFilter,
    /// Sliding window of full hashes for false positive resolution
    hash_window: VecDeque<[u8; 32]>,
    /// Maximum age of tracked actions
    window_duration: Duration,
}

impl ReplayGuard {
    pub fn check_and_record(&mut self, action: &Action) -> Result<(), ReplayError> {
        let hash = action.content_hash();

        // Fast path: bloom filter check
        if self.recent_actions.might_contain(&hash) {
            // Slow path: verify against full hash window
            if self.hash_window.contains(&hash) {
                return Err(ReplayError::DuplicateAction { hash });
            }
        }

        // Record action
        self.recent_actions.insert(&hash);
        self.hash_window.push_back(hash);
        self.prune_old_entries();

        Ok(())
    }
}
```

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRUST BOUNDARY: GATE CORE                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  • E-process computation    • Min-cut evaluation                 │  │
│  │  • Conformal prediction     • Decision logic                     │  │
│  │  • Receipt signing          • Key material                       │  │
│  │                                                                   │  │
│  │  Invariants:                                                      │  │
│  │  - All inputs validated before use                               │  │
│  │  - All outputs signed before release                             │  │
│  │  - No external calls during decision                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                         (authenticated channel)                         │
│                                    │                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                    TRUST BOUNDARY: AGENT INTERFACE                      │
│                                    │                                    │
│  • Action submission (validated)   │  • Decision receipt (verified)    │
│  • Context provision (sanitized)   │  • Witness query (authenticated)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Optimization

### Identified Bottlenecks & Solutions

#### 1. E-Process History Management

**Problem**: Unbounded history growth in `EProcess.history: Vec<EValue>`

**Solution**: Ring buffer with configurable retention

```rust
pub struct EProcess {
    /// Current accumulated value (always maintained)
    current: f64,

    /// Bounded history ring buffer
    history: RingBuffer<EValueSummary>,

    /// Checkpoint for long-term audit (sampled)
    checkpoints: Vec<EProcessCheckpoint>,
}

/// Compact summary for history
pub struct EValueSummary {
    value: f32,           // Reduced precision for storage
    timestamp: u32,       // Relative to epoch
    flags: u8,            // Metadata bits
}

impl EProcess {
    const HISTORY_CAPACITY: usize = 1024;
    const CHECKPOINT_INTERVAL: usize = 100;

    pub fn update(&mut self, e: EValue) {
        // Update current (always)
        self.current = self.update_rule.apply(self.current, e.value);

        // Add to ring buffer (bounded)
        self.history.push(e.to_summary());

        // Periodic checkpoint for audit
        if self.history.len() % Self::CHECKPOINT_INTERVAL == 0 {
            self.checkpoints.push(self.checkpoint());
        }
    }
}
```

#### 2. Min-Cut Hierarchy Updates

**Problem**: Sequential iteration over all hierarchy levels

**Solution**: Lazy propagation with dirty tracking

```rust
pub struct LazyHierarchy {
    levels: Vec<HierarchyLevel>,
    /// Bitmap of levels needing update
    dirty_levels: u64,
    /// Deferred updates queue
    pending_updates: VecDeque<DeferredUpdate>,
}

impl LazyHierarchy {
    pub fn insert(&mut self, edge: Edge) {
        // Only update lowest level immediately
        self.levels[0].insert(edge);
        self.dirty_levels |= 1;

        // Defer higher level updates
        self.pending_updates.push_back(DeferredUpdate::Insert(edge));
    }

    pub fn get_cut(&mut self) -> CutValue {
        // Propagate only if needed for query
        if self.dirty_levels != 0 {
            self.propagate_lazy();
        }
        self.levels.last().unwrap().cut_value()
    }

    fn propagate_lazy(&mut self) {
        // Process only dirty levels
        while self.dirty_levels != 0 {
            let level = self.dirty_levels.trailing_zeros() as usize;
            self.update_level(level);
            self.dirty_levels &= !(1 << level);
        }
    }
}
```

#### 3. SIMD-Optimized E-Value Computation

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Batch e-value computation with SIMD
pub fn compute_mixture_evalue_simd(
    likelihoods_h1: &[f64],
    likelihoods_h0: &[f64],
    weights: &[f64],
) -> f64 {
    assert_eq!(likelihoods_h1.len(), likelihoods_h0.len());
    assert_eq!(likelihoods_h1.len(), weights.len());

    #[cfg(target_feature = "avx2")]
    unsafe {
        let mut sum = _mm256_setzero_pd();

        for i in (0..likelihoods_h1.len()).step_by(4) {
            let h1 = _mm256_loadu_pd(likelihoods_h1.as_ptr().add(i));
            let h0 = _mm256_loadu_pd(likelihoods_h0.as_ptr().add(i));
            let w = _mm256_loadu_pd(weights.as_ptr().add(i));

            let ratio = _mm256_div_pd(h1, h0);
            let weighted = _mm256_mul_pd(ratio, w);
            sum = _mm256_add_pd(sum, weighted);
        }

        // Horizontal sum
        horizontal_sum_pd(sum)
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        // Scalar fallback
        likelihoods_h1.iter()
            .zip(likelihoods_h0.iter())
            .zip(weights.iter())
            .map(|((h1, h0), w)| (h1 / h0) * w)
            .sum()
    }
}
```

#### 4. Receipt Serialization Optimization

```rust
/// Zero-copy receipt serialization
pub struct ReceiptBuffer {
    /// Pre-allocated buffer pool
    pool: BufferPool,
    /// Current buffer
    current: Buffer,
}

impl WitnessReceipt {
    /// Serialize to pre-allocated buffer (zero-copy)
    pub fn serialize_into(&self, buffer: &mut [u8]) -> Result<usize, SerializeError> {
        let mut cursor = 0;

        // Fixed-size header (no allocation)
        cursor += self.write_header(&mut buffer[cursor..])?;

        // Structural witness (fixed size)
        cursor += self.structural.write_to(&mut buffer[cursor..])?;

        // Predictive witness (bounded size)
        cursor += self.predictive.write_to(&mut buffer[cursor..])?;

        // Evidential witness (fixed size)
        cursor += self.evidential.write_to(&mut buffer[cursor..])?;

        // Hash and signature (fixed size)
        buffer[cursor..cursor + 32].copy_from_slice(&self.receipt_hash);
        cursor += 32;
        buffer[cursor..cursor + 64].copy_from_slice(&self.signature.to_bytes());
        cursor += 64;

        Ok(cursor)
    }
}
```

### Latency Budget (Revised)

| Component | Budget | Optimization | Measured p99 |
|-----------|--------|--------------|--------------|
| Min-cut query | 10ms | Lazy propagation | TBD |
| Conformal prediction | 15ms | Cached quantiles | TBD |
| E-process update | 5ms | SIMD mixture | TBD |
| Decision logic | 5ms | Short-circuit | TBD |
| Receipt generation | 10ms | Zero-copy serialize | TBD |
| Signing | 5ms | Ed25519 batch | TBD |
| **Total** | **50ms** | | |

---

## Distributed Coordination

### Multi-Agent Gate Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED COHERENCE GATE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   REGIONAL      │    │   REGIONAL      │    │   REGIONAL      │     │
│  │   GATE (Raft)   │    │   GATE (Raft)   │    │   GATE (Raft)   │     │
│  │                 │    │                 │    │                 │     │
│  │  • Local cuts   │    │  • Local cuts   │    │  • Local cuts   │     │
│  │  • Local conf   │    │  • Local conf   │    │  • Local conf   │     │
│  │  • Local e-proc │    │  • Local e-proc │    │  • Local e-proc │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │              │
│           └──────────────────────┼──────────────────────┘              │
│                                  │                                     │
│                    ┌─────────────▼─────────────┐                       │
│                    │   GLOBAL COORDINATOR      │                       │
│                    │   (DAG Consensus)         │                       │
│                    │                           │                       │
│                    │  • Cross-region cuts      │                       │
│                    │  • Aggregated e-process   │                       │
│                    │  • Boundary arbitration   │                       │
│                    └───────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Hierarchical Decision Protocol

```rust
/// Distributed gate with hierarchical coordination
pub struct DistributedGateController {
    /// Local gate for fast-path decisions
    local_gate: AnytimeGateController,

    /// Regional coordinator (Raft consensus)
    regional: RegionalCoordinator,

    /// Global coordinator (DAG consensus)
    global: GlobalCoordinator,

    /// Decision routing policy
    routing: DecisionRoutingPolicy,
}

pub enum DecisionScope {
    /// Action affects only local partition
    Local,
    /// Action crosses regional boundary
    Regional,
    /// Action has global implications
    Global,
}

impl DistributedGateController {
    pub async fn evaluate(&mut self, action: &Action, context: &Context) -> GateResult {
        // 1. Determine scope
        let scope = self.routing.classify(action, context);

        // 2. Route to appropriate level
        match scope {
            DecisionScope::Local => {
                // Fast path: local decision only
                self.local_gate.evaluate(action, context)
            }

            DecisionScope::Regional => {
                // Medium path: coordinate with regional peers
                let local_result = self.local_gate.evaluate(action, context);
                let regional_result = self.regional.coordinate(action, &local_result).await?;
                self.merge_results(local_result, regional_result)
            }

            DecisionScope::Global => {
                // Slow path: full coordination
                let local_result = self.local_gate.evaluate(action, context);
                let regional_result = self.regional.coordinate(action, &local_result).await?;
                let global_result = self.global.arbitrate(action, &regional_result).await?;
                self.merge_all_results(local_result, regional_result, global_result)
            }
        }
    }
}
```

### Distributed E-Process Aggregation

```rust
/// E-process that aggregates across distributed gates
pub struct DistributedEProcess {
    /// Local e-process
    local: EProcess,

    /// Peer e-process summaries (received via gossip)
    peer_summaries: HashMap<NodeId, EProcessSummary>,

    /// Aggregation method
    aggregation: AggregationMethod,
}

pub enum AggregationMethod {
    /// Conservative: minimum across all nodes
    Minimum,
    /// Average with confidence weighting
    WeightedAverage,
    /// Consensus-based (requires agreement)
    Consensus { threshold: f64 },
}

impl DistributedEProcess {
    /// Get aggregated e-value for distributed decision
    pub fn aggregated_value(&self) -> f64 {
        match self.aggregation {
            AggregationMethod::Minimum => {
                let local = self.local.current_value();
                let peer_min = self.peer_summaries.values()
                    .map(|s| s.current_value)
                    .fold(f64::INFINITY, f64::min);
                local.min(peer_min)
            }

            AggregationMethod::WeightedAverage => {
                let total_weight: f64 = 1.0 + self.peer_summaries.values()
                    .map(|s| s.confidence_weight)
                    .sum::<f64>();

                let weighted_sum = self.local.current_value()
                    + self.peer_summaries.values()
                        .map(|s| s.current_value * s.confidence_weight)
                        .sum::<f64>();

                weighted_sum / total_weight
            }

            AggregationMethod::Consensus { threshold } => {
                // Requires threshold fraction of nodes to agree
                let values: Vec<f64> = std::iter::once(self.local.current_value())
                    .chain(self.peer_summaries.values().map(|s| s.current_value))
                    .collect();

                // Return median if sufficient agreement, else conservative min
                if self.check_agreement(&values, threshold) {
                    statistical_median(&values)
                } else {
                    values.iter().cloned().fold(f64::INFINITY, f64::min)
                }
            }
        }
    }
}
```

### Fault Tolerance

```rust
/// Fault-tolerant gate with automatic failover
pub struct FaultTolerantGate {
    /// Primary gate
    primary: AnytimeGateController,

    /// Standby gates (hot standbys)
    standbys: Vec<AnytimeGateController>,

    /// Health monitor
    health: HealthMonitor,

    /// Failover policy
    failover: FailoverPolicy,
}

pub struct FailoverPolicy {
    /// Maximum consecutive failures before failover
    max_failures: u32,
    /// Health check interval
    check_interval: Duration,
    /// Recovery grace period
    recovery_grace: Duration,
}

impl FaultTolerantGate {
    pub async fn evaluate(&mut self, action: &Action, context: &Context) -> GateResult {
        // Try primary
        match self.try_primary(action, context).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                self.health.record_failure(&e);
            }
        }

        // Failover to standbys
        for (idx, standby) in self.standbys.iter_mut().enumerate() {
            match standby.evaluate(action, context) {
                Ok(result) => {
                    // Promote standby if primary unhealthy
                    if self.health.should_failover() {
                        self.promote_standby(idx);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    self.health.record_standby_failure(idx, &e);
                }
            }
        }

        // All gates failed - safe default
        Ok(GateResult {
            decision: GateDecision::Deny,
            reason: "All gates unavailable - failing safe".into(),
            ..Default::default()
        })
    }
}
```

### Integration with RuVector Consensus

| Consensus Layer | RuVector Module | Gate Integration |
|-----------------|-----------------|------------------|
| Regional (Raft) | `ruvector-raft` | Local cut coordination, leader-based decisions |
| Global (DAG) | `ruvector-cluster` | Cross-region boundary arbitration |
| State Sync | `ruvector-sync` | E-process summary propagation |
| Receipt Chain | `ruvector-merkle` | Distributed receipt verification |

---

## Consequences

### Benefits

1. **Formal Guarantees**: Type I error control at any stopping time
2. **Distribution Shift Robustness**: Conformal prediction adapts without retraining
3. **Computational Efficiency**: O(n^{o(1)}) update time from subpolynomial min-cut
4. **Audit Trail**: Every decision has cryptographic witness receipt
5. **Defense in Depth**: Three independent signals must concur for permit
6. **Cryptographic Integrity**: All receipts signed with Ed25519
7. **Attack Resistance**: E-value bounds, replay guards, race condition prevention
8. **Distributed Scalability**: Hierarchical coordination with regional and global tiers
9. **Fault Tolerance**: Automatic failover with safe defaults

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Computational overhead | Lazy evaluation; batch updates; SIMD optimization |
| E-value power under uncertainty | Mixture e-values for robustness |
| Graph model mismatch | Learn graph structure from trajectories |
| Threshold tuning | Adaptive thresholds via meta-learning |
| Receipt forgery | Mandatory Ed25519 signing; chain linkage |
| E-value manipulation | Input bounds; clamping with security logging |
| Race conditions | Atomic decisions with sequence numbers |
| Replay attacks | Bloom filter + sliding window guard |
| Network partitions | Hierarchical decisions; local autonomy |
| Byzantine nodes | Consensus-based aggregation; safe defaults |

### Complexity Analysis

| Operation | Current | With AVCG | Distributed AVCG |
|-----------|---------|-----------|------------------|
| Edge update | O(n^{o(1)}) | O(n^{o(1)}) | O(n^{o(1)}) + network |
| Gate evaluation | O(1) | O(k) prediction set | O(k) + O(R) regional |
| Witness generation | O(m) | O(m) amortized | O(m) + signing |
| Certificate verification | O(n) | O(n + log T) | O(n + log T) + sig verify |
| Receipt signing | N/A | O(1) Ed25519 | O(1) + HSM latency |
| Distributed consensus | N/A | N/A | O(log N) Raft |
| E-process aggregation | N/A | O(1) | O(P) peers |

Where: k = prediction set size, T = history length, R = regional peers, N = cluster size, P = peer count

## References

### Dynamic Min-Cut
1. El-Hayek, Henzinger, Li. "Deterministic and Exact Fully-dynamic Minimum Cut of Superpolylogarithmic Size in Subpolynomial Time." arXiv:2512.13105, December 2025.
2. Jin, Sun, Thorup. "Fully Dynamic Exact Minimum Cut in Subpolynomial Time." SODA 2024.

### Online Conformal Prediction
3. "Online Conformal Inference with Retrospective Adjustment for Faster Adaptation to Distribution Shift." arXiv:2511.04275, November 2025.
4. "Distribution-informed Online Conformal Prediction (COP)." December 2025.
5. "CORE: Conformal Regression under Distribution Shift via Reinforcement Learning." October 2025.

### E-Values and E-Processes
6. Ramdas, Wang. "Hypothesis Testing with E-values." Foundations and Trends in Statistics, 2025.
7. ICML 2025 Tutorial: "Game-theoretic Statistics and Sequential Anytime-Valid Inference."
8. "Sequential Randomization Tests Using e-values." arXiv:2512.04366, December 2025.

### AI Agent Control
9. "Bounded Autonomy: A Pragmatic Response to Concerns About Fully Autonomous AI Agents." XMPRO, 2025.
10. "Customizable Runtime Enforcement for Safe and Reliable LLM Agents." arXiv:2503.18666, 2025.

## Appendix: Mathematical Foundations

### E-Value Composition

For independent e-values e₁, e₂:
```
e_combined = e₁ · e₂
E[e_combined] = E[e₁] · E[e₂] ≤ 1 · 1 = 1
```

This enables **optional continuation**: evidence accumulates validly across sessions.

### Conformal Coverage

Under exchangeability or bounded distribution shift:
```
P(Y_{t+1} ∈ C_t(X_{t+1})) ≥ 1 - α - δ_t
```

Where δ_t → 0 as the algorithm adapts via retrospective adjustment.

### Anytime-Valid Stopping

For any stopping time τ (possibly data-dependent):
```
P_H₀(E_τ ≥ 1/α) ≤ α
```

This holds because E_t is a nonnegative supermartingale with E[E_0] = 1.
