# ruQu: Classical Nervous System for Quantum Machines

<p align="center">
  <strong>Structural Self-Awareness for Fault-Tolerant Quantum Computing</strong>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#the-paradigm-shift">The Paradigm Shift</a> |
  <a href="#architecture">Architecture</a>
</p>

---

**Created by [ruv.io](https://ruv.io) and [RuVector](https://github.com/ruvnet/ruvector)**

**SDK**: Claude-Flow | **Status**: Research & Design Phase

---

## Introduction

ruQu turns quantum machines from fragile laboratory instruments into operable production systems by providing **structural self-awareness** at microsecond timescales.

### The Problem

Quantum computers today face four critical blockers:

| Blocker | Current Reality |
|---------|-----------------|
| **Unpredictable downtime** | Full resets when anything fails |
| **Correlated failures** | Detected only after logical errors spike |
| **Control latency** | Decoders can't keep up with syndrome rates |
| **Enormous overhead** | Treating the whole device as equally fragile |

### The Solution

ruQu combines two layers working together:

1. **RuVector Memory Layer**: "Have we seen this failure shape before, and what fixed it?"
2. **Dynamic Min-Cut Gate**: "Is the system structurally coherent enough to trust action right now?"

This gives quantum machines a question they couldn't ask before:

> **"Is this system still internally consistent enough to trust action?"**

That question, answered continuously at microsecond scales, enables behaviors that were structurally impossible before.

---

## Quick Start

```rust
use ruqu::{QuantumFabric, SyndromeStream, CoherenceGate};

// Initialize the 256-tile quantum control fabric
let fabric = QuantumFabric::builder()
    .tiles(256)                    // 255 workers + TileZero
    .patch_map(surface_code_d7())  // Surface code layout
    .syndrome_buffer(1024)         // Ring buffer depth
    .build()?;

// Stream syndromes from quantum hardware
let syndromes = SyndromeStream::from_hardware(qpu_interface);

// Each cycle: update, evaluate, act
loop {
    // Ingest syndrome delta
    fabric.ingest_syndromes(&syndromes.next_batch())?;

    // Get coherence gate decision
    let decision = fabric.gate.evaluate()?;

    match decision {
        GateDecision::Safe { region_mask } => {
            // Full speed ahead on stable regions
            decoder.run_fast_path(region_mask);
        }
        GateDecision::Cautious { region_mask, lead_time } => {
            // Increase syndrome rounds only where needed
            decoder.run_conservative(region_mask);
            calibrator.schedule_targeted(region_mask, lead_time);
        }
        GateDecision::Unsafe { quarantine_mask } => {
            // Isolate fragile regions, keep rest running
            scheduler.quarantine(quarantine_mask);
            recovery.trigger_local(quarantine_mask);
        }
    }
}
```

---

## The Paradigm Shift

### What This Enables That Did Not Exist Before

#### 1. Real-Time Coherence Gate

**Before**: Decoders react after errors accumulate. Control systems assume the system is always safe to act.

**With ruQu**: The machine knows, every cycle, whether it is structurally safe to continue, to learn, or to intervene.

This enables:
- Pausing learning without halting correction
- Narrowing action to only coherent regions
- Refusing to apply risky corrections even if a decoder suggests one

**This is a new primitive: permission to act, not just what action to take.**

#### 2. Sub-Microsecond Structural Awareness

**Before**: Correlated failures are detected indirectly, often only after logical error rates spike or calibrations drift visibly.

**With ruQu**: Correlated structure is detected as it forms, not after it manifests.

This enables:
- Detecting correlated noise before it becomes logical failure
- Catching cross-qubit or cross-coupler coupling in real time
- Seeing failure modes that never show up as single-qubit errors

**This is not error correction. It is early warning.**

#### 3. Partitioned Quantum Behavior

**Before**: A quantum device is treated as one monolithic object per cycle.

**With ruQu**: The device becomes many semi-independent regions, each allowed to act differently.

This enables:
- Running aggressive schedules on stable regions
- Conservative handling on fragile regions
- Local recalibration without global interruption

**This is how biological systems survive damage. Quantum systems do not do this today.**

#### 4. Control Loops That Are Not Fixed

**Before**: Control logic is static. You choose a decoder, a schedule, a cadence.

**With ruQu**: Control becomes conditional and reflexive.

This enables:
- Decoder switching in real time
- Extra syndrome rounds only where needed
- Adaptive gate timing based on structural stability

**This is a nervous system, not a script.**

#### 5. A New Scaling Path

**Before**: Scaling requires slower cycles or more hardware just to keep up.

**With ruQu**:
- Classical effort scales with structure, not system size
- Latency stays bounded as qubits increase
- Control does not collapse under correlated noise

**This is one of the few plausible paths to large-scale fault-tolerant systems without absurd overhead.**

---

<details>
<summary><h2>Architecture</h2></summary>

### Two-Layer Classical Nervous System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUANTUM HARDWARE LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Qubits │ Couplers │ Readout Chains │ Control Lines │ Temperature   │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   SYNDROME STREAM     │
                    │   Detection Events    │
                    │   Readout Confidence  │
                    │   Timing Jitter       │
                    │   Drift Signals       │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────────────────┐
│                        ruQu FABRIC (256 Tiles)                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       TILE ZERO (Coordinator)                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  SUPERGRAPH │  │ GLOBAL CUT  │  │ PERMIT      │  │ RECEIPT    │  │   │
│  │  │  MERGE      │  │ EVALUATION  │  │ TOKEN       │  │ CHAIN      │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│            ┌────────────────────┼────────────────────┐                     │
│            ▼                    ▼                    ▼                      │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐            │
│  │ WORKER TILES     │ │ WORKER TILES     │ │ WORKER TILES     │  × 255    │
│  │ [1-85]           │ │ [86-170]         │ │ [171-255]        │            │
│  │                  │ │                  │ │                  │            │
│  │ • Patch Graph    │ │ • Patch Graph    │ │ • Patch Graph    │            │
│  │ • Syndrome Ring  │ │ • Syndrome Ring  │ │ • Syndrome Ring  │            │
│  │ • Local Min-Cut  │ │ • Local Min-Cut  │ │ • Local Min-Cut  │            │
│  │ • E-Accumulator  │ │ • E-Accumulator  │ │ • E-Accumulator  │            │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   COHERENCE GATE      │
                    │   Safe / Cautious /   │
                    │   Unsafe + Region     │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────────────────┐
│                        CONTROL LAYER                                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │ DECODER        │  │ CALIBRATOR     │  │ SCHEDULER      │                │
│  │ Fast/Slow Path │  │ Targeted Only  │  │ Region-Aware   │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Operational Graph Model

The operational graph includes:

| Node Type | Examples |
|-----------|----------|
| **Qubits** | Data qubits, ancilla qubits, flag qubits |
| **Couplers** | ZZ couplers, XY couplers, tunable couplers |
| **Readout Chains** | Resonators, amplifiers, digitizers |
| **Control Lines** | Flux lines, microwave lines, DC bias |
| **Classical** | Clocks, temperature sensors, calibration state |
| **Decoder Workers** | FPGA tiles, GPU threads, ASIC units |

### Three Stacked Filters

```
┌─────────────────────────────────────────────────────────────────┐
│                    FILTER 1: STRUCTURAL                         │
│                                                                 │
│  Workers detect local fragility (partition drift)              │
│  TileZero confirms with global cut on reduced graph            │
│                                                                 │
│  Cut Value ≥ Threshold  →  Structurally Coherent               │
│  Cut Value < Threshold  →  Boundary Forming (Quarantine)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FILTER 2: SHIFT                              │
│                                                                 │
│  Workers compute cheap nonconformity scores                    │
│  TileZero aggregates into single "shift pressure" value        │
│                                                                 │
│  Shift < Threshold  →  Distribution Stable                     │
│  Shift ≥ Threshold  →  Drift Detected (Conservative Mode)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FILTER 3: EVIDENCE                           │
│                                                                 │
│  Workers maintain running evidence accumulators                │
│  TileZero checks thresholds, can stop immediately              │
│                                                                 │
│  E-Value ≥ τ_permit  →  Accept (Permit immediately)            │
│  E-Value ≤ τ_deny    →  Reject (Deny immediately)              │
│  Otherwise           →  Continue (Gather more evidence)        │
└─────────────────────────────────────────────────────────────────┘
```

### Tile Memory Layout (64KB Budget)

```
┌─────────────────────────────────────────────────────────────────┐
│                     WORKER TILE (64KB)                          │
├─────────────────────────────────────────────────────────────────┤
│  Patch Graph (Compact)                              ~32 KB     │
│  ├── Vertices: ~512 qubits                                     │
│  ├── Edges: ~2048 couplings                                    │
│  └── Adjacency + Weights                                       │
├─────────────────────────────────────────────────────────────────┤
│  Syndrome Ring Buffer                               ~16 KB     │
│  ├── 1024 syndrome rounds                                      │
│  └── Detection events + timing                                 │
├─────────────────────────────────────────────────────────────────┤
│  Evidence Accumulator                               ~4 KB      │
│  ├── Hypothesis states                                         │
│  ├── Log e-values                                              │
│  └── Sliding window statistics                                 │
├─────────────────────────────────────────────────────────────────┤
│  Local Min-Cut State                                ~8 KB      │
│  ├── Boundary candidates (top-k)                               │
│  ├── Cut value cache                                           │
│  └── Witness fragments                                         │
├─────────────────────────────────────────────────────────────────┤
│  Control / Scratch                                  ~4 KB      │
│  ├── Delta buffer (64 deltas)                                  │
│  ├── Report scratch                                            │
│  └── Stack                                                     │
└─────────────────────────────────────────────────────────────────┘
```

</details>

---

<details>
<summary><h2>Technical Deep Dive</h2></summary>

### Complexity Guarantees

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Edge Update | O(n^{o(1)}) | O(m log n) | Subpolynomial, amortized |
| Min-Cut Query | O(1) | Cached | Pre-computed per tick |
| Syndrome Ingest | O(1) | O(ring size) | Ring buffer append |
| Local Cut Eval | O(patch size) | O(1) | Per-worker |
| Global Merge | O(num_workers) | O(1) | TileZero only |
| Gate Decision | O(1) | O(1) | Three threshold checks |
| Witness Fragment | O(boundary) | O(k) | Top-k edges only |

### Latency Budget

| Component | Target | Critical Path |
|-----------|--------|---------------|
| Syndrome Ingest | < 100 ns | Ring buffer append |
| Worker Tick | < 500 ns | Local cut + report |
| Report Merge | < 1 μs | 255 reports → supergraph |
| Global Cut | < 500 ns | Reduced graph query |
| Gate Decision | < 100 ns | Three comparisons |
| Permit Signing | < 1 μs | Ed25519 signature |
| **Total** | **< 4 μs** | **End-to-end** |

### Syndrome Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYNDROME PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Hardware  ──┬──►  Syndrome    ──►  Worker    ──►  TileZero   │
│   Interface   │     Dispatcher      Tiles          Arbiter     │
│               │                                                 │
│   ┌───────────▼───────────┐                                    │
│   │ Per-Cycle Input:      │                                    │
│   │ • Detection events    │                                    │
│   │ • Readout confidence  │                                    │
│   │ • Timing jitter       │                                    │
│   │ • Hardware health     │                                    │
│   │ • Drift signals       │                                    │
│   └───────────────────────┘                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Integration with Existing Decoders

ruQu does not replace decoders—it tells them when and how hard to work.

```
┌─────────────────────────────────────────────────────────────────┐
│                 DECODER INTEGRATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ruQu Gate Decision                                           │
│         │                                                       │
│         ├── Safe → Fast Decoder Path (union-find, O(n α(n)))   │
│         │                                                       │
│         ├── Cautious → Slow Decoder Path (BP+OSD, MWPM)        │
│         │              + Extra syndrome rounds                  │
│         │              + Targeted pulse adjustments             │
│         │                                                       │
│         └── Unsafe → Quarantine region                          │
│                      + Local recalibration                      │
│                      + Reroute workloads                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### FPGA/ASIC Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARDWARE TARGET                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FPGA Path (Development):                                     │
│   • AMD VU19P or similar                                       │
│   • 256 soft tiles in fabric                                   │
│   • < 1 μs latency achievable                                  │
│   • ~10W power budget                                          │
│                                                                 │
│   ASIC Path (Production):                                      │
│   • Custom 256-tile fabric                                     │
│   • < 250 ns latency target                                    │
│   • ~100mW power (cryo-compatible)                             │
│   • 4K operation possible                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### RuVector Integration Points

| RuVector Component | ruQu Usage |
|--------------------|------------|
| `SubpolynomialMinCut` | O(n^{o(1)}) dynamic cut tracking |
| `WitnessTree` | Cryptographic cut certificates |
| `CutCertificate` | Audit-ready decision proofs |
| `CompactGraph` | Memory-efficient patch storage |
| `EvidenceAccumulator` | Anytime-valid hypothesis testing |
| `TileZero` | Central arbiter pattern |
| `PermitToken` | Signed action authorization |
| `ReceiptLog` | Hash-chained audit trail |

</details>

---

<details>
<summary><h2>Tutorials & Examples</h2></summary>

### Tutorial 1: Basic Coherence Monitoring

```rust
use ruqu::{QuantumFabric, CoherenceMonitor};

// Create a coherence monitor for a surface code
let monitor = CoherenceMonitor::new()
    .code(SurfaceCode::new(7))  // Distance-7 surface code
    .threshold(0.01)            // 1% error rate threshold
    .build()?;

// Stream syndromes and monitor coherence
for round in 0.. {
    let syndromes = hardware.read_syndromes()?;
    monitor.ingest(&syndromes)?;

    let coherence = monitor.coherence_score();
    println!("Round {}: Coherence = {:.3}", round, coherence);

    if coherence < 0.5 {
        println!("  WARNING: Coherence degrading!");
    }
}
```

### Tutorial 2: Correlated Error Detection

```rust
use ruqu::{CorrelationDetector, AlertLevel};

// Detect correlated errors before they become logical failures
let detector = CorrelationDetector::new()
    .window_size(100)           // Look at last 100 syndrome rounds
    .baseline_period(1000)      // Establish baseline over 1000 rounds
    .alert_threshold(3.0)       // Alert at 3σ deviation
    .build()?;

loop {
    let syndromes = hardware.read_syndromes()?;

    match detector.analyze(&syndromes)? {
        AlertLevel::Normal => {
            // Business as usual
        }
        AlertLevel::Elevated { lead_time, region } => {
            println!("Correlation forming in region {:?}", region);
            println!("Lead time: {} cycles before expected failure", lead_time);

            // Proactive mitigation
            decoder.switch_to_conservative(region);
            calibrator.schedule_check(region, lead_time / 2);
        }
        AlertLevel::Critical { region } => {
            println!("CRITICAL: Correlated failure imminent in {:?}", region);
            scheduler.quarantine(region);
        }
    }
}
```

### Tutorial 3: Partitioned Operation

```rust
use ruqu::{PartitionedControl, RegionPolicy};

// Different policies for different regions
let control = PartitionedControl::new()
    .add_region("stable", RegionPolicy::Aggressive {
        syndrome_rounds: 1,
        decoder: DecoderMode::Fast,
    })
    .add_region("fragile", RegionPolicy::Conservative {
        syndrome_rounds: 3,
        decoder: DecoderMode::Accurate,
        recalibrate_every: 100,
    })
    .add_region("recovering", RegionPolicy::Quarantine {
        max_isolation_time: Duration::from_millis(10),
        recovery_procedure: RecoveryMode::LocalReset,
    })
    .build()?;

loop {
    let gate_decision = fabric.gate.evaluate()?;

    // Classify regions based on coherence
    let classification = control.classify_regions(&gate_decision)?;

    for (region, policy) in classification {
        match policy {
            RegionPolicy::Aggressive { .. } => {
                // Fast path
                decoder.run_region_fast(region)?;
            }
            RegionPolicy::Conservative { .. } => {
                // Extra care
                decoder.run_region_careful(region)?;
            }
            RegionPolicy::Quarantine { .. } => {
                // Isolation
                scheduler.isolate_region(region)?;
            }
        }
    }
}
```

### Tutorial 4: Audit Trail and Compliance

```rust
use ruqu::{AuditLog, ComplianceReport};

// Every gate decision is logged with cryptographic proof
let audit = AuditLog::new()
    .storage(AuditStorage::Persistent("/var/log/ruqu/"))
    .retention(Duration::from_days(90))
    .build()?;

// During operation
loop {
    let decision = fabric.gate.evaluate()?;
    let receipt = fabric.gate.receipt()?;

    // Receipt contains:
    // - Cryptographic hash of decision inputs
    // - Signed decision output
    // - Link to previous receipt (hash chain)
    audit.append(receipt)?;
}

// Generate compliance report
let report = ComplianceReport::generate(&audit)
    .time_range(last_24_hours)
    .include_decision_distribution()
    .include_latency_percentiles()
    .include_quarantine_events()
    .build()?;

report.export_pdf("compliance_report.pdf")?;
```

### Example: Integration with Stim Simulation

```rust
use ruqu::simulation::{StimIntegration, SimulatedHardware};
use stim::Circuit;

// Load a surface code circuit
let circuit = Circuit::from_file("surface_d7.stim")?;

// Create simulated hardware
let hardware = SimulatedHardware::new()
    .circuit(circuit)
    .noise_model(DepolarizingNoise::new(0.001))
    .inject_correlated_burst_at(cycle: 5000, duration: 100)
    .build()?;

// Run ruQu against simulation
let fabric = QuantumFabric::builder()
    .hardware(hardware)
    .build()?;

// Measure three key metrics
let mut metrics = Metrics::new();

for cycle in 0..10000 {
    hardware.advance_cycle()?;
    let decision = fabric.gate.evaluate()?;

    metrics.record_gate_latency(decision.latency);
    metrics.record_decision(decision);

    if cycle == 5000 {
        // Correlated burst starts - measure lead time
        metrics.start_burst_detection_timer();
    }
}

// Report results
println!("Gate Latency p99: {:?}", metrics.gate_latency_p99());
println!("Burst Detection Lead Time: {:?}", metrics.burst_lead_time());
println!("Logical Error Rate vs Overhead: {:?}", metrics.pareto_curve());
```

</details>

---

<details>
<summary><h2>Super Advanced Usage Scenarios</h2></summary>

### Scenario 1: Multi-Chip Federated Control

For systems with multiple quantum processors connected via quantum links:

```rust
use ruqu::federation::{FederatedFabric, ChipTopology, CrossChipCoherence};

// Define multi-chip topology
let topology = ChipTopology::new()
    .add_chip("chip_a", QuantumChip::surface_17())
    .add_chip("chip_b", QuantumChip::surface_17())
    .add_chip("chip_c", QuantumChip::surface_17())
    .add_link("chip_a", "chip_b", QuantumLink::optical())
    .add_link("chip_b", "chip_c", QuantumLink::optical())
    .build()?;

// Create federated fabric
let fabric = FederatedFabric::new(topology)
    .cross_chip_coherence(CrossChipCoherence::Hierarchical {
        local_gate_latency: Duration::from_micros(4),
        global_gate_latency: Duration::from_millis(1),
    })
    .build()?;

// Federated operation
loop {
    // Local gates run at chip level
    let local_decisions = fabric.evaluate_local()?;

    // Global gate runs less frequently
    if cycle % 1000 == 0 {
        let global_decision = fabric.evaluate_global()?;

        if global_decision.cross_chip_coherence < 0.5 {
            // Cross-chip correlation detected
            fabric.isolate_link("chip_a", "chip_b")?;
        }
    }
}
```

### Scenario 2: Learning from Historical Patterns

Using RuVector's memory layer to learn from past failure patterns:

```rust
use ruqu::learning::{PatternMemory, MitigationPlaybook};

// Initialize pattern memory with historical data
let memory = PatternMemory::new()
    .load_historical("patterns.ruvec")?
    .embedding_dim(128)
    .similarity_threshold(0.85)
    .build()?;

// Create playbook of known mitigations
let playbook = MitigationPlaybook::new()
    .add("cosmic_ray_burst", Mitigation::pause_then_reset(Duration::from_millis(5)))
    .add("coupler_drift", Mitigation::targeted_recalibration())
    .add("readout_crosstalk", Mitigation::adjust_readout_timing())
    .build()?;

loop {
    let syndromes = hardware.read_syndromes()?;
    let hardware_telemetry = hardware.read_telemetry()?;

    // Embed current state
    let current_state = memory.embed(&syndromes, &hardware_telemetry)?;

    // Find similar historical patterns
    let matches = memory.find_similar(current_state, k: 5)?;

    if let Some(best_match) = matches.first() {
        if best_match.similarity > 0.9 {
            // High-confidence match
            let pattern_name = best_match.pattern_name();
            let mitigation = playbook.get(pattern_name)?;

            println!("Recognized pattern: {} (similarity: {:.2})",
                     pattern_name, best_match.similarity);
            println!("Applying mitigation: {:?}", mitigation);

            mitigation.apply(&mut hardware)?;
        }
    }

    // Learn from outcomes
    if logical_error_detected {
        memory.record_failure(current_state, syndromes)?;
    } else {
        memory.record_success(current_state, syndromes)?;
    }
}
```

### Scenario 3: Adaptive Threshold Optimization

Automatically tuning gate thresholds based on observed performance:

```rust
use ruqu::optimization::{ThresholdOptimizer, ObjectiveFunction};

// Define optimization objective
let objective = ObjectiveFunction::pareto()
    .minimize("logical_error_rate")
    .minimize("syndrome_overhead")
    .minimize("decoder_compute")
    .constraint("max_latency", Duration::from_micros(10))
    .build()?;

// Create optimizer
let mut optimizer = ThresholdOptimizer::new()
    .objective(objective)
    .search_space(ThresholdSpace {
        min_cut: (1.0, 20.0),
        max_shift: (0.1, 1.0),
        tau_deny: (0.001, 0.1),
        tau_permit: (10.0, 1000.0),
    })
    .algorithm(BayesianOptimization::new())
    .build()?;

// Optimization loop
for epoch in 0..100 {
    let thresholds = optimizer.suggest()?;
    fabric.gate.set_thresholds(thresholds)?;

    // Run for evaluation period
    let metrics = run_evaluation_period(&mut fabric, Duration::from_secs(60))?;

    // Report results
    optimizer.observe(thresholds, metrics)?;

    println!("Epoch {}: Best Pareto front = {:?}", epoch, optimizer.pareto_front());
}

// Apply best thresholds
let optimal = optimizer.best_compromise()?;
fabric.gate.set_thresholds(optimal)?;
```

### Scenario 4: Integration with External Calibration Systems

Coordinating ruQu decisions with automated calibration:

```rust
use ruqu::calibration::{CalibrationCoordinator, CalibrationRequest};

let coordinator = CalibrationCoordinator::new()
    .calibration_system(external_calibration_api)
    .max_concurrent_calibrations(4)
    .min_stable_period(Duration::from_secs(10))
    .build()?;

loop {
    let decision = fabric.gate.evaluate()?;

    // Check if calibration should be triggered
    if let Some(request) = coordinator.should_calibrate(&decision)? {
        match request {
            CalibrationRequest::Targeted { qubits, priority } => {
                // Calibrate only specific qubits
                coordinator.request_calibration(qubits, priority)?;
            }
            CalibrationRequest::Regional { region, priority } => {
                // Calibrate a region
                coordinator.request_regional_calibration(region, priority)?;
            }
            CalibrationRequest::Full { reason } => {
                // Full device calibration needed
                println!("Full calibration requested: {}", reason);
                coordinator.request_full_calibration()?;
            }
        }
    }

    // Adjust gate behavior based on active calibrations
    let active_calibrations = coordinator.active_calibrations()?;
    for region in active_calibrations {
        fabric.gate.set_region_mode(region, GateMode::Cautious)?;
    }
}
```

### Scenario 5: Real-Time Visualization and Monitoring

```rust
use ruqu::visualization::{DashboardServer, MetricStream};

// Start dashboard server
let dashboard = DashboardServer::new()
    .port(8080)
    .update_rate(Duration::from_millis(100))
    .build()?;

// Stream metrics to dashboard
let metrics = MetricStream::new();

tokio::spawn(async move {
    loop {
        let decision = fabric.gate.evaluate()?;

        metrics.record("gate_latency_ns", decision.latency.as_nanos());
        metrics.record("coherence_score", decision.coherence);
        metrics.record("active_regions", decision.active_region_count);
        metrics.record("quarantined_regions", decision.quarantine_count);

        for (region_id, region) in decision.regions() {
            metrics.record_region(region_id, "cut_value", region.cut_value);
            metrics.record_region(region_id, "shift_pressure", region.shift);
            metrics.record_region(region_id, "e_value", region.evidence);
        }

        dashboard.push(metrics.snapshot())?;
    }
});

// Dashboard now available at http://localhost:8080
// Shows:
// - Real-time coherence heatmap
// - Gate decision distribution
// - Latency histograms
// - Region status grid
// - Correlated error detection timeline
```

</details>

---

## What to Demo First

A minimal proof that investors and engineers will respect:

### Demo Setup

1. **Simulation stream** using Stim
2. **Baseline decoder** (PyMatching or union-find on CPU)
3. **ruQu chip** runs the gate and partition only
4. **Controller** switches between fast and slow decode based on risk token

### Three Plots to Show

1. **Latency distribution of the gate kernel**
   - Target: p99 < 4μs
   - Show bounded latency under worst-case event bursts

2. **Correlated event detection lead time**
   - At fixed false alarm rate
   - Show cycles of warning before logical failure

3. **Logical error vs overhead curve**
   - Pareto front: same error rate with less overhead
   - OR: lower error rate at same overhead

**If those three move in the right direction together, you are not selling a better decoder. You are selling operability.**

---

## Metrics to Prove

| Metric | What It Proves |
|--------|----------------|
| **Time to recover** | Localized recovery beats full reset |
| **Jobs completed without full reset** | Operability at scale |
| **Lead time before logical failure** | Early warning works |
| **Logical error vs overhead Pareto** | Selective overhead beats uniform |
| **Gate latency tail** | Bounded real-time performance |

---

## Implementation Blueprint

### v0.1: Structural Coherence + Witness Receipt

1. Define fixed patch map of the lattice
2. Each chip tile owns one patch plus small overlap band
3. Each tile maintains tiny ring buffer of syndromes + incremental graph deltas
4. Each tile outputs local fracture score + boundary summary every cycle
5. Coordinator tile merges boundaries into global risk + region mask
6. Controller chooses from small action set based on mask

**Skip the fancy parts initially**. Start with structural coherence + witness receipt. Add shift and anytime-evidence once the loop is stable.

---

## References

### Documentation

- [ADR-001: ruQu Architecture Decision Record](docs/adr/ADR-001-ruqu-architecture.md)
- [DDD-001: Domain-Driven Design - Coherence Gate](docs/ddd/DDD-001-coherence-gate-domain.md)
- [DDD-002: Domain-Driven Design - Syndrome Processing](docs/ddd/DDD-002-syndrome-processing-domain.md)
- [Simulation Integration Guide](docs/SIMULATION-INTEGRATION.md) - Using Stim, stim-rs, and Rust quantum simulators

### External Resources

- [El-Hayek, Henzinger, Li. "Dynamic Min-Cut with Subpolynomial Update Time." arXiv:2512.13105, 2025](https://arxiv.org/abs/2512.13105)
- [Google Quantum AI. "Quantum error correction below the surface code threshold." Nature, 2024](https://www.nature.com/articles/s41586-024-08449-y)
- [Riverlane. "Collision Clustering Decoder." Nature Communications, 2025](https://www.nature.com/articles/s41467-024-54738-z)
- [Stim: High-performance Quantum Error Correction Simulator](https://github.com/quantumlib/Stim)

---

## License

MIT OR Apache-2.0

---

<p align="center">
  <em>"The question is not 'what action to take.' The question is 'permission to act.'"</em>
</p>

<p align="center">
  <strong>This is structural self-awareness. And those are the things that quietly define eras.</strong>
</p>
