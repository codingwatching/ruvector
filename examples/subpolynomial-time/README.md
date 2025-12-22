# Dynamic Minimum Cut Demo

**Find the weakest link in any network — instantly, even as it changes.**

[![GitHub](https://img.shields.io/badge/GitHub-ruvnet%2Fruvector-blue?logo=github)](https://github.com/ruvnet/ruvector)
[![ruv.io](https://img.shields.io/badge/ruv.io-AI%20Infrastructure-orange)](https://ruv.io)

---

## What Is This?

This demo shows how to use the `ruvector-mincut` library to analyze network connectivity in real-time. It answers questions like:

- **"How many connections must fail before my network splits?"**
- **"Which links are critical to keep my system connected?"**
- **"How does adding/removing a connection affect network strength?"**

## The Simple Explanation

Imagine you have a network of computers, roads, or social connections. The **minimum cut** is the smallest number of connections you'd need to remove to split the network in two.

```
Before: A connected network          After: Network split by removing 1 edge

    [A]---[B]                              [A]   [B]
     |     |                                |     |
    [C]---[D]                              [C]   [D]

    Min cut = 1 (one weak link)            Now disconnected!
```

The lower the minimum cut, the more vulnerable your network.

## Why This Matters

| Application | Question Answered |
|-------------|-------------------|
| **Server Infrastructure** | Which servers are single points of failure? |
| **Social Networks** | What's the weakest link between communities? |
| **Supply Chain** | Which suppliers would cripple operations if lost? |
| **Road Networks** | Which roads would divide the city if closed? |

## What Makes This Special

Traditional algorithms must **reanalyze the entire network** whenever something changes. For large networks, that's too slow for real-time use.

This implementation uses the [December 2025 breakthrough algorithm](https://arxiv.org/abs/2512.13105):
- **Updates in subpolynomial time** — much faster than starting over
- **Handles changes instantly** — add or remove connections without delay
- **Exact results** — no approximations or guesses

## Quick Start

### Run the Demo

```bash
# From this directory
cargo run --release

# Or from repository root
cargo run --release --example subpolynomial-time-mincut-demo
```

### Basic Code Example

```rust
use ruvector_mincut::prelude::*;

// Create a simple network
let mut network = MinCutBuilder::new()
    .exact()
    .with_edges(vec![
        (1, 2, 1.0),  // Connection from node 1 to 2
        (2, 3, 1.0),  // Connection from node 2 to 3
        (3, 1, 1.0),  // Connection from node 3 to 1
    ])
    .build()?;

// How vulnerable is this network?
println!("Minimum cut: {}", network.min_cut_value());
// Output: 2 (need to remove 2 connections to split)

// Add a new connection
network.insert_edge(3, 4, 1.0)?;

// Remove a connection
network.delete_edge(1, 2)?;

// Get the cut instantly (no recomputation needed!)
println!("New minimum cut: {}", network.min_cut_value());
```

## What the Demo Shows

### Demo 1: Basic Usage
Create a simple graph and query its minimum cut.

### Demo 2: Dynamic Updates
Add and remove edges while tracking how connectivity changes.

### Demo 3: Exact vs Approximate
Compare perfect accuracy with faster approximate mode.

### Demo 4: Real-time Monitoring
Set up alerts when network strength drops below thresholds.

### Demo 5: Network Resilience
Analyze which connections are critical and what happens if they fail.

### Demo 6: Performance Scaling
See how the algorithm performs on different network sizes.

## Integration with RuVector Ecosystem

### With ruvector-graph (Graph Database)

```rust
use ruvector_graph::GraphDB;
use ruvector_mincut::integration::RuVectorGraphAnalyzer;

// Analyze connectivity of vectors in a similarity graph
let graph_db = GraphDB::open("my_vectors.db")?;
let analyzer = RuVectorGraphAnalyzer::from_graph(&graph_db);

let connectivity = analyzer.min_cut();
let communities = analyzer.detect_communities();
```

### With ruvector-postgres (PostgreSQL Extension)

```sql
-- Install the extension
CREATE EXTENSION ruvector_mincut;

-- Create a graph from a table
SELECT ruvector_mincut_build('my_edges_table', 'source', 'target', 'weight');

-- Query minimum cut
SELECT ruvector_mincut_value();

-- Dynamic update
SELECT ruvector_mincut_insert_edge(10, 20, 1.0);
SELECT ruvector_mincut_delete_edge(5, 6);

-- Get critical edges
SELECT * FROM ruvector_mincut_cut_edges();
```

## Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Query minimum cut | **Instant** | O(1) - no matter how big the network |
| Add connection | **Very fast** | O(n^{o(1)}) - subpolynomial time |
| Remove connection | **Very fast** | O(n^{o(1)}) - includes finding replacement |
| Build from scratch | Fast | O(m log n) - one-time setup |

## Configuration Options

### Exact Mode (Default)
Perfect accuracy, suitable for most use cases:
```rust
let network = MinCutBuilder::new()
    .exact()
    .build()?;
```

### Approximate Mode
Trade small accuracy loss for speed on very large networks:
```rust
let network = MinCutBuilder::new()
    .approximate(0.1)  // 10% approximation (1.1x at most)
    .build()?;
```

### Real-time Monitoring
Get alerts when connectivity changes:
```rust
let monitor = MonitorBuilder::new()
    .threshold_below(2.0, "warning")   // Alert when min cut < 2
    .threshold_below(1.0, "critical")  // Alert when min cut < 1
    .on_event_type(EventType::Disconnected, "alert", |_| {
        println!("Network is split!");
    })
    .build();
```

## Project Structure

```
examples/subpolynomial-time/
├── Cargo.toml          # Dependencies
├── README.md           # This file
└── src/
    └── main.rs         # Demo code (6 demonstrations)
```

## Further Reading

- **Paper**: [Deterministic Exact Subpolynomial-Time Algorithms for Global Minimum Cut](https://arxiv.org/abs/2512.13105) (December 2025)
- **Crate Docs**: [docs.rs/ruvector-mincut](https://docs.rs/ruvector-mincut)
- **ruv.io**: [AI Infrastructure Platform](https://ruv.io)

## License

Part of the [RuVector project](https://github.com/ruvnet/ruvector). See LICENSE for details.
