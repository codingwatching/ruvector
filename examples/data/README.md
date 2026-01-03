# RuVector Dataset Discovery Framework

Comprehensive examples demonstrating RuVector's capabilities for novel discovery across world-scale datasets.

## The Discovery Thesis

RuVector's unique combination of **vector memory**, **graph structures**, and **dynamic minimum cut algorithms** enables discoveries that most analysis tools miss:

- **Emerging patterns before they have names**: Detect topic splits and merges as cut boundaries shift over time
- **Non-obvious cross-domain bridges**: Find small "connector" subgraphs where disciplines quietly start citing each other
- **Causal leverage maps**: Link funders, labs, venues, and downstream citations to spot high-impact intervention points
- **Regime shifts in time series**: Use coherence breaks to flag fundamental changes in system behavior

## Recommended Datasets

### 1. OpenAlex (Research Intelligence)
**Best for**: Emerging field detection, cross-discipline bridges, funding-to-output causality

OpenAlex is the cleanest "already a graph" public dataset at world scale:
- 250M+ works, 90M+ authors, 100K+ institutions
- Native graph structure (works, authors, institutions, topics, funders)
- Designed for bulk download + API access

```rust
use ruvector_data::openalex::{OpenAlexIngester, TopicBoundaryDetector};

// Detect emerging research frontiers
let detector = TopicBoundaryDetector::new(mincut_engine, vector_db);
let frontiers = detector.find_emerging_fields(
    start_year: 2020,
    end_year: 2024,
    min_coherence_shift: 0.3
).await?;
```

### 2. NOAA + NASA Earthdata (Climate Intelligence)
**Best for**: Regime shift detection, anomaly prediction, economic risk modeling

Best public source for time series with real economic pull:
- NOAA: Weather observations, forecasts, historical climate data
- NASA Earthdata: Satellite imagery, remote sensing, global datasets

```rust
use ruvector_data::climate::{ClimateIngester, RegimeShiftDetector};

// Detect climate regime shifts
let detector = RegimeShiftDetector::new(mincut_engine, vector_db);
let shifts = detector.detect_regime_changes(
    sensor_network: "noaa_ghcn",
    window_days: 90,
    coherence_threshold: 0.7
).await?;
```

### 3. SEC EDGAR (Financial Intelligence)
**Best for**: Corporate risk signals, peer divergence detection, narrative drift analysis

High-value structured financial data:
- XBRL financial statements (standardized accounting data)
- 10-K/10-Q filings (narrative + numbers)
- Peer group relationships

```rust
use ruvector_data::edgar::{EdgarIngester, CoherenceWatch};

// Detect peer group coherence breaks
let watch = CoherenceWatch::new(mincut_engine, vector_db);
let alerts = watch.detect_peer_divergence(
    sector: "technology",
    peer_group_size: 20,
    narrative_weight: 0.4
).await?;
```

## Directory Structure

```
examples/data/
├── README.md                 # This file
├── framework/               # Core discovery framework
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Core traits and types
│       ├── ingester.rs      # Data ingestion pipeline
│       ├── coherence.rs     # Coherence signal computation
│       └── discovery.rs     # Novel pattern detection
├── openalex/               # OpenAlex integration
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Main library
│       ├── schema.rs        # OpenAlex entity schemas
│       ├── ingester.rs      # Bulk download + streaming
│       └── frontier.rs      # Research frontier detection
├── climate/                # NOAA/NASA integration
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs           # Main library
│       ├── noaa.rs          # NOAA data ingestion
│       ├── nasa.rs          # NASA Earthdata ingestion
│       └── regime.rs        # Regime shift detection
└── edgar/                  # SEC EDGAR integration
    ├── Cargo.toml
    └── src/
        ├── lib.rs           # Main library
        ├── xbrl.rs          # XBRL parsing
        ├── filings.rs       # 10-K/10-Q ingestion
        └── coherence.rs     # Financial coherence watch
```

## Quick Start

```bash
# Build all data examples
cd examples/data
cargo build --workspace

# Run OpenAlex demo
cargo run --package ruvector-data-openalex --example frontier_radar

# Run Climate demo
cargo run --package ruvector-data-climate --example regime_detector

# Run EDGAR demo
cargo run --package ruvector-data-edgar --example coherence_watch
```

## Scoring Rule for Dataset Selection

When choosing which dataset to prioritize:

**Score = Impact × Feasibility × Structural Fit**

| Dataset | Impact | Feasibility | Structural Fit | Total |
|---------|--------|-------------|----------------|-------|
| OpenAlex | High (research funding) | High (bulk API) | Very High (native graph) | ★★★★★ |
| NOAA+NASA | High (insurance/energy) | Medium (format variety) | High (time series → graph) | ★★★★☆ |
| SEC EDGAR | Very High (finance) | Medium (XBRL parsing) | High (peer networks) | ★★★★☆ |

## Demo Ideas (Achievable in Weeks)

1. **Research Frontier Radar** (OpenAlex)
   - Live map of topics where boundary motion predicts breakout areas
   - Metric: Min-cut value over topic citation graph

2. **Climate Regime Shift Detector** (NOAA+NASA)
   - Coherence breaks for weather extremes and grid stress
   - Metric: Spectral gap changes in sensor correlation network

3. **Financial Coherence Watch** (EDGAR)
   - Peer group divergence alerts when fundamentals ≠ narrative
   - Metric: Cross-embedding distance between XBRL features and filing text

## References

- [OpenAlex Documentation](https://docs.openalex.org/)
- [NOAA Open Data Dissemination](https://www.noaa.gov/information-technology/open-data-dissemination)
- [NASA Earthdata](https://earthdata.nasa.gov/)
- [SEC EDGAR](https://www.sec.gov/edgar)
- [XBRL Financial Datasets](https://www.sec.gov/dera/data/financial-statement-data-sets)

## License

MIT OR Apache-2.0
