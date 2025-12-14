<p align="center">
  <img src="https://raw.githubusercontent.com/ruvnet/ruvector/main/assets/synth-logo.png" alt="Agentic Synth" width="140" height="140" />
</p>

<h1 align="center">Agentic Synth</h1>

<p align="center">
  <strong>Large-Scale Simulation Engine with Self-Learning AI</strong>
</p>

<p align="center">
  <a href="https://apify.com/ruv/ai-synthetic-data-generator"><img src="https://img.shields.io/badge/Apify-Actor-FF9900?style=for-the-badge&logo=apify&logoColor=white" alt="Apify Actor"></a>
  <a href="https://github.com/ruvnet/ruvector"><img src="https://img.shields.io/badge/RuVector-Powered-4A90D9?style=for-the-badge" alt="RuVector"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/10,000_records-53ms-brightgreen?style=flat-square" alt="10K in 53ms">
  <img src="https://img.shields.io/badge/188K_records/sec-purple?style=flat-square" alt="188K/sec">
  <img src="https://img.shields.io/badge/18_data_types-orange?style=flat-square" alt="18 Data Types">
  <img src="https://img.shields.io/badge/SONA_Self--Learning-blue?style=flat-square" alt="Self-Learning">
  <img src="https://img.shields.io/badge/version-2.5-green?style=flat-square" alt="Version">
</p>

---

## Overview

Agentic Synth is a self-learning simulation engine that generates realistic synthetic data at scale. Unlike static generators that produce random values, this engine learns from every run—extracting patterns from your data to improve quality over time. Generate 100 records in 1ms or 10,000 records in 53ms across 18 different domains.

**Self-Learning Neural Architecture (SONA)** powers the engine with three learning tiers:

| Tier | What It Does | Example |
|------|--------------|---------|
| **Instant** | Learns patterns during generation | "Electronics products cluster around $200-500" |
| **Background** | Trains on batch completion | "Bloomberg buy ratings correlate with sector performance" |
| **Deep** | Cross-session pattern retention | "Medical diagnoses improve ICD-10 code accuracy over time" |

The engine extracts data-type specific patterns: price distributions correlate with product categories, analyst recommendations match rating distributions, medical billing codes align with procedures, and supply chain lead times reflect regional logistics.

**Key Capabilities:**
- **150x faster** than JavaScript generators (Rust/WASM powered by RuVector)
- **5 embedding models** for semantic search (all-MiniLM-L6-v2, bge-small, all-mpnet, e5-small, gte-small)
- **Real brand matching** per category (Samsung for Electronics, Nike for Sports, LEGO for Toys)
- **Consistent data logic** (stock counts match availability, shipping prices match free flags)
- **Neural pattern training** per data type with EWC++ memory protection

For developers, it eliminates rate limits and captchas. For enterprises, it provides compliant test data without legal risks. For AI teams, it generates unlimited training data with semantic embeddings.

The simulation mode streams data in batches—push 50 records every 2 seconds for real-time pipeline testing. Seeds ensure reproducible results for CI/CD. Pairs with [AI Memory Engine](https://apify.com/ruv/ai-memory-engine) for semantic search and RAG applications.

**Benchmarks:** 100 records in 1ms | 1,000 in 7ms | 5,000 in 34ms | 10,000 in 53ms (188,679 records/sec)

---

## Quick Start

```json
{
  "dataType": "demo",
  "count": 100
}
```

Run this to get 100 sample records across all types: products, social posts, job listings, and news.

---

## Common Uses

| You Want To... | Data Type | What You Get |
|----------------|-----------|--------------|
| Test an e-commerce scraper | `ecommerce` | Products, prices, reviews, sellers |
| Build a trading dashboard | `bloomberg` | Stock quotes, fundamentals, analytics |
| Train a healthcare AI | `medical` | Patient records, diagnoses, billing |
| Test a job board | `jobs` | Listings, salaries, company info |
| Prototype logistics software | `supply_chain` | Shipments, inventory, tracking |
| Load test a social app | `social` | Posts, likes, comments, followers |

---

## Why Synthetic Data?

| Real Data Problem | Agentic Synth Solution |
|-------------------|------------------------|
| Websites block after 100 requests | Generate 10,000 records instantly |
| Captchas and anti-bot detection | No restrictions |
| Bloomberg terminal: $24,000/year | Similar data for pennies |
| HIPAA/GDPR compliance issues | 100% synthetic = 100% legal |
| Inconsistent formats | Clean, predictable JSON |

---

## 18 Data Types

### For Web Developers & Scrapers
| Type | What You Get | Example Use |
|------|--------------|-------------|
| `ecommerce` | Products, prices, reviews, sellers | Test your Amazon scraper |
| `social` | Posts, likes, comments, profiles | Build a Twitter dashboard |
| `jobs` | Listings, salaries, companies | Test Indeed clone |
| `real_estate` | Properties, addresses, prices | Zillow-like app testing |
| `search_results` | SERPs, snippets, rankings | SEO tool development |
| `news` | Articles, authors, engagement | News aggregator testing |
| `api_response` | JSON responses, pagination | Mock backend APIs |

### For Enterprise & Finance
| Type | What You Get | Example Use |
|------|--------------|-------------|
| `bloomberg` | Full terminal data: quotes, fundamentals, analytics | Trading system testing |
| `stock_trading` | OHLCV, orders, market data | Backtest trading algorithms |
| `financial` | Transactions, accounts, fraud data | Banking app development |
| `company` | Org structure, financials, leadership | CRM/sales tool testing |
| `supply_chain` | Shipments, inventory, suppliers | Logistics system testing |

### For Healthcare & Research
| Type | What You Get | Example Use |
|------|--------------|-------------|
| `medical` | Patient records, diagnoses, billing | EHR system testing |
| `timeseries` | Time-stamped metrics, trends | IoT/sensor dashboards |
| `embeddings` | Vector data (384-768 dimensions) | RAG/ML model training |
| `structured` | Your custom schema | Any specialized need |
| `events` | Page views, clicks, form data | Analytics testing |
| `demo` | Mix of all types | Quick exploration |

---

## Practical Examples

### E-Commerce Products

```json
{ "dataType": "ecommerce", "count": 1000 }
```

**You get:**
```json
{
  "url": "https://example-store.com/products/premium-headphones-123",
  "title": "TechPro Premium Headphones",
  "price": 149.99,
  "originalPrice": 199.99,
  "currency": "USD",
  "category": "Electronics",
  "brand": "TechPro",
  "rating": 4.5,
  "reviewCount": 2847,
  "inStock": true,
  "seller": {
    "name": "Seller847",
    "rating": 4.8,
    "totalSales": 15420
  }
}
```

### Bloomberg Terminal Data

```json
{ "dataType": "bloomberg", "count": 500 }
```

**You get:**
```json
{
  "terminalId": "BBG1734012345678",
  "security": {
    "ticker": "MSFT",
    "name": "Microsoft Corp",
    "assetClass": "equity",
    "sector": "Technology"
  },
  "pricing": {
    "last": 378.50,
    "bid": 378.45,
    "ask": 378.55,
    "volume": 18500000
  },
  "fundamentals": {
    "marketCap": "2800B",
    "peRatio": 35.2,
    "roe": 38.5
  },
  "analytics": {
    "beta": 0.92,
    "sharpeRatio": 1.45,
    "volatility": 22.5
  },
  "consensus": {
    "recommendation": "buy",
    "targetPrice": 420.00,
    "numAnalysts": 42
  }
}
```

### Medical Records

```json
{ "dataType": "medical", "count": 200 }
```

**You get:**
```json
{
  "recordId": "MED1734012345678",
  "patient": {
    "id": "PAT847291",
    "age": 45,
    "gender": "F",
    "bloodType": "O+"
  },
  "diagnosis": {
    "primary": "Hypertension",
    "icdCode": "I10.9",
    "severity": "moderate"
  },
  "vitals": {
    "bloodPressure": "145/92",
    "heartRate": 78,
    "oxygenSaturation": 98
  },
  "billing": {
    "insurer": "Blue Cross",
    "totalCharges": 2450,
    "claimStatus": "approved"
  }
}
```

### Supply Chain & Logistics

```json
{ "dataType": "supply_chain", "count": 300 }
```

**You get:**
```json
{
  "shipmentId": "SHP1734012345678",
  "order": {
    "orderId": "ORD8472910",
    "priority": "express",
    "status": "in_transit"
  },
  "product": {
    "sku": "SKU-847291",
    "name": "Electronics Item 482",
    "quantity": 500,
    "unitPrice": 45.99
  },
  "supplier": {
    "name": "Johnson Supply Co",
    "country": "China",
    "leadTime": 21
  },
  "logistics": {
    "carrier": "Maersk",
    "mode": "sea",
    "eta": "2024-12-20"
  }
}
```

### Custom Schema (Any Structure)

```json
{
  "dataType": "structured",
  "count": 1000,
  "schema": {
    "userId": "string",
    "action": "string (click, view, purchase)",
    "timestamp": "date",
    "value": "number (1-100)"
  }
}
```

Define any fields you need—the generator builds data to match your structure.

---

## How Self-Learning Works

The AI learns from the data it generates. Turn it on with one parameter:

```json
{ "dataType": "ecommerce", "count": 1000, "sonaEnabled": true }
```

### What It Learns

| Data Type | Patterns the AI Discovers |
|-----------|---------------------------|
| **E-Commerce** | High-priced items get more reviews. Electronics have lower stock. 4.5★ ratings cluster around $100-200. |
| **Bloomberg** | Tech stocks move together. Volume spikes after earnings. Beta varies by sector. |
| **Medical** | Hypertension → BP meds, not antibiotics. Age correlates with certain diagnoses. Billing matches procedures. |
| **Supply Chain** | China = 21 day lead time. Heavy items go by sea. Q4 has electronics spikes. |

### Quality Improves Over Time

| Records Generated | Data Realism |
|-------------------|--------------|
| First 100 | Good structure, random values |
| After 1,000 | Patterns emerge (price ↔ quality) |
| After 10,000 | Industry-accurate correlations |
| After 100,000 | Near production-quality data |

### Advanced Options

For fine-tuning (most users don't need these):

| Parameter | What It Does | Default |
|-----------|--------------|---------|
| `sonaEnabled` | Turn learning on/off | `true` |
| `ewcLambda` | How strongly to remember patterns | `2000` |
| `patternThreshold` | Confidence needed to save a pattern | `0.7` |
| `sonaLearningTiers` | Which learning speeds to use | `["instant", "background"]` |

---

## Simulation Mode

For testing real-time data pipelines, enable simulation mode to stream data in batches:

```json
{
  "dataType": "ecommerce",
  "count": 1000,
  "simulationMode": true,
  "batchSize": 50,
  "delayBetweenBatches": 2000
}
```

This pushes 50 products every 2 seconds—perfect for testing scrapers that poll for updates, streaming dashboards, or message queue consumers.

### Simulation Use Cases

| Scenario | Configuration |
|----------|---------------|
| Test real-time dashboard | `batchSize: 10`, `delay: 1000` (10 records/sec) |
| Load test message queue | `batchSize: 100`, `delay: 500` (200 records/sec) |
| Simulate daily data feed | `batchSize: 1000`, `delay: 60000` (1000/min) |

---

## Reproducible Results

Use seeds to get identical data every run:

```json
{
  "dataType": "ecommerce",
  "count": 100,
  "seed": "my-test-suite-v1"
}
```

Same seed = same data. Perfect for CI/CD pipelines where tests need consistent fixtures.

---

## Parameters Reference

### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataType` | string | `ecommerce` | Type of data to generate |
| `count` | integer | 100 | Number of records (1-10,000) |
| `seed` | string | - | Random seed for reproducibility |
| `quality` | number | 0.8 | Quality level (0.1-1.0) |

### AI Provider Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | string | `gemini` | AI provider (gemini, openrouter) |
| `apiKey` | string | - | API key for AI enhancement |
| `model` | string | `gemini-2.0-flash-exp` | AI model to use |

### SONA / TRM Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sonaEnabled` | boolean | true | Enable TRM/SONA self-learning |
| `ewcLambda` | number | 2000 | EWC++ pattern preservation strength |
| `patternThreshold` | number | 0.7 | Pattern recognition confidence (0-1) |
| `sonaLearningTiers` | array | ["instant", "background"] | Learning tiers to enable |

### Simulation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulationMode` | boolean | false | Enable batch simulation |
| `batchSize` | integer | 100 | Records per batch |
| `delayBetweenBatches` | integer | 0 | Delay between batches (ms) |

---

## Integration Examples

### Python

```python
from apify_client import ApifyClient

client = ApifyClient("your-api-token")
run = client.actor("ruv/ai-synthetic-data-generator").call(run_input={
    "dataType": "bloomberg",
    "count": 1000,
    "sonaEnabled": True
})
data = client.dataset(run["defaultDatasetId"]).list_items().items

# Train your AI model with realistic data
for record in data:
    train_model(record["data"])
```

### JavaScript

```javascript
import { ApifyClient } from 'apify-client';

const client = new ApifyClient({ token: 'your-api-token' });
const run = await client.actor('ruv/ai-synthetic-data-generator').call({
    dataType: 'medical',
    count: 500,
    sonaEnabled: true
});
const { items } = await client.dataset(run.defaultDatasetId).listItems();

// Test your healthcare app
items.forEach(record => processPatientRecord(record.data));
```

### CI/CD Pipeline

```yaml
# .github/workflows/test-with-synthetic-data.yml
jobs:
  test:
    steps:
      - name: Generate test data
        run: |
          curl -X POST "https://api.apify.com/v2/acts/ruv~ai-synthetic-data-generator/runs?token=$APIFY_TOKEN" \
            -H "Content-Type: application/json" \
            -d '{"dataType": "ecommerce", "count": 100, "seed": "ci-test-v1"}'

      - name: Run tests with synthetic data
        run: npm test
```

### AI Memory Engine Integration

Generate synthetic data and store it directly in [AI Memory Engine](https://apify.com/ruv/ai-memory-engine) for semantic search and RAG:

```python
from apify_client import ApifyClient

client = ApifyClient("your-api-token")

# Step 1: Generate synthetic product data
synth_run = client.actor("ruv/ai-synthetic-data-generator").call(run_input={
    "dataType": "ecommerce",
    "count": 1000,
    "generateEmbeddings": True
})
products = client.dataset(synth_run["defaultDatasetId"]).list_items().items

# Step 2: Store in AI Memory Engine for semantic search
memory_run = client.actor("ruv/ai-memory-engine").call(run_input={
    "action": "store",
    "memories": [{"text": p["data"]["title"], "metadata": p["data"]} for p in products]
})

# Step 3: Now search semantically
search_run = client.actor("ruv/ai-memory-engine").call(run_input={
    "action": "search",
    "query": "wireless headphones under $100"
})
```

**Use Cases:**
- Generate training data → Store in memory → Build RAG chatbots
- Create product catalogs → Enable semantic product search
- Simulate customer conversations → Train support AI
- Generate medical records → Build healthcare knowledge base

---

## Performance

### Benchmark Results (Rust/WASM Engine)

| Records | Time | Records/sec | Use Case |
|---------|------|-------------|----------|
| 100 | 1ms | 100,000 | Unit tests, quick validation |
| 1,000 | 7ms | 142,857 | Integration tests |
| 5,000 | 34ms | 147,058 | Load testing |
| 10,000 | 53ms | 188,679 | Full stress tests |

### By Data Type (1,000 records each)

| Data Type | Time | Notes |
|-----------|------|-------|
| E-commerce | 7ms | Products, prices, reviews |
| Bloomberg | 12ms | Full terminal with analytics |
| Medical | 15ms | HIPAA-format records with billing |
| Supply Chain | 11ms | Shipments, inventory, logistics |

### vs Traditional Data Generation

| Approach | 10,000 Records | Memory | Setup |
|----------|----------------|--------|-------|
| **Agentic Synth** | **53ms** | 256MB | Zero config |
| Faker.js | ~800ms | 512MB | npm install |
| Python Faker | ~1,200ms | 1GB | pip install |
| Database fixtures | ~5,000ms | 2GB+ | Schema + seeds |
| Manual JSON | Hours | - | Hand-crafted |

**150x faster than JavaScript generators** — Powered by RuVector's native Rust/WASM engine with SIMD optimizations

---

## Pricing (Apify Pay-per-event)

### Core Events
| Event | Price | Description |
|-------|-------|-------------|
| Actor Start | $0.00005 | Per event (1 event per GB memory) |
| E-commerce Record | $0.001 | Per product generated |
| Social Media Post | $0.001 | Per post generated |
| Job Listing | $0.001 | Per listing generated |
| Real Estate Listing | $0.001 | Per property generated |
| News Article | $0.001 | Per article generated |
| Search Result | $0.0005 | Per SERP entry |
| API Response | $0.0005 | Per mock response |

### Enterprise Events
| Event | Price | Description |
|-------|-------|-------------|
| Bloomberg Terminal Record | $0.005 | Full market data with analytics |
| Medical Record | $0.003 | Patient records with HIPAA-safe format |
| Company Record | $0.003 | Org structure, financials, leadership |
| Supply Chain Record | $0.002 | Shipments, inventory, logistics |
| Financial Transaction | $0.002 | Banking transactions, fraud data |
| Stock Trading Record | $0.002 | OHLCV, quotes, market analytics |

### Simulation Events
| Event | Price | Description |
|-------|-------|-------------|
| Simulation Session | $0.10 | Long-running simulation with batches |
| Simulation Batch | $0.01 | Per batch pushed with delay |
| AI-Enhanced Record | $0.01 | Per record with AI generation |

**Example Cost:**
- 1,000 e-commerce products: ~$1.00
- 500 Bloomberg records: ~$2.50
- 200 medical records: ~$0.60

---

## FAQ

**Q: Does this scrape real websites?**
A: No. All data is synthetically generated. No real websites are accessed.

**Q: Is the data realistic?**
A: Yes. Data structures match real websites/APIs exactly. TRM/SONA improves quality through pattern learning.

**Q: What is SONA?**
A: Self-Optimizing Neural Architecture - learns from generation patterns to improve data quality over time.

**Q: Can I use custom schemas?**
A: Yes. Use `dataType: "structured"` with your own schema definition.

**Q: Is there a free tier?**
A: Yes. Algorithmic generation works without any API keys.

---

## Links

- [Agentic Synth on Apify](https://apify.com/ruv/ai-synthetic-data-generator) - This actor
- [AI Memory Engine](https://apify.com/ruv/ai-memory-engine) - Companion actor for persistent AI memory
- [GitHub Repository](https://github.com/ruvnet/ruvector) - Source code and documentation
- [Report Issues](https://github.com/ruvnet/ruvector/issues)

---

**Built with [RuVector](https://github.com/ruvnet/ruvector)** — High-performance synthetic data generation with SONA self-learning. Pairs with [AI Memory Engine](https://apify.com/ruv/ai-memory-engine) for complete AI data solutions.
