# AI Synthetic Data Generator - MCP Server & Actor Integration

**Generate unlimited synthetic data** grounded in real-world patterns. **One-click integration** with 13 popular Apify web scrapers (Google Maps, Instagram, TikTok, Amazon, LinkedIn) lets you transform real scraped data into AI-ready formats for RAG systems, agent memory, and model training.

**Why grounding matters:** Pure synthetic data can drift from reality. By integrating with live Apify scrapers, your synthetic data inherits real naming conventions, price distributions, engagement patterns, and business characteristics - making your AI models and tests far more realistic.

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-blue)](https://apify.com/ruv/ai-synthetic-data-generator)
[![MCP Server](https://img.shields.io/badge/MCP-Server-purple)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.2-green)](https://github.com/ruvnet/ruvector)

## What's New in v2.2

| Feature | Description |
|---------|-------------|
| **One-Click Scraper Integration** | Ground synthetic data with real patterns from Google Maps, Instagram, TikTok, YouTube, Amazon, LinkedIn, and 7 more scrapers |
| **MCP Server** | Use as AI agent tool (Claude, GPT) via Model Context Protocol |
| **6 Use Case Templates** | Lead Intelligence, Competitor Monitor, Support RAG, and more |
| **Enhanced Grounding** | Transform scraped data into AI-ready formats that preserve real-world distributions |
| **Webhook Support** | POST results to your endpoint for async workflows |
| **Output Formats** | JSON, JSONL, CSV export options |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Scraper Integration** | One-click grounding with 13 popular Apify scrapers for realistic data |
| **MCP Server** | Integrate with Claude Code, GPT, and AI agents |
| **6 Templates** | Pre-built workflows for common use cases |
| **TRM** | 7M parameter recursive reasoning (83% on GSM8K) |
| **SONA** | 3-tier self-learning (Instant/Background/Deep) |
| **EWC++** | Pattern preservation across generations (lambda=2000) |
| **18 Data Types** | E-commerce, Bloomberg, medical, social, and more |

---

## MCP Server for AI Agents

Use as a tool in Claude Code, Claude Desktop, or any MCP-compatible AI agent via the Apify Actors MCP Server.

### Add to Claude Code

```bash
claude mcp add apify-synth -- npx -y @apify/actors-mcp-server --actors "ruv/ai-synthetic-data-generator"
```

### Add to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "apify-synth": {
      "command": "npx",
      "args": ["-y", "@apify/actors-mcp-server", "--actors", "ruv/ai-synthetic-data-generator"]
    }
  }
}
```

### Environment Setup

Set your Apify API token:

```bash
export APIFY_TOKEN=your_apify_token_here
```

### Available MCP Tools

Once connected, you get these tools in your AI agent:

| Tool | Description |
|------|-------------|
| `ruv_ai-synthetic-data-generator` | Generate synthetic data with all 18 types and integration modes |

### Example: Generate Data via MCP

```json
{
  "tool": "ruv_ai-synthetic-data-generator",
  "arguments": {
    "dataType": "ecommerce",
    "count": 100,
    "seed": "my-test"
  }
}
```

---

## One-Click Actor Integration

Transform data from popular Apify scrapers into AI-ready format.

### Supported Actors

| Actor | Category | What You Get |
|-------|----------|--------------|
| **apify/google-maps-scraper** | Local Business | Business profiles, reviews, contact info |
| **apify/instagram-scraper** | Social Media | Posts, engagement, hashtags, profiles |
| **apify/tiktok-scraper** | Social Media | Videos, engagement, music, authors |
| **apify/youtube-scraper** | Video | Videos, channels, views, comments |
| **apify/twitter-scraper** | Social Media | Tweets, engagement, profiles |
| **apify/amazon-scraper** | E-commerce | Products, prices, reviews, sellers |
| **apify/shopify-scraper** | E-commerce | Store products, variants, pricing |
| **apify/google-search-scraper** | Search | SERPs, snippets, rankings |
| **apify/website-content-crawler** | Content | Full site content, markdown |
| **apify/linkedin-scraper** | Professional | Jobs, profiles, companies |
| **apify/web-scraper** | General | Any web page data |
| **apify/cheerio-scraper** | General | Structured extraction |
| **apify/news-scraper** | News | Articles, authors, sources |

### Example: Integrate Google Maps Data

```json
{
  "mode": "integrate",
  "integrateActorId": "apify/google-maps-scraper",
  "integrateRunId": "latest",
  "memorizeFields": ["title", "description", "address", "phone", "rating", "reviews"],
  "count": 1000
}
```

---

## Use Case Templates

One-click deployment for common AI/RAG scenarios.

| Template | Use Case | Target Users | Suggested Actors |
|----------|----------|--------------|------------------|
| **lead-intelligence** | Memorize prospect data for sales | Sales, BD | Google Maps, LinkedIn |
| **competitor-monitor** | Track competitor mentions | Marketing, Strategy | Google Search, Twitter, News |
| **support-knowledge** | Customer support RAG | Support Teams | Website Crawler, Web Scraper |
| **research-assistant** | Academic/market research | Researchers | Google Search, News, Content |
| **content-library** | Content creators' reference | Creators | Instagram, TikTok, YouTube |
| **product-catalog** | E-commerce product memory | E-commerce | Amazon, Shopify, Google Maps |

### Example: Lead Intelligence Template

```json
{
  "mode": "template",
  "useTemplate": "lead-intelligence",
  "integrateActorId": "apify/google-maps-scraper",
  "integrateRunId": "latest",
  "count": 500
}
```

**Output includes:**
- Prospect ID and company name
- Contact information (phone, website, email)
- Business profile and key insights
- Lead score (1-100)
- Suggested outreach approach

---

## 18 Data Types

### Web Scraping Data
| Type | Examples | What You Get |
|------|----------|--------------|
| **E-Commerce** | Amazon, eBay, Shopify | Products, prices, reviews, sellers |
| **Social Media** | Twitter, Instagram, TikTok | Posts, engagement, hashtags, profiles |
| **Job Boards** | LinkedIn, Indeed, Glassdoor | Listings, companies, salaries |
| **Real Estate** | Zillow, Realtor, Redfin | Properties, addresses, prices |
| **Search Results** | Google, Bing, DuckDuckGo | SERPs, snippets, rankings |
| **News Sites** | CNN, BBC, TechCrunch | Articles, authors, engagement |
| **APIs** | Any REST API | JSON responses, pagination |

### Enterprise Simulators
| Type | Examples | What You Get |
|------|----------|--------------|
| **Stock Trading** | Bloomberg, E*TRADE | OHLCV, quotes, orders, analytics |
| **Medical** | Epic, Cerner | Patient records, diagnoses, billing |
| **Company** | D&B, Crunchbase | Org structure, financials |
| **Supply Chain** | SAP, Oracle | Shipments, inventory, logistics |
| **Financial** | Bank APIs | Transactions, fraud detection |
| **Bloomberg Terminal** | Full terminal | Pricing, fundamentals, news |

### Technical Data
| Type | Description |
|------|-------------|
| **Time-Series** | Stock prices, IoT sensors, metrics |
| **Web Events** | Page views, clicks, form submissions |
| **Embeddings** | Vector data for ML/RAG testing |
| **Custom Schema** | Define your own data structure |

---

## Quick Start

### Generate Synthetic Data

```json
{
  "mode": "generate",
  "dataType": "ecommerce",
  "count": 100
}
```

### Integrate Actor Data

```json
{
  "mode": "integrate",
  "integrateActorId": "apify/instagram-scraper",
  "integrateRunId": "latest",
  "count": 500
}
```

### Use a Template

```json
{
  "mode": "template",
  "useTemplate": "competitor-monitor",
  "integrateActorId": "apify/twitter-scraper",
  "count": 1000
}
```

---

## Tutorial: Build a Lead Intelligence System

### Step 1: Scrape Business Data

First, run Google Maps Scraper to collect business data:

```json
// Google Maps Scraper input
{
  "searchStringsArray": ["restaurants in San Francisco"],
  "maxCrawledPlacesPerSearch": 100
}
```

### Step 2: Transform with Lead Intelligence Template

```json
{
  "mode": "template",
  "useTemplate": "lead-intelligence",
  "integrateActorId": "apify/google-maps-scraper",
  "integrateRunId": "latest",
  "memorizeFields": ["title", "address", "phone", "website", "rating", "reviewsCount"]
}
```

### Step 3: Use in Your AI Agent

```python
from apify_client import ApifyClient

client = ApifyClient("your-token")
run = client.actor("ruv/ai-synthetic-data-generator").call(run_input={
    "mode": "template",
    "useTemplate": "lead-intelligence",
    "integrateActorId": "apify/google-maps-scraper",
    "integrateRunId": "latest"
})

leads = client.dataset(run["defaultDatasetId"]).list_items().items

# Feed into your RAG system
for lead in leads:
    add_to_vector_db(lead["data"])
```

---

## Tutorial: Build a Competitor Monitor

### Step 1: Set Up Twitter Monitoring

```json
{
  "mode": "template",
  "useTemplate": "competitor-monitor",
  "integrateActorId": "apify/twitter-scraper",
  "integrateRunId": "latest",
  "memorizeFields": ["text", "likes", "retweets", "author", "createdAt"]
}
```

### Step 2: Add News Sources

```json
{
  "mode": "integrate",
  "integrateActorId": "apify/news-scraper",
  "integrateRunId": "latest",
  "memorizeFields": ["title", "text", "source", "publishedAt"]
}
```

### Step 3: Combine in Webhook

```json
{
  "mode": "template",
  "useTemplate": "competitor-monitor",
  "webhookUrl": "https://your-app.com/api/competitor-update"
}
```

---

## Tutorial: Build a Support Knowledge Base

### Step 1: Crawl Documentation

Run Website Content Crawler on your docs:

```json
// Website Content Crawler input
{
  "startUrls": ["https://docs.example.com"],
  "maxCrawlPages": 500
}
```

### Step 2: Transform for RAG

```json
{
  "mode": "template",
  "useTemplate": "support-knowledge",
  "integrateActorId": "apify/website-content-crawler",
  "integrateRunId": "latest",
  "generateEmbeddings": true,
  "embeddingDimensions": 1536
}
```

### Step 3: Feed to Vector Database

```javascript
const { items } = await client.dataset(runId).listItems();

// Add to Pinecone, Weaviate, etc.
for (const doc of items) {
  await vectorDb.upsert({
    id: doc.id,
    values: doc.embedding,
    metadata: {
      title: doc.title,
      content: doc.content,
      url: doc.url
    }
  });
}
```

---

## Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `json` | Standard JSON array | API consumption |
| `jsonl` | JSON Lines (one per line) | Streaming, large datasets |
| `csv` | Comma-separated values | Spreadsheets, databases |

```json
{
  "dataType": "ecommerce",
  "count": 1000,
  "outputFormat": "csv"
}
```

---

## SONA Self-Learning

Enable TRM/SONA for intelligent pattern learning:

```json
{
  "dataType": "ecommerce",
  "count": 1000,
  "sonaEnabled": true,
  "ewcLambda": 2000,
  "patternThreshold": 0.7,
  "sonaLearningTiers": ["instant", "background"]
}
```

**Features:**
- **Instant Learning**: Real-time pattern recognition
- **Background Learning**: Async optimization
- **EWC++**: Preserves patterns across runs
- **Trajectory Tracking**: Optimizes generation paths

---

## Parameters Reference

### Mode Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | `generate` | Operation mode: generate, integrate, template |
| `useTemplate` | string | - | Template ID for template mode |

### Integration Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `integrateActorId` | string | - | Apify actor ID to pull data from |
| `integrateRunId` | string | `latest` | Run ID or 'latest' |
| `integrateDatasetId` | string | - | Direct dataset ID (alternative) |
| `memorizeFields` | array | [] | Fields to extract for RAG |

### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataType` | string | `ecommerce` | Type of data to generate |
| `count` | integer | 100 | Number of records (1-10,000) |
| `seed` | string | - | Random seed for reproducibility |
| `quality` | number | 0.8 | Quality level (0.1-1.0) |

### Output Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outputFormat` | string | `json` | json, jsonl, or csv |
| `webhookUrl` | string | - | URL to POST results |
| `generateEmbeddings` | boolean | false | Generate vector embeddings |
| `embeddingDimensions` | integer | 384 | Vector size (384, 768, 1536) |

### SONA / TRM Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sonaEnabled` | boolean | true | Enable TRM/SONA learning |
| `ewcLambda` | number | 2000 | Pattern preservation strength |
| `patternThreshold` | number | 0.7 | Pattern confidence (0-1) |
| `sonaLearningTiers` | array | ["instant", "background"] | Learning tiers |

---

## Integration Examples

### Python with RAG

```python
from apify_client import ApifyClient
import chromadb

client = ApifyClient("your-token")
chroma = chromadb.Client()
collection = chroma.create_collection("knowledge")

# Generate and embed data
run = client.actor("ruv/ai-synthetic-data-generator").call(run_input={
    "mode": "template",
    "useTemplate": "support-knowledge",
    "integrateActorId": "apify/website-content-crawler",
    "integrateRunId": "latest",
    "generateEmbeddings": True
})

items = client.dataset(run["defaultDatasetId"]).list_items().items

# Add to ChromaDB
for item in items:
    collection.add(
        ids=[item["id"]],
        embeddings=[item["embedding"]],
        documents=[item["content"]],
        metadatas=[{"title": item["title"], "url": item["url"]}]
    )
```

### JavaScript with LangChain

```javascript
import { ApifyClient } from 'apify-client';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

const client = new ApifyClient({ token: 'your-token' });

const run = await client.actor('ruv/ai-synthetic-data-generator').call({
    mode: 'template',
    useTemplate: 'research-assistant',
    integrateActorId: 'apify/news-scraper',
    generateEmbeddings: true
});

const { items } = await client.dataset(run.defaultDatasetId).listItems();

// Create LangChain vector store
const vectorStore = new MemoryVectorStore();
for (const item of items) {
    await vectorStore.addDocuments([{
        pageContent: item.content,
        metadata: { title: item.title, source: item.source }
    }]);
}
```

### MCP in Claude Code

```bash
# Add Apify MCP server with this actor
claude mcp add apify-synth -- npx -y @apify/actors-mcp-server --actors "ruv/ai-synthetic-data-generator"

# Set your Apify token
export APIFY_TOKEN=your_token_here

# Then in Claude Code conversation:
# "Generate 50 e-commerce products for testing"
# Claude will use the ruv_ai-synthetic-data-generator tool automatically
```

---

## Pricing (Apify Pay-per-event)

### Core Events
| Event | Price | Description |
|-------|-------|-------------|
| E-commerce Record | $0.001 | Per product generated |
| Social Media Post | $0.001 | Per post generated |
| Job Listing | $0.001 | Per listing generated |
| Search Result | $0.0005 | Per SERP entry |

### Enterprise Events
| Event | Price | Description |
|-------|-------|-------------|
| Bloomberg Record | $0.005 | Full market data |
| Medical Record | $0.003 | HIPAA-safe format |
| Company Record | $0.003 | Org structure |
| Supply Chain | $0.002 | Logistics data |

### Integration Events
| Event | Price | Description |
|-------|-------|-------------|
| Actor Integration | $0.01 | Per integration run |
| Template Execution | $0.02 | Per template use |
| Embedding Generation | $0.001 | Per vector generated |

---

## FAQ

**Q: Can I use this with Claude Code?**
A: Yes! Add the Apify MCP server: `claude mcp add apify-synth -- npx -y @apify/actors-mcp-server --actors "ruv/ai-synthetic-data-generator"`

**Q: How does actor integration work?**
A: Run any supported scraper first, then use this actor to transform the data into AI-ready format.

**Q: What's the difference between modes?**
A: `generate` creates synthetic data, `integrate` transforms scraper data, `template` uses pre-built workflows.

**Q: Does this access real websites?**
A: No. Synthetic generation creates fake data. Integration transforms data you've already scraped.

**Q: Can I combine multiple actors?**
A: Yes, run multiple integrations with the same seed, or use template mode which combines data from suggested actors.

---

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Apify Store](https://apify.com/ruv/ai-synthetic-data-generator)
- [MCP Protocol](https://modelcontextprotocol.io)
- [Report Issues](https://github.com/ruvnet/ruvector/issues)

---

**Built with RuVector** - AI-ready synthetic data with MCP server, actor integration, and TRM/SONA self-learning.
