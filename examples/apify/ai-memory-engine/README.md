# AI Memory Engine - Smart Database That Learns & Remembers

**Give your AI persistent memory.** Store conversations, search semantically, build knowledge graphs, and watch your AI get smarter with every interaction. The ultimate memory solution for AI agents, chatbots, and intelligent automation.

🧠 **Self-Learning** · 🔍 **Semantic Search** · 🕸️ **Knowledge Graphs** · ⚡ **Sub-millisecond** · 🔗 **LLM Agnostic** · 🔌 **One-Click Integrations**

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-blue)](https://apify.com/ruv/ai-memory-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Why AI Memory Engine?

**The Problem:** Every AI application needs memory, but building it is complex. You need embeddings, vector databases, search algorithms, and persistence - all expensive and time-consuming to set up.

**The Solution:** AI Memory Engine gives you production-ready AI memory in one click. Built on [RuvLLM](https://github.com/ruvnet/ruvector) with native SIMD acceleration, HNSW indexing, and SONA self-learning - no external dependencies required.

### What Makes This Different

| Feature | AI Memory Engine | Traditional Solutions |
|---------|------------------|----------------------|
| **Setup Time** | 1 minute | Hours to days |
| **Cost** | Pay per use ($0.001/operation) | $50-500/month subscriptions |
| **Memory Persistence** | Built-in cross-session | Requires external DB |
| **Self-Learning** | SONA neural architecture | Manual tuning |
| **Apify Integration** | One-click from any actor | Custom code required |
| **Export Options** | 6 vector DB formats | Limited |

### Key Capabilities

**27 Actions** including:
- **Core**: store, search, get, list, update, delete, clear
- **Advanced Search**: batch_search, hybrid_search, find_duplicates, deduplicate
- **AI Features**: chat, recommend, analyze, build_knowledge, learn
- **Integration**: integrate_actor, integrate_synthetic, integrate_scraper, template
- **Utilities**: natural language commands, cluster, export_vectordb, feedback

**One-Click Actor Integration** with 10+ popular scrapers:
- Google Maps, Instagram, TikTok, YouTube, Twitter, Amazon, TripAdvisor, LinkedIn, and more
- Automatically memorize any scraper results with semantic search

**6 Pre-Built Templates** for instant deployment:
- Lead Intelligence, Customer Support, Research Assistant
- Competitor Intelligence, Content Library, Product Catalog

---

## What Does This Do?

| Without AI Memory | With AI Memory Engine |
|-------------------|----------------------|
| AI forgets everything between sessions | **Remembers all conversations** |
| Same questions, same generic answers | **Personalized, context-aware responses** |
| No learning from interactions | **Gets smarter with every use** |
| Expensive vector DB subscriptions | **Built-in, no external dependencies** |
| Complex RAG setup | **One-click semantic search** |
| Manual integration with scrapers | **One-click memory from any actor** |
| Learn new tools for each vector DB | **Export to Pinecone, Weaviate, ChromaDB** |

---

## Use Cases

### 💬 Chatbots with Memory
```json
{
  "action": "chat",
  "chatMessage": "What products did I look at last week?",
  "sessionId": "customer-123"
}
```
Your chatbot remembers every conversation and provides personalized responses.

### 📚 RAG (Retrieval Augmented Generation)
```json
{
  "action": "store",
  "memories": [
    {"text": "Product X requires 220V power supply", "metadata": {"type": "specs"}},
    {"text": "Product X comes with 2-year warranty", "metadata": {"type": "warranty"}}
  ]
}
```
Then search with natural language:
```json
{
  "action": "search",
  "query": "What's the warranty on Product X?"
}
```

### 🛍️ Recommendation Engine
```json
{
  "action": "recommend",
  "query": "customer interested in home automation"
}
```
Get personalized recommendations based on learned patterns.

### 🧠 Knowledge Graph
```json
{
  "action": "build_knowledge",
  "memories": [
    "John works at TechCorp as a Senior Engineer",
    "TechCorp is located in San Francisco",
    "John manages the AI team"
  ]
}
```
Automatically extract entities and relationships.

---

## Quick Start (1 Minute)

### Try the Demo
```json
{
  "action": "demo"
}
```

This will:
1. Store sample memories (customer preferences, product info, support tickets)
2. Run semantic searches
3. Generate recommendations
4. Show you what's possible

**Output:**
```json
{
  "demo": true,
  "memoriesStored": 8,
  "sampleSearch": {
    "query": "What does the customer prefer?",
    "results": [
      {
        "text": "Customer prefers eco-friendly products and fast shipping",
        "similarity": 0.89
      }
    ]
  }
}
```

---

## Core Features

### 1. Store Memories
Add any text to your AI's memory:

```json
{
  "action": "store",
  "memories": [
    {"text": "Customer John prefers phone support over email", "metadata": {"customerId": "C001", "type": "preference"}},
    {"text": "Issue resolved by offering free shipping upgrade", "metadata": {"type": "resolution", "success": true}},
    {"text": "User mentioned they have a small apartment", "metadata": {"type": "context", "userId": "U001"}}
  ]
}
```

**Output:**
```json
{
  "stored": 3,
  "totalMemories": 3,
  "memories": [
    {"id": "mem_1702489200_0", "text": "Customer John prefers phone support..."}
  ]
}
```

### 2. Semantic Search
Find relevant memories using natural language:

```json
{
  "action": "search",
  "query": "How do we usually resolve customer complaints?",
  "topK": 5,
  "similarityThreshold": 0.6
}
```

**Output:**
```json
{
  "query": "How do we usually resolve customer complaints?",
  "resultsFound": 2,
  "results": [
    {
      "text": "Issue resolved by offering free shipping upgrade",
      "similarity": 0.82,
      "metadata": {"type": "resolution", "success": true}
    },
    {
      "text": "Customer complaint resolved by offering 20% discount",
      "similarity": 0.78
    }
  ]
}
```

### 3. Chat with Memory
Have conversations where AI remembers everything:

```json
{
  "action": "chat",
  "chatMessage": "What do we know about customer John?",
  "chatHistory": [
    {"role": "user", "content": "I need to call John today"},
    {"role": "assistant", "content": "I can help you prepare for the call."}
  ],
  "sessionId": "support-session-1",
  "provider": "gemini",
  "apiKey": "your-gemini-key"
}
```

**Output:**
```json
{
  "message": "What do we know about customer John?",
  "response": "Based on our records, John prefers phone support over email. He mentioned having a small apartment, which might be relevant for product recommendations.",
  "contextUsed": 2,
  "relevantMemories": [
    {"text": "Customer John prefers phone support over email", "similarity": 0.91}
  ]
}
```

### 4. Build Knowledge Graphs
Automatically extract entities and relationships:

```json
{
  "action": "build_knowledge",
  "memories": [
    "Apple Inc was founded by Steve Jobs in California",
    "Steve Jobs was CEO of Apple until 2011",
    "Tim Cook became CEO of Apple in 2011",
    "Apple headquarters is in Cupertino, California"
  ]
}
```

**Output:**
```json
{
  "nodesCreated": 6,
  "edgesCreated": 8,
  "topEntities": [
    {"label": "Apple", "mentions": 4},
    {"label": "Steve Jobs", "mentions": 2},
    {"label": "California", "mentions": 2}
  ]
}
```

### 5. Analyze Patterns
Get insights from your stored memories:

```json
{
  "action": "analyze"
}
```

**Output:**
```json
{
  "totalMemories": 150,
  "insights": [
    "You have 150 memories stored",
    "42 searches performed with 128 results returned",
    "Most common metadata keys: type, customerId, category",
    "Top keyword: 'shipping' (23 occurrences)"
  ],
  "topKeywords": [
    {"word": "shipping", "count": 23},
    {"word": "customer", "count": 18}
  ]
}
```

---

## Integrations

### 🔗 Integrate with Synthetic Data Generator

Generate test data and automatically memorize it:

```json
{
  "action": "integrate_synthetic",
  "integrationConfig": {
    "syntheticDataActor": "ruv/ai-synthetic-data-generator",
    "dataType": "ecommerce",
    "count": 1000,
    "memorizeFields": ["title", "description", "category"]
  }
}
```

This calls the [AI Synthetic Data Generator](https://apify.com/ruv/ai-synthetic-data-generator) and stores the results as searchable memories.

**Supported data types:**
- `ecommerce` - Product catalogs
- `jobs` - Job listings
- `real_estate` - Property listings
- `social` - Social media posts
- `stock_trading` - Market data
- `medical` - Healthcare records
- `company` - Corporate data

### 🌐 Integrate with Web Scraper

Scrape websites and build a searchable knowledge base:

```json
{
  "action": "integrate_scraper",
  "scraperConfig": {
    "urls": [
      "https://docs.example.com/getting-started",
      "https://docs.example.com/api-reference"
    ],
    "selector": "article",
    "maxPages": 50
  }
}
```

Perfect for:
- Documentation search
- Competitor analysis
- Content aggregation
- Research databases

### 🔌 One-Click Actor Integration

**Instantly memorize results from any Apify actor:**

```json
{
  "action": "integrate_actor",
  "actorId": "apify/google-maps-scraper",
  "actorConfig": {
    "runId": "latest",
    "memorizeFields": ["title", "address", "totalScore", "categoryName"],
    "limit": 500
  }
}
```

**Supported actors (one-click ready):**
| Actor | What Gets Memorized |
|-------|---------------------|
| `apify/google-maps-scraper` | Business name, address, rating, reviews |
| `apify/instagram-scraper` | Captions, hashtags, engagement |
| `apify/tiktok-scraper` | Video text, author, play count |
| `apify/youtube-scraper` | Titles, descriptions, view counts |
| `apify/twitter-scraper` | Tweets, authors, engagement |
| `apify/amazon-scraper` | Products, prices, ratings |
| `apify/tripadvisor-scraper` | Venues, ratings, locations |
| `apify/linkedin-scraper` | Profiles, headlines, companies |
| `apify/web-scraper` | Any website content |
| `apify/website-content-crawler` | Full page content |

**Example: Build a local business database:**
```json
{
  "action": "integrate_actor",
  "actorId": "apify/google-maps-scraper",
  "actorConfig": {
    "actorInput": {
      "searchStringsArray": ["restaurants near San Francisco"],
      "maxCrawledPlaces": 100
    }
  }
}
```

Then search naturally:
```json
{
  "action": "search",
  "query": "highly rated Italian restaurants with parking"
}
```

---

## Pre-Built Templates

**Get started instantly with industry templates:**

```json
{
  "action": "template",
  "template": "customer-support"
}
```

**Available templates:**
| Template | Use Case |
|----------|----------|
| `lead-intelligence` | Sales lead tracking, CRM enrichment |
| `customer-support` | FAQ knowledge base, ticket resolution |
| `research-assistant` | Academic & market research |
| `competitor-intelligence` | Market tracking, competitive analysis |
| `content-library` | Content ideas, editorial planning |
| `product-catalog` | E-commerce knowledge base |

Each template includes:
- Sample memories for immediate use
- Suggested search queries
- Optimized metadata structure

---

## Natural Language Commands

**Talk to your memory database in plain English:**

```json
{
  "action": "natural",
  "command": "remember that John prefers email communication"
}
```

**Supported commands:**
| Command | Action |
|---------|--------|
| `remember [text]` | Store a new memory |
| `forget about [topic]` | Remove related memories |
| `what do you know about [topic]` | Semantic search |
| `find [query]` | Search memories |
| `how many memories` | Get stats |
| `list memories` | Show all memories |
| `clear everything` | Reset database |
| `analyze` | Get insights |
| `find duplicates` | Detect duplicates |
| `export` | Download data |
| `help` | Show commands |

**Examples:**
```json
{"action": "natural", "command": "what do you know about customer preferences"}
{"action": "natural", "command": "forget about old pricing"}
{"action": "natural", "command": "analyze"}
```

---

## Memory Clustering

**Automatically group memories by similarity:**

```json
{
  "action": "cluster",
  "numClusters": 5
}
```

**Output:**
```json
{
  "totalMemories": 150,
  "numClusters": 5,
  "clusters": [
    {
      "id": 0,
      "label": "customer, preferences, shipping",
      "keywords": ["customer", "preferences", "shipping", "support", "email"],
      "size": 34,
      "sampleMemories": [...]
    },
    {
      "id": 1,
      "label": "product, pricing, features",
      "keywords": ["product", "pricing", "features", "specifications"],
      "size": 28,
      "sampleMemories": [...]
    }
  ]
}
```

**Use cases:**
- Discover topics in large memory collections
- Identify content gaps
- Organize knowledge automatically
- Find related memories without a query

---

## Vector DB Export

**Export to any vector database:**

```json
{
  "action": "export_vectordb",
  "vectorDbFormat": "pinecone"
}
```

**Supported formats:**
| Format | Database | Output Structure |
|--------|----------|------------------|
| `pinecone` | Pinecone | `{id, values, metadata}` |
| `weaviate` | Weaviate | `{id, vector, properties}` |
| `chromadb` | ChromaDB | `{ids, embeddings, documents, metadatas}` |
| `qdrant` | Qdrant | `{id, vector, payload}` |
| `langchain` | LangChain | `{pageContent, metadata, embeddings}` |
| `openai` | OpenAI-compatible | `{input, embedding}` |

**Example: Migrate to Pinecone**
```json
{
  "action": "export_vectordb",
  "vectorDbFormat": "pinecone"
}
```

**Output:**
```json
{
  "format": "pinecone",
  "vectors": [
    {
      "id": "mem_123",
      "values": [0.123, 0.456, ...],
      "metadata": {"text": "Customer prefers...", "type": "preference"}
    }
  ],
  "namespace": "default"
}
```

---

## Session Persistence

Keep memories across multiple runs:

```json
{
  "action": "store",
  "memories": [{"text": "New customer preference discovered"}],
  "sessionId": "my-project-memory"
}
```

Later, in another run:
```json
{
  "action": "search",
  "query": "customer preferences",
  "sessionId": "my-project-memory"
}
```

All memories from the session are automatically restored.

---

## Export & Import

### Export Memories
```json
{
  "action": "export",
  "exportFormat": "json"
}
```

Formats available:
- `json` - Full export with metadata and embeddings
- `csv` - Spreadsheet compatible
- `embeddings` - Raw vectors for ML pipelines

### Import Memories
```json
{
  "action": "import",
  "importData": {
    "memories": [
      {"text": "Imported memory 1", "metadata": {}, "embedding": [...]}
    ]
  }
}
```

---

## Configuration Options

### Embedding Models
| Model | Dimensions | Speed | Quality | API Required |
|-------|------------|-------|---------|--------------|
| `local-384` | 384 | ⚡⚡⚡ Fastest | Good | No |
| `local-768` | 768 | ⚡⚡ Fast | Better | No |
| `gemini` | 768 | ⚡ Normal | Best | Yes (free tier) |
| `openai` | 1536 | ⚡ Normal | Best | Yes |

### Distance Metrics
| Metric | Best For |
|--------|----------|
| `cosine` | Text similarity (default) |
| `euclidean` | Numerical data |
| `dot_product` | Normalized vectors |
| `manhattan` | Outlier-resistant |

### AI Providers
| Provider | Models Available |
|----------|-----------------|
| `local` | No LLM (search only) |
| `gemini` | Gemini 2.0 Flash, 1.5 Pro |
| `openrouter` | GPT-4o, Claude 3.5, Llama 3.3, 100+ models |

---

## API Integration

### Python
```python
from apify_client import ApifyClient

client = ApifyClient("your-api-token")

# Store memories
client.actor("ruv/ai-memory-engine").call(run_input={
    "action": "store",
    "memories": [{"text": "Customer feedback: Great product!", "metadata": {"type": "feedback"}}],
    "sessionId": "my-app"
})

# Search memories
result = client.actor("ruv/ai-memory-engine").call(run_input={
    "action": "search",
    "query": "What feedback have we received?",
    "sessionId": "my-app"
})

print(result["defaultDatasetId"])
```

### JavaScript
```javascript
import { ApifyClient } from 'apify-client';

const client = new ApifyClient({ token: 'your-api-token' });

// Chat with memory
const run = await client.actor('ruv/ai-memory-engine').call({
    action: 'chat',
    chatMessage: 'What do customers like about our product?',
    sessionId: 'support-bot',
    provider: 'gemini',
    apiKey: process.env.GEMINI_API_KEY
});

const { items } = await client.dataset(run.defaultDatasetId).listItems();
console.log(items[0].result.response);
```

### cURL
```bash
curl -X POST "https://api.apify.com/v2/acts/ruv~ai-memory-engine/runs?token=YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "search",
    "query": "shipping preferences",
    "sessionId": "customer-data"
  }'
```

---

## Pricing

This actor uses Pay-Per-Event pricing:

| Event | Price | Description |
|-------|-------|-------------|
| `apify-actor-start` | $0.00005 | Actor startup (per GB memory) |
| `apify-default-dataset-item` | $0.00001 | Result in dataset |
| `memory-store` | $0.001 | Store memory with RuvLLM embeddings |
| `memory-search` | $0.001 | Semantic search with HNSW |
| `chat-interaction` | $0.003 | AI chat with memory context |
| `knowledge-graph-build` | $0.005 | Build knowledge graph |
| `recommendation` | $0.002 | Get recommendations (per batch) |
| `pattern-analysis` | $0.003 | Analyze patterns with SONA |
| `memory-export` | $0.002 | Export database |
| `memory-import` | $0.002 | Import data |
| `synthetic-integration` | $0.005 | Integrate with synthetic data |
| `scraper-integration` | $0.01 | Integrate with web scraper |
| `learning-cycle` | $0.005 | Force SONA learning cycle |

**Example costs:**
- Store 1,000 memories: $1.00
- 1,000 searches: $1.00
- 100 chat interactions: $0.30
- Build knowledge graph: $0.005

---

## Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Store memory | ~2ms | 500/sec |
| Search (1000 memories) | ~5ms | 200/sec |
| Search (10000 memories) | ~20ms | 50/sec |
| Chat with context | ~100ms | 10/sec |

---

## FAQ

**Q: How long are memories stored?**
A: Memories persist as long as you use a `sessionId`. Without a session, memories exist only for the run.

**Q: Can I use this with my own LLM?**
A: Yes! Use OpenRouter to access 100+ models including GPT-4, Claude, Llama, Mistral, and more.

**Q: Is there a limit on memories?**
A: No hard limit. Performance is optimized for up to 100,000 memories per session.

**Q: Can I use this for production?**
A: Absolutely! The actor is designed for production workloads with session persistence and high throughput.

**Q: Does it work without an API key?**
A: Yes! Local embeddings and search work without any API. LLM features require Gemini or OpenRouter key.

---

## MCP Integration (Model Context Protocol)

AI Memory Engine is fully compatible with [Apify's MCP server](https://mcp.apify.com), allowing AI agents (Claude, GPT-4, etc.) to directly interact with your memory database.

### Quick Setup

**Add to Claude Code (one command):**
```bash
claude mcp add ai-memory --transport sse --url "https://mcp.apify.com/sse?token=YOUR_APIFY_TOKEN&actors=ruv/ai-memory-engine"
```

> Replace `YOUR_APIFY_TOKEN` with your [Apify API token](https://console.apify.com/settings/integrations).

**For Claude Desktop / VS Code / Cursor (config file):**
```json
{
  "mcpServers": {
    "ai-memory": {
      "transport": "sse",
      "url": "https://mcp.apify.com/sse?token=YOUR_APIFY_TOKEN&actors=ruv/ai-memory-engine"
    }
  }
}
```

**Alternative: Local MCP Server**
```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "npx",
      "args": ["-y", "@apify/actors-mcp-server", "--actors", "ruv/ai-memory-engine"],
      "env": {
        "APIFY_TOKEN": "your-apify-token"
      }
    }
  }
}
```

### What AI Agents Can Do

Once connected, AI agents can autonomously:
- **Store memories** from conversations
- **Search semantically** through stored knowledge
- **Build knowledge graphs** from unstructured data
- **Get recommendations** based on patterns
- **Manage sessions** across conversations

### Example AI Agent Workflow

```
User: "Remember that I prefer dark mode and fast responses"
Agent: [Calls AI Memory Engine store action]
       [Stores: {"text": "User prefers dark mode and fast responses", "metadata": {"type": "preference"}}]
       "Got it! I'll remember your preference for dark mode and fast responses."

User: "What are my preferences?"
Agent: [Calls AI Memory Engine search action with query "user preferences"]
       [Returns stored memory with similarity 0.94]
       "Based on our conversation, you prefer dark mode and fast responses."
```

### MCP Resources

- [Apify MCP Documentation](https://docs.apify.com/platform/integrations/mcp)
- [MCP Server Configuration](https://mcp.apify.com/)
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)

---

## Related Actors

- [AI Synthetic Data Generator](https://apify.com/ruv/ai-synthetic-data-generator) - Generate mock data for testing
- [Self-Learning AI Memory](https://apify.com/ruv/self-learning-ai-memory) - PostgreSQL-based vector storage

---

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Report Issues](https://github.com/ruvnet/ruvector/issues)

---

**Built with [RuVector](https://github.com/ruvnet/ruvector)** - High-performance vector database for AI applications.
