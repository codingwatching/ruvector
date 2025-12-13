# AI Memory Engine - Smart Database That Learns & Remembers

**Give your AI persistent memory.** Store conversations, search semantically, build knowledge graphs, and watch your AI get smarter with every interaction.

🧠 **Self-Learning** · 🔍 **Semantic Search** · 🕸️ **Knowledge Graphs** · ⚡ **Sub-millisecond** · 🔗 **LLM Agnostic**

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-blue)](https://apify.com/ruv/ai-memory-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What Does This Do?

| Without AI Memory | With AI Memory Engine |
|-------------------|----------------------|
| AI forgets everything between sessions | **Remembers all conversations** |
| Same questions, same generic answers | **Personalized, context-aware responses** |
| No learning from interactions | **Gets smarter with every use** |
| Expensive vector DB subscriptions | **Built-in, no external dependencies** |
| Complex RAG setup | **One-click semantic search** |

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
| `memory-store` | $0.001/memory | Store memories |
| `memory-search` | $0.002/search | Semantic search |
| `chat-interaction` | $0.005/chat | Chat with memory |
| `knowledge-graph-build` | $0.01/build | Build knowledge graph |
| `recommendation` | $0.001/result | Get recommendations |
| `pattern-analysis` | $0.01/analysis | Analyze patterns |
| `memory-export` | $0.005/export | Export database |
| `memory-import` | $0.005/import | Import data |
| `synthetic-integration` | $0.02/run | Integrate with synthetic data |
| `scraper-integration` | $0.02/run | Integrate with web scraper |

**Example costs:**
- Store 1,000 memories: $1.00
- 100 searches: $0.20
- 50 chat interactions: $0.25
- Build knowledge graph: $0.01

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

## Related Actors

- [AI Synthetic Data Generator](https://apify.com/ruv/ai-synthetic-data-generator) - Generate mock data for testing
- [Self-Learning AI Memory](https://apify.com/ruv/self-learning-ai-memory) - PostgreSQL-based vector storage

---

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [Report Issues](https://github.com/ruvnet/ruvector/issues)

---

**Built with [RuVector](https://github.com/ruvnet/ruvector)** - High-performance vector database for AI applications.
