# RAG Knowledge Base Builder - Crawl, Chunk, Embed & Index for AI

**Build production-ready knowledge bases for RAG (Retrieval Augmented Generation) systems in minutes.**

This Apify actor crawls websites, intelligently chunks content, generates embeddings, and exports to popular vector databases. Perfect for building AI chatbots, semantic search engines, and LLM-powered assistants with accurate, up-to-date knowledge.

## 🌟 Features

### Multi-Source Content Ingestion
- **URLs List**: Crawl from a list of starting URLs
- **XML Sitemaps**: Process entire sites via sitemap.xml
- **Actor Datasets**: Use output from other Apify actors
- **File Upload**: Process uploaded documents (coming soon)

### 4 Intelligent Chunking Strategies

1. **Fixed Size** - Consistent token-based chunks with configurable overlap
2. **Semantic** - Topic-aware chunking that preserves context
3. **Paragraph-based** - Natural paragraph boundaries
4. **Sentence-based** - Sentence-level granularity for fine-grained search

### Embedding Model Support
- OpenAI text-embedding-3-small (1536 dimensions)
- OpenAI text-embedding-3-large (3072 dimensions)
- Cohere Embed English v3 (1024 dimensions)
- Local MiniLM (384 dimensions)
- Simulated embeddings (for testing without API keys)

### Vector Database Export Formats
- **Pinecone** - Ready-to-upload format
- **Weaviate** - Class-based schema format
- **Qdrant** - Point-based format
- **Chroma** - Collection-ready format
- **AgentDB** - Optimized for [ruv.io AgentDB](https://ruv.io)
- **JSONL** - Generic format for any vector DB

### Advanced Features
- **Smart Deduplication** - Removes duplicate and highly similar chunks
- **Rich Metadata Preservation** - Source URLs, titles, timestamps, custom fields
- **Configurable Crawl Depth** - Control link following (1-5 levels)
- **CSS Selector Control** - Include/exclude specific page elements
- **Custom Metadata** - Add your own fields to all chunks
- **Chunking Statistics** - Track tokens, chunk counts, and performance

## 🚀 Quick Start

### Example 1: Build Documentation Knowledge Base

```json
{
  "urls": [
    { "url": "https://docs.python.org/3/tutorial/index.html" }
  ],
  "crawlDepth": 3,
  "chunkStrategy": "semantic",
  "chunkSize": 512,
  "chunkOverlap": 128,
  "embeddingModel": "text-embedding-3-small",
  "outputFormat": "pinecone",
  "deduplication": true
}
```

**Result**: Creates a searchable Python documentation knowledge base with ~500 semantic chunks, ready for RAG.

### Example 2: Product Catalog for E-commerce Chatbot

```json
{
  "urls": [
    { "url": "https://example-store.com/products" }
  ],
  "crawlDepth": 2,
  "chunkStrategy": "paragraph",
  "chunkSize": 384,
  "embeddingModel": "text-embedding-3-small",
  "outputFormat": "weaviate",
  "includeSelectors": [".product-description", ".product-specs"],
  "metadata": {
    "source_type": "product_catalog",
    "company": "Example Store"
  }
}
```

**Result**: Indexed product descriptions and specs, optimized for customer support chatbots.

### Example 3: Support Articles Knowledge Base

```json
{
  "urls": [
    { "url": "https://help.example.com" }
  ],
  "crawlDepth": 2,
  "chunkStrategy": "fixed_size",
  "chunkSize": 768,
  "chunkOverlap": 192,
  "embeddingModel": "cohere-embed-english-v3",
  "outputFormat": "qdrant",
  "excludeSelectors": ["nav", "footer", ".related-articles"],
  "deduplication": true,
  "similarityThreshold": 0.90
}
```

**Result**: Clean support article chunks with aggressive deduplication, perfect for customer service AI.

## 🔌 Apify MCP Integration

Integrate this actor with Claude Code using Apify MCP:

```bash
# Add to your Claude Code MCP servers
claude mcp add rag-builder -- npx -y @apify/actors-mcp-server --actors "ruv/rag-knowledge-builder"
```

Then use in Claude Code:

```javascript
// Run via MCP
mcp__apify__run_actor({
  actorId: "ruv/rag-knowledge-builder",
  input: {
    urls: [{ url: "https://docs.example.com" }],
    chunkStrategy: "semantic",
    outputFormat: "agentdb"
  }
})
```

## 📚 Tutorials

### Tutorial 1: Build Knowledge Base from Documentation Site

**Goal**: Create a searchable knowledge base from your product documentation.

**Steps**:
1. Configure input with your docs URL
2. Set `crawlDepth: 3` to get all pages
3. Use `chunkStrategy: "semantic"` for topic coherence
4. Set `chunkSize: 512` for balanced context
5. Choose embedding model (OpenAI recommended)
6. Select output format matching your vector DB
7. Run actor and download dataset

**Best Practices**:
- Use `excludeSelectors` to remove navigation, footers
- Enable `deduplication` to avoid redundancy
- Add `metadata` to track documentation version

### Tutorial 2: Create Product Catalog RAG from E-commerce Scrape

**Goal**: Build a product knowledge base for chatbot recommendations.

**Steps**:
1. First, scrape product pages with Apify Web Scraper
2. Use scraped dataset as input: `sourceType: "actor_dataset"`
3. Set `includeSelectors: [".product-description", ".specs"]`
4. Use `chunkStrategy: "paragraph"` for natural product info chunks
5. Add custom metadata: `{ category, brand, price_range }`
6. Export to vector DB format
7. Import into your RAG system

**Integration**:
```javascript
// Chain actors: Scraper → RAG Builder
const scraperRun = await apifyClient.actor('scraper').call(scrapingInput);
const ragRun = await apifyClient.actor('rag-builder').call({
  sourceType: 'actor_dataset',
  actorDatasetId: scraperRun.defaultDatasetId,
  chunkStrategy: 'paragraph'
});
```

### Tutorial 3: Index Support Articles for Chatbot

**Goal**: Power customer support chatbot with knowledge base.

**Steps**:
1. Configure with support site URL
2. Set `crawlDepth: 2` for main articles only
3. Use `excludeSelectors` to remove FAQs, contact forms
4. Set `chunkSize: 768` for detailed answers
5. Enable `deduplication` with high threshold (0.95)
6. Export to Pinecone/Weaviate/Qdrant
7. Connect to LangChain/LlamaIndex RAG pipeline

**RAG Pipeline**:
```python
# Use chunks in LangChain
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

# Load chunks from Apify dataset
chunks = load_apify_dataset(run_id)

# Create vector store
vectorstore = Pinecone.from_documents(
    chunks,
    OpenAIEmbeddings(),
    index_name="support-kb"
)

# Query for relevant context
docs = vectorstore.similarity_search(user_question, k=5)
```

### Tutorial 4: Combine Multiple Sources into Unified Knowledge Base

**Goal**: Merge documentation, blog posts, and support articles.

**Steps**:
1. Run actor separately for each source:
   - Docs: `crawlDepth: 3, chunkStrategy: semantic`
   - Blog: `crawlDepth: 2, chunkStrategy: paragraph`
   - Support: `crawlDepth: 1, chunkStrategy: fixed_size`
2. Add different metadata to each run
3. Combine datasets using Apify's dataset merge
4. Use unified dataset for comprehensive RAG

**Advanced**: Use different chunk sizes per source type for optimal retrieval.

## 🔧 Configuration Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sourceType` | enum | `urls` | Content source: urls, sitemap, actor_dataset, file_upload |
| `urls` | array | `[]` | List of start URLs to crawl |
| `crawlDepth` | integer | `1` | Maximum link depth (1-5) |
| `maxPages` | integer | `100` | Maximum pages to crawl (0 = unlimited) |
| `chunkStrategy` | enum | `fixed_size` | Chunking strategy: fixed_size, semantic, paragraph, sentence |
| `chunkSize` | integer | `512` | Target chunk size in tokens (256-2048) |
| `chunkOverlap` | integer | `128` | Token overlap between chunks (0-256) |
| `embeddingModel` | enum | `text-embedding-3-small` | Embedding model to use |
| `outputFormat` | enum | `jsonl` | Export format: jsonl, pinecone, weaviate, qdrant, chroma, agentdb |
| `includeMetadata` | boolean | `true` | Include rich metadata in chunks |
| `deduplication` | boolean | `true` | Remove duplicate chunks |
| `excludeSelectors` | array | `[nav, header, footer]` | CSS selectors to exclude |
| `includeSelectors` | array | `[]` | CSS selectors to include (empty = all) |
| `metadata` | object | `{}` | Custom metadata for all chunks |

### Chunking Strategies Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Fixed Size** | General purpose | Consistent size, predictable costs | May split mid-sentence |
| **Semantic** | Documentation, articles | Preserves topic coherence | Variable chunk sizes |
| **Paragraph** | Natural text | Natural boundaries | Size variability |
| **Sentence** | Q&A, definitions | High precision | Many small chunks |

### Recommended Configurations

**Documentation Sites**:
```json
{
  "chunkStrategy": "semantic",
  "chunkSize": 512,
  "chunkOverlap": 128,
  "crawlDepth": 3
}
```

**Product Catalogs**:
```json
{
  "chunkStrategy": "paragraph",
  "chunkSize": 384,
  "chunkOverlap": 64,
  "crawlDepth": 2
}
```

**Blog Posts**:
```json
{
  "chunkStrategy": "semantic",
  "chunkSize": 768,
  "chunkOverlap": 192,
  "crawlDepth": 1
}
```

## 📊 Output Format Examples

### JSONL (Generic)
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "Python is a high-level programming language...",
  "embedding": [0.123, -0.456, 0.789, ...],
  "tokenCount": 512,
  "metadata": {
    "source": "https://docs.python.org/3/tutorial/intro.html",
    "title": "Introduction to Python",
    "chunkIndex": 0,
    "crawledAt": "2025-12-13T10:30:00Z"
  }
}
```

### Pinecone Format
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "values": [0.123, -0.456, 0.789, ...],
  "metadata": {
    "text": "Python is a high-level programming language...",
    "source": "https://docs.python.org/3/tutorial/intro.html",
    "title": "Introduction to Python",
    "tokenCount": 512
  }
}
```

### AgentDB Format (ruv.io)
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "vector": [0.123, -0.456, 0.789, ...],
  "text": "Python is a high-level programming language...",
  "metadata": {
    "tokenCount": 512,
    "source": "https://docs.python.org/3/tutorial/intro.html",
    "title": "Introduction to Python",
    "indexed_at": "2025-12-13T10:30:00Z",
    "source": "apify-rag-builder"
  }
}
```

## 🎯 RAG Implementation Guide

### Step 1: Build Knowledge Base
Run this actor with your content sources.

### Step 2: Load into Vector Database
```python
# Pinecone example
import pinecone
from apify_client import ApifyClient

client = ApifyClient('YOUR_APIFY_TOKEN')
dataset = client.dataset('YOUR_DATASET_ID').list_items().items

pinecone.init(api_key='YOUR_PINECONE_KEY')
index = pinecone.Index('knowledge-base')

# Upload chunks
index.upsert(vectors=[
    (item['id'], item['values'], item['metadata'])
    for item in dataset
])
```

### Step 3: Implement RAG Query
```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Create retriever from vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=retriever
)

# Query knowledge base
answer = qa.run("How do I install Python packages?")
```

## 🔍 SEO Keywords

This actor is optimized for:
- **RAG (Retrieval Augmented Generation)** - Build knowledge bases for RAG systems
- **Knowledge Base** - Create searchable knowledge bases from web content
- **Embeddings** - Generate vector embeddings for semantic search
- **Vector Database** - Export to Pinecone, Weaviate, Qdrant, Chroma
- **Semantic Search** - Enable semantic search with embeddings
- **LLM** - Augment large language models with custom knowledge
- **Chatbot** - Power AI chatbots with accurate, up-to-date information
- **AI Assistant** - Build intelligent assistants with domain-specific knowledge
- **Text Chunking** - Intelligent text chunking for optimal retrieval
- **Document Indexing** - Index documents for AI-powered search
- **OpenAI Embeddings** - Generate OpenAI text-embedding-3 vectors
- **Cohere Embeddings** - Use Cohere Embed models for embeddings
- **Web Crawling for AI** - Crawl and process web content for AI applications
- **Documentation Indexing** - Index documentation sites for AI assistants
- **Knowledge Graph** - Build structured knowledge for AI systems

## 🌐 Use Cases

1. **Customer Support Chatbots** - Index support articles, FAQs, product docs
2. **Documentation Assistants** - Create AI assistants for technical documentation
3. **E-commerce Recommendations** - Build product knowledge bases for shopping assistants
4. **Research Tools** - Index academic papers, research articles
5. **Internal Knowledge Management** - Organize company wikis, policies, procedures
6. **Content Discovery** - Enable semantic search across blog posts, articles
7. **Legal Document Search** - Index contracts, case law, regulations
8. **Educational Platforms** - Build Q&A systems for course materials
9. **Healthcare Information** - Index medical knowledge bases (ensure compliance)
10. **News & Media** - Create searchable news archives with semantic search

## 🔗 Integration Examples

### LangChain Integration
```python
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from apify_client import ApifyClient

# Load chunks from Apify
client = ApifyClient('YOUR_TOKEN')
chunks = client.dataset('DATASET_ID').list_items().items

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_texts(
    [chunk['text'] for chunk in chunks],
    embeddings,
    metadatas=[chunk['metadata'] for chunk in chunks]
)
```

### LlamaIndex Integration
```python
from llama_index import VectorStoreIndex, Document
from apify_client import ApifyClient

# Load chunks
client = ApifyClient('YOUR_TOKEN')
chunks = client.dataset('DATASET_ID').list_items().items

# Create documents
documents = [
    Document(text=chunk['text'], metadata=chunk['metadata'])
    for chunk in chunks
]

# Build index
index = VectorStoreIndex.from_documents(documents)
```

### Haystack Integration
```python
from haystack.document_stores import PineconeDocumentStore
from apify_client import ApifyClient

# Initialize document store
document_store = PineconeDocumentStore(
    api_key='YOUR_PINECONE_KEY',
    index='knowledge-base'
)

# Load and write chunks
client = ApifyClient('YOUR_TOKEN')
chunks = client.dataset('DATASET_ID').list_items().items

document_store.write_documents([
    {"content": chunk['text'], "meta": chunk['metadata']}
    for chunk in chunks
])
```

## 🛠️ Advanced Configuration

### Custom Chunking Logic
For specialized chunking needs, fork this actor and modify the `ChunkingEngine` class in `src/main.js`.

### Embedding Model Customization
Add your own embedding model by extending the `EmbeddingEngine` class:

```javascript
async customEmbedding(text) {
    // Your custom embedding logic
    const response = await fetch('YOUR_EMBEDDING_API', {
        method: 'POST',
        body: JSON.stringify({ text })
    });
    return response.json().embedding;
}
```

### Output Format Extensions
Add new vector database formats by extending the `OutputFormatter` class:

```javascript
customDbFormat(chunk, embedding) {
    return {
        // Your custom format
    };
}
```

## 📈 Performance Tips

1. **Optimize Chunk Size**: 512 tokens balances context and retrieval accuracy
2. **Use Overlap**: 128-256 token overlap prevents context loss at boundaries
3. **Enable Deduplication**: Reduces storage costs and improves search quality
4. **Exclude Navigation**: Remove menus, footers with `excludeSelectors`
5. **Limit Crawl Depth**: Deep crawls may include low-quality pages
6. **Batch Processing**: Process large sites in chunks with `maxPages`
7. **Monitor Token Usage**: Track `tokenCount` to estimate embedding costs

## 💡 Best Practices

### Content Preparation
- Clean HTML with `excludeSelectors` before chunking
- Use `includeSelectors` for focused content extraction
- Add custom metadata to improve search filtering

### Chunking Strategy Selection
- **Documentation**: Semantic chunking preserves topic coherence
- **Q&A**: Sentence chunking for precise answers
- **Mixed Content**: Fixed-size chunking for consistency

### Embedding Selection
- **OpenAI text-embedding-3-small**: Best cost/performance balance
- **OpenAI text-embedding-3-large**: Highest accuracy, higher cost
- **Cohere**: Good alternative to OpenAI
- **Local**: Privacy-focused, no API costs

### Vector Database Selection
- **Pinecone**: Managed, scalable, easy setup
- **Weaviate**: Open-source, GraphQL API
- **Qdrant**: High performance, Rust-based
- **Chroma**: Lightweight, Python-first
- **AgentDB**: Optimized for ruv.io ecosystem

## 🆘 Troubleshooting

**Issue**: No chunks created
- **Solution**: Check `includeSelectors` and `excludeSelectors` configuration

**Issue**: Chunks too small/large
- **Solution**: Adjust `chunkSize` parameter (recommended: 384-768)

**Issue**: Too many duplicates
- **Solution**: Increase `similarityThreshold` or enable `deduplication`

**Issue**: Missing embeddings
- **Solution**: Verify API keys for OpenAI/Cohere models

**Issue**: Crawl stops early
- **Solution**: Increase `maxPages` or check `crawlDepth` setting

## 📞 Support & Resources

- **Actor Source**: [GitHub - ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Documentation**: [ruv.io RAG Guide](https://ruv.io/docs/rag)
- **Vector Database**: [AgentDB by ruv.io](https://ruv.io/agentdb)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruvector/issues)
- **Community**: [Discord Community](https://discord.gg/ruvio)

## 📜 License

Apache 2.0 - See LICENSE file for details

## 👨‍💻 Author

**rUv**
- Website: [https://ruv.io](https://ruv.io)
- Email: info@ruv.io
- GitHub: [@ruvnet](https://github.com/ruvnet)

---

**Built with ❤️ by rUv | Powered by [Apify](https://apify.com) | Part of the [ruvector](https://github.com/ruvnet/ruvector) ecosystem**

## 🚀 Get Started Now

1. **[Run this actor on Apify](https://console.apify.com/actors)** - Sign up for free tier
2. **Configure your knowledge sources** - Add URLs, sitemap, or datasets
3. **Choose chunking strategy** - Select from 4 intelligent methods
4. **Generate embeddings** - Use OpenAI, Cohere, or local models
5. **Export to vector DB** - Pinecone, Weaviate, Qdrant, Chroma, AgentDB
6. **Build your RAG system** - Integrate with LangChain, LlamaIndex, or custom

**🎯 From web content to production RAG in minutes!**
