# Agent Training Data Factory - AI Agent Memory & RAG Dataset Generator

**Generate high-quality training datasets for AI agents** including conversations, Q&A pairs, tool calls, reasoning chains, memory patterns, and embeddings. Perfect for RAG systems, LLM fine-tuning, and agent development.

Built by [rUv](https://ruv.io) | Part of the [ruvector](https://github.com/ruvnet/ruvector) ecosystem

## Introduction

The **Agent Training Data Factory** is a specialized Apify actor that generates synthetic training data grounded in real-world patterns. Whether you're building conversational AI, developing RAG systems, or fine-tuning large language models, this tool provides production-ready datasets that accelerate your development cycle.

**Key Innovation**: Unlike pure synthetic data generators, this actor can ground its outputs in real scraped data from any Apify actor, ensuring your training data reflects actual patterns from your domain.

Integration with the [agentic-flow](https://www.npmjs.com/package/agentic-flow) npm package enables seamless orchestration of multi-agent systems using the generated training data.

## Features

### 6 Dataset Types

1. **Conversations** - Multi-turn dialogue datasets for conversational AI
2. **Q&A Pairs** - Question-answer pairs for training and fine-tuning
3. **Tool Calls** - Function calling sequences with realistic schemas
4. **Reasoning Chains** - Chain-of-Thought and Tree-of-Thought patterns
5. **Memory Patterns** - Short-term, long-term, episodic, semantic, and procedural memory
6. **Embeddings** - Vector embeddings for RAG systems

### Domain-Specific Generation

- **Customer Support** - Help desk, troubleshooting, account management
- **Coding** - Programming Q&A, debugging, code review
- **Research** - Academic questions, scientific explanations
- **Sales** - Product inquiries, objection handling
- **Healthcare** - Medical questions, patient support
- **Education** - Learning materials, tutoring conversations
- **Custom Domains** - Define your own domain for specialized datasets

### Complexity Levels

- **Simple** - Basic single-step examples
- **Moderate** - Multi-step with some reasoning
- **Complex** - Advanced multi-step with context
- **Expert** - Sophisticated reasoning and edge cases

### Advanced Capabilities

- Ground synthetic data in real scraped patterns from Apify actors
- HuggingFace-compatible output formats
- Vector embeddings for RAG using OpenAI or local models
- Tool call simulation with realistic API schemas
- Memory pattern generation for agent state management
- Reproducible generation with random seeds
- Custom prompts for guided generation

## Quick Start

### 1. Basic Conversation Dataset

```javascript
{
  "datasetType": "conversations",
  "domain": "customer_support",
  "count": 100,
  "complexity": "moderate",
  "outputFormat": "jsonl"
}
```

### 2. Coding Q&A with Embeddings

```javascript
{
  "datasetType": "qa_pairs",
  "domain": "coding",
  "count": 500,
  "complexity": "complex",
  "generateEmbeddings": true,
  "embeddingModel": "text-embedding-3-small",
  "outputFormat": "huggingface"
}
```

### 3. Grounded Training Data

```javascript
{
  "datasetType": "conversations",
  "domain": "research",
  "count": 200,
  "complexity": "expert",
  "groundingActorId": "apify/web-scraper",
  "outputFormat": "jsonl"
}
```

## Apify MCP Integration

Use this actor directly in Claude Desktop with the Apify MCP server:

```bash
# Install Apify MCP server
claude mcp add agent-factory -- npx -y @apify/actors-mcp-server --actors "ruv/agent-training-factory"
```

Then in Claude:

```
Generate 100 customer support conversation examples with moderate complexity
```

The MCP integration enables:
- Natural language actor invocation
- Automatic dataset retrieval
- Seamless workflow integration
- Real-time progress monitoring

## Detailed Tutorials

### Tutorial 1: Generate Customer Support Conversation Dataset

**Objective**: Create realistic customer support conversations for training a support chatbot.

**Steps**:

1. Navigate to the actor in Apify Console
2. Configure input:

```json
{
  "datasetType": "conversations",
  "domain": "customer_support",
  "count": 500,
  "complexity": "moderate",
  "includeMetadata": true,
  "outputFormat": "jsonl"
}
```

3. Run the actor
4. Download the dataset from the Output tab
5. Use for fine-tuning or RAG system population

**Expected Output**:

```json
{
  "type": "conversation",
  "messages": [
    {
      "role": "user",
      "content": "I can't log into my account. It keeps saying invalid password."
    },
    {
      "role": "assistant",
      "content": "I understand you're having trouble logging in. Let me help you with that. First, could you confirm the email address associated with your account?"
    }
  ],
  "metadata": {
    "id": "conv_000001",
    "domain": "customer_support",
    "complexity": "moderate",
    "turn_count": 2,
    "generated_at": "2025-12-13T10:30:00Z",
    "grounded": false
  }
}
```

### Tutorial 2: Create Coding Assistant Q&A Pairs

**Objective**: Build a dataset of programming questions and expert answers.

**Steps**:

1. Configure for coding domain with higher complexity:

```json
{
  "datasetType": "qa_pairs",
  "domain": "coding",
  "count": 1000,
  "complexity": "complex",
  "includeMetadata": true,
  "outputFormat": "huggingface"
}
```

2. Run and download
3. Load into HuggingFace datasets:

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='dataset.jsonl')
```

4. Fine-tune your model:

```python
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("your-base-model")
trainer = Trainer(model=model, train_dataset=dataset)
trainer.train()
```

### Tutorial 3: Build RAG Embedding Dataset from Scraped Data

**Objective**: Create vector embeddings grounded in real documentation for a RAG system.

**Steps**:

1. First, scrape your documentation using Apify Web Scraper
2. Note the dataset ID from the scraper run
3. Configure the training factory:

```json
{
  "datasetType": "embeddings",
  "domain": "research",
  "count": 1000,
  "complexity": "expert",
  "groundingDatasetId": "your-scraper-dataset-id",
  "generateEmbeddings": true,
  "embeddingModel": "text-embedding-3-large",
  "openaiApiKey": "your-openai-key",
  "outputFormat": "jsonl"
}
```

4. Run the actor
5. Ingest embeddings into your vector database:

```python
import pinecone
import json

# Initialize Pinecone
pinecone.init(api_key="your-key")
index = pinecone.Index("your-index")

# Load embeddings
with open('dataset.jsonl') as f:
    for line in f:
        item = json.loads(line)
        index.upsert([(
            item['metadata']['id'],
            item['vector'],
            {'text': item['text']}
        )])
```

### Tutorial 4: Train Agent with Tool Call Sequences

**Objective**: Generate realistic function calling patterns for agent training.

**Steps**:

1. Configure for tool calls with expert complexity:

```json
{
  "datasetType": "tool_calls",
  "domain": "coding",
  "count": 500,
  "complexity": "expert",
  "includeMetadata": true,
  "outputFormat": "jsonl"
}
```

2. Example output shows realistic multi-step tool usage:

```json
{
  "type": "tool_call_sequence",
  "calls": [
    {
      "id": "call_0",
      "type": "function",
      "function": {
        "name": "execute_code",
        "arguments": "{\"language\":\"python\",\"code\":\"print('Hello')\"}"
      },
      "result": {
        "output": "Hello\n",
        "exit_code": 0
      }
    }
  ],
  "complexity": "expert"
}
```

3. Use for fine-tuning function-calling models
4. Validate agent behavior against generated patterns

## Input Schema Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `datasetType` | enum | `conversations` | Type of dataset to generate |
| `domain` | string | `customer_support` | Domain or subject area |
| `count` | integer | 100 | Number of examples to generate |
| `complexity` | enum | `moderate` | Complexity level of examples |

### Enhancement Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `includeMetadata` | boolean | `true` | Include metadata fields |
| `groundingActorId` | string | `null` | Apify actor to fetch grounding data from |
| `groundingDatasetId` | string | `null` | Existing dataset ID for grounding |
| `customPrompts` | array | `null` | Custom prompts to guide generation |
| `seed` | integer | `null` | Random seed for reproducibility |

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outputFormat` | enum | `jsonl` | Output format (jsonl, parquet, huggingface) |
| `generateEmbeddings` | boolean | `false` | Generate vector embeddings |
| `embeddingModel` | enum | `text-embedding-3-small` | Model for embeddings |
| `openaiApiKey` | string | `null` | OpenAI API key for real embeddings |

## Output Formats

### JSONL Format

Standard JSON Lines format, one record per line:

```jsonl
{"type":"conversation","messages":[...],"metadata":{...}}
{"type":"qa_pair","question":"...","answer":"...","metadata":{...}}
```

### Parquet Format

Columnar storage format optimized for analytics:

```
dataset.parquet
├── type
├── messages.0.role
├── messages.0.content
└── metadata.id
```

### HuggingFace Format

Compatible with HuggingFace datasets library:

```json
{
  "conversations": [...],
  "_hf_metadata": {
    "dataset_name": "agent_training_data",
    "version": "1.0.0",
    "license": "apache-2.0"
  }
}
```

## Use Cases

### 1. RAG System Development

Generate grounded embeddings for retrieval-augmented generation:

- Extract real documentation with Apify scrapers
- Generate embeddings with domain-specific context
- Populate vector databases (Pinecone, Weaviate, Qdrant)
- Build semantic search capabilities

### 2. LLM Fine-Tuning

Create training datasets for model customization:

- Domain-specific conversations
- Task-oriented dialogues
- Multi-turn reasoning examples
- Function calling patterns

### 3. Agent Memory Systems

Build memory patterns for stateful agents:

- Short-term working memory
- Long-term knowledge storage
- Episodic memory sequences
- Procedural learning patterns

### 4. Evaluation Benchmarks

Generate test datasets for model evaluation:

- Complexity-graded examples
- Domain-specific challenges
- Reasoning chain validation
- Tool use assessment

## Integration with agentic-flow

This actor integrates seamlessly with the [agentic-flow](https://www.npmjs.com/package/agentic-flow) npm package:

```javascript
import { AgenticFlow } from 'agentic-flow';

// Use generated training data to initialize agent memory
const flow = new AgenticFlow({
  memoryPatterns: loadFromApifyDataset('memory_patterns'),
  conversationExamples: loadFromApifyDataset('conversations'),
  toolSchemas: loadFromApifyDataset('tool_calls'),
});

// Agent learns from generated patterns
await flow.train();
```

## Performance & Scaling

### Memory Requirements

- Simple datasets: 256 MB
- Moderate datasets: 512 MB
- Complex datasets: 1024 MB
- Expert with embeddings: 2048-4096 MB

### Generation Speed

- Conversations: ~100 examples/minute
- Q&A Pairs: ~150 examples/minute
- Tool Calls: ~80 examples/minute
- Embeddings (mock): ~200 examples/minute
- Embeddings (OpenAI): ~10-20 examples/minute (API rate limited)

### Scaling Strategies

1. **Parallel Runs** - Launch multiple actor instances with different seeds
2. **Batch Processing** - Generate 10,000+ examples per run
3. **Grounding Optimization** - Use cached grounding datasets
4. **Embedding Batching** - Process embeddings in batches of 100

## SEO & Keywords

**Primary Keywords**: AI agent training, RAG dataset, LLM fine-tuning, conversation synthesis, agent memory, tool calls, vector embeddings, training data generation

**Secondary Keywords**: synthetic data, conversational AI, chain-of-thought, tree-of-thought, episodic memory, semantic memory, procedural memory, function calling, OpenAI embeddings, HuggingFace datasets

**Use Cases**: chatbot training, customer support AI, coding assistants, research agents, sales automation, healthcare AI, educational AI, retrieval augmented generation

## Technical Architecture

```
Agent Training Factory
├── Generators
│   ├── Conversations - Multi-turn dialogue generation
│   ├── Q&A Pairs - Question-answer dataset creation
│   ├── Tool Calls - Function calling sequence synthesis
│   ├── Reasoning Chains - CoT/ToT pattern generation
│   ├── Memory Patterns - Memory system data creation
│   └── Embeddings - Vector embedding generation
├── Utils
│   ├── Grounding - Real data integration from Apify
│   ├── Formatters - Output format conversion
│   └── Embeddings - Vector generation utilities
└── Output
    ├── JSONL - Standard JSON Lines format
    ├── Parquet - Columnar storage format
    └── HuggingFace - HF datasets compatible
```

## Best Practices

### 1. Start Small, Scale Up

Begin with 100 examples to validate quality, then scale to thousands.

### 2. Use Grounding for Production

Ground synthetic data in real patterns from your domain using Apify scrapers.

### 3. Combine Complexity Levels

Mix simple, moderate, and complex examples for robust training.

### 4. Validate Outputs

Review generated data quality before using in production systems.

### 5. Version Your Datasets

Use the `seed` parameter to ensure reproducible generation.

### 6. Optimize Embeddings

Use mock embeddings for development, real embeddings for production.

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce `count` or disable `generateEmbeddings`. Use lower `complexity` level.

### Issue: Slow Embedding Generation

**Solution**: Use mock embeddings during development. For production, batch process in smaller chunks.

### Issue: Grounding Data Not Loading

**Solution**: Verify `groundingActorId` or `groundingDatasetId` exists and has successful runs.

### Issue: Output Format Incompatibility

**Solution**: Try different `outputFormat` options. JSONL is most universally compatible.

## License

Apache-2.0

## Support & Resources

- **Documentation**: [https://ruv.io/docs/agent-training-factory](https://ruv.io)
- **GitHub**: [https://github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruvector/issues)
- **Website**: [https://ruv.io](https://ruv.io)
- **Apify Platform**: [https://apify.com](https://apify.com)

## Related Projects

- **ruvector** - Rust-powered vector database with hybrid search
- **agentic-flow** - Multi-agent orchestration framework
- **claude-flow** - Claude-based agent coordination system

## Author

**rUv**
- Email: info@ruv.io
- Website: [https://ruv.io](https://ruv.io)
- GitHub: [@ruvnet](https://github.com/ruvnet)

---

**Built with Apify** | **Powered by rUv** | **Part of the ruvector ecosystem**

Generate production-ready AI agent training data in minutes, not months.
