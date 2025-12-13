# Quick Start Guide - Agent Training Data Factory

## 5-Minute Setup

### 1. Deploy to Apify (2 minutes)

```bash
cd /workspaces/ruvector/examples/apify/agent-training-factory
apify login
apify push
```

### 2. Run Your First Dataset (1 minute)

In Apify Console, use this input:

```json
{
  "datasetType": "conversations",
  "domain": "customer_support",
  "count": 50,
  "complexity": "moderate",
  "outputFormat": "jsonl"
}
```

Click "Start" and wait ~30 seconds.

### 3. Download Results (1 minute)

```bash
apify dataset download
```

You now have 50 customer support conversations ready for training!

## Quick Examples

### Generate Q&A for Fine-Tuning

```json
{
  "datasetType": "qa_pairs",
  "domain": "coding",
  "count": 1000,
  "complexity": "complex",
  "outputFormat": "huggingface"
}
```

### Create RAG Embeddings

```json
{
  "datasetType": "embeddings",
  "domain": "research",
  "count": 500,
  "generateEmbeddings": true,
  "openaiApiKey": "sk-..."
}
```

### Ground in Real Data

```json
{
  "datasetType": "conversations",
  "count": 200,
  "groundingActorId": "apify/web-scraper",
  "outputFormat": "jsonl"
}
```

## Claude Desktop Integration

```bash
claude mcp add agent-factory -- npx -y @apify/actors-mcp-server --actors "your-username/agent-training-factory"
```

Then ask Claude:

> "Generate 100 coding conversation examples with expert complexity"

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [examples/run-examples.js](examples/run-examples.js) for more configurations
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
- See [SUMMARY.md](SUMMARY.md) for complete technical details

---

**Need Help?** info@ruv.io | https://ruv.io
