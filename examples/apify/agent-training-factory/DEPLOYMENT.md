# Deployment Guide - Agent Training Data Factory

## Prerequisites

- Apify account (free tier available)
- Apify CLI installed: `npm install -g apify-cli`
- Git repository access (optional)

## Quick Deployment

### Option 1: Deploy via Apify CLI

```bash
# Navigate to the actor directory
cd /workspaces/ruvector/examples/apify/agent-training-factory

# Login to Apify
apify login

# Initialize Apify project (if not already done)
apify init

# Deploy to Apify platform
apify push
```

### Option 2: Deploy via Apify Console

1. Navigate to [Apify Console](https://console.apify.com)
2. Click "Actors" > "Create new"
3. Select "From scratch"
4. Copy all files from this directory
5. Click "Build" to create the actor
6. Run with example configuration

### Option 3: Deploy from GitHub

```bash
# Push to GitHub repository
git add examples/apify/agent-training-factory
git commit -m "Add Agent Training Data Factory actor"
git push

# In Apify Console:
# 1. Create new Actor
# 2. Select "From GitHub repository"
# 3. Enter repository URL
# 4. Set path to: examples/apify/agent-training-factory
# 5. Click "Create"
```

## Configuration

### Environment Variables

For production deployments with OpenAI embeddings, add these secrets in Apify Console:

```bash
OPENAI_API_KEY=your-api-key-here
```

Navigate to: Actor Settings > Environment Variables > Add Secret

### Actor Settings

Recommended settings for production:

- **Memory**: 1024 MB (2048 MB for embeddings)
- **Timeout**: 3600 seconds
- **Build tag**: latest
- **Restart on error**: Yes

## Testing

### Test Locally

```bash
# Install dependencies
npm install

# Set environment variables (if using embeddings)
export OPENAI_API_KEY=your-key

# Run locally with example input
apify run --purge
```

### Test on Platform

1. Navigate to your actor
2. Click "Input" tab
3. Use example configuration:

```json
{
  "datasetType": "conversations",
  "domain": "customer_support",
  "count": 10,
  "complexity": "moderate",
  "outputFormat": "jsonl"
}
```

4. Click "Start"
5. Monitor logs and dataset output

## Publishing

### Publish to Apify Store

1. Ensure all metadata is complete in `.actor/actor.json`
2. Add comprehensive README.md (already included)
3. Test thoroughly with multiple configurations
4. In Apify Console: Actor > Settings > Publish to Store
5. Fill in store listing details
6. Submit for review

### Version Management

```bash
# Update version in actor.json
# Then push new version
apify push --version-number 1.0.1
```

## MCP Server Integration

After deployment, users can integrate with Claude Desktop:

```bash
claude mcp add agent-factory -- npx -y @apify/actors-mcp-server --actors "your-username/agent-training-factory"
```

## Monitoring

### View Runs

```bash
# List recent runs
apify runs ls

# Get run details
apify run info <RUN_ID>

# Download dataset
apify dataset download <DATASET_ID>
```

### Monitor Logs

- Real-time: Apify Console > Actor > Runs > Select Run > Log
- CLI: `apify logs <RUN_ID>`

### Performance Metrics

Track these metrics in production:

- Average run duration
- Success rate
- Memory usage
- Dataset quality scores
- API costs (for embeddings)

## Troubleshooting

### Build Failures

```bash
# Check Dockerfile syntax
docker build -t test .

# Verify package.json dependencies
npm install --dry-run
```

### Runtime Errors

Common issues and solutions:

1. **Out of Memory**: Increase memory allocation or reduce `count`
2. **API Rate Limits**: Add delays between API calls
3. **Invalid Input**: Validate input schema matches expected format
4. **Grounding Data Not Found**: Verify actor/dataset IDs exist

### Performance Optimization

```javascript
// Reduce memory usage
{
  "count": 100,  // Lower count
  "generateEmbeddings": false,  // Disable embeddings
  "complexity": "simple"  // Use simpler generation
}

// Increase generation speed
{
  "outputFormat": "jsonl",  // Fastest format
  "includeMetadata": false,  // Skip metadata
  "groundingActorId": null  // Skip grounding
}
```

## Maintenance

### Update Dependencies

```bash
# Update to latest Apify SDK
npm update apify crawlee

# Test after updates
npm test
```

### Backup Data

```bash
# Backup all datasets
apify datasets ls --json > datasets.json

# Download specific dataset
apify dataset download <DATASET_ID> --output data/
```

## Security

### API Key Management

Never commit API keys to git:

```bash
# Use Apify secrets
# Set in: Console > Actor > Settings > Environment Variables

# Or use .env file (excluded from git)
echo "OPENAI_API_KEY=your-key" > .env
```

### Access Control

Configure actor visibility:

- **Private**: Only you can access
- **Unlisted**: Anyone with link can access
- **Public**: Listed in Apify Store

## Scaling

### Parallel Execution

Run multiple instances with different seeds:

```javascript
// Instance 1
{ "seed": 42, "count": 1000 }

// Instance 2
{ "seed": 43, "count": 1000 }

// Instance 3
{ "seed": 44, "count": 1000 }
```

### Batch Processing

For large datasets (10,000+ examples):

```javascript
{
  "count": 10000,
  "complexity": "moderate",
  "outputFormat": "parquet"  // More efficient for large datasets
}
```

## Cost Optimization

### Free Tier Limits

- 100 actor hours/month
- 5 GB storage
- 50 GB data transfer

### Cost Reduction Tips

1. Use mock embeddings for development
2. Batch similar generation types
3. Cache grounding data
4. Use lower complexity for large datasets
5. Set appropriate memory limits

### Estimated Costs

| Dataset Type | Count | Memory | Duration | Cost (USD) |
|--------------|-------|--------|----------|------------|
| Conversations | 1,000 | 512 MB | 10 min | $0.02 |
| Q&A Pairs | 5,000 | 512 MB | 30 min | $0.06 |
| Embeddings (mock) | 1,000 | 1 GB | 5 min | $0.03 |
| Embeddings (OpenAI) | 1,000 | 1 GB | 20 min | $0.10 + API |

## Support

### Resources

- [Apify Documentation](https://docs.apify.com)
- [Actor Development Guide](https://docs.apify.com/actors)
- [Apify SDK Reference](https://docs.apify.com/sdk/js)
- [GitHub Issues](https://github.com/ruvnet/ruvector/issues)

### Community

- [Apify Discord](https://discord.com/invite/jyEM2PRvMU)
- [Apify Forum](https://community.apify.com)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/apify)

---

**Happy Deploying!** 🚀

For questions or issues, contact: info@ruv.io
