<div align="center">

# Market Research Swarm
### AI-Powered Competitive Intelligence & Analysis

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-06C?style=for-the-badge&logo=apify&logoColor=white)](https://apify.com/ruv/market-research-swarm)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Multi-Agent AI](https://img.shields.io/badge/AI-Multi--Agent-purple?style=for-the-badge&logo=artificial-intelligence)](https://github.com/ruvnet/ruvector)
[![Enterprise Ready](https://img.shields.io/badge/Enterprise-Ready-blue?style=for-the-badge&logo=enterprise)](https://ruv.io)

**Orchestrate 10 specialized AI agents to conduct comprehensive market research, competitor analysis, and business intelligence in minutes, not weeks.**

Built by [rUv](https://ruv.io) | Part of the [ruvector](https://github.com/ruvnet/ruvector) AI agent ecosystem

[Get Started](#quick-start) · [Documentation](#documentation) · [Examples](#example-use-cases) · [API Reference](#api-reference)

</div>

---

## Introduction

Market Research Swarm is an advanced Apify actor that deploys AI agent swarms (1-10 agents) to conduct systematic market research across multiple dimensions. By orchestrating specialized agents for competitor analysis, pricing intelligence, sentiment tracking, trend detection, and more, this actor generates comprehensive business intelligence reports with actionable recommendations.

Unlike traditional market research tools that require manual data collection and analysis, Market Research Swarm automates the entire process by:
- **Orchestrating multiple AI agents** in parallel, each specializing in different aspects (pricing, features, sentiment, trends)
- **Integrating data from Apify scrapers** (Google Search, website crawlers, social media, reviews, e-commerce)
- **Analyzing competitors** across pricing, features, positioning, and market presence
- **Detecting market trends** including emerging technologies, declining patterns, and seasonal cycles
- **Generating executive summaries** with market share estimates and strategic recommendations
- **Identifying opportunities and gaps** for competitive advantage

Perfect for SaaS companies, e-commerce businesses, fintech startups, marketing agencies, and any organization needing competitive intelligence.

---

## Features

### Multi-Agent Swarm Orchestration
- Deploy **1-10 specialized AI agents** concurrently
- Agent roles: competitor analyzer, pricing analyst, feature mapper, sentiment tracker, trend detector, market share estimator, gap identifier, opportunity finder, content analyzer, social listener
- Parallel execution for fast comprehensive analysis
- Configurable swarm size based on research depth

### 5 Research Types
1. **Competitor Analysis** - Deep dive into competitor strengths, weaknesses, positioning, market presence
2. **Market Trends** - Detect emerging/declining trends, seasonal patterns, growth rates, predictions
3. **Product Comparison** - Feature matrices, pricing comparison, quality scores, differentiators
4. **Pricing Intelligence** - Price points, pricing models, value propositions, discount patterns
5. **Sentiment Analysis** - Brand sentiment, customer satisfaction, pain points, positive drivers

### Data Integration
- **Integrate data from multiple Apify scrapers**:
  - Google Search Scraper
  - Website Content Crawler
  - Social Media Scrapers (Twitter, Facebook, LinkedIn)
  - Review Scrapers (G2, Capterra, Trustpilot)
  - E-commerce Scrapers (Amazon, Shopify)
- Automatic data categorization (websites, social, search, reviews, pricing)
- Simulated data support for testing

### Comprehensive Analysis
- **Market share estimation** based on online presence and engagement
- **Executive summary generation** with key findings and metrics
- **Automated recommendations** across strategy, pricing, product, marketing, customer experience
- **Opportunity identification** including market gaps, product innovations, pricing optimization
- **Gap analysis** across geographic, feature, and channel dimensions

### Output Formats
- **Report format** - Comprehensive research report with all findings
- **Dashboard format** - Metrics summary with top findings and recommendations
- **JSON format** - Structured data for programmatic access

### Flexible Configuration
- Industry specification (SaaS, e-commerce, fintech, healthcare, etc.)
- Competitor list (company names or domains)
- Time range analysis (7d, 30d, 90d, 1y)
- Custom keywords tracking
- Analysis depth (1=basic, 2=standard, 3=comprehensive)
- Optional market share estimation
- Configurable recommendation generation

---

## Quick Start

### Running on Apify Platform

1. **Open the actor**: [Market Research Swarm](https://apify.com/ruv/market-research-swarm)
2. **Configure input**:
   ```json
   {
     "researchType": "competitor_analysis",
     "industry": "saas",
     "competitors": [
       "competitor1.com",
       "competitor2.com",
       "competitor3.com"
     ],
     "dataSourceActors": [
       "apify/google-search-scraper",
       "apify/website-content-crawler"
     ],
     "timeRange": "30d",
     "outputFormat": "report",
     "includeRecommendations": true,
     "swarmSize": 5
   }
   ```
3. **Run the actor** and view results in the dataset

### Example Research Scenarios

#### 1. SaaS Competitor Landscape Analysis
```json
{
  "researchType": "competitor_analysis",
  "industry": "saas",
  "competitors": ["salesforce.com", "hubspot.com", "zoho.com"],
  "dataSourceActors": ["apify/google-search-scraper", "apify/website-content-crawler"],
  "swarmSize": 8,
  "includeMarketShare": true,
  "maxDepth": 3
}
```

**Output**: Comprehensive competitor profiles with strengths/weaknesses, market positioning, estimated market share, feature comparison matrices, and strategic recommendations.

#### 2. E-commerce Pricing Trends
```json
{
  "researchType": "pricing_intelligence",
  "industry": "ecommerce",
  "competitors": ["amazon.com", "walmart.com", "target.com"],
  "dataSourceActors": ["apify/amazon-scraper", "apify/google-shopping-scraper"],
  "timeRange": "90d",
  "swarmSize": 6
}
```

**Output**: Pricing models, price points, discount patterns, price elasticity estimates, value propositions, and pricing strategy recommendations.

#### 3. Product Comparison Report
```json
{
  "researchType": "product_comparison",
  "industry": "fintech",
  "competitors": ["stripe.com", "square.com", "paypal.com"],
  "keywords": ["payment processing", "fraud detection", "API"],
  "swarmSize": 7,
  "includeRecommendations": true
}
```

**Output**: Feature comparison matrices, quality scores, user experience analysis, unique differentiators, and product development recommendations.

---

## Apify MCP Integration

Integrate Market Research Swarm with Claude for Desktop using the Apify MCP server:

```bash
# Install Apify MCP Server
claude mcp add research-swarm -- npx -y @apify/actors-mcp-server --actors "ruv/market-research-swarm"
```

Once configured, you can trigger market research from Claude:

```
"Analyze my SaaS competitors in the CRM space: salesforce.com, hubspot.com, zoho.com.
Include market share estimation and recommendations."
```

Claude will execute the actor and return comprehensive research findings.

---

## Tutorials

### Tutorial 1: Analyze SaaS Competitor Landscape

**Objective**: Understand competitor strengths, weaknesses, and market positioning in the CRM SaaS market.

**Steps**:
1. Set `researchType` to `competitor_analysis`
2. Set `industry` to `saas`
3. Add competitors: `["salesforce.com", "hubspot.com", "zoho.com", "pipedrive.com"]`
4. Configure data sources: `["apify/google-search-scraper", "apify/website-content-crawler"]`
5. Set `swarmSize` to `8` for comprehensive analysis
6. Enable `includeMarketShare` and `includeRecommendations`
7. Run actor

**Expected Output**:
- Detailed competitor profiles with strengths (e.g., "Strong enterprise presence") and weaknesses (e.g., "Complex pricing")
- Market share estimates based on online presence
- Feature comparison across all competitors
- Strategic recommendations for differentiation

---

### Tutorial 2: Track E-commerce Pricing Trends

**Objective**: Monitor pricing strategies and identify optimization opportunities in consumer electronics.

**Steps**:
1. Set `researchType` to `pricing_intelligence`
2. Set `industry` to `ecommerce`
3. Add competitors: `["amazon.com", "bestbuy.com", "newegg.com"]`
4. Configure data sources: `["apify/amazon-scraper"]`
5. Set `timeRange` to `90d` for quarterly trends
6. Set `swarmSize` to `6`
7. Add keywords: `["laptop", "smartphone", "tablet"]`
8. Run actor

**Expected Output**:
- Price point distributions across competitors
- Pricing models (subscription, one-time, usage-based)
- Discount patterns (seasonal, volume, promotional)
- Price elasticity estimates
- Recommendations for pricing strategy

---

### Tutorial 3: Generate Product Comparison Report

**Objective**: Create a feature-by-feature comparison report for project management tools.

**Steps**:
1. Set `researchType` to `product_comparison`
2. Set `industry` to `saas`
3. Add competitors: `["asana.com", "monday.com", "clickup.com", "notion.so"]`
4. Configure data sources: `["apify/website-content-crawler"]`
5. Set `swarmSize` to `7`
6. Set `maxDepth` to `3` for comprehensive feature extraction
7. Set `outputFormat` to `report`
8. Run actor

**Expected Output**:
- Feature comparison matrix showing which features each competitor offers
- Quality scores based on user reviews and sentiment
- User experience analysis
- Unique differentiators for each product
- Recommendations for product development

---

### Tutorial 4: Monitor Social Sentiment for Brand

**Objective**: Track brand perception and customer sentiment across social media and review platforms.

**Steps**:
1. Set `researchType` to `sentiment_analysis`
2. Set `industry` to `consumer_goods`
3. Add competitors: `["nike.com", "adidas.com", "underarmour.com"]`
4. Configure data sources: `["apify/twitter-scraper", "apify/google-reviews-scraper"]`
5. Set `timeRange` to `30d`
6. Set `swarmSize` to `5`
7. Add keywords: `["quality", "comfort", "style", "durability"]`
8. Run actor

**Expected Output**:
- Overall sentiment scores (positive/negative/neutral ratios)
- Brand sentiment comparison across competitors
- Customer satisfaction estimates
- Identified pain points (e.g., "sizing issues", "delivery delays")
- Positive drivers (e.g., "excellent quality", "great customer service")
- Recommendations for brand improvement

---

## Input Schema Reference

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `researchType` | enum | Yes | Type of research: `competitor_analysis`, `market_trends`, `product_comparison`, `pricing_intelligence`, `sentiment_analysis` |
| `industry` | string | Yes | Industry or market segment (e.g., "saas", "ecommerce", "fintech") |
| `competitors` | array | No | List of competitor company names or domains |
| `dataSourceActors` | array | No | Apify actor IDs to pull data from |
| `timeRange` | enum | No | Time range for analysis: `7d`, `30d`, `90d`, `1y` (default: `30d`) |
| `outputFormat` | enum | No | Output format: `report`, `json`, `dashboard` (default: `report`) |
| `includeRecommendations` | boolean | No | Generate actionable recommendations (default: `true`) |
| `swarmSize` | integer | No | Number of AI agents (1-10, default: `5`) |
| `keywords` | array | No | Specific keywords or topics to track |
| `includeMarketShare` | boolean | No | Estimate market share (default: `true`) |
| `maxDepth` | integer | No | Analysis depth 1-3 (default: `2`) |

---

## Output Schema

### Report Format
```json
{
  "researchType": "competitor_analysis",
  "industry": "saas",
  "competitors": ["competitor1.com", "competitor2.com"],
  "analyzedAt": "2025-12-13T10:30:00.000Z",
  "agents": [
    {
      "role": "competitor_analyzer",
      "status": "completed",
      "findingsCount": 12,
      "executedAt": "2025-12-13T10:35:00.000Z"
    }
  ],
  "findings": [
    {
      "type": "competitor_profile",
      "competitor": "competitor1.com",
      "strengths": ["Strong brand", "Feature-rich"],
      "weaknesses": ["Complex pricing", "Steep learning curve"],
      "positioning": "Premium enterprise solution",
      "marketPresence": 850
    }
  ],
  "metrics": {
    "competitorCount": 2,
    "averagePrice": 99,
    "totalFeatures": 47,
    "averageSentiment": 0.72,
    "socialMentions": 1250
  },
  "marketShare": {
    "competitor1.com": {
      "estimatedShare": "55.20",
      "presenceScore": 850,
      "rank": 1
    },
    "competitor2.com": {
      "estimatedShare": "44.80",
      "presenceScore": 690,
      "rank": 2
    }
  },
  "executiveSummary": "Market Research Analysis: competitor_analysis - saas...",
  "recommendations": [
    {
      "category": "Strategic",
      "priority": "High",
      "recommendation": "Focus on differentiation in underserved segments",
      "rationale": "Analysis reveals gaps in competitor coverage",
      "expectedImpact": "High market share gain potential"
    }
  ],
  "opportunities": [
    {
      "type": "Market Gap",
      "description": "Underserved mid-market segment",
      "confidence": "High",
      "potentialImpact": "Significant revenue opportunity"
    }
  ],
  "gaps": [
    {
      "category": "Feature",
      "description": "Missing integration capabilities",
      "severity": "High"
    }
  ]
}
```

---

## SEO Keywords & Use Cases

**Keywords**: competitive intelligence, market research, competitor analysis, business intelligence, market trends, AI swarm, pricing intelligence, sentiment analysis, product comparison, market share estimation, competitor monitoring, market analysis automation, AI-powered research, strategic planning

**Use Cases**:
- **SaaS Companies**: Analyze competitor features, pricing strategies, and market positioning
- **E-commerce Businesses**: Track pricing trends, product comparisons, and consumer sentiment
- **Marketing Agencies**: Generate competitive intelligence reports for clients
- **Product Managers**: Identify feature gaps and product development opportunities
- **Business Strategists**: Estimate market share and identify growth opportunities
- **Investment Firms**: Conduct market due diligence and competitor assessments
- **Startup Founders**: Understand competitive landscape before market entry
- **Brand Managers**: Monitor brand sentiment and customer satisfaction

---

## Advanced Configuration

### Custom Agent Swarm Composition

While the actor automatically deploys agents based on `swarmSize`, you can optimize for specific research types:

- **Competitor Analysis**: Use swarmSize 8-10 (deploy all agent types)
- **Pricing Intelligence**: Use swarmSize 4-5 (focus on pricing_analyst, competitor_analyzer, trend_detector)
- **Market Trends**: Use swarmSize 5-6 (focus on trend_detector, social_listener, content_analyzer)
- **Sentiment Analysis**: Use swarmSize 3-4 (focus on sentiment_tracker, social_listener)
- **Product Comparison**: Use swarmSize 6-7 (focus on feature_mapper, competitor_analyzer, gap_identifier)

### Integrating Custom Data Sources

Connect your own Apify actors for specialized data:

```json
{
  "dataSourceActors": [
    "apify/google-search-scraper",
    "apify/website-content-crawler",
    "apify/twitter-scraper",
    "your-username/custom-industry-scraper"
  ]
}
```

### Time Range Strategy

- **7d**: Real-time monitoring, breaking news, rapid trend detection
- **30d**: Monthly reporting, short-term trend analysis
- **90d**: Quarterly business reviews, seasonal pattern detection
- **1y**: Annual strategic planning, long-term trend identification

---

## Performance & Limitations

### Performance
- **Execution time**: 2-15 minutes depending on swarmSize and maxDepth
- **Memory usage**: 512 MB - 8 GB (scales with swarm size)
- **Concurrent agents**: Up to 10 agents in parallel
- **Data volume**: Can process 1000s of data points from source actors

### Limitations
- Requires data source actors for comprehensive analysis (works with simulated data for testing)
- Market share estimates are relative, not absolute market data
- Sentiment analysis accuracy depends on data quality from source actors
- Some competitor data may be limited by public availability

---

## Support & Resources

- **Documentation**: [https://github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Issues**: [https://github.com/ruvnet/ruvector/issues](https://github.com/ruvnet/ruvector/issues)
- **Author**: rUv - [https://ruv.io](https://ruv.io)
- **Email**: info@ruv.io
- **Apify Platform**: [https://apify.com](https://apify.com)

---

## Related Actors

- **Google Search Scraper** - Extract search results for competitor analysis
- **Website Content Crawler** - Scrape competitor websites for features and content
- **Twitter Scraper** - Collect social sentiment data
- **Amazon Scraper** - Track e-commerce pricing and reviews
- **G2 Review Scraper** - Analyze product reviews and ratings

---

## License

Apache-2.0 License - See LICENSE file for details

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/ruvnet/ruvector/blob/main/CONTRIBUTING.md).

---

**Built with Apify by [rUv](https://ruv.io) | Part of the ruvector AI agent ecosystem**
