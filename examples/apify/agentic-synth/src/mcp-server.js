#!/usr/bin/env node
/**
 * MCP Server for AI Synthetic Data Generator
 *
 * Exposes synthetic data generation as MCP tools for AI agents (Claude, GPT, etc.)
 *
 * Usage:
 *   npx @apify/agentic-synth mcp start
 *
 * Or add to Claude Code MCP config:
 *   claude mcp add agentic-synth npx @apify/agentic-synth mcp start
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

// Import data generators
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Data type definitions for tool descriptions
const DATA_TYPES = {
  ecommerce: 'E-commerce product listings (Amazon, eBay, Shopify style)',
  social: 'Social media posts (Twitter, Instagram, TikTok style)',
  api_response: 'REST API mock responses with pagination',
  search_results: 'Search engine results (Google, Bing style)',
  real_estate: 'Real estate listings (Zillow, Realtor style)',
  jobs: 'Job listings (LinkedIn, Indeed style)',
  news: 'News articles with engagement metrics',
  stock_trading: 'Stock trading data (OHLCV, quotes, orders)',
  medical: 'Medical/healthcare records (HIPAA-safe format)',
  company: 'Company/corporate data (org structure, financials)',
  supply_chain: 'Supply chain data (shipments, inventory, logistics)',
  financial: 'Financial transactions and fraud detection data',
  bloomberg: 'Bloomberg terminal-style market data',
  structured: 'Custom schema-defined data',
  timeseries: 'Time-series data (stock prices, IoT sensors)',
  events: 'Web event tracking data (clicks, pageviews)',
  embeddings: 'Vector embeddings for ML/RAG testing',
  demo: 'Demo data sampling all types'
};

class AgenticSynthMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'agentic-synth',
        version: '2.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    this.setupErrorHandling();
  }

  setupErrorHandling() {
    this.server.onerror = (error) => {
      console.error('[MCP Error]', error);
    };

    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'generate_synthetic_data',
            description: 'Generate high-quality synthetic data for testing, training AI models, and development. Supports 18 data types including e-commerce, Bloomberg terminal, medical records, social media, and more. Powered by TRM/SONA self-learning.',
            inputSchema: {
              type: 'object',
              properties: {
                dataType: {
                  type: 'string',
                  enum: Object.keys(DATA_TYPES),
                  description: 'Type of data to generate: ' + Object.entries(DATA_TYPES).map(([k, v]) => `${k} (${v})`).join(', ')
                },
                count: {
                  type: 'integer',
                  minimum: 1,
                  maximum: 1000,
                  default: 10,
                  description: 'Number of records to generate (1-1000 for MCP, use Apify for larger)'
                },
                seed: {
                  type: 'string',
                  description: 'Random seed for reproducible results'
                },
                schema: {
                  type: 'object',
                  description: 'Custom schema for structured data type. Example: {"name": "string", "price": "number (10-500)"}'
                },
                quality: {
                  type: 'number',
                  minimum: 0.1,
                  maximum: 1.0,
                  default: 0.8,
                  description: 'Data quality level (0.1-1.0)'
                }
              },
              required: ['dataType']
            }
          },
          {
            name: 'list_data_types',
            description: 'List all available synthetic data types with descriptions',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          },
          {
            name: 'get_data_schema',
            description: 'Get the output schema/structure for a specific data type',
            inputSchema: {
              type: 'object',
              properties: {
                dataType: {
                  type: 'string',
                  enum: Object.keys(DATA_TYPES),
                  description: 'Data type to get schema for'
                }
              },
              required: ['dataType']
            }
          },
          {
            name: 'compose_data',
            description: 'Generate composite data from multiple sources with relationships',
            inputSchema: {
              type: 'object',
              properties: {
                sources: {
                  type: 'array',
                  items: {
                    type: 'object',
                    properties: {
                      dataType: { type: 'string', enum: Object.keys(DATA_TYPES) },
                      count: { type: 'integer', minimum: 1, maximum: 100 },
                      alias: { type: 'string', description: 'Alias for this data source' }
                    },
                    required: ['dataType', 'count']
                  },
                  description: 'Array of data sources to combine'
                },
                seed: {
                  type: 'string',
                  description: 'Random seed for reproducibility'
                }
              },
              required: ['sources']
            }
          }
        ]
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'generate_synthetic_data':
            return await this.generateData(args);
          case 'list_data_types':
            return this.listDataTypes();
          case 'get_data_schema':
            return this.getDataSchema(args.dataType);
          case 'compose_data':
            return await this.composeData(args);
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({ error: error.message }, null, 2)
            }
          ],
          isError: true
        };
      }
    });
  }

  async generateData(args) {
    const { dataType, count = 10, seed, schema, quality = 0.8 } = args;

    // Limit count for MCP (use Apify for larger datasets)
    const limitedCount = Math.min(count, 1000);

    // Dynamic import of generator functions
    const generators = await this.loadGenerators();

    let data;
    switch (dataType) {
      case 'ecommerce':
        data = await generators.generateEcommerceData(limitedCount, seed);
        break;
      case 'social':
        data = await generators.generateSocialMediaData(limitedCount, seed);
        break;
      case 'api_response':
        data = await generators.generateApiResponseData(limitedCount, '/api/resource', seed);
        break;
      case 'search_results':
        data = await generators.generateSearchResultsData(limitedCount, seed);
        break;
      case 'real_estate':
        data = await generators.generateRealEstateData(limitedCount, seed);
        break;
      case 'jobs':
        data = await generators.generateJobListingsData(limitedCount, seed);
        break;
      case 'news':
        data = await generators.generateNewsData(limitedCount, seed);
        break;
      case 'stock_trading':
        data = await generators.generateStockTradingData(limitedCount, seed);
        break;
      case 'medical':
        data = await generators.generateMedicalData(limitedCount, seed);
        break;
      case 'company':
        data = await generators.generateCompanyData(limitedCount, seed);
        break;
      case 'supply_chain':
        data = await generators.generateSupplyChainData(limitedCount, seed);
        break;
      case 'financial':
        data = await generators.generateFinancialData(limitedCount, seed);
        break;
      case 'bloomberg':
        data = await generators.generateBloombergData(limitedCount, seed);
        break;
      case 'structured':
        data = await generators.generateStructuredData(limitedCount, schema || {}, null, null, seed);
        break;
      case 'timeseries':
        data = await generators.generateTimeSeriesData(limitedCount, {}, seed);
        break;
      case 'events':
        data = await generators.generateEventData(limitedCount, ['page_view', 'click', 'scroll'], seed);
        break;
      case 'embeddings':
        data = await generators.generateEmbeddingData(limitedCount, 384, seed);
        break;
      case 'demo':
        data = await generators.generateDemoData(limitedCount, null, null);
        break;
      default:
        throw new Error(`Unknown data type: ${dataType}`);
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: true,
            dataType,
            count: data.length,
            seed: seed || 'random',
            quality,
            data: data
          }, null, 2)
        }
      ]
    };
  }

  listDataTypes() {
    const types = Object.entries(DATA_TYPES).map(([type, description]) => ({
      type,
      description,
      categories: this.categorizeType(type)
    }));

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            availableTypes: types,
            totalTypes: types.length,
            categories: {
              webScraping: ['ecommerce', 'social', 'search_results', 'real_estate', 'jobs', 'news'],
              enterprise: ['stock_trading', 'medical', 'company', 'supply_chain', 'financial', 'bloomberg'],
              technical: ['api_response', 'structured', 'timeseries', 'events', 'embeddings'],
              utility: ['demo']
            }
          }, null, 2)
        }
      ]
    };
  }

  categorizeType(type) {
    const webScraping = ['ecommerce', 'social', 'search_results', 'real_estate', 'jobs', 'news'];
    const enterprise = ['stock_trading', 'medical', 'company', 'supply_chain', 'financial', 'bloomberg'];
    const technical = ['api_response', 'structured', 'timeseries', 'events', 'embeddings'];

    if (webScraping.includes(type)) return ['webScraping'];
    if (enterprise.includes(type)) return ['enterprise'];
    if (technical.includes(type)) return ['technical'];
    return ['utility'];
  }

  getDataSchema(dataType) {
    const schemas = {
      ecommerce: {
        url: 'string (product URL)',
        title: 'string (product name)',
        price: 'number (current price)',
        originalPrice: 'number|null (before discount)',
        currency: 'string (USD, EUR, etc.)',
        category: 'string',
        brand: 'string',
        rating: 'number (1-5)',
        reviewCount: 'integer',
        inStock: 'boolean',
        seller: { name: 'string', rating: 'number', totalSales: 'integer' },
        shipping: { free: 'boolean', estimatedDays: 'integer', price: 'number' },
        images: 'array<string>',
        specifications: 'object'
      },
      social: {
        url: 'string (post URL)',
        platform: 'string (twitter, instagram, etc.)',
        postType: 'string (text, image, video, link)',
        author: { username: 'string', displayName: 'string', verified: 'boolean', followers: 'integer' },
        content: { text: 'string', hashtags: 'array<string>', mentions: 'array<string>' },
        engagement: { likes: 'integer', comments: 'integer', shares: 'integer', views: 'integer' },
        timestamp: 'ISO 8601 date string'
      },
      bloomberg: {
        terminalId: 'string',
        security: { ticker: 'string', name: 'string', assetClass: 'string', sector: 'string' },
        pricing: { last: 'number', bid: 'number', ask: 'number', volume: 'integer' },
        fundamentals: { marketCap: 'string', peRatio: 'number', roe: 'number', eps: 'number' },
        analytics: { beta: 'number', sharpeRatio: 'number', volatility: 'number' },
        consensus: { recommendation: 'string', targetPrice: 'number', numAnalysts: 'integer' }
      },
      medical: {
        recordId: 'string',
        patient: { id: 'string', age: 'integer', gender: 'string', bloodType: 'string' },
        encounter: { type: 'string', department: 'string', admitDate: 'ISO date' },
        diagnosis: { primary: 'string', icdCode: 'string', severity: 'string' },
        vitals: { bloodPressure: 'string', heartRate: 'integer', temperature: 'number' },
        billing: { insurer: 'string', totalCharges: 'number', claimStatus: 'string' }
      },
      // Add more schemas as needed
    };

    const schema = schemas[dataType] || { note: 'Schema varies based on data type. Generate sample data to see structure.' };

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            dataType,
            description: DATA_TYPES[dataType],
            schema,
            example: `Use generate_synthetic_data with dataType="${dataType}" and count=1 to see a full example.`
          }, null, 2)
        }
      ]
    };
  }

  async composeData(args) {
    const { sources, seed } = args;
    const generators = await this.loadGenerators();
    const result = {};

    for (const source of sources) {
      const { dataType, count, alias } = source;
      const key = alias || dataType;

      // Generate data for each source
      const generateMethod = `generate${this.capitalize(dataType)}Data`;
      if (generators[generateMethod]) {
        result[key] = await generators[generateMethod](Math.min(count, 100), seed);
      } else {
        result[key] = { error: `Unknown data type: ${dataType}` };
      }
    }

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: true,
            sources: sources.map(s => s.alias || s.dataType),
            seed: seed || 'random',
            data: result
          }, null, 2)
        }
      ]
    };
  }

  capitalize(str) {
    const map = {
      ecommerce: 'Ecommerce',
      social: 'SocialMedia',
      api_response: 'ApiResponse',
      search_results: 'SearchResults',
      real_estate: 'RealEstate',
      jobs: 'JobListings',
      news: 'News',
      stock_trading: 'StockTrading',
      medical: 'Medical',
      company: 'Company',
      supply_chain: 'SupplyChain',
      financial: 'Financial',
      bloomberg: 'Bloomberg',
      structured: 'Structured',
      timeseries: 'TimeSeries',
      events: 'Event',
      embeddings: 'Embedding',
      demo: 'Demo'
    };
    return map[str] || str.charAt(0).toUpperCase() + str.slice(1);
  }

  async loadGenerators() {
    // Import generator module dynamically
    // In production, these would be imported from the main module
    const module = await import('./generators.js');
    return module;
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Agentic Synth MCP Server running on stdio');
  }
}

// Start server if run directly
const server = new AgenticSynthMCPServer();
server.run().catch(console.error);

export { AgenticSynthMCPServer };
