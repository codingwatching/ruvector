#!/usr/bin/env node
/**
 * Agentic Synth CLI
 *
 * Usage:
 *   agentic-synth mcp start   - Start MCP server for AI agent integration
 *   agentic-synth generate    - Generate synthetic data (interactive)
 *   agentic-synth list        - List available data types and templates
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
const command = args[0];

switch (command) {
  case 'mcp':
    if (args[1] === 'start') {
      // Start MCP server
      const mcpServer = join(__dirname, 'mcp-server.js');
      const child = spawn('node', [mcpServer], {
        stdio: 'inherit',
        env: process.env
      });

      child.on('error', (err) => {
        console.error('Failed to start MCP server:', err.message);
        process.exit(1);
      });

      child.on('exit', (code) => {
        process.exit(code || 0);
      });
    } else {
      console.log('Usage: agentic-synth mcp start');
    }
    break;

  case 'list':
    console.log(`
Agentic Synth - Available Data Types
=====================================

Web Scraping Data:
  - ecommerce     : Amazon, eBay, Shopify style product listings
  - social        : Twitter, Instagram, TikTok style posts
  - api_response  : REST API mock responses with pagination
  - search_results: Google, Bing style search results
  - real_estate   : Zillow, Realtor style property listings
  - jobs          : LinkedIn, Indeed style job listings
  - news          : News articles with engagement metrics

Enterprise Simulators:
  - stock_trading : OHLCV, quotes, orders, market data
  - medical       : Patient records, diagnoses, billing, vitals
  - company       : Org structure, financials, workforce
  - supply_chain  : Shipments, inventory, logistics
  - financial     : Transactions, accounts, fraud detection
  - bloomberg     : Full terminal-style market data

Technical Data:
  - structured    : Custom schema-defined data
  - timeseries    : Stock prices, IoT sensors, metrics
  - events        : Page views, clicks, form submissions
  - embeddings    : Vector data for ML/RAG testing
  - demo          : Sample data from all types

Use Case Templates:
  - lead-intelligence  : Sales teams memorizing prospect data
  - competitor-monitor : Track competitor mentions/changes
  - support-knowledge  : Customer support RAG system
  - research-assistant : Academic/market research
  - content-library    : Content creators' reference
  - product-catalog    : E-commerce product memory

Supported Apify Actor Integrations:
  - apify/google-maps-scraper
  - apify/google-search-scraper
  - apify/instagram-scraper
  - apify/tiktok-scraper
  - apify/youtube-scraper
  - apify/twitter-scraper
  - apify/amazon-scraper
  - apify/shopify-scraper
  - apify/web-scraper
  - apify/website-content-crawler
  - apify/linkedin-scraper
`);
    break;

  case 'help':
  case '--help':
  case '-h':
  default:
    console.log(`
Agentic Synth v2.2.0
====================

AI Synthetic Data Generator with TRM/SONA self-learning,
MCP server support, and Apify actor integrations.

Commands:
  mcp start   Start MCP server for AI agent integration
  list        List available data types and templates
  help        Show this help message

MCP Integration:
  Add to Claude Code:
    claude mcp add agentic-synth npx agentic-synth mcp start

  Or run directly:
    npx agentic-synth mcp start

Apify Usage:
  Run on Apify:
    https://apify.com/ruv/ai-synthetic-data-generator

  Deploy to Apify:
    apify push

Documentation:
  https://github.com/ruvnet/ruvector
`);
    break;
}
