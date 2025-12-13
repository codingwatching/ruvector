#!/usr/bin/env node

/**
 * RvLite CLI - Standalone vector database with SQL, SPARQL, and Cypher
 *
 * Usage:
 *   rvlite serve [--port 3000]     Start the dashboard server
 *   rvlite repl                    Start interactive REPL
 *   rvlite query <query>           Execute a query (SQL/SPARQL/Cypher)
 *   rvlite --help                  Show help
 */

import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, dirname, extname } from 'path';
import { fileURLToPath } from 'url';
import { createInterface } from 'readline';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const DASHBOARD_DIR = join(__dirname, '..', 'dashboard');

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

function serve(port = 3000) {
  const server = createServer((req, res) => {
    let filePath = join(DASHBOARD_DIR, req.url === '/' ? 'index.html' : req.url);

    // Security: prevent directory traversal
    if (!filePath.startsWith(DASHBOARD_DIR)) {
      res.writeHead(403);
      res.end('Forbidden');
      return;
    }

    if (!existsSync(filePath)) {
      // SPA fallback
      filePath = join(DASHBOARD_DIR, 'index.html');
    }

    try {
      const content = readFileSync(filePath);
      const ext = extname(filePath);
      const contentType = MIME_TYPES[ext] || 'application/octet-stream';

      res.writeHead(200, {
        'Content-Type': contentType,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      res.end(content);
    } catch (err) {
      res.writeHead(500);
      res.end('Internal Server Error');
    }
  });

  server.listen(port, () => {
    console.log(`
  ╭─────────────────────────────────────────────────╮
  │                                                 │
  │   RvLite Dashboard                              │
  │   Vector Database + RuvLLM                      │
  │                                                 │
  │   Running at: http://localhost:${port}            │
  │                                                 │
  │   Features:                                     │
  │   • Vector search with SQL/SPARQL/Cypher        │
  │   • RuvLLM TRM configuration                    │
  │   • MicroLoRA / BaseLoRA settings               │
  │   • WASM-powered, browser-native                │
  │                                                 │
  │   Press Ctrl+C to stop                          │
  │                                                 │
  ╰─────────────────────────────────────────────────╯
`);
  });
}

async function repl() {
  console.log(`
  RvLite REPL - Interactive Query Shell
  Type SQL, SPARQL, or Cypher queries
  Commands: .help, .tables, .stats, .exit
  `);

  // Dynamic import for Node.js WASM
  let db;
  try {
    const { RvLite, RvLiteConfig, init } = await import('../wasm/rvlite.js');
    await init();
    const config = new RvLiteConfig(384);
    db = new RvLite(config);
    console.log('  Database initialized (384 dimensions, cosine similarity)\n');
  } catch (err) {
    console.error('  Error: WASM module requires browser environment for full functionality.');
    console.error('  Use "rvlite serve" to start the browser dashboard.\n');
    process.exit(1);
  }

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: 'rvlite> ',
  });

  rl.prompt();

  rl.on('line', (line) => {
    const input = line.trim();

    if (!input) {
      rl.prompt();
      return;
    }

    if (input === '.exit' || input === '.quit') {
      console.log('Goodbye!');
      process.exit(0);
    }

    if (input === '.help') {
      console.log(`
  Commands:
    .help     Show this help
    .tables   List tables
    .stats    Show database stats
    .exit     Exit REPL

  Query Types:
    SQL:      SELECT * FROM vectors WHERE ...
    SPARQL:   SELECT ?s ?p ?o WHERE { ?s ?p ?o }
    Cypher:   MATCH (n:Label) RETURN n
      `);
      rl.prompt();
      return;
    }

    if (input === '.stats') {
      console.log(`  Vectors: ${db.len()}`);
      console.log(`  Triples: ${db.triple_count()}`);
      const cypherStats = db.cypher_stats();
      console.log(`  Cypher:  ${JSON.stringify(cypherStats)}`);
      rl.prompt();
      return;
    }

    // Try to detect query type and execute
    try {
      let result;
      const upper = input.toUpperCase();

      if (upper.startsWith('SELECT ?') || upper.startsWith('ASK ') || upper.startsWith('CONSTRUCT ')) {
        result = db.sparql(input);
      } else if (upper.startsWith('MATCH ') || upper.startsWith('CREATE (') || upper.startsWith('MERGE ')) {
        result = db.cypher(input);
      } else {
        result = db.sql(input);
      }

      console.log(JSON.stringify(result, null, 2));
    } catch (err) {
      console.error(`  Error: ${err.message}`);
    }

    rl.prompt();
  });

  rl.on('close', () => {
    console.log('\nGoodbye!');
    process.exit(0);
  });
}

function printHelp() {
  console.log(`
  RvLite - Standalone Vector Database

  Usage:
    rvlite serve [--port <port>]   Start dashboard server (default: 3000)
    rvlite repl                    Start interactive query shell
    rvlite --version               Show version
    rvlite --help                  Show this help

  Examples:
    rvlite serve                   Start dashboard on port 3000
    rvlite serve --port 8080       Start dashboard on port 8080
    rvlite repl                    Interactive SQL/SPARQL/Cypher shell

  Features:
    • Vector similarity search with SQL syntax
    • RDF triple store with SPARQL queries
    • Property graph with Cypher queries
    • Browser-based dashboard with IndexedDB persistence
    • WASM-powered, runs entirely in browser
  `);
}

// Parse arguments
const args = process.argv.slice(2);
const command = args[0];

switch (command) {
  case 'serve':
  case 'server':
  case 'dashboard': {
    const portIndex = args.indexOf('--port');
    const port = portIndex !== -1 ? parseInt(args[portIndex + 1], 10) : 3000;
    serve(port);
    break;
  }

  case 'repl':
  case 'shell':
    repl();
    break;

  case '--version':
  case '-v':
    console.log('rvlite 0.2.0');
    break;

  case '--help':
  case '-h':
  case undefined:
    printHelp();
    break;

  default:
    console.error(`Unknown command: ${command}`);
    printHelp();
    process.exit(1);
}
