import { Actor, log } from 'apify';
import { ApifyClient } from 'apify-client';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createRequire } from 'module';

// Import RuvLLM with TRM/SONA capabilities using CJS require (ESM has extension bug)
let RuvLLM, SonaCoordinator, TrajectoryBuilder, version, hasSimdSupport;
let ruvllmLoaded = false;

try {
  const require = createRequire(import.meta.url);
  const ruvllm = require('@ruvector/ruvllm');
  RuvLLM = ruvllm.RuvLLM;
  SonaCoordinator = ruvllm.SonaCoordinator;
  TrajectoryBuilder = ruvllm.TrajectoryBuilder;
  version = ruvllm.version;
  hasSimdSupport = ruvllm.hasSimdSupport;
  ruvllmLoaded = true;
  log.info('RuvLLM loaded successfully via CJS');
} catch (e) {
  log.warning(`RuvLLM native module not available: ${e.message}. Using fallback embeddings.`);
}

// ============================================
// RUVLLM-POWERED MEMORY STORE
// ============================================

class MemoryStore {
  constructor(config) {
    this.dimensions = config.dimensions || 768;
    this.distanceMetric = config.distanceMetric || 'cosine';
    this.namespace = config.namespace || 'default';
    this.learningEnabled = config.learningEnabled !== false;
    this.memories = [];
    this.knowledgeGraph = { nodes: [], edges: [] };
    this.stats = { queries: 0, stores: 0, hits: 0, trajectories: 0, patterns: 0 };

    // Initialize RuvLLM engine with SONA learning
    this.ruvllm = null;
    this.sona = null;
    this.simdAvailable = false;

    if (ruvllmLoaded) {
      try {
        this.ruvllm = new RuvLLM({
          embeddingDim: this.dimensions,
          learningEnabled: this.learningEnabled,
          ewcLambda: config.ewcLambda || 2000,
          hnswM: config.hnswM || 32,
          hnswEfConstruction: config.hnswEfConstruction || 200,
          hnswEfSearch: config.hnswEfSearch || 64,
        });

        // Initialize SONA coordinator for learning
        this.sona = new SonaCoordinator({
          instantLoopEnabled: true,
          backgroundLoopEnabled: true,
          ewcLambda: config.ewcLambda || 2000,
          patternThreshold: config.patternThreshold || 0.85,
          maxTrajectorySize: 1000,
        });

        this.simdAvailable = hasSimdSupport?.() || false;
        log.info('RuvLLM engine initialized', {
          dimensions: this.dimensions,
          simd: this.simdAvailable,
          native: this.ruvllm.isNativeLoaded?.() || false,
          version: version?.() || 'unknown'
        });
      } catch (e) {
        log.warning(`RuvLLM initialization failed: ${e.message}`);
        this.ruvllm = null;
        this.sona = null;
      }
    }
  }

  async add(text, metadata = {}, embedding = null) {
    // Start trajectory tracking
    const trajectory = this.sona ? new TrajectoryBuilder() : null;
    trajectory?.startStep('memory', `Adding: ${text.substring(0, 50)}...`);

    const vector = embedding || await this.embed(text);
    const memory = {
      id: `mem_${Date.now()}_${this.memories.length}`,
      text,
      metadata: { ...metadata, namespace: this.namespace },
      embedding: vector,
      createdAt: new Date().toISOString(),
      accessCount: 0
    };

    this.memories.push(memory);
    this.stats.stores++;

    // Add to RuvLLM HNSW memory if available
    if (this.ruvllm) {
      try {
        this.ruvllm.addMemory(text, metadata);
      } catch (e) {
        // Fallback silently
      }
    }

    // Complete trajectory
    if (trajectory) {
      trajectory.endStep(`Stored memory ${memory.id}`, 0.95);
      this.sona.recordTrajectory(trajectory.complete('success'));
      this.stats.trajectories++;
    }

    return memory;
  }

  async search(query, topK = 10, threshold = 0.7) {
    // Start trajectory tracking
    const trajectory = this.sona ? new TrajectoryBuilder() : null;
    trajectory?.startStep('query', query);

    let results;

    // Try RuvLLM native search first (HNSW + SIMD optimized)
    if (this.ruvllm && this.ruvllm.isNativeLoaded?.()) {
      try {
        const nativeResults = this.ruvllm.searchMemory(query, topK * 2); // Get more to filter
        // Calculate similarity scores for each result
        results = nativeResults
          .map(r => {
            const similarity = this.ruvllm.similarity(query, r.content);
            const localMem = this.memories.find(m => m.text === r.content);
            return {
              ...(localMem || { text: r.content, metadata: r.metadata || {} }),
              similarity
            };
          })
          .filter(r => r.similarity >= threshold)
          .sort((a, b) => b.similarity - a.similarity)
          .slice(0, topK);

        trajectory?.endStep(`Found ${results.length} via HNSW`, 0.95);
      } catch (e) {
        // Fall through to manual search
        results = null;
      }
    }

    // Fallback to manual search with RuvLLM embeddings
    if (!results || results.length === 0) {
      const queryVector = await this.embed(query);
      results = this.memories
        .map(mem => ({
          ...mem,
          similarity: this.ruvllm
            ? this.ruvllm.similarity(query, mem.text)
            : this.cosineSimilarity(queryVector, mem.embedding)
        }))
        .filter(mem => mem.similarity >= threshold)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK);

      trajectory?.endStep(`Found ${results.length} via embedding search`, 0.85);
    }

    // Update access counts for learning
    if (this.learningEnabled) {
      results.forEach(r => {
        const mem = this.memories.find(m => m.id === r.id);
        if (mem) mem.accessCount++;
      });
    }

    this.stats.queries++;
    this.stats.hits += results.length;

    // Record trajectory and signal
    if (trajectory) {
      this.sona.recordTrajectory(trajectory.complete(results.length > 0 ? 'success' : 'partial'));
      this.stats.trajectories++;

      // Record learning signal for successful searches
      if (results.length > 0) {
        this.sona.recordSignal({
          type: 'feedback',
          quality: Math.min(1, results[0]?.similarity || 0.5),
          source: 'search',
          timestamp: Date.now()
        });
      }
    }

    return results;
  }

  async embed(text) {
    // Use RuvLLM native embeddings if available (SIMD optimized)
    if (this.ruvllm) {
      try {
        const emb = this.ruvllm.embed(text);
        if (emb && emb.length > 0) {
          return Array.from(emb);
        }
      } catch (e) {
        // Fall through to fallback
      }
    }

    // Fallback: Character n-gram embeddings
    const vector = new Array(this.dimensions).fill(0);
    const normalized = text.toLowerCase().replace(/[^a-z0-9\s]/g, '');
    const words = normalized.split(/\s+/);

    // Word-level features
    words.forEach((word, wi) => {
      for (let i = 0; i < word.length; i++) {
        const charCode = word.charCodeAt(i);
        const idx = (charCode * 7 + wi * 13 + i * 31) % this.dimensions;
        vector[idx] += 1;
      }
      // Bigrams
      for (let i = 0; i < word.length - 1; i++) {
        const bigram = word.charCodeAt(i) * 256 + word.charCodeAt(i + 1);
        const idx = (bigram * 17) % this.dimensions;
        vector[idx] += 0.5;
      }
    });

    // Position encoding
    words.forEach((word, pos) => {
      const posIdx = (pos * 23) % this.dimensions;
      vector[posIdx] += 0.1 * (1 / (pos + 1));
    });

    // Normalize
    const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0)) || 1;
    return vector.map(v => v / norm);
  }

  cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
  }

  // Run SONA background learning loop
  runBackgroundLearning() {
    if (!this.sona) return { patternsLearned: 0, trajectoriesProcessed: 0 };

    const result = this.sona.runBackgroundLoop();
    this.stats.patterns += result.patternsLearned;
    return result;
  }

  // Get SONA learning statistics
  getSonaStats() {
    if (!this.sona) return null;
    return this.sona.stats();
  }

  // Force learning cycle
  forceLearn() {
    const results = {
      background: null,
      ruvllm: null
    };

    if (this.sona) {
      results.background = this.runBackgroundLearning();
    }

    if (this.ruvllm) {
      try {
        results.ruvllm = this.ruvllm.forceLearn();
      } catch (e) {
        results.ruvllm = e.message;
      }
    }

    return results;
  }

  // Provide feedback for learning
  feedback(memoryId, rating, correction) {
    if (this.ruvllm) {
      try {
        this.ruvllm.feedback({ requestId: memoryId, rating, correction });
      } catch (e) {
        // Ignore
      }
    }

    if (this.sona) {
      this.sona.recordSignal({
        type: 'feedback',
        quality: rating / 5,
        source: 'user',
        timestamp: Date.now()
      });
    }
  }

  size() {
    return this.memories.length;
  }

  getStats() {
    return {
      ...this.stats,
      simdEnabled: this.simdAvailable,
      ruvllmNative: this.ruvllm?.isNativeLoaded?.() || false,
      sonaEnabled: !!this.sona,
      ruvllmStats: this.ruvllm?.stats?.() || null,
      sonaStats: this.getSonaStats()
    };
  }

  toJSON() {
    return {
      memories: this.memories,
      knowledgeGraph: this.knowledgeGraph,
      stats: this.getStats(),
      config: {
        dimensions: this.dimensions,
        distanceMetric: this.distanceMetric,
        namespace: this.namespace
      }
    };
  }

  fromJSON(data) {
    this.memories = data.memories || [];
    this.knowledgeGraph = data.knowledgeGraph || { nodes: [], edges: [] };
    this.stats = { ...this.stats, ...data.stats };

    // Re-add memories to RuvLLM HNSW index
    if (this.ruvllm) {
      for (const mem of this.memories) {
        try {
          this.ruvllm.addMemory(mem.text, mem.metadata);
        } catch (e) {
          // Ignore
        }
      }
    }
  }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

function generateLocalSyntheticData(dataType, count) {
  const data = [];
  const random = Math.random;

  const templates = {
    ecommerce: () => ({
      title: `${['Premium', 'Ultra', 'Pro', 'Essential'][Math.floor(random() * 4)]} ${['Headphones', 'Speaker', 'Watch', 'Camera'][Math.floor(random() * 4)]}`,
      description: `High-quality ${['wireless', 'bluetooth', 'smart', 'portable'][Math.floor(random() * 4)]} device with advanced features`,
      category: ['Electronics', 'Audio', 'Wearables', 'Cameras'][Math.floor(random() * 4)],
      price: Math.floor(50 + random() * 500),
      rating: Math.round((3 + random() * 2) * 10) / 10
    }),
    social: () => ({
      title: `@user${Math.floor(random() * 1000)}`,
      description: `${['Just discovered', 'Loving this', 'Check out', 'Amazing'][Math.floor(random() * 4)]} #${['tech', 'innovation', 'trending'][Math.floor(random() * 3)]}`,
      category: 'social',
      engagement: Math.floor(random() * 10000)
    }),
    jobs: () => ({
      title: `${['Senior', 'Junior', 'Lead', 'Staff'][Math.floor(random() * 4)]} ${['Engineer', 'Designer', 'Manager', 'Analyst'][Math.floor(random() * 4)]}`,
      description: `Looking for talented professionals to join our ${['growing', 'dynamic', 'innovative'][Math.floor(random() * 3)]} team`,
      category: ['Engineering', 'Design', 'Product', 'Data'][Math.floor(random() * 4)],
      salary: Math.floor(80000 + random() * 120000)
    }),
    real_estate: () => ({
      title: `${Math.floor(1 + random() * 5)} Bed ${['House', 'Apartment', 'Condo'][Math.floor(random() * 3)]}`,
      description: `Beautiful property in ${['Downtown', 'Suburbs', 'Waterfront'][Math.floor(random() * 3)]} location`,
      category: 'real_estate',
      price: Math.floor(200000 + random() * 800000)
    })
  };

  const generator = templates[dataType] || templates.ecommerce;

  for (let i = 0; i < count; i++) {
    data.push(generator());
  }

  return data;
}

async function loadSession(memoryStore, sessionId) {
  try {
    const store = await Actor.openKeyValueStore();
    const data = await store.getValue(`session_${sessionId}`);
    if (data) {
      memoryStore.fromJSON(data);
      log.info(`Loaded session ${sessionId} with ${memoryStore.size()} memories`);
    }
  } catch (e) {
    log.warning(`Could not load session: ${e.message}`);
  }
}

async function saveSession(memoryStore, sessionId) {
  try {
    const store = await Actor.openKeyValueStore();
    await store.setValue(`session_${sessionId}`, memoryStore.toJSON());
    log.info(`Saved session ${sessionId} with ${memoryStore.size()} memories`);
  } catch (e) {
    log.warning(`Could not save session: ${e.message}`);
  }
}

// Initialize Actor
await Actor.init();

try {
  const input = await Actor.getInput() || {};

  const {
    action = 'demo',
    memories = [],
    query = '',
    chatHistory = [],
    chatMessage = '',
    topK = 10,
    similarityThreshold = 0.7,
    distanceMetric = 'cosine',
    embeddingModel = 'ruvllm-768',
    namespace = 'default',
    sessionId,
    learningEnabled = true,
    knowledgeGraphEnabled = false,
    integrationConfig = {},
    scraperConfig = {},
    exportFormat = 'json',
    importData = null,
    provider = 'gemini',
    apiKey,
    model = 'gemini-2.0-flash-exp',
    seed,
    // SONA/TRM specific options
    sonaEnabled = true,
    ewcLambda = 2000,
    patternThreshold = 0.85,
    forceBackgroundLearning = false
  } = input;

  // Determine dimensions based on embedding model
  const dimensions = embeddingModel.includes('384') ? 384 : 768;

  log.info('AI Memory Engine (RuvLLM Powered)', {
    action,
    namespace,
    embeddingModel,
    provider,
    ruvllmLoaded,
    simd: hasSimdSupport?.() || false,
    version: version?.() || 'fallback'
  });

  // Initialize memory store with RuvLLM
  const memoryStore = new MemoryStore({
    dimensions,
    distanceMetric,
    namespace,
    learningEnabled: learningEnabled && sonaEnabled,
    ewcLambda,
    patternThreshold
  });

  // Load existing session if provided
  if (sessionId) {
    await loadSession(memoryStore, sessionId);
  }

  // Get API keys
  const geminiKey = provider === 'gemini' ? (apiKey || process.env.GEMINI_API_KEY) : null;
  const openRouterKey = provider === 'openrouter' ? (apiKey || process.env.OPENROUTER_API_KEY) : null;
  const apifyToken = process.env.APIFY_TOKEN;

  let result = {};

  // Execute action
  switch (action) {
    case 'demo':
      result = await runDemo(memoryStore, geminiKey, model);
      break;

    case 'store':
      result = await storeMemories(memoryStore, memories, knowledgeGraphEnabled);
      await Actor.charge({ eventName: 'memory-store', count: memories.length });
      break;

    case 'search':
      result = await searchMemories(memoryStore, query, topK, similarityThreshold);
      await Actor.charge({ eventName: 'memory-search', count: 1 });
      break;

    case 'chat':
      result = await chatWithMemory(memoryStore, chatMessage, chatHistory, geminiKey, openRouterKey, model, provider);
      await Actor.charge({ eventName: 'chat-interaction', count: 1 });
      break;

    case 'build_knowledge':
      result = await buildKnowledgeGraph(memoryStore, memories, geminiKey, model);
      await Actor.charge({ eventName: 'knowledge-graph-build', count: 1 });
      break;

    case 'recommend':
      result = await getRecommendations(memoryStore, query, topK);
      await Actor.charge({ eventName: 'recommendation', count: topK });
      break;

    case 'analyze':
      result = await analyzePatterns(memoryStore, geminiKey, model);
      await Actor.charge({ eventName: 'pattern-analysis', count: 1 });
      break;

    case 'export':
      result = await exportMemory(memoryStore, exportFormat);
      await Actor.charge({ eventName: 'memory-export', count: 1 });
      break;

    case 'import':
      result = await importMemory(memoryStore, importData);
      await Actor.charge({ eventName: 'memory-import', count: 1 });
      break;

    case 'integrate_synthetic':
      result = await integrateSyntheticData(memoryStore, integrationConfig, apifyToken);
      await Actor.charge({ eventName: 'synthetic-integration', count: 1 });
      break;

    case 'integrate_scraper':
      result = await integrateWebScraper(memoryStore, scraperConfig, apifyToken);
      await Actor.charge({ eventName: 'scraper-integration', count: 1 });
      break;

    case 'learn':
      // Force SONA background learning cycle
      result = memoryStore.forceLearn();
      await Actor.charge({ eventName: 'learning-cycle', count: 1 });
      break;

    default:
      throw new Error(`Unknown action: ${action}`);
  }

  // Run background learning if enabled
  if (forceBackgroundLearning || (learningEnabled && sonaEnabled)) {
    const learningResult = memoryStore.runBackgroundLearning();
    if (learningResult.patternsLearned > 0) {
      log.info('Background learning completed', learningResult);
    }
  }

  // Save session if provided
  if (sessionId) {
    await saveSession(memoryStore, sessionId);
  }

  // Push results
  await Actor.pushData({
    action,
    success: true,
    result,
    metadata: {
      namespace,
      memoryCount: memoryStore.size(),
      sessionId,
      timestamp: new Date().toISOString(),
      engine: {
        ruvllmLoaded,
        simdEnabled: memoryStore.simdAvailable,
        nativeLoaded: memoryStore.ruvllm?.isNativeLoaded?.() || false,
        sonaEnabled: !!memoryStore.sona,
        version: version?.() || 'fallback'
      }
    }
  });

  log.info('Operation completed successfully', { action, memoryCount: memoryStore.size() });

} catch (error) {
  log.error('Actor failed', { error: error.message });
  await Actor.pushData({
    action: 'error',
    success: false,
    error: error.message,
    timestamp: new Date().toISOString()
  });
  throw error;
} finally {
  await Actor.exit();
}

// ============================================
// ACTION IMPLEMENTATIONS
// ============================================

async function runDemo(memoryStore, apiKey, model) {
  log.info('Running demo with RuvLLM-powered memory...');

  // Sample memories for demo
  const sampleMemories = [
    { text: "Customer prefers eco-friendly products and fast shipping", metadata: { type: "preference", customerId: "C001" } },
    { text: "Product A pairs well with Product B for complete home automation", metadata: { type: "knowledge", category: "products" } },
    { text: "Support ticket #1234: Customer asked about return policy", metadata: { type: "support", ticketId: "1234" } },
    { text: "User mentioned they work from home and need quiet equipment", metadata: { type: "context", userId: "U001" } },
    { text: "Successful upsell: Added warranty after mentioning 3-year protection", metadata: { type: "sales", outcome: "success" } },
    { text: "Customer complaint resolved by offering 20% discount on next order", metadata: { type: "resolution", strategy: "discount" } },
    { text: "Peak shopping hours are between 7-9 PM on weekdays", metadata: { type: "analytics", insight: "timing" } },
    { text: "Mobile users convert 30% better with one-click checkout", metadata: { type: "analytics", insight: "conversion" } }
  ];

  // Store memories (with trajectory tracking)
  for (const mem of sampleMemories) {
    await memoryStore.add(mem.text, mem.metadata);
  }
  log.info(`Stored ${sampleMemories.length} demo memories`);

  // Demo search with HNSW + SIMD
  const searchResults = await memoryStore.search("What does the customer prefer?", 3, 0.3);
  log.info(`Search found ${searchResults.length} relevant memories`);

  // Demo recommendations
  const recommendResults = await memoryStore.search("home automation products", 3, 0.3);

  // Force a learning cycle to show SONA in action
  const learningResult = memoryStore.forceLearn();

  return {
    demo: true,
    memoriesStored: sampleMemories.length,
    sampleSearch: {
      query: "What does the customer prefer?",
      results: searchResults.map(r => ({
        text: r.text,
        similarity: Math.round(r.similarity * 100) / 100,
        metadata: r.metadata
      }))
    },
    sampleRecommendation: {
      query: "home automation products",
      results: recommendResults.map(r => ({
        text: r.text,
        similarity: Math.round(r.similarity * 100) / 100
      }))
    },
    stats: memoryStore.getStats(),
    learningResult,
    engine: {
      ruvllmLoaded,
      simdEnabled: memoryStore.simdAvailable,
      nativeLoaded: memoryStore.ruvllm?.isNativeLoaded?.() || false,
      sonaEnabled: !!memoryStore.sona,
      version: version?.() || 'fallback'
    },
    capabilities: {
      trm: "Tiny Recursive Models - 7M params recursive reasoning",
      sona: "Self-Optimizing Neural Architecture - 3-tier learning",
      ewc: "EWC++ anti-forgetting during retraining",
      hnsw: "HNSW indexing for O(log N) search",
      simd: "SIMD-optimized embeddings and similarity"
    },
    nextSteps: [
      "Use 'store' action to add your own memories",
      "Use 'search' action to find similar content",
      "Use 'chat' action for conversational AI with memory",
      "Use 'learn' action to force a SONA learning cycle",
      "Use 'integrate_synthetic' to generate test data",
      "Use 'build_knowledge' to create knowledge graphs"
    ]
  };
}

async function storeMemories(memoryStore, memories, buildGraph = false) {
  log.info(`Storing ${memories.length} memories with RuvLLM...`);
  const stored = [];

  for (const mem of memories) {
    const text = typeof mem === 'string' ? mem : mem.text;
    const metadata = typeof mem === 'string' ? {} : mem.metadata || {};
    const memory = await memoryStore.add(text, metadata);
    stored.push({
      id: memory.id,
      text: text.substring(0, 100) + (text.length > 100 ? '...' : ''),
      metadata
    });
  }

  return {
    stored: stored.length,
    memories: stored,
    totalMemories: memoryStore.size(),
    sonaStats: memoryStore.getSonaStats()
  };
}

async function searchMemories(memoryStore, query, topK, threshold) {
  log.info(`Searching for: "${query.substring(0, 50)}..." using RuvLLM HNSW`);

  if (memoryStore.size() === 0) {
    return {
      query,
      results: [],
      message: "No memories stored yet. Use 'store' action first."
    };
  }

  const results = await memoryStore.search(query, topK, threshold);

  return {
    query,
    resultsFound: results.length,
    results: results.map(r => ({
      text: r.text,
      similarity: Math.round(r.similarity * 1000) / 1000,
      metadata: r.metadata,
      id: r.id
    })),
    stats: memoryStore.getStats()
  };
}

async function chatWithMemory(memoryStore, message, history, geminiKey, openRouterKey, model, provider) {
  log.info(`Chat message: "${message.substring(0, 50)}..."`);

  // Search for relevant context using RuvLLM
  const context = memoryStore.size() > 0
    ? await memoryStore.search(message, 5, 0.3)
    : [];

  const contextText = context.length > 0
    ? `Relevant memories:\n${context.map(c => `- ${c.text}`).join('\n')}\n\n`
    : '';

  const systemPrompt = `You are a helpful AI assistant with access to a self-learning memory database powered by RuvLLM with TRM reasoning and SONA adaptive learning. Use the relevant memories to provide personalized, context-aware responses.

${contextText}

Previous conversation:
${history.map(h => `${h.role}: ${h.content}`).join('\n')}

User: ${message}`;

  let response;

  if (geminiKey && provider === 'gemini') {
    const genAI = new GoogleGenerativeAI(geminiKey);
    const gemini = genAI.getGenerativeModel({ model });
    const result = await gemini.generateContent(systemPrompt);
    response = result.response.text();
  } else if (openRouterKey && provider === 'openrouter') {
    const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openRouterKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://apify.com',
        'X-Title': 'AI Memory Engine'
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: systemPrompt }],
        temperature: 0.7
      })
    });
    const data = await res.json();
    response = data.choices?.[0]?.message?.content || 'I apologize, I could not generate a response.';
  } else {
    // Local response without API
    response = context.length > 0
      ? `Based on my memory, I found ${context.length} relevant pieces of information:\n\n${context.map((c, i) => `${i + 1}. ${c.text}`).join('\n\n')}\n\nFor a more conversational response, please provide a Gemini or OpenRouter API key.`
      : `I don't have any relevant memories about "${message}". Try storing some information first using the 'store' action.`;
  }

  // Store the interaction as a memory
  await memoryStore.add(`User asked: ${message}`, { type: 'conversation', role: 'user' });
  await memoryStore.add(`Assistant responded: ${response.substring(0, 200)}`, { type: 'conversation', role: 'assistant' });

  // Provide feedback for learning
  memoryStore.feedback(`chat_${Date.now()}`, context.length > 0 ? 4 : 2, null);

  return {
    message,
    response,
    contextUsed: context.length,
    relevantMemories: context.map(c => ({
      text: c.text.substring(0, 100),
      similarity: Math.round(c.similarity * 100) / 100
    })),
    updatedHistory: [
      ...history,
      { role: 'user', content: message },
      { role: 'assistant', content: response }
    ],
    sonaStats: memoryStore.getSonaStats()
  };
}

async function buildKnowledgeGraph(memoryStore, memories, apiKey, model) {
  log.info('Building knowledge graph from memories...');

  // First store the memories if not already stored
  if (memories.length > 0) {
    await storeMemories(memoryStore, memories);
  }

  // Extract entities and relationships
  const nodes = new Map();
  const edges = [];

  for (const mem of memoryStore.memories) {
    // Simple entity extraction (words/phrases)
    const text = mem.text;
    const words = text.match(/[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g) || [];
    const entities = [...new Set(words)];

    entities.forEach(entity => {
      if (!nodes.has(entity)) {
        nodes.set(entity, {
          id: `node_${nodes.size}`,
          label: entity,
          type: 'entity',
          mentions: 1
        });
      } else {
        nodes.get(entity).mentions++;
      }
    });

    // Create edges between co-occurring entities
    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        edges.push({
          source: entities[i],
          target: entities[j],
          relation: 'co-occurs',
          weight: 1,
          context: mem.text.substring(0, 100)
        });
      }
    }
  }

  memoryStore.knowledgeGraph = {
    nodes: Array.from(nodes.values()),
    edges
  };

  return {
    nodesCreated: nodes.size,
    edgesCreated: edges.length,
    topEntities: Array.from(nodes.values())
      .sort((a, b) => b.mentions - a.mentions)
      .slice(0, 10)
      .map(n => ({ label: n.label, mentions: n.mentions })),
    sampleEdges: edges.slice(0, 5)
  };
}

async function getRecommendations(memoryStore, query, topK) {
  log.info(`Getting recommendations for: "${query}"`);

  const results = await memoryStore.search(query, topK, 0.2);

  // Group by metadata categories
  const byCategory = {};
  results.forEach(r => {
    const category = r.metadata?.category || r.metadata?.type || 'general';
    if (!byCategory[category]) byCategory[category] = [];
    byCategory[category].push(r);
  });

  return {
    query,
    recommendations: results.map(r => ({
      text: r.text,
      score: Math.round(r.similarity * 100),
      category: r.metadata?.category || 'general',
      metadata: r.metadata
    })),
    byCategory: Object.entries(byCategory).map(([cat, items]) => ({
      category: cat,
      count: items.length,
      topItem: items[0]?.text.substring(0, 100)
    }))
  };
}

async function analyzePatterns(memoryStore, apiKey, model) {
  log.info('Analyzing patterns in memories with SONA...');

  const stats = memoryStore.getStats();
  const memories = memoryStore.memories;

  // Analyze metadata patterns
  const metadataStats = {};
  memories.forEach(mem => {
    Object.entries(mem.metadata || {}).forEach(([key, value]) => {
      if (!metadataStats[key]) metadataStats[key] = {};
      const v = String(value);
      metadataStats[key][v] = (metadataStats[key][v] || 0) + 1;
    });
  });

  // Find most accessed memories
  const topAccessed = [...memories]
    .sort((a, b) => b.accessCount - a.accessCount)
    .slice(0, 5);

  // Word frequency
  const wordFreq = {};
  memories.forEach(mem => {
    mem.text.toLowerCase().split(/\s+/).forEach(word => {
      if (word.length > 3) wordFreq[word] = (wordFreq[word] || 0) + 1;
    });
  });
  const topWords = Object.entries(wordFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);

  // Get SONA learning insights
  const sonaStats = memoryStore.getSonaStats();

  return {
    totalMemories: memories.length,
    stats,
    sonaStats,
    metadataDistribution: metadataStats,
    mostAccessedMemories: topAccessed.map(m => ({
      text: m.text.substring(0, 100),
      accessCount: m.accessCount
    })),
    topKeywords: topWords.map(([word, count]) => ({ word, count })),
    insights: [
      `You have ${memories.length} memories stored`,
      `${stats.queries} searches performed with ${stats.hits} results returned`,
      `${stats.trajectories} learning trajectories recorded`,
      `${stats.patterns} patterns learned via SONA`,
      `Most common metadata keys: ${Object.keys(metadataStats).join(', ') || 'none'}`,
      `Top keyword: "${topWords[0]?.[0] || 'N/A'}" (${topWords[0]?.[1] || 0} occurrences)`,
      `SIMD acceleration: ${stats.simdEnabled ? 'enabled' : 'disabled'}`,
      `RuvLLM native: ${stats.ruvllmNative ? 'loaded' : 'fallback'}`
    ]
  };
}

async function exportMemory(memoryStore, format) {
  log.info(`Exporting memories in ${format} format...`);

  const data = memoryStore.toJSON();

  if (format === 'csv') {
    const rows = data.memories.map(m => ({
      id: m.id,
      text: m.text.replace(/,/g, ';'),
      metadata: JSON.stringify(m.metadata),
      createdAt: m.createdAt,
      accessCount: m.accessCount
    }));
    return {
      format: 'csv',
      headers: ['id', 'text', 'metadata', 'createdAt', 'accessCount'],
      rows,
      rowCount: rows.length
    };
  }

  if (format === 'embeddings') {
    return {
      format: 'embeddings',
      dimensions: memoryStore.dimensions,
      vectors: data.memories.map(m => ({
        id: m.id,
        embedding: m.embedding
      })),
      count: data.memories.length
    };
  }

  return {
    format: 'json',
    data,
    memoryCount: data.memories.length,
    graphNodes: data.knowledgeGraph.nodes.length,
    graphEdges: data.knowledgeGraph.edges.length
  };
}

async function importMemory(memoryStore, importData) {
  log.info('Importing memories...');

  if (!importData) {
    return { error: 'No import data provided' };
  }

  if (importData.memories) {
    memoryStore.fromJSON(importData);
    return {
      imported: true,
      memoriesLoaded: memoryStore.size(),
      graphNodes: memoryStore.knowledgeGraph.nodes.length
    };
  }

  // Handle array of texts
  if (Array.isArray(importData)) {
    for (const item of importData) {
      const text = typeof item === 'string' ? item : item.text;
      const metadata = typeof item === 'string' ? {} : item.metadata || {};
      await memoryStore.add(text, metadata, item.embedding);
    }
    return {
      imported: true,
      memoriesLoaded: importData.length
    };
  }

  return { error: 'Invalid import data format' };
}

async function integrateSyntheticData(memoryStore, config, apifyToken) {
  log.info('Integrating with Synthetic Data Generator...');

  const {
    syntheticDataActor = 'ruv/ai-synthetic-data-generator',
    dataType = 'ecommerce',
    count = 100,
    memorizeFields = ['title', 'description', 'category']
  } = config;

  if (!apifyToken) {
    // Generate local synthetic data
    log.info('No Apify token - generating local synthetic data');
    const localData = generateLocalSyntheticData(dataType, count);

    for (const item of localData) {
      const text = memorizeFields.map(f => item[f]).filter(Boolean).join('. ');
      await memoryStore.add(text, { source: 'synthetic', dataType, ...item.metadata });
    }

    return {
      integrated: true,
      source: 'local',
      memoriesCreated: localData.length,
      dataType
    };
  }

  try {
    const client = new ApifyClient({ token: apifyToken });

    log.info(`Calling ${syntheticDataActor}...`);
    const run = await client.actor(syntheticDataActor).call({
      dataType,
      count
    });

    const { items } = await client.dataset(run.defaultDatasetId).listItems();
    log.info(`Got ${items.length} items from synthetic data generator`);

    // Store in memory
    for (const item of items) {
      const data = item.data || item;
      const text = memorizeFields.map(f => data[f]).filter(Boolean).join('. ');
      if (text) {
        await memoryStore.add(text, {
          source: 'synthetic',
          dataType,
          originalId: item.id
        });
      }
    }

    return {
      integrated: true,
      source: syntheticDataActor,
      itemsReceived: items.length,
      memoriesCreated: memoryStore.size(),
      dataType
    };
  } catch (error) {
    log.warning(`Integration failed: ${error.message}. Using local generation.`);
    const localData = generateLocalSyntheticData(dataType, count);

    for (const item of localData) {
      const text = memorizeFields.map(f => item[f]).filter(Boolean).join('. ');
      await memoryStore.add(text, { source: 'synthetic-local', dataType });
    }

    return {
      integrated: true,
      source: 'local-fallback',
      memoriesCreated: localData.length,
      note: error.message
    };
  }
}

async function integrateWebScraper(memoryStore, config, apifyToken) {
  log.info('Integrating with Web Scraper...');

  const {
    urls = [],
    selector = 'article',
    maxPages = 10
  } = config;

  if (!apifyToken || urls.length === 0) {
    return {
      integrated: false,
      error: 'Apify token and URLs required for web scraping'
    };
  }

  try {
    const client = new ApifyClient({ token: apifyToken });

    log.info(`Scraping ${urls.length} URLs...`);
    const run = await client.actor('apify/cheerio-scraper').call({
      startUrls: urls.map(url => ({ url })),
      pageFunction: async function pageFunction(context) {
        const { $, request } = context;
        const items = [];
        $(context.input.selector || 'article, .content, main').each((i, el) => {
          const text = $(el).text().trim();
          if (text.length > 50) {
            items.push({
              url: request.url,
              text: text.substring(0, 2000),
              title: $('title').text() || ''
            });
          }
        });
        return items;
      },
      maxRequestsPerCrawl: maxPages
    });

    const { items } = await client.dataset(run.defaultDatasetId).listItems();
    log.info(`Scraped ${items.length} content items`);

    // Store in memory
    for (const item of items) {
      await memoryStore.add(item.text, {
        source: 'web',
        url: item.url,
        title: item.title
      });
    }

    return {
      integrated: true,
      urlsScraped: urls.length,
      contentItemsFound: items.length,
      memoriesCreated: memoryStore.size()
    };
  } catch (error) {
    return {
      integrated: false,
      error: error.message
    };
  }
}
