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

  // Get a single memory by ID
  get(id) {
    return this.memories.find(m => m.id === id) || null;
  }

  // List memories with pagination and filtering
  list(options = {}) {
    const { limit = 100, offset = 0, filter = null, sortBy = 'createdAt', sortOrder = 'desc' } = options;

    let filtered = [...this.memories];

    // Apply metadata filter
    if (filter) {
      filtered = filtered.filter(m => {
        for (const [key, value] of Object.entries(filter)) {
          if (m.metadata[key] !== value) return false;
        }
        return true;
      });
    }

    // Sort
    filtered.sort((a, b) => {
      const aVal = sortBy === 'accessCount' ? a.accessCount : new Date(a.createdAt).getTime();
      const bVal = sortBy === 'accessCount' ? b.accessCount : new Date(b.createdAt).getTime();
      return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
    });

    // Paginate
    const paginated = filtered.slice(offset, offset + limit);

    return {
      items: paginated,
      total: filtered.length,
      limit,
      offset,
      hasMore: offset + limit < filtered.length
    };
  }

  // Delete a memory by ID
  delete(id) {
    const index = this.memories.findIndex(m => m.id === id);
    if (index === -1) return null;

    const deleted = this.memories.splice(index, 1)[0];
    this.stats.deletes = (this.stats.deletes || 0) + 1;
    return deleted;
  }

  // Delete multiple memories by IDs
  deleteMany(ids) {
    const deleted = [];
    for (const id of ids) {
      const result = this.delete(id);
      if (result) deleted.push(result);
    }
    return deleted;
  }

  // Delete by metadata filter
  deleteByFilter(filter) {
    const toDelete = this.memories.filter(m => {
      for (const [key, value] of Object.entries(filter)) {
        if (m.metadata[key] !== value) return false;
      }
      return true;
    });

    const deletedIds = toDelete.map(m => m.id);
    return this.deleteMany(deletedIds);
  }

  // Update a memory
  async update(id, updates) {
    const memory = this.memories.find(m => m.id === id);
    if (!memory) return null;

    // Update text if provided
    if (updates.text && updates.text !== memory.text) {
      memory.text = updates.text;
      memory.embedding = await this.embed(updates.text);
      memory.updatedAt = new Date().toISOString();
    }

    // Update metadata if provided
    if (updates.metadata) {
      memory.metadata = { ...memory.metadata, ...updates.metadata };
    }

    this.stats.updates = (this.stats.updates || 0) + 1;
    return memory;
  }

  // Clear all memories
  clear() {
    const count = this.memories.length;
    this.memories = [];
    this.knowledgeGraph = { nodes: [], edges: [] };
    this.stats.clears = (this.stats.clears || 0) + 1;
    return count;
  }

  // Hybrid search (semantic + keyword)
  async hybridSearch(query, options = {}) {
    const { topK = 10, threshold = 0.5, keywordWeight = 0.3, semanticWeight = 0.7 } = options;

    // Semantic search
    const semanticResults = await this.search(query, topK * 2, 0);

    // Keyword search (simple BM25-like scoring)
    const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
    const keywordScores = this.memories.map(mem => {
      const text = mem.text.toLowerCase();
      let score = 0;
      for (const term of queryTerms) {
        if (text.includes(term)) {
          const tf = (text.match(new RegExp(term, 'g')) || []).length;
          score += Math.log(1 + tf);
        }
      }
      return { id: mem.id, keywordScore: score / (queryTerms.length || 1) };
    });

    // Combine scores
    const combined = semanticResults.map(r => {
      const kw = keywordScores.find(k => k.id === r.id);
      const kwScore = kw?.keywordScore || 0;
      const combinedScore = (r.similarity * semanticWeight) + (kwScore * keywordWeight);
      return { ...r, keywordScore: kwScore, combinedScore };
    });

    // Sort by combined score and filter
    return combined
      .filter(r => r.combinedScore >= threshold)
      .sort((a, b) => b.combinedScore - a.combinedScore)
      .slice(0, topK);
  }

  // Batch search - multiple queries at once
  async batchSearch(queries, options = {}) {
    const { topK = 10, threshold = 0.7 } = options;
    const results = [];

    for (const query of queries) {
      const queryResults = await this.search(query, topK, threshold);
      results.push({
        query,
        results: queryResults,
        count: queryResults.length
      });
    }

    return results;
  }

  // Find duplicates based on similarity threshold
  async findDuplicates(threshold = 0.95) {
    const duplicates = [];

    for (let i = 0; i < this.memories.length; i++) {
      for (let j = i + 1; j < this.memories.length; j++) {
        const similarity = this.ruvllm
          ? this.ruvllm.similarity(this.memories[i].text, this.memories[j].text)
          : this.cosineSimilarity(this.memories[i].embedding, this.memories[j].embedding);

        if (similarity >= threshold) {
          duplicates.push({
            memory1: { id: this.memories[i].id, text: this.memories[i].text.substring(0, 100) },
            memory2: { id: this.memories[j].id, text: this.memories[j].text.substring(0, 100) },
            similarity
          });
        }
      }
    }

    return duplicates;
  }

  // Deduplicate - remove similar memories, keeping the oldest
  async deduplicate(threshold = 0.95) {
    const duplicates = await this.findDuplicates(threshold);
    const toRemove = new Set();

    for (const dup of duplicates) {
      // Keep the one with more access count, or the older one
      const mem1 = this.get(dup.memory1.id);
      const mem2 = this.get(dup.memory2.id);

      if (mem1 && mem2) {
        const removeId = mem1.accessCount >= mem2.accessCount ? dup.memory2.id : dup.memory1.id;
        toRemove.add(removeId);
      }
    }

    const removed = this.deleteMany([...toRemove]);
    return { duplicatesFound: duplicates.length, removed: removed.length };
  }

  getStats() {
    return {
      ...this.stats,
      memoryCount: this.memories.length,
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
    // Use named store for cross-run persistence (sanitize sessionId for store name)
    const storeName = `ai-memory-${sessionId.replace(/[^a-zA-Z0-9-_]/g, '_').substring(0, 50)}`;
    const store = await Actor.openKeyValueStore(storeName);
    const data = await store.getValue('memory_data');
    if (data) {
      memoryStore.fromJSON(data);
      log.info(`Loaded session ${sessionId} with ${memoryStore.size()} memories from store ${storeName}`);
    } else {
      log.info(`No existing session data found for ${sessionId}`);
    }
  } catch (e) {
    log.warning(`Could not load session: ${e.message}`);
  }
}

async function saveSession(memoryStore, sessionId) {
  try {
    // Use named store for cross-run persistence (sanitize sessionId for store name)
    const storeName = `ai-memory-${sessionId.replace(/[^a-zA-Z0-9-_]/g, '_').substring(0, 50)}`;
    const store = await Actor.openKeyValueStore(storeName);
    await store.setValue('memory_data', memoryStore.toJSON());
    log.info(`Saved session ${sessionId} with ${memoryStore.size()} memories to store ${storeName}`);
  } catch (e) {
    log.warning(`Could not save session: ${e.message}`);
  }
}

// ============================================
// ONE-CLICK ACTOR INTEGRATION
// ============================================

const POPULAR_ACTORS = {
  'apify/google-maps-scraper': {
    name: 'Google Maps Scraper',
    defaultFields: ['title', 'description', 'address', 'phone', 'website', 'totalScore', 'reviewsCount', 'categoryName'],
    textTemplate: (item) => `${item.title || ''} - ${item.address || ''} - ${item.categoryName || ''} - Rating: ${item.totalScore || 'N/A'} (${item.reviewsCount || 0} reviews)`
  },
  'apify/instagram-scraper': {
    name: 'Instagram Scraper',
    defaultFields: ['caption', 'hashtags', 'likesCount', 'commentsCount', 'ownerUsername'],
    textTemplate: (item) => `@${item.ownerUsername || 'unknown'}: ${item.caption || ''} ${(item.hashtags || []).map(h => '#' + h).join(' ')}`
  },
  'apify/tiktok-scraper': {
    name: 'TikTok Scraper',
    defaultFields: ['text', 'authorMeta', 'hashtags', 'diggCount', 'playCount'],
    textTemplate: (item) => `@${item.authorMeta?.name || 'unknown'}: ${item.text || ''} - ${item.playCount || 0} plays`
  },
  'apify/youtube-scraper': {
    name: 'YouTube Scraper',
    defaultFields: ['title', 'description', 'channelName', 'viewCount', 'likes'],
    textTemplate: (item) => `${item.title || ''} by ${item.channelName || 'unknown'} - ${item.viewCount || 0} views`
  },
  'apify/web-scraper': {
    name: 'Web Scraper',
    defaultFields: ['url', 'title', 'text'],
    textTemplate: (item) => `${item.title || item.url || ''}: ${(item.text || '').substring(0, 500)}`
  },
  'apify/website-content-crawler': {
    name: 'Website Content Crawler',
    defaultFields: ['url', 'title', 'text', 'markdown'],
    textTemplate: (item) => `${item.title || ''} (${item.url || ''}): ${(item.text || item.markdown || '').substring(0, 500)}`
  },
  'apify/twitter-scraper': {
    name: 'Twitter/X Scraper',
    defaultFields: ['text', 'author', 'replyCount', 'retweetCount', 'likeCount'],
    textTemplate: (item) => `@${item.author?.userName || 'unknown'}: ${item.text || ''}`
  },
  'apify/amazon-scraper': {
    name: 'Amazon Scraper',
    defaultFields: ['title', 'description', 'price', 'stars', 'reviewsCount'],
    textTemplate: (item) => `${item.title || ''} - $${item.price || 'N/A'} - ${item.stars || 'N/A'} stars (${item.reviewsCount || 0} reviews)`
  },
  'apify/tripadvisor-scraper': {
    name: 'TripAdvisor Scraper',
    defaultFields: ['name', 'description', 'rating', 'reviewsCount', 'address'],
    textTemplate: (item) => `${item.name || ''} - ${item.address || ''} - Rating: ${item.rating || 'N/A'}`
  },
  'apify/linkedin-scraper': {
    name: 'LinkedIn Scraper',
    defaultFields: ['name', 'headline', 'summary', 'company', 'location'],
    textTemplate: (item) => `${item.name || ''} - ${item.headline || ''} at ${item.company || 'N/A'}`
  }
};

async function integrateActorResults(memoryStore, actorId, config, apifyToken) {
  const client = new ApifyClient({ token: apifyToken });

  const actorConfig = POPULAR_ACTORS[actorId] || {
    name: actorId,
    defaultFields: config.memorizeFields || ['title', 'description', 'text'],
    textTemplate: (item) => JSON.stringify(item).substring(0, 500)
  };

  let items = [];

  // Get results from specified run or latest
  if (config.runId) {
    const run = config.runId === 'latest'
      ? await client.actor(actorId).lastRun().get()
      : await client.run(config.runId).get();

    if (run) {
      const dataset = await client.dataset(run.defaultDatasetId).listItems({ limit: config.limit || 1000 });
      items = dataset.items;
    }
  } else if (config.datasetId) {
    const dataset = await client.dataset(config.datasetId).listItems({ limit: config.limit || 1000 });
    items = dataset.items;
  } else {
    // Run the actor with provided input
    const run = await client.actor(actorId).call(config.actorInput || {}, {
      memory: config.memory || 1024,
      timeout: config.timeout || 300
    });
    const dataset = await client.dataset(run.defaultDatasetId).listItems({ limit: config.limit || 1000 });
    items = dataset.items;
  }

  log.info(`Retrieved ${items.length} items from ${actorConfig.name}`);

  // Memorize items
  const memorizeFields = config.memorizeFields || actorConfig.defaultFields;
  const stored = [];

  for (const item of items) {
    const text = config.customTemplate
      ? config.customTemplate(item)
      : actorConfig.textTemplate(item);

    if (text && text.trim()) {
      const metadata = {
        source: actorId,
        sourceActor: actorConfig.name,
        ...Object.fromEntries(memorizeFields.filter(f => item[f] !== undefined).map(f => [f, item[f]]))
      };

      await memoryStore.add(text, metadata);
      stored.push({ text: text.substring(0, 100), metadata });
    }
  }

  return {
    actorId,
    actorName: actorConfig.name,
    itemsRetrieved: items.length,
    memoriesStored: stored.length,
    sampleMemories: stored.slice(0, 5)
  };
}

// ============================================
// PRE-BUILT TEMPLATES
// ============================================

const TEMPLATES = {
  'lead-intelligence': {
    name: 'Lead Intelligence',
    description: 'Sales lead tracking and enrichment',
    sampleMemories: [
      { text: 'Lead: John Smith, CEO at TechCorp, interested in enterprise solutions, budget $50k+', metadata: { type: 'lead', stage: 'qualified' }},
      { text: 'Company: TechCorp - 500 employees, Series B, HQ in San Francisco', metadata: { type: 'company', industry: 'technology' }},
      { text: 'Meeting notes: Discussed Q1 implementation timeline, decision maker is CTO', metadata: { type: 'notes', date: new Date().toISOString() }}
    ],
    suggestedQueries: ['qualified leads', 'decision makers', 'enterprise prospects', 'budget over 50k']
  },
  'customer-support': {
    name: 'Customer Support Knowledge Base',
    description: 'Support ticket resolution and FAQ',
    sampleMemories: [
      { text: 'Issue: Login problems - Solution: Clear browser cache and cookies, try incognito mode', metadata: { type: 'solution', category: 'authentication' }},
      { text: 'FAQ: How to reset password - Go to Settings > Security > Reset Password', metadata: { type: 'faq', category: 'account' }},
      { text: 'Escalation: Billing issues over $1000 should go to finance team', metadata: { type: 'process', priority: 'high' }}
    ],
    suggestedQueries: ['login issues', 'password reset', 'billing problems', 'escalation process']
  },
  'research-assistant': {
    name: 'Research Assistant',
    description: 'Academic and market research',
    sampleMemories: [
      { text: 'Study: AI adoption in enterprises grew 35% in 2024 (Source: Gartner)', metadata: { type: 'statistic', topic: 'AI', year: 2024 }},
      { text: 'Market insight: Vector databases market expected to reach $5B by 2028', metadata: { type: 'market', topic: 'vector-db' }},
      { text: 'Research finding: RAG systems improve accuracy by 40% over base LLMs', metadata: { type: 'research', topic: 'RAG' }}
    ],
    suggestedQueries: ['AI adoption statistics', 'market projections', 'RAG performance']
  },
  'competitor-intelligence': {
    name: 'Competitor Intelligence',
    description: 'Track competitor activities and market position',
    sampleMemories: [
      { text: 'Competitor A launched new pricing tier - $99/mo for startups', metadata: { type: 'pricing', competitor: 'A' }},
      { text: 'Competitor B announced partnership with Microsoft for AI integration', metadata: { type: 'partnership', competitor: 'B' }},
      { text: 'Market share Q4: We have 15%, Competitor A 25%, Competitor B 20%', metadata: { type: 'market-share', quarter: 'Q4' }}
    ],
    suggestedQueries: ['competitor pricing', 'partnerships', 'market share']
  },
  'content-library': {
    name: 'Content Library',
    description: 'Content ideas and reference material',
    sampleMemories: [
      { text: 'Blog idea: "10 Ways AI is Transforming Customer Service" - target SEO keywords', metadata: { type: 'idea', format: 'blog' }},
      { text: 'Case study template: Problem > Solution > Results > Quote > CTA', metadata: { type: 'template', format: 'case-study' }},
      { text: 'Best performing content: Video tutorials get 3x engagement vs text', metadata: { type: 'insight', topic: 'engagement' }}
    ],
    suggestedQueries: ['blog ideas', 'content templates', 'high engagement content']
  },
  'product-catalog': {
    name: 'Product Catalog',
    description: 'E-commerce product knowledge base',
    sampleMemories: [
      { text: 'Product: Wireless Headphones Pro - $199, 40hr battery, ANC, Bluetooth 5.3', metadata: { type: 'product', category: 'audio', sku: 'WHP-001' }},
      { text: 'Inventory: Low stock alert for SKU WHP-001 - 23 units remaining', metadata: { type: 'inventory', priority: 'high' }},
      { text: 'Customer feedback: Headphones comfort rated 4.8/5, sound quality 4.6/5', metadata: { type: 'feedback', sku: 'WHP-001' }}
    ],
    suggestedQueries: ['low stock items', 'best rated products', 'product specifications']
  }
};

async function loadTemplate(memoryStore, templateId) {
  const template = TEMPLATES[templateId];
  if (!template) {
    throw new Error(`Unknown template: ${templateId}. Available: ${Object.keys(TEMPLATES).join(', ')}`);
  }

  // Store sample memories
  for (const mem of template.sampleMemories) {
    await memoryStore.add(mem.text, { ...mem.metadata, template: templateId });
  }

  // Run sample searches
  const searchResults = [];
  for (const query of template.suggestedQueries.slice(0, 2)) {
    const results = await memoryStore.search(query, 3, 0.5);
    searchResults.push({ query, results: results.slice(0, 2) });
  }

  return {
    template: templateId,
    name: template.name,
    description: template.description,
    memoriesLoaded: template.sampleMemories.length,
    suggestedQueries: template.suggestedQueries,
    sampleSearches: searchResults
  };
}

// ============================================
// NATURAL LANGUAGE COMMANDS
// ============================================

const NL_PATTERNS = [
  { pattern: /^remember\s+(?:that\s+)?(.+)/i, action: 'store', extract: (m) => ({ text: m[1] }) },
  { pattern: /^forget\s+(?:about\s+)?(.+)/i, action: 'delete_by_query', extract: (m) => ({ query: m[1] }) },
  { pattern: /^(?:what|tell me|show)\s+(?:do\s+(?:you|we)\s+know\s+about|about)\s+(.+)/i, action: 'search', extract: (m) => ({ query: m[1] }) },
  { pattern: /^(?:find|search|look\s+for)\s+(.+)/i, action: 'search', extract: (m) => ({ query: m[1] }) },
  { pattern: /^(?:how\s+many|count)\s+memories/i, action: 'stats', extract: () => ({}) },
  { pattern: /^(?:list|show)\s+(?:all\s+)?memories/i, action: 'list', extract: () => ({}) },
  { pattern: /^clear\s+(?:all\s+)?(?:memories|everything)/i, action: 'clear', extract: () => ({}) },
  { pattern: /^(?:analyze|insights|patterns)/i, action: 'analyze', extract: () => ({}) },
  { pattern: /^(?:similar|duplicates|find\s+duplicates)/i, action: 'find_duplicates', extract: () => ({}) },
  { pattern: /^(?:export|download|backup)/i, action: 'export', extract: () => ({}) },
  { pattern: /^(?:help|commands|what\s+can\s+you\s+do)/i, action: 'help', extract: () => ({}) }
];

function parseNaturalLanguage(command) {
  for (const { pattern, action, extract } of NL_PATTERNS) {
    const match = command.match(pattern);
    if (match) {
      return { action, params: extract(match), matched: true };
    }
  }
  // Default to search if no pattern matches
  return { action: 'search', params: { query: command }, matched: false };
}

async function executeNaturalLanguage(memoryStore, command, topK = 5) {
  const { action, params, matched } = parseNaturalLanguage(command);

  let result = { command, interpretedAs: action, matched };

  switch (action) {
    case 'store':
      await memoryStore.add(params.text, { source: 'natural_language', originalCommand: command });
      result.stored = { text: params.text };
      result.message = `Remembered: "${params.text.substring(0, 100)}..."`;
      break;

    case 'delete_by_query':
      const toDelete = await memoryStore.search(params.query, 5, 0.8);
      if (toDelete.length > 0) {
        for (const item of toDelete) {
          memoryStore.delete(item.id);
        }
        result.deleted = toDelete.length;
        result.message = `Forgot ${toDelete.length} memories related to "${params.query}"`;
      } else {
        result.deleted = 0;
        result.message = `No memories found about "${params.query}"`;
      }
      break;

    case 'search':
      const searchResults = await memoryStore.search(params.query, topK, 0.5);
      result.results = searchResults;
      result.message = `Found ${searchResults.length} memories about "${params.query}"`;
      break;

    case 'stats':
      result.stats = memoryStore.getStats();
      result.message = `You have ${memoryStore.size()} memories stored`;
      break;

    case 'list':
      result.memories = memoryStore.list({ limit: 20 });
      result.message = `Showing ${result.memories.items.length} of ${result.memories.total} memories`;
      break;

    case 'clear':
      const count = memoryStore.clear();
      result.cleared = count;
      result.message = `Cleared all ${count} memories`;
      break;

    case 'analyze':
      result.analysis = memoryStore.analyze();
      result.message = 'Analysis complete';
      break;

    case 'find_duplicates':
      const dups = await memoryStore.findDuplicates(0.9);
      result.duplicates = dups;
      result.message = `Found ${dups.length} potential duplicates`;
      break;

    case 'export':
      result.export = memoryStore.export('json');
      result.message = `Exported ${memoryStore.size()} memories`;
      break;

    case 'help':
      result.commands = [
        'remember [text] - Store a new memory',
        'forget [topic] - Remove memories about a topic',
        'what do you know about [topic] - Search memories',
        'find [query] - Search memories',
        'how many memories - Show count',
        'list memories - Show all memories',
        'clear everything - Remove all memories',
        'analyze - Get insights and patterns',
        'find duplicates - Find similar memories',
        'export - Download all memories'
      ];
      result.message = 'Available natural language commands';
      break;
  }

  return result;
}

// ============================================
// MEMORY CLUSTERING
// ============================================

async function clusterMemories(memoryStore, numClusters = 5) {
  const memories = memoryStore.memories;
  if (memories.length < numClusters) {
    return { clusters: [{ id: 0, label: 'All', memories: memories.map(m => m.id) }], message: 'Not enough memories to cluster' };
  }

  // Simple k-means-like clustering using cosine similarity
  const clusters = [];
  const assigned = new Set();

  // Select initial centroids (first n most different memories)
  const centroids = [memories[0]];
  for (let i = 0; i < numClusters - 1 && centroids.length < numClusters; i++) {
    let maxMinDist = -1;
    let bestMem = null;

    for (const mem of memories) {
      if (centroids.includes(mem)) continue;
      const minDist = Math.min(...centroids.map(c => cosineSimilarity(mem.embedding, c.embedding)));
      if (minDist > maxMinDist) {
        maxMinDist = minDist;
        bestMem = mem;
      }
    }
    if (bestMem) centroids.push(bestMem);
  }

  // Assign memories to nearest centroid
  for (let i = 0; i < numClusters; i++) {
    clusters.push({ id: i, centroid: centroids[i]?.text?.substring(0, 50), memories: [], keywords: [] });
  }

  for (const mem of memories) {
    let bestCluster = 0;
    let bestSim = -1;

    for (let i = 0; i < centroids.length; i++) {
      const sim = cosineSimilarity(mem.embedding, centroids[i].embedding);
      if (sim > bestSim) {
        bestSim = sim;
        bestCluster = i;
      }
    }

    clusters[bestCluster].memories.push({ id: mem.id, text: mem.text.substring(0, 100), similarity: bestSim });
  }

  // Extract keywords from each cluster
  for (const cluster of clusters) {
    const words = {};
    for (const mem of cluster.memories) {
      const memText = memories.find(m => m.id === mem.id)?.text || '';
      memText.toLowerCase().split(/\W+/).filter(w => w.length > 3).forEach(w => {
        words[w] = (words[w] || 0) + 1;
      });
    }
    cluster.keywords = Object.entries(words)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([word]) => word);
    cluster.label = cluster.keywords.slice(0, 3).join(', ') || `Cluster ${cluster.id}`;
  }

  return {
    totalMemories: memories.length,
    numClusters: clusters.length,
    clusters: clusters.map(c => ({
      id: c.id,
      label: c.label,
      keywords: c.keywords,
      size: c.memories.length,
      sampleMemories: c.memories.slice(0, 3)
    }))
  };
}

function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

// ============================================
// VECTOR DB EXPORT FORMATS
// ============================================

function exportToFormat(memoryStore, format) {
  const memories = memoryStore.memories;

  switch (format) {
    case 'pinecone':
      return {
        format: 'pinecone',
        vectors: memories.map(m => ({
          id: m.id,
          values: m.embedding,
          metadata: { text: m.text, ...m.metadata }
        })),
        namespace: memoryStore.namespace
      };

    case 'weaviate':
      return {
        format: 'weaviate',
        class: 'Memory',
        objects: memories.map(m => ({
          id: m.id,
          vector: m.embedding,
          properties: { text: m.text, ...m.metadata, createdAt: m.createdAt }
        }))
      };

    case 'chromadb':
      return {
        format: 'chromadb',
        collection: memoryStore.namespace,
        ids: memories.map(m => m.id),
        embeddings: memories.map(m => m.embedding),
        documents: memories.map(m => m.text),
        metadatas: memories.map(m => m.metadata)
      };

    case 'qdrant':
      return {
        format: 'qdrant',
        collection: memoryStore.namespace,
        points: memories.map(m => ({
          id: m.id,
          vector: m.embedding,
          payload: { text: m.text, ...m.metadata }
        }))
      };

    case 'langchain':
      return {
        format: 'langchain',
        documents: memories.map(m => ({
          pageContent: m.text,
          metadata: { id: m.id, ...m.metadata }
        })),
        embeddings: memories.map(m => m.embedding)
      };

    case 'openai':
      return {
        format: 'openai_compatible',
        data: memories.map(m => ({
          input: m.text,
          embedding: m.embedding
        }))
      };

    default:
      return memoryStore.export(format);
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
    queries = [],  // For batch_search
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
    forceBackgroundLearning = false,
    // New action parameters
    memoryId = null,        // For get/delete/update single memory
    memoryIds = [],         // For batch delete
    updates = {},           // For update action
    metadataFilter = null,  // For filtering operations
    limit = 100,            // For list action
    offset = 0,             // For list action
    sortBy = 'createdAt',   // For list action
    sortOrder = 'desc',     // For list action
    keywordWeight = 0.3,    // For hybrid search
    semanticWeight = 0.7,   // For hybrid search
    duplicateThreshold = 0.95,  // For deduplication
    feedbackRating = null,  // For feedback action
    feedbackCorrection = null,  // For feedback action
    // NEW: Actor integration parameters
    actorId = null,         // Actor to integrate with
    actorConfig = {},       // Config for actor integration (runId, datasetId, memorizeFields, etc.)
    // NEW: Template parameters
    template = null,        // Pre-built template to load
    // NEW: Natural language parameters
    command = null,         // Natural language command
    // NEW: Clustering parameters
    numClusters = 5,        // Number of clusters for clustering action
    // NEW: Export format for vector DBs
    vectorDbFormat = null   // pinecone, weaviate, chromadb, qdrant, langchain, openai
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

  // Safe billing helper - handles unconfigured events gracefully
  const safeCharge = async (eventName, count = 1) => {
    try {
      await Actor.charge({ eventName, count });
    } catch (e) {
      // Event may not be configured - log but don't fail
      log.debug(`Billing event ${eventName} not charged: ${e.message}`);
    }
  };

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

    // ================== NEW ACTIONS ==================

    case 'get':
      // Get single memory by ID
      if (!memoryId) {
        throw new Error('memoryId is required for get action');
      }
      const memory = memoryStore.get(memoryId);
      result = memory ? { found: true, memory } : { found: false, message: `Memory ${memoryId} not found` };
      await safeCharge('memory-get', 1);
      break;

    case 'list':
      // List memories with pagination
      result = memoryStore.list({ limit, offset, filter: metadataFilter, sortBy, sortOrder });
      await safeCharge('memory-list', 1);
      break;

    case 'delete':
      // Delete memories by ID, IDs array, or filter
      if (memoryId) {
        const deleted = memoryStore.delete(memoryId);
        result = deleted ? { deleted: 1, memory: deleted } : { deleted: 0, message: `Memory ${memoryId} not found` };
      } else if (memoryIds && memoryIds.length > 0) {
        const deleted = memoryStore.deleteMany(memoryIds);
        result = { deleted: deleted.length, memories: deleted.map(m => m.id) };
      } else if (metadataFilter) {
        const deleted = memoryStore.deleteByFilter(metadataFilter);
        result = { deleted: deleted.length, memories: deleted.map(m => m.id) };
      } else {
        throw new Error('Provide memoryId, memoryIds, or metadataFilter for delete action');
      }
      await safeCharge('memory-delete', result.deleted || 1);
      break;

    case 'update':
      // Update a memory
      if (!memoryId) {
        throw new Error('memoryId is required for update action');
      }
      const updated = await memoryStore.update(memoryId, updates);
      result = updated ? { updated: true, memory: updated } : { updated: false, message: `Memory ${memoryId} not found` };
      await safeCharge('memory-update', 1);
      break;

    case 'clear':
      // Clear all memories in session
      const clearedCount = memoryStore.clear();
      result = { cleared: clearedCount, message: `Cleared ${clearedCount} memories` };
      await safeCharge('session-clear', 1);
      break;

    case 'stats':
      // Get session statistics (free action)
      result = {
        stats: memoryStore.getStats(),
        sessionId,
        namespace,
        timestamp: new Date().toISOString()
      };
      // Stats is free - no charge
      break;

    case 'batch_search':
      // Multiple queries at once
      if (!queries || queries.length === 0) {
        throw new Error('queries array is required for batch_search action');
      }
      result = await memoryStore.batchSearch(queries, { topK, threshold: similarityThreshold });
      await safeCharge('batch-search', queries.length);
      break;

    case 'hybrid_search':
      // Semantic + keyword combined search
      if (!query) {
        throw new Error('query is required for hybrid_search action');
      }
      result = await memoryStore.hybridSearch(query, {
        topK,
        threshold: similarityThreshold,
        keywordWeight,
        semanticWeight
      });
      await safeCharge('hybrid-search', 1);
      break;

    case 'find_duplicates':
      // Find similar/duplicate memories
      const duplicates = await memoryStore.findDuplicates(duplicateThreshold);
      result = { duplicates, count: duplicates.length };
      await safeCharge('find-duplicates', 1);
      break;

    case 'deduplicate':
      // Remove duplicate memories
      result = await memoryStore.deduplicate(duplicateThreshold);
      await safeCharge('deduplication', 1);
      break;

    case 'feedback':
      // Provide feedback for learning
      if (!memoryId || feedbackRating === null) {
        throw new Error('memoryId and feedbackRating are required for feedback action');
      }
      memoryStore.feedback(memoryId, feedbackRating, feedbackCorrection);
      result = { recorded: true, memoryId, rating: feedbackRating };
      await safeCharge('feedback', 1);
      break;

    case 'integrate_actor':
      // One-click integration with popular Apify actors
      if (!actorId) {
        throw new Error('actorId is required for integrate_actor action. Try "apify/google-maps-scraper", "apify/instagram-scraper", etc.');
      }
      if (!apifyToken) {
        throw new Error('APIFY_TOKEN environment variable is required for actor integration');
      }
      result = await integrateActorResults(memoryStore, actorId, actorConfig || {}, apifyToken);
      await safeCharge('actor-integration', result.memoriesStored || 1);
      break;

    case 'template':
      // Load pre-built templates
      if (!template) {
        result = {
          availableTemplates: Object.entries(TEMPLATES).map(([id, t]) => ({
            id,
            name: t.name,
            description: t.description
          })),
          message: 'Specify a template to load'
        };
      } else {
        result = await loadTemplate(memoryStore, template);
        await safeCharge('template-load', 1);
      }
      break;

    case 'natural':
      // Natural language command parsing
      if (!command) {
        result = {
          message: 'Provide a natural language command',
          examples: [
            'remember that John prefers email communication',
            'what do you know about customers',
            'find shipping preferences',
            'how many memories',
            'forget about old data',
            'analyze',
            'help'
          ]
        };
      } else {
        result = await executeNaturalLanguage(memoryStore, command, topK);
        await safeCharge('natural-language', 1);
      }
      break;

    case 'cluster':
      // Cluster memories into groups
      result = await clusterMemories(memoryStore, numClusters);
      await safeCharge('memory-clustering', 1);
      break;

    case 'export_vectordb':
      // Export to vector DB formats
      if (!vectorDbFormat) {
        result = {
          availableFormats: ['pinecone', 'weaviate', 'chromadb', 'qdrant', 'langchain', 'openai'],
          message: 'Specify vectorDbFormat to export'
        };
      } else {
        result = exportToFormat(memoryStore, vectorDbFormat);
        await safeCharge('vectordb-export', 1);
      }
      break;

    case 'integrate_trading':
      // Integration with Neural Trader System
      result = await integrateNeuralTrader(memoryStore, {
        symbols: input.tradingSymbols || ['BTC', 'ETH'],
        strategy: input.tradingStrategy || 'ensemble',
        memorizeSignals: input.memorizeSignals !== false,
        memorizeMarketData: input.memorizeMarketData || false,
        signalThreshold: input.signalConfidenceThreshold || 70,
        actorConfig: input.tradingActorConfig || {},
        searchHistory: input.searchTradingHistory || false,
        historyQuery: input.tradingHistoryQuery || ''
      }, apifyToken);
      await safeCharge('trading-integration', result.signalsMemorized || 1);
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

// ============================================
// NEURAL TRADER SYSTEM INTEGRATION
// ============================================

async function integrateNeuralTrader(memoryStore, config, apifyToken) {
  log.info('Integrating with Neural Trader System...');

  const {
    symbols = ['BTC', 'ETH'],
    strategy = 'neural_ensemble',
    memorizeSignals = true,
    memorizeMarketData = false,
    signalThreshold = 70,
    actorConfig = {},
    searchHistory = false,
    historyQuery = ''
  } = config;

  // If searching history, perform semantic search on trading memories
  if (searchHistory && historyQuery) {
    log.info(`Searching trading history for: "${historyQuery}"`);
    const results = await memoryStore.search(historyQuery, 20, 0.5);
    const tradingResults = results.filter(r =>
      r.metadata?.source === 'neural-trader' ||
      r.metadata?.type === 'trading-signal' ||
      r.metadata?.type === 'market-data'
    );

    return {
      action: 'search_history',
      query: historyQuery,
      resultsFound: tradingResults.length,
      results: tradingResults.map(r => ({
        text: r.text,
        similarity: Math.round(r.similarity * 100) / 100,
        metadata: r.metadata,
        id: r.id
      })),
      insight: tradingResults.length > 0
        ? `Found ${tradingResults.length} relevant trading memories`
        : 'No matching trading history found. Try running analyze action first.'
    };
  }

  // Call Neural Trader System actor
  if (!apifyToken) {
    // Generate simulated trading signals locally (demo mode)
    log.info('No Apify token - generating simulated trading signals');
    const simulatedSignals = generateSimulatedTradingSignals(symbols, strategy);

    // Store signals in memory
    let signalsMemorized = 0;
    if (memorizeSignals) {
      for (const signal of simulatedSignals) {
        const signalText = `${signal.signal} signal for ${signal.symbol} at $${signal.price.toFixed(2)} - ` +
          `Confidence: ${signal.confidence}% - Strategy: ${signal.strategy} - ` +
          `Target: $${signal.target?.toFixed(2) || 'N/A'}, Stop: $${signal.stopLoss?.toFixed(2) || 'N/A'} - ` +
          `Reasons: ${signal.reasons.join('; ')}`;

        await memoryStore.add(signalText, {
          source: 'neural-trader',
          type: 'trading-signal',
          symbol: signal.symbol,
          signal: signal.signal,
          confidence: signal.confidence,
          strategy: signal.strategy,
          price: signal.price,
          target: signal.target,
          stopLoss: signal.stopLoss,
          timestamp: signal.timestamp,
          simulated: true
        });
        signalsMemorized++;
      }
    }

    return {
      integrated: true,
      mode: 'simulated',
      symbols,
      strategy,
      signalsGenerated: simulatedSignals.length,
      signalsMemorized,
      signals: simulatedSignals,
      message: 'Simulated signals generated. Connect to Neural Trader System actor for live signals.',
      suggestedQueries: [
        'BUY signals with high confidence',
        'trading signals for BTC',
        'recent SELL recommendations',
        `${strategy} strategy performance`
      ]
    };
  }

  try {
    const client = new ApifyClient({ token: apifyToken });
    const neuralTraderActor = actorConfig.actorId || 'ruv/neural-trader-system';

    log.info(`Calling ${neuralTraderActor} for ${symbols.join(', ')}...`);

    // Prepare input for Neural Trader System
    const traderInput = {
      action: actorConfig.action || 'analyze',
      symbols: symbols,
      strategy: strategy,
      confidenceThreshold: signalThreshold,
      riskProfile: actorConfig.riskProfile || 'moderate',
      ...actorConfig
    };

    // Run the Neural Trader System
    const run = await client.actor(neuralTraderActor).call(traderInput, {
      memory: actorConfig.memory || 2048,
      timeout: actorConfig.timeout || 300
    });

    // Get results from the dataset
    const { items } = await client.dataset(run.defaultDatasetId).listItems();
    log.info(`Received ${items.length} items from Neural Trader System`);

    // Process and store trading signals
    let signalsMemorized = 0;
    let marketDataMemorized = 0;
    const processedSignals = [];

    for (const item of items) {
      // Handle signals
      if (item.signals || item.signal) {
        const signals = item.signals || [item];
        for (const signal of signals) {
          if (memorizeSignals && signal.confidence >= signalThreshold) {
            const signalText = `${signal.signal} signal for ${signal.symbol} at $${signal.price?.toFixed(2) || 'N/A'} - ` +
              `Confidence: ${signal.confidence}% - Strategy: ${signal.strategy || strategy} - ` +
              `Target: $${signal.target?.toFixed(2) || 'N/A'}, Stop: $${signal.stopLoss?.toFixed(2) || 'N/A'} - ` +
              `Patterns: ${(signal.patterns || []).join(', ')} - ` +
              `Reasons: ${(signal.reasons || []).join('; ')}`;

            await memoryStore.add(signalText, {
              source: 'neural-trader',
              type: 'trading-signal',
              symbol: signal.symbol,
              signal: signal.signal,
              confidence: signal.confidence,
              strategy: signal.strategy || strategy,
              price: signal.price,
              target: signal.target,
              stopLoss: signal.stopLoss,
              patterns: signal.patterns,
              timestamp: signal.timestamp || new Date().toISOString()
            });
            signalsMemorized++;
            processedSignals.push(signal);
          }
        }
      }

      // Handle market data
      if (memorizeMarketData && item.marketData) {
        const md = item.marketData;
        const mdText = `Market data for ${md.symbol}: Price $${md.price?.toFixed(2)} - ` +
          `24h Change: ${md.change24h?.toFixed(2)}% - Volume: $${(md.volume / 1e6)?.toFixed(2)}M - ` +
          `RSI: ${md.rsi?.toFixed(1)} - MACD: ${md.macd?.toFixed(4)}`;

        await memoryStore.add(mdText, {
          source: 'neural-trader',
          type: 'market-data',
          symbol: md.symbol,
          price: md.price,
          change24h: md.change24h,
          volume: md.volume,
          rsi: md.rsi,
          macd: md.macd,
          timestamp: new Date().toISOString()
        });
        marketDataMemorized++;
      }

      // Handle portfolio recommendations
      if (item.portfolio) {
        const portfolio = item.portfolio;
        const portfolioText = `Portfolio recommendation: Expected return ${portfolio.expectedReturn?.toFixed(2)}% - ` +
          `Risk score: ${portfolio.riskScore?.toFixed(2)} - Sharpe ratio: ${portfolio.sharpeRatio?.toFixed(2)} - ` +
          `Positions: ${portfolio.positions?.map(p => `${p.symbol} ${p.weight}%`).join(', ')}`;

        await memoryStore.add(portfolioText, {
          source: 'neural-trader',
          type: 'portfolio-recommendation',
          expectedReturn: portfolio.expectedReturn,
          riskScore: portfolio.riskScore,
          sharpeRatio: portfolio.sharpeRatio,
          positions: portfolio.positions,
          timestamp: new Date().toISOString()
        });
      }
    }

    return {
      integrated: true,
      mode: 'live',
      actorId: neuralTraderActor,
      runId: run.id,
      symbols,
      strategy,
      itemsReceived: items.length,
      signalsMemorized,
      marketDataMemorized,
      signals: processedSignals.slice(0, 10), // Return first 10 signals
      totalMemories: memoryStore.size(),
      suggestedQueries: [
        'high confidence BUY signals',
        'recent trading recommendations',
        `${symbols[0]} price targets`,
        'portfolio risk analysis'
      ]
    };

  } catch (error) {
    log.warning(`Neural Trader integration failed: ${error.message}. Using simulation.`);

    // Fallback to simulated signals
    const simulatedSignals = generateSimulatedTradingSignals(symbols, strategy);

    let signalsMemorized = 0;
    if (memorizeSignals) {
      for (const signal of simulatedSignals) {
        const signalText = `${signal.signal} signal for ${signal.symbol} - Confidence: ${signal.confidence}% (simulated)`;
        await memoryStore.add(signalText, {
          source: 'neural-trader-fallback',
          type: 'trading-signal',
          ...signal,
          simulated: true
        });
        signalsMemorized++;
      }
    }

    return {
      integrated: true,
      mode: 'fallback',
      error: error.message,
      symbols,
      strategy,
      signalsMemorized,
      signals: simulatedSignals,
      message: 'Using simulated signals due to actor error'
    };
  }
}

function generateSimulatedTradingSignals(symbols, strategy) {
  const signals = [];
  const strategies = {
    'ensemble': { weight: 0.9, patterns: ['trend_following', 'mean_reversion', 'multi_model'] },
    'neural_momentum': { weight: 0.85, patterns: ['momentum', 'breakout', 'trend'] },
    'lstm_prediction': { weight: 0.87, patterns: ['time_series', 'sequence', 'recurrent'] },
    'transformer_attention': { weight: 0.88, patterns: ['attention', 'cross_asset', 'context'] },
    'reinforcement': { weight: 0.82, patterns: ['adaptive', 'reward_optimization', 'policy'] }
  };

  const strategyConfig = strategies[strategy] || strategies['ensemble'];

  for (const symbol of symbols) {
    // Simulate market conditions
    const basePrice = symbol === 'BTC' ? 45000 : symbol === 'ETH' ? 2500 : 100;
    const volatility = 0.02 + Math.random() * 0.03;
    const price = basePrice * (1 + (Math.random() - 0.5) * volatility);

    // Neural prediction simulation
    const prediction = Math.random();
    const confidence = Math.floor(60 + Math.random() * 35);

    let signalType = 'HOLD';
    let target = null;
    let stopLoss = null;
    const reasons = [];

    if (prediction > 0.65 && confidence >= 70) {
      signalType = 'BUY';
      target = price * (1 + 0.05 + Math.random() * 0.1);
      stopLoss = price * (1 - 0.02 - Math.random() * 0.03);
      reasons.push(`Neural prediction: ${(prediction * 100).toFixed(1)}% bullish`);
      reasons.push(`Strategy confidence: ${confidence}%`);
      if (Math.random() > 0.5) reasons.push('RSI oversold recovery');
    } else if (prediction < 0.35 && confidence >= 70) {
      signalType = 'SELL';
      target = price * (1 - 0.05 - Math.random() * 0.1);
      stopLoss = price * (1 + 0.02 + Math.random() * 0.03);
      reasons.push(`Neural prediction: ${((1 - prediction) * 100).toFixed(1)}% bearish`);
      reasons.push(`Strategy confidence: ${confidence}%`);
      if (Math.random() > 0.5) reasons.push('Resistance level detected');
    } else {
      reasons.push('Insufficient signal strength');
      reasons.push(`Prediction: ${(prediction * 100).toFixed(1)}%, Confidence: ${confidence}%`);
    }

    signals.push({
      timestamp: new Date().toISOString(),
      symbol,
      price,
      signal: signalType,
      confidence,
      strategy,
      target,
      stopLoss,
      reasons,
      patterns: strategyConfig.patterns,
      metrics: {
        prediction,
        volatility: volatility * 100,
        strategyWeight: strategyConfig.weight
      }
    });
  }

  return signals;
}
