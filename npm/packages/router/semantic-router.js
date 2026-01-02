/**
 * SemanticRouter - High-level semantic routing on top of VectorDb
 *
 * @example
 * ```javascript
 * const { SemanticRouter } = require('@ruvector/router');
 *
 * const router = new SemanticRouter({ dimension: 384 });
 *
 * router.addIntent({
 *   name: 'weather',
 *   utterances: ['What is the weather today?', 'Will it rain tomorrow?'],
 *   metadata: { handler: 'weather_agent' }
 * });
 *
 * const results = router.routeWithEmbedding(queryEmbedding);
 * console.log(results[0].intent); // 'weather'
 * ```
 */

const nativeModule = (() => {
  const { platform, arch } = process;
  const path = require('path');

  const platformMap = {
    'linux': {
      'x64': { package: '@ruvector/router-linux-x64-gnu', file: 'ruvector-router.linux-x64-gnu.node' },
      'arm64': { package: '@ruvector/router-linux-arm64-gnu', file: 'ruvector-router.linux-arm64-gnu.node' }
    },
    'darwin': {
      'x64': { package: '@ruvector/router-darwin-x64', file: 'ruvector-router.darwin-x64.node' },
      'arm64': { package: '@ruvector/router-darwin-arm64', file: 'ruvector-router.darwin-arm64.node' }
    },
    'win32': {
      'x64': { package: '@ruvector/router-win32-x64-msvc', file: 'ruvector-router.win32-x64-msvc.node' }
    }
  };

  const platformInfo = platformMap[platform]?.[arch];
  if (!platformInfo) return null;

  try {
    const localPath = path.join(__dirname, platformInfo.file);
    return require(localPath);
  } catch {
    try {
      return require(platformInfo.package);
    } catch {
      return null;
    }
  }
})();

/**
 * @typedef {Object} Intent
 * @property {string} name - Intent name
 * @property {string[]} utterances - Example utterances for this intent
 * @property {Object} [metadata] - Optional metadata
 */

/**
 * @typedef {Object} RouteResult
 * @property {string} intent - Matched intent name
 * @property {number} score - Similarity score (0-1)
 * @property {Object} [metadata] - Intent metadata
 */

/**
 * @typedef {Object} RouterConfig
 * @property {number} dimension - Vector dimension
 * @property {number} [metric=1] - Distance metric (0=Euclidean, 1=Cosine, 2=DotProduct)
 * @property {number} [m=16] - HNSW M parameter
 * @property {number} [efConstruction=200] - HNSW ef_construction
 * @property {number} [efSearch=100] - HNSW ef_search
 */

class SemanticRouter {
  /**
   * Create a new SemanticRouter
   * @param {RouterConfig} config - Router configuration
   */
  constructor(config) {
    if (!config || typeof config.dimension !== 'number') {
      throw new Error('SemanticRouter requires a dimension in config');
    }

    this.dimension = config.dimension;
    this.metric = config.metric ?? 1; // Default Cosine
    this.intents = new Map(); // name -> { utterances, metadata, vectorIds }
    this.vectorToIntent = new Map(); // vectorId -> intentName
    this._nextId = 0;

    // Initialize VectorDb if native module available
    if (nativeModule && nativeModule.VectorDb) {
      this.db = new nativeModule.VectorDb({
        dimensions: config.dimension,
        distanceMetric: this.metric,
        hnswM: config.m ?? 16,
        hnswEfConstruction: config.efConstruction ?? 200,
        hnswEfSearch: config.efSearch ?? 100,
      });
    } else {
      // Fallback: in-memory storage
      this.db = null;
      this._vectors = new Map();
    }
  }

  /**
   * Add an intent with utterances
   * @param {Intent} intent - Intent definition
   */
  addIntent(intent) {
    if (!intent.name || !Array.isArray(intent.utterances)) {
      throw new Error('Intent must have name and utterances array');
    }

    const vectorIds = [];

    // For each utterance, we expect the caller to provide embeddings
    // or we store placeholders. In a real scenario, embeddings would be generated.
    // Here we store the intent for later matching with pre-computed embeddings.

    this.intents.set(intent.name, {
      utterances: intent.utterances,
      metadata: intent.metadata || {},
      vectorIds,
    });
  }

  /**
   * Add an intent with pre-computed embeddings
   * @param {string} name - Intent name
   * @param {Float32Array[]} embeddings - Embeddings for each utterance
   * @param {Object} [metadata] - Optional metadata
   */
  addIntentWithEmbeddings(name, embeddings, metadata = {}) {
    if (!name || !Array.isArray(embeddings)) {
      throw new Error('Must provide name and embeddings array');
    }

    const vectorIds = [];

    for (const embedding of embeddings) {
      const id = `${name}_${this._nextId++}`;

      if (this.db) {
        this.db.insert(id, embedding);
      } else {
        this._vectors.set(id, embedding);
      }

      vectorIds.push(id);
      this.vectorToIntent.set(id, name);
    }

    this.intents.set(name, {
      utterances: [],
      metadata,
      vectorIds,
    });
  }

  /**
   * Route a query using a pre-computed embedding
   * @param {Float32Array} embedding - Query embedding
   * @param {number} [k=5] - Number of results
   * @returns {RouteResult[]} - Matched intents with scores
   */
  routeWithEmbedding(embedding, k = 5) {
    if (!(embedding instanceof Float32Array)) {
      throw new Error('Embedding must be Float32Array');
    }

    let results;

    if (this.db) {
      results = this.db.search(embedding, k * 2); // Get more for deduplication
    } else {
      // Fallback cosine similarity search
      results = this._fallbackSearch(embedding, k * 2);
    }

    // Aggregate by intent (take best score per intent)
    const intentScores = new Map();

    for (const result of results) {
      const intentName = this.vectorToIntent.get(result.id);
      if (!intentName) continue;

      const existing = intentScores.get(intentName);
      if (!existing || result.score > existing.score) {
        const intentData = this.intents.get(intentName);
        intentScores.set(intentName, {
          intent: intentName,
          score: result.score,
          metadata: intentData?.metadata || {},
        });
      }
    }

    // Sort by score descending and take top k
    return Array.from(intentScores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  /**
   * Remove an intent
   * @param {string} name - Intent name
   * @returns {boolean} - true if removed
   */
  removeIntent(name) {
    const intent = this.intents.get(name);
    if (!intent) return false;

    // Remove vectors
    for (const id of intent.vectorIds) {
      if (this.db) {
        this.db.delete(id);
      } else {
        this._vectors.delete(id);
      }
      this.vectorToIntent.delete(id);
    }

    this.intents.delete(name);
    return true;
  }

  /**
   * Get all intent names
   * @returns {string[]}
   */
  getIntents() {
    return Array.from(this.intents.keys());
  }

  /**
   * Get intent details
   * @param {string} name - Intent name
   * @returns {Intent|null}
   */
  getIntent(name) {
    const data = this.intents.get(name);
    if (!data) return null;
    return {
      name,
      utterances: data.utterances,
      metadata: data.metadata,
    };
  }

  /**
   * Clear all intents
   */
  clear() {
    this.intents.clear();
    this.vectorToIntent.clear();
    this._nextId = 0;

    if (!this.db) {
      this._vectors.clear();
    }
    // Note: VectorDb doesn't have a clear method, recreate if needed
  }

  /**
   * Get total vector count
   * @returns {number}
   */
  count() {
    if (this.db) {
      return this.db.count();
    }
    return this._vectors.size;
  }

  /**
   * Fallback search using cosine similarity
   * @private
   */
  _fallbackSearch(query, k) {
    const results = [];

    for (const [id, vec] of this._vectors) {
      const score = this._cosineSimilarity(query, vec);
      results.push({ id, score });
    }

    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  /**
   * Compute cosine similarity
   * @private
   */
  _cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }
}

module.exports = SemanticRouter;
