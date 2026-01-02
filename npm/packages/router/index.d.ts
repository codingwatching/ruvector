/**
 * Distance metric for vector similarity
 */
export enum DistanceMetric {
  /** Euclidean (L2) distance */
  Euclidean = 0,
  /** Cosine similarity */
  Cosine = 1,
  /** Dot product similarity */
  DotProduct = 2,
  /** Manhattan (L1) distance */
  Manhattan = 3
}

/**
 * Intent definition for SemanticRouter
 */
export interface Intent {
  /** Intent name */
  name: string;
  /** Example utterances for this intent */
  utterances: string[];
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Route result from SemanticRouter
 */
export interface RouteResult {
  /** Matched intent name */
  intent: string;
  /** Similarity score (0-1) */
  score: number;
  /** Intent metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Configuration for SemanticRouter
 */
export interface RouterConfig {
  /** Vector dimension (required) */
  dimension: number;
  /** Distance metric (optional, default: Cosine) */
  metric?: DistanceMetric;
  /** HNSW M parameter (optional, default: 16) */
  m?: number;
  /** HNSW ef_construction (optional, default: 200) */
  efConstruction?: number;
  /** HNSW ef_search (optional, default: 100) */
  efSearch?: number;
}

/**
 * High-level semantic router for intent matching
 *
 * @example
 * ```typescript
 * import { SemanticRouter } from '@ruvector/router';
 *
 * const router = new SemanticRouter({ dimension: 384 });
 *
 * // Add intent with pre-computed embeddings
 * router.addIntentWithEmbeddings('weather', [embedding1, embedding2], { handler: 'weather_agent' });
 *
 * // Route query
 * const results = router.routeWithEmbedding(queryEmbedding, 5);
 * console.log(results[0].intent); // 'weather'
 * ```
 */
export class SemanticRouter {
  /**
   * Create a new SemanticRouter
   * @param config Router configuration
   */
  constructor(config: RouterConfig);

  /**
   * Add an intent with utterances (for later embedding)
   * @param intent Intent definition
   */
  addIntent(intent: Intent): void;

  /**
   * Add an intent with pre-computed embeddings
   * @param name Intent name
   * @param embeddings Embeddings for each utterance
   * @param metadata Optional metadata
   */
  addIntentWithEmbeddings(name: string, embeddings: Float32Array[], metadata?: Record<string, unknown>): void;

  /**
   * Route a query using a pre-computed embedding
   * @param embedding Query embedding
   * @param k Number of results (default: 5)
   * @returns Matched intents with scores
   */
  routeWithEmbedding(embedding: Float32Array, k?: number): RouteResult[];

  /**
   * Remove an intent
   * @param name Intent name
   * @returns true if removed
   */
  removeIntent(name: string): boolean;

  /**
   * Get all intent names
   */
  getIntents(): string[];

  /**
   * Get intent details
   * @param name Intent name
   */
  getIntent(name: string): Intent | null;

  /**
   * Clear all intents
   */
  clear(): void;

  /**
   * Get total vector count
   */
  count(): number;
}

/**
 * Options for creating a VectorDb instance
 */
export interface DbOptions {
  /** Vector dimension size (required) */
  dimensions: number;
  /** Maximum number of elements (optional) */
  maxElements?: number;
  /** Distance metric for similarity (optional, default: Cosine) */
  distanceMetric?: DistanceMetric;
  /** HNSW M parameter (optional, default: 16) */
  hnswM?: number;
  /** HNSW ef_construction parameter (optional, default: 200) */
  hnswEfConstruction?: number;
  /** HNSW ef_search parameter (optional, default: 100) */
  hnswEfSearch?: number;
  /** Storage path for persistence (optional) */
  storagePath?: string;
}

/**
 * Search result from a vector query
 */
export interface SearchResult {
  /** Vector ID */
  id: string;
  /** Similarity score */
  score: number;
}

/**
 * High-performance vector database for semantic search
 *
 * @example
 * ```typescript
 * import { VectorDb, DistanceMetric } from '@ruvector/router';
 *
 * // Create a vector database
 * const db = new VectorDb({
 *   dimensions: 384,
 *   distanceMetric: DistanceMetric.Cosine
 * });
 *
 * // Insert vectors
 * const embedding = new Float32Array(384).fill(0.5);
 * db.insert('doc-1', embedding);
 *
 * // Search for similar vectors
 * const results = db.search(embedding, 5);
 * console.log(results[0].id);    // 'doc-1'
 * console.log(results[0].score); // ~1.0
 * ```
 */
export class VectorDb {
  /**
   * Create a new vector database
   * @param options Database options
   */
  constructor(options: DbOptions);

  /**
   * Insert a vector into the database
   * @param id Unique identifier
   * @param vector Vector data (Float32Array)
   * @returns The inserted ID
   */
  insert(id: string, vector: Float32Array): string;

  /**
   * Insert a vector asynchronously
   * @param id Unique identifier
   * @param vector Vector data (Float32Array)
   * @returns Promise resolving to the inserted ID
   */
  insertAsync(id: string, vector: Float32Array): Promise<string>;

  /**
   * Search for similar vectors
   * @param queryVector Query embedding
   * @param k Number of results to return
   * @returns Array of search results
   */
  search(queryVector: Float32Array, k: number): SearchResult[];

  /**
   * Search for similar vectors asynchronously
   * @param queryVector Query embedding
   * @param k Number of results to return
   * @returns Promise resolving to search results
   */
  searchAsync(queryVector: Float32Array, k: number): Promise<SearchResult[]>;

  /**
   * Delete a vector by ID
   * @param id Vector ID to delete
   * @returns true if deleted, false if not found
   */
  delete(id: string): boolean;

  /**
   * Get the total count of vectors
   * @returns Number of vectors in the database
   */
  count(): number;

  /**
   * Get all vector IDs
   * @returns Array of all IDs
   */
  getAllIds(): string[];
}
