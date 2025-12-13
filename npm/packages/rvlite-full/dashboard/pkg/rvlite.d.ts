/* tslint:disable */
/* eslint-disable */

export class CypherEngine {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new Cypher engine with empty graph
   */
  constructor();
  /**
   * Clear the graph
   */
  clear(): void;
  /**
   * Get graph statistics
   */
  stats(): any;
  /**
   * Execute a Cypher query and return JSON results
   */
  execute(query: string): any;
}

export class RvLite {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Add an RDF triple
   *
   * # Arguments
   * * `subject` - Subject IRI or blank node (e.g., "<http://example.org/s>" or "_:b1")
   * * `predicate` - Predicate IRI (e.g., "<http://example.org/p>")
   * * `object` - Object IRI, blank node, or literal (e.g., "<http://example.org/o>" or '"value"')
   */
  add_triple(subject: string, predicate: string, object: string): void;
  /**
   * Get configuration
   */
  get_config(): any;
  /**
   * Export database state as JSON (for manual backup)
   */
  export_json(): any;
  /**
   * Get version string
   */
  get_version(): string;
  /**
   * Import database state from JSON
   */
  import_json(json: any): void;
  /**
   * Clear the Cypher graph
   */
  cypher_clear(): void;
  /**
   * Get Cypher graph statistics
   */
  cypher_stats(): any;
  /**
   * Get enabled features
   */
  get_features(): any;
  /**
   * Initialize IndexedDB storage for persistence
   * Must be called before save() or load()
   */
  init_storage(): Promise<any>;
  /**
   * Get the number of triples in the store
   */
  triple_count(): number;
  /**
   * Clear saved state from IndexedDB
   */
  static clear_storage(): Promise<any>;
  /**
   * Clear all triples
   */
  clear_triples(): void;
  /**
   * Insert a vector with a specific ID
   */
  insert_with_id(id: string, vector: Float32Array, metadata: any): void;
  /**
   * Check if saved state exists in IndexedDB
   */
  static has_saved_state(): Promise<any>;
  /**
   * Search with metadata filter
   */
  search_with_filter(query_vector: Float32Array, k: number, filter: any): any;
  /**
   * Check if IndexedDB is available in the browser
   */
  static is_storage_available(): boolean;
  /**
   * Get a vector by ID
   */
  get(id: string): any;
  /**
   * Get the number of vectors in the database
   */
  len(): number;
  /**
   * Create a new RvLite database
   */
  constructor(config: RvLiteConfig);
  /**
   * Execute SQL query
   *
   * Supported syntax:
   * - CREATE TABLE vectors (id TEXT PRIMARY KEY, vector VECTOR(384))
   * - SELECT * FROM vectors WHERE id = 'x'
   * - SELECT id, vector <-> '[...]' AS distance FROM vectors ORDER BY distance LIMIT 10
   * - INSERT INTO vectors (id, vector) VALUES ('x', '[...]')
   * - DELETE FROM vectors WHERE id = 'x'
   */
  sql(query: string): any;
  /**
   * Load database from IndexedDB
   * Returns a Promise<RvLite> with the restored database
   */
  static load(config: RvLiteConfig): Promise<any>;
  /**
   * Save database state to IndexedDB
   * Returns a Promise that resolves when save is complete
   */
  save(): Promise<any>;
  /**
   * Execute Cypher query
   *
   * Supported operations:
   * - CREATE (n:Label {prop: value})
   * - MATCH (n:Label) WHERE n.prop = value RETURN n
   * - CREATE (a)-[r:REL]->(b)
   * - DELETE n
   */
  cypher(query: string): any;
  /**
   * Delete a vector by ID
   */
  delete(id: string): boolean;
  /**
   * Insert a vector with optional metadata
   * Returns the vector ID
   */
  insert(vector: Float32Array, metadata: any): string;
  /**
   * Search for similar vectors
   */
  search(query_vector: Float32Array, k: number): any;
  /**
   * Execute SPARQL query
   *
   * Supported operations:
   * - SELECT ?s ?p ?o WHERE { ?s ?p ?o }
   * - SELECT ?s WHERE { ?s <predicate> ?o }
   * - ASK { ?s ?p ?o }
   */
  sparql(query: string): any;
  /**
   * Create with default configuration (384 dimensions, cosine similarity)
   */
  static default(): RvLite;
  /**
   * Check if database is empty
   */
  is_empty(): boolean;
  /**
   * Check if database is ready
   */
  is_ready(): boolean;
}

export class RvLiteConfig {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Get dimensions
   */
  get_dimensions(): number;
  /**
   * Get distance metric name
   */
  get_distance_metric(): string;
  /**
   * Set distance metric (euclidean, cosine, dotproduct, manhattan)
   */
  with_distance_metric(metric: string): RvLiteConfig;
  constructor(dimensions: number);
}

export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_cypherengine_free: (a: number, b: number) => void;
  readonly __wbg_rvlite_free: (a: number, b: number) => void;
  readonly __wbg_rvliteconfig_free: (a: number, b: number) => void;
  readonly cypherengine_clear: (a: number) => void;
  readonly cypherengine_execute: (a: number, b: number, c: number, d: number) => void;
  readonly cypherengine_new: () => number;
  readonly cypherengine_stats: (a: number, b: number) => void;
  readonly rvlite_add_triple: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly rvlite_clear_triples: (a: number) => void;
  readonly rvlite_cypher: (a: number, b: number, c: number, d: number) => void;
  readonly rvlite_cypher_clear: (a: number) => void;
  readonly rvlite_cypher_stats: (a: number, b: number) => void;
  readonly rvlite_default: (a: number) => void;
  readonly rvlite_delete: (a: number, b: number, c: number, d: number) => void;
  readonly rvlite_export_json: (a: number, b: number) => void;
  readonly rvlite_get: (a: number, b: number, c: number, d: number) => void;
  readonly rvlite_get_config: (a: number, b: number) => void;
  readonly rvlite_get_features: (a: number, b: number) => void;
  readonly rvlite_get_version: (a: number, b: number) => void;
  readonly rvlite_import_json: (a: number, b: number, c: number) => void;
  readonly rvlite_init_storage: (a: number) => number;
  readonly rvlite_insert: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly rvlite_insert_with_id: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly rvlite_is_empty: (a: number, b: number) => void;
  readonly rvlite_is_ready: (a: number) => number;
  readonly rvlite_is_storage_available: () => number;
  readonly rvlite_len: (a: number, b: number) => void;
  readonly rvlite_load: (a: number) => number;
  readonly rvlite_new: (a: number, b: number) => void;
  readonly rvlite_save: (a: number) => number;
  readonly rvlite_search: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly rvlite_search_with_filter: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly rvlite_sparql: (a: number, b: number, c: number, d: number) => void;
  readonly rvlite_sql: (a: number, b: number, c: number, d: number) => void;
  readonly rvlite_triple_count: (a: number) => number;
  readonly rvliteconfig_get_dimensions: (a: number) => number;
  readonly rvliteconfig_get_distance_metric: (a: number, b: number) => void;
  readonly rvliteconfig_new: (a: number) => number;
  readonly rvliteconfig_with_distance_metric: (a: number, b: number, c: number) => number;
  readonly init: () => void;
  readonly rvlite_clear_storage: () => number;
  readonly rvlite_has_saved_state: () => number;
  readonly __wasm_bindgen_func_elem_1368: (a: number, b: number, c: number) => void;
  readonly __wasm_bindgen_func_elem_1367: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_180: (a: number, b: number, c: number) => void;
  readonly __wasm_bindgen_func_elem_179: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_1414: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_export: (a: number, b: number) => number;
  readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export3: (a: number) => void;
  readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
