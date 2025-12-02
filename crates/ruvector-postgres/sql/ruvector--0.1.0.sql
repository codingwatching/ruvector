-- RuVector PostgreSQL Extension
-- Version: 0.1.0
-- High-performance vector similarity search with SIMD optimizations

-- Complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION ruvector" to load this file. \quit

-- ============================================================================
-- Utility Functions
-- ============================================================================

-- Get extension version
CREATE OR REPLACE FUNCTION ruvector_version()
RETURNS text
AS 'MODULE_PATHNAME', 'ruvector_version_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Get SIMD info
CREATE OR REPLACE FUNCTION ruvector_simd_info()
RETURNS text
AS 'MODULE_PATHNAME', 'ruvector_simd_info_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Get memory stats
CREATE OR REPLACE FUNCTION ruvector_memory_stats()
RETURNS jsonb
AS 'MODULE_PATHNAME', 'ruvector_memory_stats_wrapper'
LANGUAGE C VOLATILE PARALLEL SAFE;

-- Index maintenance
CREATE OR REPLACE FUNCTION ruvector_index_maintenance(index_name text)
RETURNS text
AS 'MODULE_PATHNAME', 'ruvector_index_maintenance_wrapper'
LANGUAGE C VOLATILE PARALLEL SAFE;

-- ============================================================================
-- Distance Functions (array-based)
-- ============================================================================

-- L2 (Euclidean) distance between two float arrays
CREATE OR REPLACE FUNCTION l2_distance_arr(a real[], b real[])
RETURNS real
AS 'MODULE_PATHNAME', 'l2_distance_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Inner product between two float arrays
CREATE OR REPLACE FUNCTION inner_product_arr(a real[], b real[])
RETURNS real
AS 'MODULE_PATHNAME', 'inner_product_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Negative inner product (for ORDER BY ASC nearest neighbor)
CREATE OR REPLACE FUNCTION neg_inner_product_arr(a real[], b real[])
RETURNS real
AS 'MODULE_PATHNAME', 'neg_inner_product_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Cosine distance between two float arrays
CREATE OR REPLACE FUNCTION cosine_distance_arr(a real[], b real[])
RETURNS real
AS 'MODULE_PATHNAME', 'cosine_distance_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Cosine similarity between two float arrays
CREATE OR REPLACE FUNCTION cosine_similarity_arr(a real[], b real[])
RETURNS real
AS 'MODULE_PATHNAME', 'cosine_similarity_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- L1 (Manhattan) distance between two float arrays
CREATE OR REPLACE FUNCTION l1_distance_arr(a real[], b real[])
RETURNS real
AS 'MODULE_PATHNAME', 'l1_distance_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- Vector Utility Functions
-- ============================================================================

-- Normalize a vector to unit length
CREATE OR REPLACE FUNCTION vector_normalize(v real[])
RETURNS real[]
AS 'MODULE_PATHNAME', 'vector_normalize_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Add two vectors element-wise
CREATE OR REPLACE FUNCTION vector_add(a real[], b real[])
RETURNS real[]
AS 'MODULE_PATHNAME', 'vector_add_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Subtract two vectors element-wise
CREATE OR REPLACE FUNCTION vector_sub(a real[], b real[])
RETURNS real[]
AS 'MODULE_PATHNAME', 'vector_sub_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Multiply vector by scalar
CREATE OR REPLACE FUNCTION vector_mul_scalar(v real[], scalar real)
RETURNS real[]
AS 'MODULE_PATHNAME', 'vector_mul_scalar_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Get vector dimensions
CREATE OR REPLACE FUNCTION vector_dims(v real[])
RETURNS integer
AS 'MODULE_PATHNAME', 'vector_dims_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Get vector L2 norm
CREATE OR REPLACE FUNCTION vector_norm(v real[])
RETURNS real
AS 'MODULE_PATHNAME', 'vector_norm_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Average two vectors
CREATE OR REPLACE FUNCTION vector_avg2(a real[], b real[])
RETURNS real[]
AS 'MODULE_PATHNAME', 'vector_avg2_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- Quantization Functions
-- ============================================================================

-- Binary quantize a vector
CREATE OR REPLACE FUNCTION binary_quantize_arr(v real[])
RETURNS bytea
AS 'MODULE_PATHNAME', 'binary_quantize_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Scalar quantize a vector (SQ8)
CREATE OR REPLACE FUNCTION scalar_quantize_arr(v real[])
RETURNS jsonb
AS 'MODULE_PATHNAME', 'scalar_quantize_arr_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- Helper Functions for RuVector type
-- ============================================================================

-- Create a ruvector from a float array (returns JSON)
CREATE OR REPLACE FUNCTION ruvector_from_array(arr real[])
RETURNS jsonb
AS 'MODULE_PATHNAME', 'ruvector_from_array_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Parse a ruvector from string format [1.0, 2.0, 3.0]
CREATE OR REPLACE FUNCTION ruvector_parse(input text)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'ruvector_parse_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Get vector as float array from JSON representation
CREATE OR REPLACE FUNCTION ruvector_to_array(v jsonb)
RETURNS real[]
AS 'MODULE_PATHNAME', 'ruvector_to_array_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- HalfVec Helper Functions
-- ============================================================================

-- Create a halfvec from a float array
CREATE OR REPLACE FUNCTION halfvec_from_array(arr real[])
RETURNS jsonb
AS 'MODULE_PATHNAME', 'halfvec_from_array_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Parse a halfvec from string format [1.0, 2.0, 3.0]
CREATE OR REPLACE FUNCTION halfvec_parse(input text)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'halfvec_parse_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- SparseVec Helper Functions
-- ============================================================================

-- Parse a sparse vector from string format {1:0.5, 3:0.7}/10
CREATE OR REPLACE FUNCTION sparsevec_parse(input text)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'sparsevec_parse_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON FUNCTION ruvector_version() IS 'Returns the ruvector extension version';
COMMENT ON FUNCTION ruvector_simd_info() IS 'Returns SIMD capability information';
COMMENT ON FUNCTION l2_distance_arr(real[], real[]) IS 'Compute L2 (Euclidean) distance between arrays';
COMMENT ON FUNCTION cosine_distance_arr(real[], real[]) IS 'Compute cosine distance between arrays';
COMMENT ON FUNCTION inner_product_arr(real[], real[]) IS 'Compute inner product between arrays';
COMMENT ON FUNCTION vector_normalize(real[]) IS 'Normalize vector to unit length';
COMMENT ON FUNCTION vector_dims(real[]) IS 'Get vector dimensions';
COMMENT ON FUNCTION vector_norm(real[]) IS 'Get vector L2 norm';
