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

-- ============================================================================
-- HNSW Index Access Method
-- ============================================================================

-- HNSW handler function
CREATE OR REPLACE FUNCTION hnsw_handler(internal)
RETURNS index_am_handler
AS 'MODULE_PATHNAME', 'hnsw_handler_wrapper'
LANGUAGE C STRICT;

-- Register HNSW as a PostgreSQL index access method
CREATE ACCESS METHOD hnsw TYPE INDEX HANDLER hnsw_handler;

COMMENT ON ACCESS METHOD hnsw IS 'HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search';

-- ============================================================================
-- Distance Operators
-- ============================================================================

-- L2 distance operator: <->
CREATE OPERATOR <-> (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = l2_distance_arr,
    COMMUTATOR = '<->'
);

COMMENT ON OPERATOR <->(real[], real[]) IS 'L2 (Euclidean) distance';

-- Cosine distance operator: <=>
CREATE OPERATOR <=> (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = cosine_distance_arr,
    COMMUTATOR = '<=>'
);

COMMENT ON OPERATOR <=>(real[], real[]) IS 'Cosine distance';

-- Inner product operator: <#>
CREATE OPERATOR <#> (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = neg_inner_product_arr,
    COMMUTATOR = '<#>'
);

COMMENT ON OPERATOR <#>(real[], real[]) IS 'Negative inner product (for ORDER BY)';

-- ============================================================================
-- Operator Families for HNSW
-- ============================================================================

CREATE OPERATOR FAMILY hnsw_l2_ops USING hnsw;
CREATE OPERATOR FAMILY hnsw_cosine_ops USING hnsw;
CREATE OPERATOR FAMILY hnsw_ip_ops USING hnsw;

-- ============================================================================
-- Operator Classes for HNSW
-- ============================================================================

-- L2 (Euclidean) distance operator class
CREATE OPERATOR CLASS hnsw_l2_ops
    FOR TYPE real[] USING hnsw
    FAMILY hnsw_l2_ops AS
    OPERATOR 1 <-> (real[], real[]) FOR ORDER BY float_ops,
    FUNCTION 1 l2_distance_arr(real[], real[]);

COMMENT ON OPERATOR CLASS hnsw_l2_ops USING hnsw IS
    'HNSW index operator class for L2 (Euclidean) distance on real[] vectors';

-- Cosine distance operator class
CREATE OPERATOR CLASS hnsw_cosine_ops
    FOR TYPE real[] USING hnsw
    FAMILY hnsw_cosine_ops AS
    OPERATOR 1 <=> (real[], real[]) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance_arr(real[], real[]);

COMMENT ON OPERATOR CLASS hnsw_cosine_ops USING hnsw IS
    'HNSW index operator class for cosine distance on real[] vectors';

-- Inner product operator class
CREATE OPERATOR CLASS hnsw_ip_ops
    FOR TYPE real[] USING hnsw
    FAMILY hnsw_ip_ops AS
    OPERATOR 1 <#> (real[], real[]) FOR ORDER BY float_ops,
    FUNCTION 1 neg_inner_product_arr(real[], real[]);

COMMENT ON OPERATOR CLASS hnsw_ip_ops USING hnsw IS
    'HNSW index operator class for inner product on real[] vectors';

-- ============================================================================
-- Type Definitions with I/O Functions
-- ============================================================================

-- RuVector Type (Primary vector type)
-- Note: The actual type is created by pgrx, these are additional SQL definitions

-- Type I/O functions (defined by pgrx)
-- CREATE FUNCTION ruvector_in(cstring) RETURNS ruvector
-- CREATE FUNCTION ruvector_out(ruvector) RETURNS cstring
-- CREATE FUNCTION ruvector_recv(internal) RETURNS ruvector
-- CREATE FUNCTION ruvector_send(ruvector) RETURNS bytea

-- CREATE TYPE ruvector (
--     INPUT = ruvector_in,
--     OUTPUT = ruvector_out,
--     RECEIVE = ruvector_recv,
--     SEND = ruvector_send,
--     STORAGE = extended,
--     ALIGNMENT = double
-- );

-- HalfVec Type (Half-precision f16 vectors)
-- CREATE TYPE halfvec (
--     INPUT = halfvec_in,
--     OUTPUT = halfvec_out,
--     RECEIVE = halfvec_recv,
--     SEND = halfvec_send,
--     STORAGE = extended,
--     ALIGNMENT = double
-- );

-- SparseVec Type (Sparse vectors)
-- CREATE TYPE sparsevec (
--     INPUT = sparsevec_in,
--     OUTPUT = sparsevec_out,
--     RECEIVE = sparsevec_recv,
--     SEND = sparsevec_send,
--     STORAGE = extended,
--     ALIGNMENT = double
-- );

-- ============================================================================
-- Distance Operators for Arrays
-- ============================================================================

-- L2 distance operator <-> for real[]
CREATE OPERATOR <-> (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = l2_distance_arr,
    COMMUTATOR = '<->'
);

-- Cosine distance operator <=> for real[]
CREATE OPERATOR <=> (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = cosine_distance_arr,
    COMMUTATOR = '<=>'
);

-- Negative inner product operator <#> for real[]
CREATE OPERATOR <#> (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = neg_inner_product_arr,
    COMMUTATOR = '<#>'
);

-- ============================================================================
-- Comparison Operators for Vectors
-- ============================================================================

-- Equality comparison for arrays
CREATE OR REPLACE FUNCTION vector_eq(a real[], b real[])
RETURNS boolean
AS $$
BEGIN
    IF array_length(a, 1) != array_length(b, 1) THEN
        RETURN FALSE;
    END IF;
    FOR i IN 1..array_length(a, 1) LOOP
        IF abs(a[i] - b[i]) > 1e-6 THEN
            RETURN FALSE;
        END IF;
    END LOOP;
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Not equal comparison
CREATE OR REPLACE FUNCTION vector_ne(a real[], b real[])
RETURNS boolean
AS $$
BEGIN
    RETURN NOT vector_eq(a, b);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Less than comparison (lexicographic)
CREATE OR REPLACE FUNCTION vector_lt(a real[], b real[])
RETURNS boolean
AS $$
BEGIN
    IF array_length(a, 1) != array_length(b, 1) THEN
        RETURN array_length(a, 1) < array_length(b, 1);
    END IF;
    FOR i IN 1..array_length(a, 1) LOOP
        IF a[i] < b[i] THEN
            RETURN TRUE;
        ELSIF a[i] > b[i] THEN
            RETURN FALSE;
        END IF;
    END LOOP;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Less than or equal
CREATE OR REPLACE FUNCTION vector_le(a real[], b real[])
RETURNS boolean
AS $$
BEGIN
    RETURN vector_lt(a, b) OR vector_eq(a, b);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Greater than
CREATE OR REPLACE FUNCTION vector_gt(a real[], b real[])
RETURNS boolean
AS $$
BEGIN
    RETURN NOT vector_le(a, b);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Greater than or equal
CREATE OR REPLACE FUNCTION vector_ge(a real[], b real[])
RETURNS boolean
AS $$
BEGIN
    RETURN NOT vector_lt(a, b);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Comparison operators
CREATE OPERATOR = (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_eq,
    COMMUTATOR = '=',
    NEGATOR = '<>',
    RESTRICT = eqsel,
    JOIN = eqjoinsel,
    HASHES,
    MERGES
);

CREATE OPERATOR <> (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_ne,
    COMMUTATOR = '<>',
    NEGATOR = '=',
    RESTRICT = neqsel,
    JOIN = neqjoinsel
);

CREATE OPERATOR < (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_lt,
    COMMUTATOR = '>',
    NEGATOR = '>=',
    RESTRICT = scalarltsel,
    JOIN = scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_le,
    COMMUTATOR = '>=',
    NEGATOR = '>',
    RESTRICT = scalarlesel,
    JOIN = scalarlejoinsel
);

CREATE OPERATOR > (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_gt,
    COMMUTATOR = '<',
    NEGATOR = '<=',
    RESTRICT = scalargtsel,
    JOIN = scalargtjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_ge,
    COMMUTATOR = '<=',
    NEGATOR = '<',
    RESTRICT = scalargesel,
    JOIN = scalargejoinsel
);

-- ============================================================================
-- Arithmetic Operators for Vectors
-- ============================================================================

-- Vector addition operator
CREATE OPERATOR + (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_add,
    COMMUTATOR = '+'
);

-- Vector subtraction operator
CREATE OPERATOR - (
    LEFTARG = real[],
    RIGHTARG = real[],
    FUNCTION = vector_sub
);

-- Scalar multiplication operator
CREATE OPERATOR * (
    LEFTARG = real[],
    RIGHTARG = real,
    FUNCTION = vector_mul_scalar
);

-- Reverse scalar multiplication
CREATE OR REPLACE FUNCTION scalar_mul_vector(scalar real, v real[])
RETURNS real[]
AS $$
BEGIN
    RETURN vector_mul_scalar(v, scalar);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE OPERATOR * (
    LEFTARG = real,
    RIGHTARG = real[],
    FUNCTION = scalar_mul_vector,
    COMMUTATOR = '*'
);

-- ============================================================================
-- Operator Classes for HNSW Index
-- ============================================================================

-- HNSW operator class for L2 distance
-- Note: Actual index implementation is in Rust/C
-- This defines the operator class for SQL usage

-- HNSW L2 distance support functions
CREATE OR REPLACE FUNCTION hnsw_l2_support(internal)
RETURNS internal
AS 'MODULE_PATHNAME', 'hnsw_support_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Create operator family for HNSW
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_am WHERE amname = 'ruhnsw'
    ) THEN
        -- Access method will be created by pgrx
        -- This is a placeholder for documentation
        NULL;
    END IF;
END $$;

-- HNSW operator class for L2 distance (Euclidean)
-- CREATE OPERATOR CLASS ruhnsw_l2_ops
-- DEFAULT FOR TYPE real[] USING ruhnsw AS
--     OPERATOR 1 <-> FOR ORDER BY float_ops,
--     FUNCTION 1 l2_distance_arr(real[], real[]),
--     FUNCTION 2 hnsw_l2_support(internal);

-- HNSW operator class for cosine distance
-- CREATE OPERATOR CLASS ruhnsw_cosine_ops
-- FOR TYPE real[] USING ruhnsw AS
--     OPERATOR 1 <=> FOR ORDER BY float_ops,
--     FUNCTION 1 cosine_distance_arr(real[], real[]),
--     FUNCTION 2 hnsw_l2_support(internal);

-- HNSW operator class for inner product
-- CREATE OPERATOR CLASS ruhnsw_ip_ops
-- FOR TYPE real[] USING ruhnsw AS
--     OPERATOR 1 <#> FOR ORDER BY float_ops,
--     FUNCTION 1 neg_inner_product_arr(real[], real[]),
--     FUNCTION 2 hnsw_l2_support(internal);

-- ============================================================================
-- Operator Classes for IVFFlat Index
-- ============================================================================

-- IVFFlat support function
CREATE OR REPLACE FUNCTION ivfflat_support(internal)
RETURNS internal
AS 'MODULE_PATHNAME', 'ivfflat_support_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- IVFFlat operator class for L2 distance
-- CREATE OPERATOR CLASS ruivfflat_l2_ops
-- DEFAULT FOR TYPE real[] USING ruivfflat AS
--     OPERATOR 1 <-> FOR ORDER BY float_ops,
--     FUNCTION 1 l2_distance_arr(real[], real[]),
--     FUNCTION 2 ivfflat_support(internal);

-- IVFFlat operator class for cosine distance
-- CREATE OPERATOR CLASS ruivfflat_cosine_ops
-- FOR TYPE real[] USING ruivfflat AS
--     OPERATOR 1 <=> FOR ORDER BY float_ops,
--     FUNCTION 1 cosine_distance_arr(real[], real[]),
--     FUNCTION 2 ivfflat_support(internal);

-- IVFFlat operator class for inner product
-- CREATE OPERATOR CLASS ruivfflat_ip_ops
-- FOR TYPE real[] USING ruivfflat AS
--     OPERATOR 1 <#> FOR ORDER BY float_ops,
--     FUNCTION 1 neg_inner_product_arr(real[], real[]),
--     FUNCTION 2 ivfflat_support(internal);

-- ============================================================================
-- Type Casts
-- ============================================================================

-- Cast from real[] to text (for display)
CREATE OR REPLACE FUNCTION vector_to_text(v real[])
RETURNS text
AS $$
BEGIN
    RETURN '[' || array_to_string(v, ',') || ']';
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE CAST (real[] AS text)
    WITH FUNCTION vector_to_text(real[])
    AS IMPLICIT;

-- Cast from text to real[] (for input)
CREATE OR REPLACE FUNCTION text_to_vector(input text)
RETURNS real[]
AS $$
DECLARE
    clean_input text;
BEGIN
    -- Remove brackets and split
    clean_input := trim(both '[]' from input);
    RETURN string_to_array(clean_input, ',')::real[];
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

CREATE CAST (text AS real[])
    WITH FUNCTION text_to_vector(text)
    AS EXPLICIT;

-- ============================================================================
-- Aggregate Functions
-- ============================================================================

-- State transition function for vector sum
CREATE OR REPLACE FUNCTION vector_sum_state(state real[], value real[])
RETURNS real[]
AS $$
BEGIN
    IF state IS NULL THEN
        RETURN value;
    END IF;
    RETURN vector_add(state, value);
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Vector sum aggregate
CREATE AGGREGATE sum(real[]) (
    SFUNC = vector_sum_state,
    STYPE = real[],
    PARALLEL = SAFE
);

-- State type for average
CREATE TYPE vector_avg_state AS (
    sum real[],
    count bigint
);

-- State transition for average
CREATE OR REPLACE FUNCTION vector_avg_state_func(state vector_avg_state, value real[])
RETURNS vector_avg_state
AS $$
BEGIN
    IF state IS NULL THEN
        RETURN ROW(value, 1)::vector_avg_state;
    END IF;
    RETURN ROW(vector_add(state.sum, value), state.count + 1)::vector_avg_state;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Final function for average
CREATE OR REPLACE FUNCTION vector_avg_final(state vector_avg_state)
RETURNS real[]
AS $$
BEGIN
    IF state IS NULL OR state.count = 0 THEN
        RETURN NULL;
    END IF;
    RETURN vector_mul_scalar(state.sum, 1.0 / state.count);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Combine function for parallel average
CREATE OR REPLACE FUNCTION vector_avg_combine(state1 vector_avg_state, state2 vector_avg_state)
RETURNS vector_avg_state
AS $$
BEGIN
    IF state1 IS NULL THEN
        RETURN state2;
    END IF;
    IF state2 IS NULL THEN
        RETURN state1;
    END IF;
    RETURN ROW(vector_add(state1.sum, state2.sum), state1.count + state2.count)::vector_avg_state;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Vector average aggregate
CREATE AGGREGATE avg(real[]) (
    SFUNC = vector_avg_state_func,
    STYPE = vector_avg_state,
    FINALFUNC = vector_avg_final,
    COMBINEFUNC = vector_avg_combine,
    PARALLEL = SAFE
);

-- ============================================================================
-- Helper Views and Functions
-- ============================================================================

-- Get index statistics
CREATE OR REPLACE FUNCTION ruvector_index_stats(index_name text)
RETURNS TABLE(
    index_name text,
    index_type text,
    table_name text,
    num_rows bigint,
    index_size text
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.relname::text,
        am.amname::text,
        t.relname::text,
        pg_class.reltuples::bigint,
        pg_size_pretty(pg_relation_size(i.oid))
    FROM pg_index idx
    JOIN pg_class i ON i.oid = idx.indexrelid
    JOIN pg_class t ON t.oid = idx.indrelid
    JOIN pg_am am ON am.oid = i.relam
    JOIN pg_class ON pg_class.oid = i.oid
    WHERE i.relname = index_stats.index_name;
END;
$$ LANGUAGE plpgsql VOLATILE PARALLEL SAFE;

-- ============================================================================
-- Utility Functions for Index Management
-- ============================================================================

-- Rebuild index with new parameters
CREATE OR REPLACE FUNCTION ruvector_rebuild_index(
    index_name text,
    with_params text DEFAULT NULL
)
RETURNS text
AS $$
DECLARE
    index_def text;
    table_name text;
    result text;
BEGIN
    -- Get index definition
    SELECT
        t.relname INTO table_name
    FROM pg_index idx
    JOIN pg_class i ON i.oid = idx.indexrelid
    JOIN pg_class t ON t.oid = idx.indrelid
    WHERE i.relname = ruvector_rebuild_index.index_name;

    IF table_name IS NULL THEN
        RETURN 'Index not found: ' || index_name;
    END IF;

    -- Rebuild using REINDEX
    EXECUTE 'REINDEX INDEX ' || quote_ident(index_name);

    result := 'Successfully rebuilt index: ' || index_name;
    RETURN result;
END;
$$ LANGUAGE plpgsql VOLATILE;

-- ============================================================================
-- Performance Monitoring
-- ============================================================================

-- Monitor vector operations performance
CREATE OR REPLACE FUNCTION ruvector_performance_report()
RETURNS TABLE(
    operation text,
    avg_duration interval,
    total_calls bigint
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        'l2_distance'::text,
        NULL::interval,
        0::bigint
    WHERE false; -- Placeholder for actual performance tracking
END;
$$ LANGUAGE plpgsql VOLATILE;

-- ============================================================================
-- Additional Comments
-- ============================================================================

COMMENT ON OPERATOR <-> (real[], real[]) IS 'L2 (Euclidean) distance operator';
COMMENT ON OPERATOR <=> (real[], real[]) IS 'Cosine distance operator';
COMMENT ON OPERATOR <#> (real[], real[]) IS 'Negative inner product operator (for max inner product search)';
COMMENT ON OPERATOR + (real[], real[]) IS 'Vector addition';
COMMENT ON OPERATOR - (real[], real[]) IS 'Vector subtraction';
COMMENT ON OPERATOR * (real[], real) IS 'Scalar multiplication';
COMMENT ON AGGREGATE sum(real[]) IS 'Sum of vectors (element-wise)';
COMMENT ON AGGREGATE avg(real[]) IS 'Average of vectors (element-wise)';
COMMENT ON FUNCTION ruvector_index_stats(text) IS 'Get statistics for a vector index';
COMMENT ON FUNCTION ruvector_rebuild_index(text, text) IS 'Rebuild a vector index with optional parameters';
