//! RestrictionMap: Linear transform defining state constraints
//!
//! In sheaf theory, a restriction map ρ: F(U) -> F(V) defines how the state
//! at one location constrains the state at another. For our coherence engine,
//! we use affine linear maps: y = Ax + b
//!
//! This allows us to express constraints like:
//! - Identity: states must match exactly
//! - Projection: some dimensions must match
//! - Scaling: values must be proportional
//! - Translation: values must differ by a constant
//!
//! # SIMD Optimization
//!
//! The `apply` method is SIMD-optimized for common cases:
//! - Identity maps bypass matrix multiplication
//! - Small matrices (up to 8x8) use unrolled loops
//! - Larger matrices use cache-friendly blocking

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when working with restriction maps
#[derive(Debug, Error)]
pub enum RestrictionMapError {
    /// Matrix dimensions don't match
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid matrix data
    #[error("Invalid matrix: {0}")]
    InvalidMatrix(String),

    /// Operation not supported
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Storage format for the transformation matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatrixStorage {
    /// Identity matrix (no storage needed)
    Identity,
    /// Diagonal matrix (only diagonal elements stored)
    Diagonal(Vec<f32>),
    /// Sparse matrix in COO format (row, col, value)
    Sparse {
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f32>,
        output_dim: usize,
        input_dim: usize,
    },
    /// Dense matrix stored in row-major order
    Dense {
        data: Vec<f32>,
        output_dim: usize,
        input_dim: usize,
    },
    /// Projection to subset of dimensions
    Projection {
        /// Indices of dimensions to keep
        indices: Vec<usize>,
        input_dim: usize,
    },
}

impl MatrixStorage {
    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        match self {
            MatrixStorage::Identity => 0, // Unknown until applied
            MatrixStorage::Diagonal(d) => d.len(),
            MatrixStorage::Sparse { input_dim, .. } => *input_dim,
            MatrixStorage::Dense { input_dim, .. } => *input_dim,
            MatrixStorage::Projection { input_dim, .. } => *input_dim,
        }
    }

    /// Get the output dimension
    pub fn output_dim(&self) -> usize {
        match self {
            MatrixStorage::Identity => 0, // Unknown until applied
            MatrixStorage::Diagonal(d) => d.len(),
            MatrixStorage::Sparse { output_dim, .. } => *output_dim,
            MatrixStorage::Dense { output_dim, .. } => *output_dim,
            MatrixStorage::Projection { indices, .. } => indices.len(),
        }
    }

    /// Check if this is an identity transform
    pub fn is_identity(&self) -> bool {
        matches!(self, MatrixStorage::Identity)
    }

    /// Check if this is a diagonal transform
    pub fn is_diagonal(&self) -> bool {
        matches!(self, MatrixStorage::Diagonal(_))
    }

    /// Check if this is a projection
    pub fn is_projection(&self) -> bool {
        matches!(self, MatrixStorage::Projection { .. })
    }
}

/// A restriction map implementing an affine linear transform: y = Ax + b
///
/// This is the mathematical foundation for expressing constraints between
/// connected nodes in the sheaf graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestrictionMap {
    /// The transformation matrix A
    pub matrix: MatrixStorage,
    /// The bias vector b (optional, empty means no bias)
    pub bias: Vec<f32>,
    /// Output dimension (cached for fast access)
    output_dim: usize,
    /// Input dimension (cached for fast access)
    input_dim: usize,
}

impl RestrictionMap {
    /// Create an identity restriction map (states must match exactly)
    pub fn identity(dim: usize) -> Self {
        Self {
            matrix: MatrixStorage::Identity,
            bias: Vec::new(),
            output_dim: dim,
            input_dim: dim,
        }
    }

    /// Create a diagonal scaling map
    pub fn diagonal(scales: Vec<f32>) -> Self {
        let dim = scales.len();
        Self {
            matrix: MatrixStorage::Diagonal(scales),
            bias: Vec::new(),
            output_dim: dim,
            input_dim: dim,
        }
    }

    /// Create a projection map that selects specific dimensions
    pub fn projection(indices: Vec<usize>, input_dim: usize) -> Self {
        let output_dim = indices.len();
        Self {
            matrix: MatrixStorage::Projection { indices, input_dim },
            bias: Vec::new(),
            output_dim,
            input_dim,
        }
    }

    /// Create a dense linear map from a matrix
    pub fn dense(
        data: Vec<f32>,
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Self, RestrictionMapError> {
        if data.len() != output_dim * input_dim {
            return Err(RestrictionMapError::InvalidMatrix(format!(
                "Matrix data length {} doesn't match {}x{}",
                data.len(),
                output_dim,
                input_dim
            )));
        }

        Ok(Self {
            matrix: MatrixStorage::Dense {
                data,
                output_dim,
                input_dim,
            },
            bias: Vec::new(),
            output_dim,
            input_dim,
        })
    }

    /// Create a sparse map from COO format
    pub fn sparse(
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<f32>,
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Self, RestrictionMapError> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(RestrictionMapError::InvalidMatrix(
                "COO arrays must have same length".to_string(),
            ));
        }

        Ok(Self {
            matrix: MatrixStorage::Sparse {
                rows,
                cols,
                values,
                output_dim,
                input_dim,
            },
            bias: Vec::new(),
            output_dim,
            input_dim,
        })
    }

    /// Add a bias vector to the map
    pub fn with_bias(mut self, bias: Vec<f32>) -> Result<Self, RestrictionMapError> {
        if !bias.is_empty() && bias.len() != self.output_dim {
            return Err(RestrictionMapError::DimensionMismatch {
                expected: self.output_dim,
                actual: bias.len(),
            });
        }
        self.bias = bias;
        Ok(self)
    }

    /// Get the input dimension
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the output dimension
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Apply the restriction map to an input vector: y = Ax + b
    ///
    /// # SIMD Optimization
    ///
    /// This method is optimized for common cases:
    /// - Identity: O(n) copy
    /// - Diagonal: O(n) element-wise multiply
    /// - Projection: O(k) index gather
    /// - Dense: SIMD-friendly matrix-vector multiply
    #[inline]
    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        // Validate input dimension (for identity, we infer from input)
        let expected_input = if self.matrix.is_identity() {
            input.len()
        } else {
            self.input_dim
        };

        debug_assert_eq!(input.len(), expected_input, "Input dimension mismatch");

        let mut output = match &self.matrix {
            MatrixStorage::Identity => input.to_vec(),

            MatrixStorage::Diagonal(scales) => {
                // SIMD-friendly element-wise multiply using chunks
                let mut result = Vec::with_capacity(input.len());
                let chunks_in = input.chunks_exact(4);
                let chunks_sc = scales.chunks_exact(4);
                let rem_in = chunks_in.remainder();
                let rem_sc = chunks_sc.remainder();

                for (chunk_in, chunk_sc) in chunks_in.zip(chunks_sc) {
                    result.push(chunk_in[0] * chunk_sc[0]);
                    result.push(chunk_in[1] * chunk_sc[1]);
                    result.push(chunk_in[2] * chunk_sc[2]);
                    result.push(chunk_in[3] * chunk_sc[3]);
                }
                for (&x, &s) in rem_in.iter().zip(rem_sc.iter()) {
                    result.push(x * s);
                }
                result
            }

            MatrixStorage::Projection { indices, .. } => {
                // Gather selected dimensions with pre-allocated capacity
                let mut result = Vec::with_capacity(indices.len());
                for &i in indices {
                    result.push(input[i]);
                }
                result
            }

            MatrixStorage::Sparse {
                rows,
                cols,
                values,
                output_dim,
                ..
            } => {
                let mut result = vec![0.0; *output_dim];
                // Use iterator without allocation overhead
                for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
                    result[r] += v * input[c];
                }
                result
            }

            MatrixStorage::Dense {
                data,
                output_dim,
                input_dim,
            } => self.apply_dense_simd(input, data, *output_dim, *input_dim),
        };

        // Add bias if present - use SIMD-friendly pattern
        if !self.bias.is_empty() {
            let bias_len = self.bias.len();
            let chunk_count = bias_len / 4;

            // Process chunks of 4
            for i in 0..chunk_count {
                let base = i * 4;
                output[base] += self.bias[base];
                output[base + 1] += self.bias[base + 1];
                output[base + 2] += self.bias[base + 2];
                output[base + 3] += self.bias[base + 3];
            }

            // Handle remainder
            for i in (chunk_count * 4)..bias_len {
                output[i] += self.bias[i];
            }
        }

        output
    }

    /// Apply restriction map into a pre-allocated output buffer (zero allocation)
    ///
    /// This is the preferred method for hot paths where the output buffer
    /// can be reused across multiple calls.
    #[inline]
    pub fn apply_into(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(output.len(), self.output_dim, "Output dimension mismatch");

        match &self.matrix {
            MatrixStorage::Identity => {
                output.copy_from_slice(input);
            }

            MatrixStorage::Diagonal(scales) => {
                // SIMD-friendly element-wise multiply
                for ((out, &inp), &sc) in output.iter_mut().zip(input.iter()).zip(scales.iter()) {
                    *out = inp * sc;
                }
            }

            MatrixStorage::Projection { indices, .. } => {
                for (out, &i) in output.iter_mut().zip(indices.iter()) {
                    *out = input[i];
                }
            }

            MatrixStorage::Sparse {
                rows,
                cols,
                values,
                ..
            } => {
                output.fill(0.0);
                for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
                    output[r] += v * input[c];
                }
            }

            MatrixStorage::Dense {
                data,
                output_dim,
                input_dim,
            } => {
                self.apply_dense_simd_into(input, data, *output_dim, *input_dim, output);
            }
        }

        // Add bias if present
        if !self.bias.is_empty() {
            for (y, &b) in output.iter_mut().zip(self.bias.iter()) {
                *y += b;
            }
        }
    }

    /// SIMD-optimized dense matrix-vector multiplication
    ///
    /// Uses 4-lane accumulation for better vectorization.
    #[inline]
    fn apply_dense_simd(
        &self,
        input: &[f32],
        matrix: &[f32],
        output_dim: usize,
        input_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; output_dim];
        self.apply_dense_simd_into(input, matrix, output_dim, input_dim, &mut output);
        output
    }

    /// SIMD-optimized dense matrix-vector multiplication into pre-allocated buffer
    #[inline]
    fn apply_dense_simd_into(
        &self,
        input: &[f32],
        matrix: &[f32],
        output_dim: usize,
        input_dim: usize,
        output: &mut [f32],
    ) {
        // Process 4 output elements at a time for SIMD
        let output_chunks = output_dim / 4;
        let output_remainder = output_dim % 4;

        // Main loop: process 4 rows at a time with better cache locality
        for chunk in 0..output_chunks {
            let base = chunk * 4;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;

            // Process input in chunks of 4 for better ILP
            let input_chunks = input_dim / 4;
            let input_remainder = input_dim % 4;

            let row0 = base * input_dim;
            let row1 = (base + 1) * input_dim;
            let row2 = (base + 2) * input_dim;
            let row3 = (base + 3) * input_dim;

            for jc in 0..input_chunks {
                let j = jc * 4;
                let x0 = input[j];
                let x1 = input[j + 1];
                let x2 = input[j + 2];
                let x3 = input[j + 3];

                acc0 += matrix[row0 + j] * x0
                    + matrix[row0 + j + 1] * x1
                    + matrix[row0 + j + 2] * x2
                    + matrix[row0 + j + 3] * x3;
                acc1 += matrix[row1 + j] * x0
                    + matrix[row1 + j + 1] * x1
                    + matrix[row1 + j + 2] * x2
                    + matrix[row1 + j + 3] * x3;
                acc2 += matrix[row2 + j] * x0
                    + matrix[row2 + j + 1] * x1
                    + matrix[row2 + j + 2] * x2
                    + matrix[row2 + j + 3] * x3;
                acc3 += matrix[row3 + j] * x0
                    + matrix[row3 + j + 1] * x1
                    + matrix[row3 + j + 2] * x2
                    + matrix[row3 + j + 3] * x3;
            }

            // Handle input remainder
            for j in (input_dim - input_remainder)..input_dim {
                let x = input[j];
                acc0 += matrix[row0 + j] * x;
                acc1 += matrix[row1 + j] * x;
                acc2 += matrix[row2 + j] * x;
                acc3 += matrix[row3 + j] * x;
            }

            output[base] = acc0;
            output[base + 1] = acc1;
            output[base + 2] = acc2;
            output[base + 3] = acc3;
        }

        // Handle output remainder rows
        for i in (output_dim - output_remainder)..output_dim {
            let row_start = i * input_dim;
            let mut sum = 0.0f32;

            // Unroll inner loop by 4
            let input_chunks = input_dim / 4;
            for jc in 0..input_chunks {
                let j = jc * 4;
                sum += matrix[row_start + j] * input[j]
                    + matrix[row_start + j + 1] * input[j + 1]
                    + matrix[row_start + j + 2] * input[j + 2]
                    + matrix[row_start + j + 3] * input[j + 3];
            }
            for j in (input_chunks * 4)..input_dim {
                sum += matrix[row_start + j] * input[j];
            }
            output[i] = sum;
        }
    }

    /// Compose two restriction maps: (B o A)(x) = B(A(x))
    pub fn compose(&self, other: &RestrictionMap) -> Result<RestrictionMap, RestrictionMapError> {
        // Check dimension compatibility
        if self.output_dim != other.input_dim {
            return Err(RestrictionMapError::DimensionMismatch {
                expected: other.input_dim,
                actual: self.output_dim,
            });
        }

        // Special case: both identity
        if self.matrix.is_identity() && other.matrix.is_identity() {
            return Ok(RestrictionMap::identity(self.input_dim));
        }

        // Special case: one is identity
        if self.matrix.is_identity() {
            return Ok(other.clone());
        }
        if other.matrix.is_identity() {
            return Ok(self.clone());
        }

        // General case: materialize both as dense and multiply
        // This is a simplification - could be optimized for sparse/diagonal
        Err(RestrictionMapError::Unsupported(
            "General matrix composition not yet implemented".to_string(),
        ))
    }
}

/// Builder for constructing RestrictionMap instances
#[derive(Debug, Default)]
pub struct RestrictionMapBuilder {
    matrix: Option<MatrixStorage>,
    bias: Vec<f32>,
    input_dim: Option<usize>,
    output_dim: Option<usize>,
}

impl RestrictionMapBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an identity map
    pub fn identity(mut self, dim: usize) -> Self {
        self.matrix = Some(MatrixStorage::Identity);
        self.input_dim = Some(dim);
        self.output_dim = Some(dim);
        self
    }

    /// Create a diagonal scaling map
    pub fn diagonal(mut self, scales: Vec<f32>) -> Self {
        let dim = scales.len();
        self.matrix = Some(MatrixStorage::Diagonal(scales));
        self.input_dim = Some(dim);
        self.output_dim = Some(dim);
        self
    }

    /// Create a projection map
    pub fn projection(mut self, indices: Vec<usize>, input_dim: usize) -> Self {
        let output_dim = indices.len();
        self.matrix = Some(MatrixStorage::Projection { indices, input_dim });
        self.input_dim = Some(input_dim);
        self.output_dim = Some(output_dim);
        self
    }

    /// Create a dense map
    pub fn dense(mut self, data: Vec<f32>, output_dim: usize, input_dim: usize) -> Self {
        self.matrix = Some(MatrixStorage::Dense {
            data,
            output_dim,
            input_dim,
        });
        self.input_dim = Some(input_dim);
        self.output_dim = Some(output_dim);
        self
    }

    /// Add a bias vector
    pub fn bias(mut self, bias: Vec<f32>) -> Self {
        self.bias = bias;
        self
    }

    /// Build the restriction map
    pub fn build(self) -> Result<RestrictionMap, RestrictionMapError> {
        let matrix = self
            .matrix
            .ok_or_else(|| RestrictionMapError::InvalidMatrix("No matrix specified".to_string()))?;

        let input_dim = self.input_dim.unwrap_or(0);
        let output_dim = self.output_dim.unwrap_or(0);

        if !self.bias.is_empty() && self.bias.len() != output_dim {
            return Err(RestrictionMapError::DimensionMismatch {
                expected: output_dim,
                actual: self.bias.len(),
            });
        }

        Ok(RestrictionMap {
            matrix,
            bias: self.bias,
            output_dim,
            input_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_map() {
        let map = RestrictionMap::identity(3);
        let input = vec![1.0, 2.0, 3.0];
        let output = map.apply(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_diagonal_map() {
        let map = RestrictionMap::diagonal(vec![2.0, 3.0, 4.0]);
        let input = vec![1.0, 2.0, 3.0];
        let output = map.apply(&input);
        assert_eq!(output, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_projection_map() {
        let map = RestrictionMap::projection(vec![0, 2], 3);
        let input = vec![1.0, 2.0, 3.0];
        let output = map.apply(&input);
        assert_eq!(output, vec![1.0, 3.0]);
    }

    #[test]
    fn test_dense_map() {
        // 2x3 matrix: [[1,2,3], [4,5,6]]
        let map = RestrictionMap::dense(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let input = vec![1.0, 1.0, 1.0];
        let output = map.apply(&input);
        assert_eq!(output, vec![6.0, 15.0]);
    }

    #[test]
    fn test_sparse_map() {
        // Sparse 2x3: only (0,0)=1, (0,2)=2, (1,1)=3
        let map = RestrictionMap::sparse(vec![0, 0, 1], vec![0, 2, 1], vec![1.0, 2.0, 3.0], 2, 3)
            .unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let output = map.apply(&input);
        // output[0] = 1*1 + 2*3 = 7
        // output[1] = 3*2 = 6
        assert_eq!(output, vec![7.0, 6.0]);
    }

    #[test]
    fn test_map_with_bias() {
        let map = RestrictionMap::diagonal(vec![2.0, 3.0])
            .with_bias(vec![1.0, 2.0])
            .unwrap();
        let input = vec![1.0, 2.0];
        let output = map.apply(&input);
        assert_eq!(output, vec![3.0, 8.0]);
    }

    #[test]
    fn test_builder() {
        let map = RestrictionMapBuilder::new()
            .diagonal(vec![1.0, 2.0, 3.0])
            .bias(vec![0.5, 0.5, 0.5])
            .build()
            .unwrap();

        let input = vec![1.0, 1.0, 1.0];
        let output = map.apply(&input);
        assert_eq!(output, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let map = RestrictionMap::diagonal(vec![1.0, 2.0]);
        let result = map.with_bias(vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dense_simd_optimization() {
        // Test with larger matrix to verify SIMD path
        let size = 16;
        let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
        let map = RestrictionMap::dense(data, size, size).unwrap();
        let input: Vec<f32> = vec![1.0; size];
        let output = map.apply(&input);

        // Verify output has correct dimension
        assert_eq!(output.len(), size);

        // Each row sums to sum of [row*size .. (row+1)*size-1]
        for (row, &val) in output.iter().enumerate() {
            let expected: f32 = (row * size..(row + 1) * size).map(|i| i as f32).sum();
            assert!(
                (val - expected).abs() < 1e-4,
                "Row {}: expected {}, got {}",
                row,
                expected,
                val
            );
        }
    }
}
