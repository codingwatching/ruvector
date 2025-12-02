//! Primary vector type implementation (RuVector)
//!
//! This is the main vector type, compatible with pgvector's `vector` type.
//! Stores f32 elements with efficient SIMD operations.

use pgrx::prelude::*;
use pgrx::pgrx_sql_entity_graph::metadata::{
    ArgumentError, Returns, ReturnsError, SqlMapping, SqlTranslatable,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::MAX_DIMENSIONS;

/// RuVector: Primary vector type for PostgreSQL
///
/// Memory layout:
/// - Header: 8 bytes (varlena header + dimensions)
/// - Data: 4 bytes per dimension (f32)
///
/// Maximum dimensions: 16,000
#[derive(Clone, Serialize, Deserialize)]
pub struct RuVector {
    /// Vector dimensions (cached for fast access)
    dimensions: u32,
    /// Vector data (f32 elements)
    data: Vec<f32>,
}

impl RuVector {
    /// Create a new vector from a slice
    pub fn from_slice(data: &[f32]) -> Self {
        if data.len() > MAX_DIMENSIONS {
            pgrx::error!(
                "Vector dimension {} exceeds maximum {}",
                data.len(),
                MAX_DIMENSIONS
            );
        }

        Self {
            dimensions: data.len() as u32,
            data: data.to_vec(),
        }
    }

    /// Create a zero vector of given dimensions
    pub fn zeros(dimensions: usize) -> Self {
        if dimensions > MAX_DIMENSIONS {
            pgrx::error!(
                "Vector dimension {} exceeds maximum {}",
                dimensions,
                MAX_DIMENSIONS
            );
        }

        Self {
            dimensions: dimensions as u32,
            data: vec![0.0; dimensions],
        }
    }

    /// Get vector dimensions
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions as usize
    }

    /// Get vector data as slice
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable vector data
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Convert to Vec<f32>
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Calculate L2 norm
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            return self.clone();
        }
        Self {
            dimensions: self.dimensions,
            data: self.data.iter().map(|x| x / norm).collect(),
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.dimensions, other.dimensions,
            "Vector dimensions must match"
        );
        Self {
            dimensions: self.dimensions,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(
            self.dimensions, other.dimensions,
            "Vector dimensions must match"
        );
        Self {
            dimensions: self.dimensions,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        Self {
            dimensions: self.dimensions,
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }

    /// Dot product
    pub fn dot(&self, other: &Self) -> f32 {
        assert_eq!(
            self.dimensions, other.dimensions,
            "Vector dimensions must match"
        );
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    /// Memory size in bytes
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len() * std::mem::size_of::<f32>()
    }

    /// Serialize to bytes (dimensions + f32 data)
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + self.data.len() * 4);
        bytes.extend_from_slice(&self.dimensions.to_le_bytes());
        for val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes
    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < 4 {
            pgrx::error!("Invalid vector data: too short");
        }

        let dimensions = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let expected_len = 4 + (dimensions as usize) * 4;

        if bytes.len() != expected_len {
            pgrx::error!(
                "Invalid vector data: expected {} bytes, got {}",
                expected_len,
                bytes.len()
            );
        }

        let mut data = Vec::with_capacity(dimensions as usize);
        for i in 0..dimensions as usize {
            let offset = 4 + i * 4;
            let val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            data.push(val);
        }

        Self { dimensions, data }
    }
}

impl fmt::Display for RuVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for RuVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RuVector(dims={}, {:?})", self.dimensions, &self.data)
    }
}

impl FromStr for RuVector {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse format: [1.0, 2.0, 3.0] or [1,2,3]
        let s = s.trim();
        if !s.starts_with('[') || !s.ends_with(']') {
            return Err(format!("Invalid vector format: {}", s));
        }

        let inner = &s[1..s.len() - 1];
        if inner.is_empty() {
            return Ok(Self::zeros(0));
        }

        let values: Result<Vec<f32>, _> = inner
            .split(',')
            .map(|v| v.trim().parse::<f32>())
            .collect();

        match values {
            Ok(data) => Ok(Self::from_slice(&data)),
            Err(e) => Err(format!("Invalid vector element: {}", e)),
        }
    }
}

impl PartialEq for RuVector {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.data == other.data
    }
}

impl Eq for RuVector {}

// ============================================================================
// PostgreSQL Type Integration
// ============================================================================

unsafe impl SqlTranslatable for RuVector {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("ruvector")))
    }

    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("ruvector"))))
    }
}

impl pgrx::IntoDatum for RuVector {
    fn into_datum(self) -> Option<pgrx::pg_sys::Datum> {
        let bytes = self.to_bytes();
        let len = bytes.len();
        let total_size = pgrx::pg_sys::VARHDRSZ + len;

        unsafe {
            let ptr = pgrx::pg_sys::palloc(total_size) as *mut u8;
            let varlena = ptr as *mut pgrx::pg_sys::varlena;
            pgrx::varlena::set_varsize_4b(varlena, total_size as i32);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(pgrx::pg_sys::VARHDRSZ), len);
            Some(pgrx::pg_sys::Datum::from(ptr))
        }
    }

    fn type_oid() -> pgrx::pg_sys::Oid {
        pgrx::pg_sys::Oid::INVALID
    }
}

impl pgrx::FromDatum for RuVector {
    unsafe fn from_polymorphic_datum(
        datum: pgrx::pg_sys::Datum,
        is_null: bool,
        _typoid: pgrx::pg_sys::Oid,
    ) -> Option<Self> {
        if is_null {
            return None;
        }

        let ptr = datum.cast_mut_ptr::<pgrx::pg_sys::varlena>();
        let len = pgrx::varlena::varsize_any_exhdr(ptr);
        let data_ptr = pgrx::varlena::vardata_any(ptr) as *const u8;
        let bytes = std::slice::from_raw_parts(data_ptr, len);

        Some(RuVector::from_bytes(bytes))
    }
}

// ============================================================================
// SQL Helper Functions
// ============================================================================

/// Create a vector from a float array
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_from_array(arr: Vec<f32>) -> pgrx::JsonB {
    if arr.len() > MAX_DIMENSIONS {
        pgrx::error!("Vector exceeds maximum dimensions ({})", MAX_DIMENSIONS);
    }
    let v = RuVector::from_slice(&arr);
    pgrx::JsonB(serde_json::json!({
        "dimensions": v.dimensions(),
        "data": v.as_slice(),
    }))
}

/// Parse a vector from string format [1.0, 2.0, 3.0]
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_parse(input: &str) -> pgrx::JsonB {
    match RuVector::from_str(input) {
        Ok(v) => pgrx::JsonB(serde_json::json!({
            "dimensions": v.dimensions(),
            "data": v.as_slice(),
        })),
        Err(e) => pgrx::error!("Invalid vector format: {}", e),
    }
}

/// Get vector as float array from JSON representation
#[pg_extern(immutable, parallel_safe)]
pub fn ruvector_to_array(v: pgrx::JsonB) -> Vec<f32> {
    let obj = v.0.as_object().expect("Invalid vector JSON");
    let data = obj.get("data").expect("Missing data field");
    data.as_array()
        .expect("Data must be array")
        .iter()
        .map(|x| x.as_f64().expect("Invalid float") as f32)
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_slice() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.dimensions(), 3);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_zeros() {
        let v = RuVector::zeros(5);
        assert_eq!(v.dimensions(), 5);
        assert_eq!(v.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_norm() {
        let v = RuVector::from_slice(&[3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = RuVector::from_slice(&[3.0, 4.0]);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot() {
        let a = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let b = RuVector::from_slice(&[4.0, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_sub() {
        let a = RuVector::from_slice(&[1.0, 2.0]);
        let b = RuVector::from_slice(&[3.0, 4.0]);
        assert_eq!(a.add(&b).as_slice(), &[4.0, 6.0]);
        assert_eq!(b.sub(&a).as_slice(), &[2.0, 2.0]);
    }

    #[test]
    fn test_parse() {
        let v: RuVector = "[1.0, 2.0, 3.0]".parse().unwrap();
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);

        let v2: RuVector = "[1,2,3]".parse().unwrap();
        assert_eq!(v2.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_display() {
        let v = RuVector::from_slice(&[1.0, 2.5, 3.0]);
        assert_eq!(v.to_string(), "[1,2.5,3]");
    }

    #[test]
    fn test_serialization() {
        let v = RuVector::from_slice(&[1.0, 2.0, 3.0]);
        let bytes = v.to_bytes();
        let v2 = RuVector::from_bytes(&bytes);
        assert_eq!(v, v2);
    }
}
