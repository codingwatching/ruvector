//! Half-precision (f16) vector type implementation
//!
//! HalfVec stores vectors using 16-bit floating point, reducing memory
//! usage by 50% compared to f32 with minimal accuracy loss.

use half::f16;
use pgrx::prelude::*;
use pgrx::pgrx_sql_entity_graph::metadata::{
    ArgumentError, Returns, ReturnsError, SqlMapping, SqlTranslatable,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::MAX_DIMENSIONS;

/// HalfVec: Half-precision (f16) vector type
///
/// Memory layout:
/// - Header: 8 bytes (varlena header + dimensions)
/// - Data: 2 bytes per dimension (f16)
///
/// Benefits:
/// - 50% memory reduction vs f32
/// - Faster memory bandwidth
/// - Minimal accuracy loss for most embeddings
#[derive(Clone, Serialize, Deserialize)]
pub struct HalfVec {
    /// Vector dimensions
    dimensions: u32,
    /// Vector data (f16 stored as u16 for serialization)
    data: Vec<u16>,
}

impl HalfVec {
    /// Create from f32 slice
    pub fn from_f32(data: &[f32]) -> Self {
        if data.len() > MAX_DIMENSIONS {
            pgrx::error!(
                "Vector dimension {} exceeds maximum {}",
                data.len(),
                MAX_DIMENSIONS
            );
        }

        Self {
            dimensions: data.len() as u32,
            data: data.iter().map(|&x| f16::from_f32(x).to_bits()).collect(),
        }
    }

    /// Create from f16 slice
    pub fn from_f16(data: &[f16]) -> Self {
        if data.len() > MAX_DIMENSIONS {
            pgrx::error!(
                "Vector dimension {} exceeds maximum {}",
                data.len(),
                MAX_DIMENSIONS
            );
        }

        Self {
            dimensions: data.len() as u32,
            data: data.iter().map(|x| x.to_bits()).collect(),
        }
    }

    /// Create a zero vector
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
            data: vec![f16::ZERO.to_bits(); dimensions],
        }
    }

    /// Get dimensions
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions as usize
    }

    /// Get data as f16 slice
    pub fn as_f16(&self) -> Vec<f16> {
        self.data.iter().map(|&bits| f16::from_bits(bits)).collect()
    }

    /// Convert to f32 Vec
    pub fn to_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect()
    }

    /// Get raw u16 data
    pub fn as_raw(&self) -> &[u16] {
        &self.data
    }

    /// Calculate L2 norm
    pub fn norm(&self) -> f32 {
        self.to_f32().iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Memory size in bytes
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len() * std::mem::size_of::<u16>()
    }

    /// Serialize to bytes (dimensions + u16 data)
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + self.data.len() * 2);
        bytes.extend_from_slice(&self.dimensions.to_le_bytes());
        for val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes
    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() < 4 {
            pgrx::error!("Invalid halfvec data: too short");
        }

        let dimensions = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let expected_len = 4 + (dimensions as usize) * 2;

        if bytes.len() != expected_len {
            pgrx::error!(
                "Invalid halfvec data: expected {} bytes, got {}",
                expected_len,
                bytes.len()
            );
        }

        let mut data = Vec::with_capacity(dimensions as usize);
        for i in 0..dimensions as usize {
            let offset = 4 + i * 2;
            let val = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            data.push(val);
        }

        Self { dimensions, data }
    }
}

impl fmt::Display for HalfVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &bits) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", f16::from_bits(bits))?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for HalfVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HalfVec(dims={}, [...])", self.dimensions)
    }
}

impl FromStr for HalfVec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if !s.starts_with('[') || !s.ends_with(']') {
            return Err(format!("Invalid halfvec format: {}", s));
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
            Ok(data) => Ok(Self::from_f32(&data)),
            Err(e) => Err(format!("Invalid halfvec element: {}", e)),
        }
    }
}

impl PartialEq for HalfVec {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.data == other.data
    }
}

impl Eq for HalfVec {}

// ============================================================================
// PostgreSQL Type Integration
// ============================================================================

unsafe impl SqlTranslatable for HalfVec {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("halfvec")))
    }

    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("halfvec"))))
    }
}

impl pgrx::IntoDatum for HalfVec {
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

impl pgrx::FromDatum for HalfVec {
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

        Some(HalfVec::from_bytes(bytes))
    }
}

// ============================================================================
// SQL Helper Functions
// ============================================================================

/// Create a halfvec from a float array
#[pg_extern(immutable, parallel_safe)]
pub fn halfvec_from_array(arr: Vec<f32>) -> pgrx::JsonB {
    if arr.len() > MAX_DIMENSIONS {
        pgrx::error!("Vector exceeds maximum dimensions ({})", MAX_DIMENSIONS);
    }
    let v = HalfVec::from_f32(&arr);
    pgrx::JsonB(serde_json::json!({
        "dimensions": v.dimensions(),
        "data": v.to_f32(),
    }))
}

/// Parse a halfvec from string format [1.0, 2.0, 3.0]
#[pg_extern(immutable, parallel_safe)]
pub fn halfvec_parse(input: &str) -> pgrx::JsonB {
    match HalfVec::from_str(input) {
        Ok(v) => pgrx::JsonB(serde_json::json!({
            "dimensions": v.dimensions(),
            "data": v.to_f32(),
        })),
        Err(e) => pgrx::error!("Invalid halfvec format: {}", e),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f32() {
        let v = HalfVec::from_f32(&[1.0, 2.0, 3.0]);
        assert_eq!(v.dimensions(), 3);

        let f32_data = v.to_f32();
        assert!((f32_data[0] - 1.0).abs() < 0.01);
        assert!((f32_data[1] - 2.0).abs() < 0.01);
        assert!((f32_data[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_savings() {
        let f32_vec = vec![1.0f32; 1536];
        let half_vec = HalfVec::from_f32(&f32_vec);

        // HalfVec should be ~50% of f32 size
        let f32_size = f32_vec.len() * 4; // 6144 bytes
        let half_size = half_vec.data.len() * 2; // 3072 bytes

        assert_eq!(half_size, f32_size / 2);
    }

    #[test]
    fn test_precision() {
        // Test that precision is acceptable for typical embedding values
        let original = vec![0.123456, -0.654321, 0.999999, -0.000001];
        let half = HalfVec::from_f32(&original);
        let restored = half.to_f32();

        for (orig, rest) in original.iter().zip(restored.iter()) {
            // f16 has ~3 decimal digits of precision
            assert!((orig - rest).abs() < 0.001, "orig={}, restored={}", orig, rest);
        }
    }

    #[test]
    fn test_serialization() {
        let v = HalfVec::from_f32(&[1.0, 2.0, 3.0]);
        let bytes = v.to_bytes();
        let v2 = HalfVec::from_bytes(&bytes);
        assert_eq!(v, v2);
    }
}
