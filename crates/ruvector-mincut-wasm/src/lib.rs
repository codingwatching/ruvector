//! WASM bindings for RuVector MinCut
//!
//! Provides JavaScript/TypeScript API for dynamic minimum cut operations.
//!
//! ## Example Usage
//!
//! ```javascript
//! import init, { WasmMinCut } from './ruvector_mincut_wasm';
//!
//! await init();
//!
//! // Create from edges
//! const edges = [
//!   [0, 1, 1.0],
//!   [1, 2, 1.0],
//!   [0, 2, 1.0]
//! ];
//! const mincut = WasmMinCut.fromEdges(edges);
//!
//! console.log(mincut.minCutValue());
//! console.log(mincut.partition());
//! ```

use wasm_bindgen::prelude::*;
use ruvector_mincut::{DynamicMinCut, MinCutBuilder, MinCutConfig};
use serde::{Serialize, Deserialize};

/// WASM wrapper for DynamicMinCut
#[wasm_bindgen]
pub struct WasmMinCut {
    inner: DynamicMinCut,
}

#[derive(Serialize, Deserialize)]
struct EdgeInput {
    u: u64,
    v: u64,
    weight: f64,
}

#[derive(Serialize, Deserialize)]
struct Partition {
    s: Vec<u64>,
    t: Vec<u64>,
}

#[derive(Serialize, Deserialize)]
struct Edge {
    u: u64,
    v: u64,
    weight: f64,
}

#[derive(Serialize, Deserialize)]
struct Stats {
    num_vertices: usize,
    num_edges: usize,
    min_cut_value: f64,
    is_connected: bool,
    num_operations: usize,
}

#[wasm_bindgen]
impl WasmMinCut {
    /// Create a new empty minimum cut structure
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmMinCut, JsError> {
        console_error_panic_hook::set_once();

        Ok(WasmMinCut {
            inner: DynamicMinCut::new(MinCutConfig::default()),
        })
    }

    /// Create from edges array: [[u, v, weight], ...]
    ///
    /// # Arguments
    /// * `edges` - JavaScript array of [u, v, weight] tuples
    ///
    /// # Example
    /// ```javascript
    /// const edges = [[0, 1, 1.5], [1, 2, 2.0]];
    /// const mincut = WasmMinCut.fromEdges(edges);
    /// ```
    #[wasm_bindgen(js_name = "fromEdges")]
    pub fn from_edges(edges: JsValue) -> Result<WasmMinCut, JsError> {
        console_error_panic_hook::set_once();

        // Deserialize edges from JavaScript array
        let edges_vec: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(edges)
            .map_err(|e| JsError::new(&format!("Failed to parse edges: {}", e)))?;

        // Convert to tuple format expected by with_edges
        let mut edge_tuples = Vec::with_capacity(edges_vec.len());

        for edge in edges_vec {
            if edge.len() != 3 {
                return Err(JsError::new("Each edge must be [u, v, weight]"));
            }

            let u = edge[0] as u64;
            let v = edge[1] as u64;
            let weight = edge[2];

            edge_tuples.push((u, v, weight));
        }

        let inner = MinCutBuilder::new()
            .with_edges(edge_tuples)
            .build()
            .map_err(|e| JsError::new(&format!("Failed to build mincut: {}", e)))?;

        Ok(WasmMinCut { inner })
    }

    /// Insert an edge into the graph
    ///
    /// # Arguments
    /// * `u` - Source vertex
    /// * `v` - Target vertex
    /// * `weight` - Edge weight
    ///
    /// # Returns
    /// The new minimum cut value after insertion
    #[wasm_bindgen(js_name = "insertEdge")]
    pub fn insert_edge(&mut self, u: u64, v: u64, weight: f64) -> Result<f64, JsError> {
        self.inner.insert_edge(u, v, weight)
            .map_err(|e| JsError::new(&format!("Failed to insert edge: {}", e)))
    }

    /// Delete an edge from the graph
    ///
    /// # Arguments
    /// * `u` - Source vertex
    /// * `v` - Target vertex
    ///
    /// # Returns
    /// The new minimum cut value after deletion
    #[wasm_bindgen(js_name = "deleteEdge")]
    pub fn delete_edge(&mut self, u: u64, v: u64) -> Result<f64, JsError> {
        self.inner.delete_edge(u, v)
            .map_err(|e| JsError::new(&format!("Failed to delete edge: {}", e)))
    }

    /// Get the current minimum cut value
    ///
    /// # Returns
    /// The sum of edge weights in the minimum cut
    #[wasm_bindgen(js_name = "minCutValue")]
    pub fn min_cut_value(&self) -> f64 {
        self.inner.min_cut_value()
    }

    /// Get the partition as JSON: { "s": [...], "t": [...] }
    ///
    /// # Returns
    /// JavaScript object with two arrays: `s` and `t` containing vertex IDs
    ///
    /// # Example
    /// ```javascript
    /// const { s, t } = mincut.partition();
    /// console.log("S partition:", s);
    /// console.log("T partition:", t);
    /// ```
    #[wasm_bindgen]
    pub fn partition(&self) -> JsValue {
        let (s_set, t_set) = self.inner.partition();

        let partition = Partition {
            s: s_set.into_iter().collect(),
            t: t_set.into_iter().collect(),
        };

        serde_wasm_bindgen::to_value(&partition).unwrap_or(JsValue::NULL)
    }

    /// Get the cut edges as JSON array
    ///
    /// # Returns
    /// JavaScript array of edge objects: [{ u, v, weight }, ...]
    ///
    /// # Example
    /// ```javascript
    /// const edges = mincut.cutEdges();
    /// edges.forEach(e => console.log(`Edge ${e.u}-${e.v}: ${e.weight}`));
    /// ```
    #[wasm_bindgen(js_name = "cutEdges")]
    pub fn cut_edges(&self) -> JsValue {
        let edges = self.inner.cut_edges();

        let edge_list: Vec<Edge> = edges
            .into_iter()
            .map(|e| Edge { u: e.source, v: e.target, weight: e.weight })
            .collect();

        serde_wasm_bindgen::to_value(&edge_list).unwrap_or(JsValue::NULL)
    }

    /// Get the number of vertices in the graph
    #[wasm_bindgen(js_name = "numVertices")]
    pub fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    /// Get the number of edges in the graph
    #[wasm_bindgen(js_name = "numEdges")]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Check if the graph is connected
    ///
    /// # Returns
    /// `true` if there is a path between all vertex pairs
    #[wasm_bindgen(js_name = "isConnected")]
    pub fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Get comprehensive statistics as JSON
    ///
    /// # Returns
    /// JavaScript object with:
    /// - `num_vertices`: Number of vertices
    /// - `num_edges`: Number of edges
    /// - `min_cut_value`: Current minimum cut value
    /// - `is_connected`: Whether graph is connected
    /// - `num_operations`: Total operations performed
    ///
    /// # Example
    /// ```javascript
    /// const stats = mincut.stats();
    /// console.log(`Graph has ${stats.num_vertices} vertices and ${stats.num_edges} edges`);
    /// console.log(`Minimum cut value: ${stats.min_cut_value}`);
    /// ```
    #[wasm_bindgen]
    pub fn stats(&self) -> JsValue {
        let algo_stats = self.inner.stats();
        let stats = Stats {
            num_vertices: self.inner.num_vertices(),
            num_edges: self.inner.num_edges(),
            min_cut_value: self.inner.min_cut_value(),
            is_connected: self.inner.is_connected(),
            num_operations: (algo_stats.insertions + algo_stats.deletions + algo_stats.queries) as usize,
        };

        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }

    /// Update an edge weight (delete old, insert new)
    ///
    /// # Arguments
    /// * `u` - Source vertex
    /// * `v` - Target vertex
    /// * `new_weight` - New edge weight
    ///
    /// # Returns
    /// The new minimum cut value after update
    #[wasm_bindgen(js_name = "updateEdge")]
    pub fn update_edge(&mut self, u: u64, v: u64, new_weight: f64) -> Result<f64, JsError> {
        // Delete old edge (ignore error if doesn't exist)
        let _ = self.inner.delete_edge(u, v);

        // Insert with new weight
        self.inner.insert_edge(u, v, new_weight)
            .map_err(|e| JsError::new(&format!("Failed to update edge: {}", e)))
    }

    /// Batch insert multiple edges
    ///
    /// # Arguments
    /// * `edges` - JavaScript array of [u, v, weight] tuples
    ///
    /// # Returns
    /// The final minimum cut value
    ///
    /// # Example
    /// ```javascript
    /// const edges = [[0, 1, 1.0], [1, 2, 2.0], [2, 3, 1.5]];
    /// const cutValue = mincut.batchInsert(edges);
    /// ```
    #[wasm_bindgen(js_name = "batchInsert")]
    pub fn batch_insert(&mut self, edges: JsValue) -> Result<f64, JsError> {
        let edges_vec: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(edges)
            .map_err(|e| JsError::new(&format!("Failed to parse edges: {}", e)))?;

        for edge in edges_vec {
            if edge.len() != 3 {
                return Err(JsError::new("Each edge must be [u, v, weight]"));
            }

            let u = edge[0] as u64;
            let v = edge[1] as u64;
            let weight = edge[2];

            self.inner.insert_edge(u, v, weight)
                .map_err(|e| JsError::new(&format!("Failed to insert edge [{}, {}]: {}", u, v, e)))?;
        }

        Ok(self.inner.min_cut_value())
    }

    /// Batch delete multiple edges
    ///
    /// # Arguments
    /// * `edges` - JavaScript array of [u, v] tuples
    ///
    /// # Returns
    /// The final minimum cut value
    ///
    /// # Example
    /// ```javascript
    /// const edges = [[0, 1], [1, 2]];
    /// const cutValue = mincut.batchDelete(edges);
    /// ```
    #[wasm_bindgen(js_name = "batchDelete")]
    pub fn batch_delete(&mut self, edges: JsValue) -> Result<f64, JsError> {
        let edges_vec: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(edges)
            .map_err(|e| JsError::new(&format!("Failed to parse edges: {}", e)))?;

        for edge in edges_vec {
            if edge.len() < 2 {
                return Err(JsError::new("Each edge must be [u, v] or [u, v, weight]"));
            }

            let u = edge[0] as u64;
            let v = edge[1] as u64;

            self.inner.delete_edge(u, v)
                .map_err(|e| JsError::new(&format!("Failed to delete edge [{}, {}]: {}", u, v, e)))?;
        }

        Ok(self.inner.min_cut_value())
    }

    /// Clear all edges from the graph
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.inner = DynamicMinCut::new(MinCutConfig::default());
    }
}

/// Initialize the WASM module (call once at startup)
///
/// This sets up panic hooks for better error messages in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Get version information
#[wasm_bindgen(js_name = "getVersion")]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_new() {
        let mincut = WasmMinCut::new().unwrap();
        assert_eq!(mincut.num_vertices(), 0);
        assert_eq!(mincut.num_edges(), 0);
    }

    #[wasm_bindgen_test]
    fn test_insert_edge() {
        let mut mincut = WasmMinCut::new().unwrap();
        let result = mincut.insert_edge(0, 1, 1.0);
        assert!(result.is_ok());
        assert_eq!(mincut.num_edges(), 1);
    }

    #[wasm_bindgen_test]
    fn test_min_cut_value() {
        let mut mincut = WasmMinCut::new().unwrap();
        mincut.insert_edge(0, 1, 1.0).unwrap();
        mincut.insert_edge(1, 2, 2.0).unwrap();
        mincut.insert_edge(0, 2, 1.5).unwrap();

        let cut_value = mincut.min_cut_value();
        assert!(cut_value > 0.0);
    }
}
