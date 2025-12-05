//! ConceptNet to RuVector Graph Builder
//!
//! Converts ConceptNet edges into RuVector's graph database format.

use crate::api::{ConceptNode, Edge, RelationType};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use uuid::Uuid;

/// Errors during graph construction
#[derive(Error, Debug)]
pub enum GraphBuildError {
    #[error("Invalid node: {0}")]
    InvalidNode(String),

    #[error("Duplicate edge: {0}")]
    DuplicateEdge(String),

    #[error("Graph capacity exceeded: {max} nodes")]
    CapacityExceeded { max: usize },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Configuration for graph building
#[derive(Debug, Clone)]
pub struct GraphBuildConfig {
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Maximum edges per node
    pub max_edges_per_node: usize,
    /// Minimum edge weight to include
    pub min_edge_weight: f64,
    /// Languages to include (empty = all)
    pub languages: Vec<String>,
    /// Relations to include (empty = all)
    pub relations: Vec<RelationType>,
    /// Enable deduplication
    pub deduplicate: bool,
    /// Store embeddings alongside nodes
    pub store_embeddings: bool,
}

impl Default for GraphBuildConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1_000_000,
            max_edges_per_node: 1000,
            min_edge_weight: 0.5,
            languages: vec!["en".to_string()],
            relations: vec![],
            deduplicate: true,
            store_embeddings: true,
        }
    }
}

/// Node in the constructed graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub uri: String,
    pub label: String,
    pub language: String,
    pub embedding: Option<Vec<f32>>,
    pub properties: HashMap<String, GraphProperty>,
}

/// Property value types
#[derive(Debug, Clone)]
pub enum GraphProperty {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    StringArray(Vec<String>),
}

/// Edge in the constructed graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation: RelationType,
    pub weight: f64,
    pub surface_text: Option<String>,
    pub confidence: f64,
    pub sources: Vec<String>,
    pub properties: HashMap<String, GraphProperty>,
}

/// Hyperedge connecting 3+ concepts
#[derive(Debug, Clone)]
pub struct GraphHyperedge {
    pub id: String,
    pub node_ids: Vec<String>,
    pub relation: RelationType,
    pub weight: f64,
    pub properties: HashMap<String, GraphProperty>,
}

/// Statistics about the built graph
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub total_hyperedges: usize,
    pub nodes_by_language: HashMap<String, usize>,
    pub edges_by_relation: HashMap<String, usize>,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub connected_components: usize,
}

/// Builder for constructing RuVector graphs from ConceptNet data
pub struct ConceptNetGraphBuilder {
    config: GraphBuildConfig,
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
    hyperedges: Vec<GraphHyperedge>,
    edge_hashes: HashSet<u64>,
    node_degrees: HashMap<String, usize>,
}

impl ConceptNetGraphBuilder {
    /// Create a new graph builder
    pub fn new(config: GraphBuildConfig) -> Self {
        Self {
            config,
            nodes: HashMap::with_capacity(10000),
            edges: Vec::with_capacity(50000),
            hyperedges: Vec::new(),
            edge_hashes: HashSet::new(),
            node_degrees: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(GraphBuildConfig::default())
    }

    /// Add a ConceptNet edge to the graph
    pub fn add_edge(&mut self, edge: &Edge) -> Result<(), GraphBuildError> {
        // Check weight threshold
        if edge.weight < self.config.min_edge_weight {
            return Ok(());
        }

        // Check relation filter
        let relation = edge.relation_type();
        if !self.config.relations.is_empty() && !self.config.relations.contains(&relation) {
            return Ok(());
        }

        // Check language filter
        if !self.config.languages.is_empty() {
            let start_lang = edge.start.language_code().unwrap_or("unknown");
            let end_lang = edge.end.language_code().unwrap_or("unknown");
            if !self.config.languages.contains(&start_lang.to_string())
                && !self.config.languages.contains(&end_lang.to_string())
            {
                return Ok(());
            }
        }

        // Deduplicate
        if self.config.deduplicate {
            let hash = Self::edge_hash(&edge.start.id, &edge.end.id, &edge.rel.id);
            if self.edge_hashes.contains(&hash) {
                return Ok(());
            }
            self.edge_hashes.insert(hash);
        }

        // Add source node
        self.ensure_node(&edge.start)?;

        // Add target node
        self.ensure_node(&edge.end)?;

        // Check degree limits
        let source_degree = self.node_degrees.get(&edge.start.id).copied().unwrap_or(0);
        let target_degree = self.node_degrees.get(&edge.end.id).copied().unwrap_or(0);

        if source_degree >= self.config.max_edges_per_node
            || target_degree >= self.config.max_edges_per_node
        {
            return Ok(()); // Skip but don't error
        }

        // Create graph edge
        let graph_edge = GraphEdge {
            id: Uuid::new_v4().to_string(),
            source_id: edge.start.id.clone(),
            target_id: edge.end.id.clone(),
            relation,
            weight: edge.weight,
            surface_text: edge.surface_text.clone(),
            confidence: edge.confidence(),
            sources: edge.sources.iter().filter_map(|s| s.id.clone()).collect(),
            properties: Self::extract_edge_properties(edge),
        };

        self.edges.push(graph_edge);
        *self.node_degrees.entry(edge.start.id.clone()).or_insert(0) += 1;
        *self.node_degrees.entry(edge.end.id.clone()).or_insert(0) += 1;

        Ok(())
    }

    /// Add multiple edges in batch
    pub fn add_edges(&mut self, edges: &[Edge]) -> Result<usize, GraphBuildError> {
        let mut added = 0;
        for edge in edges {
            if self.nodes.len() >= self.config.max_nodes {
                return Err(GraphBuildError::CapacityExceeded {
                    max: self.config.max_nodes,
                });
            }
            self.add_edge(edge)?;
            added += 1;
        }
        Ok(added)
    }

    /// Create a hyperedge from multiple concepts
    pub fn add_hyperedge(
        &mut self,
        node_uris: &[String],
        relation: RelationType,
        weight: f64,
    ) -> Result<String, GraphBuildError> {
        if node_uris.len() < 3 {
            return Err(GraphBuildError::InvalidNode(
                "Hyperedge requires at least 3 nodes".into(),
            ));
        }

        // Ensure all nodes exist
        for uri in node_uris {
            if !self.nodes.contains_key(uri) {
                return Err(GraphBuildError::InvalidNode(format!(
                    "Node not found: {}",
                    uri
                )));
            }
        }

        let id = Uuid::new_v4().to_string();
        let hyperedge = GraphHyperedge {
            id: id.clone(),
            node_ids: node_uris.to_vec(),
            relation,
            weight,
            properties: HashMap::new(),
        };

        self.hyperedges.push(hyperedge);
        Ok(id)
    }

    /// Set embedding for a node
    pub fn set_node_embedding(&mut self, uri: &str, embedding: Vec<f32>) -> Result<(), GraphBuildError> {
        if let Some(node) = self.nodes.get_mut(uri) {
            node.embedding = Some(embedding);
            Ok(())
        } else {
            Err(GraphBuildError::InvalidNode(format!("Node not found: {}", uri)))
        }
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.values()
    }

    /// Get all edges
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Get all hyperedges
    pub fn hyperedges(&self) -> &[GraphHyperedge] {
        &self.hyperedges
    }

    /// Get a node by URI
    pub fn get_node(&self, uri: &str) -> Option<&GraphNode> {
        self.nodes.get(uri)
    }

    /// Get edges for a node
    pub fn get_node_edges(&self, uri: &str) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.source_id == uri || e.target_id == uri)
            .collect()
    }

    /// Get outgoing edges for a node
    pub fn get_outgoing_edges(&self, uri: &str) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.source_id == uri)
            .collect()
    }

    /// Get incoming edges for a node
    pub fn get_incoming_edges(&self, uri: &str) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.target_id == uri)
            .collect()
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, uri: &str) -> Vec<&GraphNode> {
        let mut neighbor_ids: HashSet<&String> = HashSet::new();

        for edge in &self.edges {
            if edge.source_id == uri {
                neighbor_ids.insert(&edge.target_id);
            } else if edge.target_id == uri {
                neighbor_ids.insert(&edge.source_id);
            }
        }

        neighbor_ids
            .into_iter()
            .filter_map(|id| self.nodes.get(id))
            .collect()
    }

    /// Compute graph statistics
    pub fn stats(&self) -> GraphStats {
        let mut nodes_by_language: HashMap<String, usize> = HashMap::new();
        let mut edges_by_relation: HashMap<String, usize> = HashMap::new();
        let mut max_degree = 0usize;

        for node in self.nodes.values() {
            *nodes_by_language.entry(node.language.clone()).or_insert(0) += 1;
        }

        for edge in &self.edges {
            let rel_name = format!("{:?}", edge.relation);
            *edges_by_relation.entry(rel_name).or_insert(0) += 1;
        }

        for &degree in self.node_degrees.values() {
            max_degree = max_degree.max(degree);
        }

        let total_degree: usize = self.node_degrees.values().sum();
        let avg_degree = if self.nodes.is_empty() {
            0.0
        } else {
            total_degree as f64 / self.nodes.len() as f64
        };

        GraphStats {
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            total_hyperedges: self.hyperedges.len(),
            nodes_by_language,
            edges_by_relation,
            avg_degree,
            max_degree,
            connected_components: self.count_components(),
        }
    }

    /// Find shortest path between two concepts
    pub fn shortest_path(&self, start_uri: &str, end_uri: &str, max_depth: usize) -> Option<Vec<String>> {
        use std::collections::VecDeque;

        if start_uri == end_uri {
            return Some(vec![start_uri.to_string()]);
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();

        queue.push_back((start_uri.to_string(), vec![start_uri.to_string()]));
        visited.insert(start_uri);

        while let Some((current, path)) = queue.pop_front() {
            if path.len() > max_depth {
                continue;
            }

            for edge in &self.edges {
                let next = if edge.source_id == current {
                    &edge.target_id
                } else if edge.target_id == current {
                    &edge.source_id
                } else {
                    continue;
                };

                if *next == end_uri {
                    let mut result = path.clone();
                    result.push(next.clone());
                    return Some(result);
                }

                if !visited.contains(next.as_str()) {
                    visited.insert(next);
                    let mut new_path = path.clone();
                    new_path.push(next.clone());
                    queue.push_back((next.clone(), new_path));
                }
            }
        }

        None
    }

    /// Generate Cypher CREATE statements for the graph
    pub fn to_cypher(&self) -> String {
        let mut cypher = String::new();

        // Create nodes
        for node in self.nodes.values() {
            cypher.push_str(&format!(
                "CREATE (n:Concept {{uri: '{}', label: '{}', language: '{}'}})\n",
                node.uri,
                node.label.replace('\'', "\\'"),
                node.language
            ));
        }

        // Create edges
        for edge in &self.edges {
            cypher.push_str(&format!(
                "MATCH (a:Concept {{uri: '{}'}}), (b:Concept {{uri: '{}'}}) \
                 CREATE (a)-[:{}{{weight: {:.4}, confidence: {:.4}}}]->(b)\n",
                edge.source_id,
                edge.target_id,
                format!("{:?}", edge.relation).to_uppercase(),
                edge.weight,
                edge.confidence
            ));
        }

        cypher
    }

    /// Export graph to JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "nodes": self.nodes.values().map(|n| {
                serde_json::json!({
                    "id": n.id,
                    "uri": n.uri,
                    "label": n.label,
                    "language": n.language,
                    "has_embedding": n.embedding.is_some()
                })
            }).collect::<Vec<_>>(),
            "edges": self.edges.iter().map(|e| {
                serde_json::json!({
                    "id": e.id,
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": format!("{:?}", e.relation),
                    "weight": e.weight,
                    "confidence": e.confidence
                })
            }).collect::<Vec<_>>(),
            "hyperedges": self.hyperedges.iter().map(|h| {
                serde_json::json!({
                    "id": h.id,
                    "nodes": h.node_ids,
                    "relation": format!("{:?}", h.relation),
                    "weight": h.weight
                })
            }).collect::<Vec<_>>(),
            "stats": {
                "total_nodes": self.nodes.len(),
                "total_edges": self.edges.len(),
                "total_hyperedges": self.hyperedges.len()
            }
        })
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.hyperedges.clear();
        self.edge_hashes.clear();
        self.node_degrees.clear();
    }

    // Private helper methods

    fn ensure_node(&mut self, concept: &ConceptNode) -> Result<(), GraphBuildError> {
        if self.nodes.contains_key(&concept.id) {
            return Ok(());
        }

        if self.nodes.len() >= self.config.max_nodes {
            return Err(GraphBuildError::CapacityExceeded {
                max: self.config.max_nodes,
            });
        }

        let node = GraphNode {
            id: Uuid::new_v4().to_string(),
            uri: concept.id.clone(),
            label: concept.label.clone().unwrap_or_else(|| concept.term_from_uri()),
            language: concept.language_code().unwrap_or("unknown").to_string(),
            embedding: None,
            properties: Self::extract_node_properties(concept),
        };

        self.nodes.insert(concept.id.clone(), node);
        Ok(())
    }

    fn extract_node_properties(concept: &ConceptNode) -> HashMap<String, GraphProperty> {
        let mut props = HashMap::new();
        if let Some(ref term) = concept.term {
            props.insert("term".to_string(), GraphProperty::String(term.clone()));
        }
        if let Some(ref sense) = concept.sense_label {
            props.insert("sense".to_string(), GraphProperty::String(sense.clone()));
        }
        props
    }

    fn extract_edge_properties(edge: &Edge) -> HashMap<String, GraphProperty> {
        let mut props = HashMap::new();
        if let Some(ref license) = edge.license {
            props.insert("license".to_string(), GraphProperty::String(license.clone()));
        }
        if let Some(ref dataset) = edge.dataset {
            props.insert("dataset".to_string(), GraphProperty::String(dataset.clone()));
        }
        props.insert(
            "source_count".to_string(),
            GraphProperty::Integer(edge.sources.len() as i64),
        );
        props
    }

    fn edge_hash(source: &str, target: &str, relation: &str) -> u64 {
        use xxhash_rust::xxh3::xxh3_64;
        let combined = format!("{}-{}-{}", source, target, relation);
        xxh3_64(combined.as_bytes())
    }

    fn count_components(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut components = 0;

        for node_uri in self.nodes.keys() {
            if !visited.contains(node_uri.as_str()) {
                // BFS from this node
                let mut stack = vec![node_uri.as_str()];
                while let Some(current) = stack.pop() {
                    if visited.insert(current) {
                        for neighbor in self.get_neighbors(current) {
                            if !visited.contains(neighbor.uri.as_str()) {
                                stack.push(&neighbor.uri);
                            }
                        }
                    }
                }
                components += 1;
            }
        }

        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ConceptNode, Edge, Relation, Source};

    fn mock_edge(start: &str, end: &str, rel: &str, weight: f64) -> Edge {
        Edge {
            id: format!("/a/[{},{},{}]", rel, start, end),
            start: ConceptNode {
                id: start.to_string(),
                label: Some(start.split('/').last().unwrap_or(start).to_string()),
                language: Some("en".to_string()),
                term: None,
                sense_label: None,
            },
            end: ConceptNode {
                id: end.to_string(),
                label: Some(end.split('/').last().unwrap_or(end).to_string()),
                language: Some("en".to_string()),
                term: None,
                sense_label: None,
            },
            rel: Relation {
                id: rel.to_string(),
                label: Some(rel.trim_start_matches("/r/").to_string()),
            },
            weight,
            surface_text: None,
            license: None,
            dataset: None,
            sources: vec![],
        }
    }

    #[test]
    fn test_add_edge() {
        let mut builder = ConceptNetGraphBuilder::default_config();
        let edge = mock_edge("/c/en/dog", "/c/en/animal", "/r/IsA", 2.0);

        builder.add_edge(&edge).unwrap();

        assert_eq!(builder.nodes.len(), 2);
        assert_eq!(builder.edges.len(), 1);
    }

    #[test]
    fn test_shortest_path() {
        let mut builder = ConceptNetGraphBuilder::default_config();

        builder.add_edge(&mock_edge("/c/en/dog", "/c/en/animal", "/r/IsA", 2.0)).unwrap();
        builder.add_edge(&mock_edge("/c/en/animal", "/c/en/living_thing", "/r/IsA", 2.0)).unwrap();

        let path = builder.shortest_path("/c/en/dog", "/c/en/living_thing", 5);
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 3);
    }
}
