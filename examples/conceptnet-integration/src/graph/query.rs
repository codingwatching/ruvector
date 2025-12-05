//! Commonsense Query Engine
//!
//! High-level query interface for commonsense reasoning over the knowledge graph.

use super::builder::{ConceptNetGraphBuilder, GraphEdge, GraphNode};
use crate::api::RelationType;
use std::collections::{HashMap, HashSet, VecDeque};

/// A commonsense query for reasoning over the knowledge graph
#[derive(Debug, Clone)]
pub struct CommonsenseQuery {
    /// Starting concept(s)
    pub start_concepts: Vec<String>,
    /// Target concept(s) or None for exploration
    pub target_concepts: Option<Vec<String>>,
    /// Relation types to traverse
    pub relations: Vec<RelationType>,
    /// Maximum path depth
    pub max_depth: usize,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Maximum results to return
    pub limit: usize,
    /// Query type
    pub query_type: QueryType,
}

/// Type of commonsense query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Find paths between concepts
    PathFinding,
    /// Explore related concepts
    Exploration,
    /// Verify a relationship
    Verification,
    /// Find common ancestors/properties
    CommonGround,
    /// Causal reasoning
    CausalChain,
    /// Analogical reasoning
    Analogy,
}

impl Default for CommonsenseQuery {
    fn default() -> Self {
        Self {
            start_concepts: vec![],
            target_concepts: None,
            relations: vec![],
            max_depth: 3,
            min_confidence: 0.5,
            limit: 10,
            query_type: QueryType::Exploration,
        }
    }
}

impl CommonsenseQuery {
    /// Create a new query
    pub fn new(start: &str) -> Self {
        Self {
            start_concepts: vec![start.to_string()],
            ..Default::default()
        }
    }

    /// Add starting concept
    pub fn from(mut self, concept: &str) -> Self {
        self.start_concepts.push(concept.to_string());
        self
    }

    /// Set target concept
    pub fn to(mut self, concept: &str) -> Self {
        self.target_concepts = Some(vec![concept.to_string()]);
        self
    }

    /// Add relation filter
    pub fn via(mut self, relation: RelationType) -> Self {
        self.relations.push(relation);
        self
    }

    /// Set maximum depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set minimum confidence
    pub fn min_confidence(mut self, conf: f64) -> Self {
        self.min_confidence = conf;
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set query type
    pub fn query_type(mut self, qt: QueryType) -> Self {
        self.query_type = qt;
        self
    }
}

/// A reasoning path through the knowledge graph
#[derive(Debug, Clone)]
pub struct ReasoningPath {
    /// Nodes in the path
    pub nodes: Vec<PathNode>,
    /// Total confidence score
    pub confidence: f64,
    /// Path length
    pub length: usize,
    /// Natural language explanation
    pub explanation: String,
}

/// A node in a reasoning path
#[derive(Debug, Clone)]
pub struct PathNode {
    pub uri: String,
    pub label: String,
    pub relation_to_next: Option<RelationType>,
    pub edge_weight: Option<f64>,
}

/// Result of a commonsense query
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Query that was executed
    pub query: CommonsenseQuery,
    /// Found paths
    pub paths: Vec<ReasoningPath>,
    /// Related concepts discovered
    pub related_concepts: Vec<RelatedConcept>,
    /// Execution statistics
    pub stats: QueryStats,
}

/// A related concept with relevance score
#[derive(Debug, Clone)]
pub struct RelatedConcept {
    pub uri: String,
    pub label: String,
    pub relevance: f64,
    pub relation: RelationType,
    pub distance: usize,
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    pub nodes_visited: usize,
    pub edges_traversed: usize,
    pub paths_found: usize,
    pub execution_time_ms: u64,
}

/// Engine for executing commonsense queries
pub struct CommonsenseQueryEngine<'a> {
    graph: &'a ConceptNetGraphBuilder,
}

impl<'a> CommonsenseQueryEngine<'a> {
    /// Create a new query engine
    pub fn new(graph: &'a ConceptNetGraphBuilder) -> Self {
        Self { graph }
    }

    /// Execute a query
    pub fn execute(&self, query: &CommonsenseQuery) -> QueryResult {
        let start = std::time::Instant::now();
        let mut stats = QueryStats::default();

        let (paths, related) = match query.query_type {
            QueryType::PathFinding => self.find_paths(query, &mut stats),
            QueryType::Exploration => self.explore(query, &mut stats),
            QueryType::Verification => self.verify(query, &mut stats),
            QueryType::CommonGround => self.find_common_ground(query, &mut stats),
            QueryType::CausalChain => self.find_causal_chain(query, &mut stats),
            QueryType::Analogy => self.find_analogy(query, &mut stats),
        };

        stats.execution_time_ms = start.elapsed().as_millis() as u64;
        stats.paths_found = paths.len();

        QueryResult {
            query: query.clone(),
            paths,
            related_concepts: related,
            stats,
        }
    }

    /// Find IsA ancestors (what something is)
    pub fn what_is(&self, concept: &str) -> Vec<RelatedConcept> {
        let query = CommonsenseQuery::new(concept)
            .via(RelationType::IsA)
            .via(RelationType::InstanceOf)
            .max_depth(3)
            .query_type(QueryType::Exploration);

        let mut stats = QueryStats::default();
        let (_, related) = self.explore(&query, &mut stats);
        related
    }

    /// Find properties (what something has)
    pub fn has_properties(&self, concept: &str) -> Vec<RelatedConcept> {
        let query = CommonsenseQuery::new(concept)
            .via(RelationType::HasA)
            .via(RelationType::HasProperty)
            .max_depth(2)
            .query_type(QueryType::Exploration);

        let mut stats = QueryStats::default();
        let (_, related) = self.explore(&query, &mut stats);
        related
    }

    /// Find uses (what something is used for)
    pub fn used_for(&self, concept: &str) -> Vec<RelatedConcept> {
        let query = CommonsenseQuery::new(concept)
            .via(RelationType::UsedFor)
            .max_depth(2)
            .query_type(QueryType::Exploration);

        let mut stats = QueryStats::default();
        let (_, related) = self.explore(&query, &mut stats);
        related
    }

    /// Find capabilities (what something can do)
    pub fn capable_of(&self, concept: &str) -> Vec<RelatedConcept> {
        let query = CommonsenseQuery::new(concept)
            .via(RelationType::CapableOf)
            .max_depth(2)
            .query_type(QueryType::Exploration);

        let mut stats = QueryStats::default();
        let (_, related) = self.explore(&query, &mut stats);
        related
    }

    /// Find locations (where something is found)
    pub fn at_location(&self, concept: &str) -> Vec<RelatedConcept> {
        let query = CommonsenseQuery::new(concept)
            .via(RelationType::AtLocation)
            .via(RelationType::LocatedNear)
            .max_depth(2)
            .query_type(QueryType::Exploration);

        let mut stats = QueryStats::default();
        let (_, related) = self.explore(&query, &mut stats);
        related
    }

    /// Find causes and effects
    pub fn causes(&self, concept: &str) -> Vec<RelatedConcept> {
        let query = CommonsenseQuery::new(concept)
            .via(RelationType::Causes)
            .via(RelationType::HasSubevent)
            .max_depth(2)
            .query_type(QueryType::CausalChain);

        let mut stats = QueryStats::default();
        let (_, related) = self.find_causal_chain(&query, &mut stats);
        related
    }

    // Private methods

    fn find_paths(
        &self,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
    ) -> (Vec<ReasoningPath>, Vec<RelatedConcept>) {
        let mut paths = Vec::new();

        if let Some(ref targets) = query.target_concepts {
            for start in &query.start_concepts {
                for target in targets {
                    if let Some(path) = self.bfs_path(start, target, query, stats) {
                        paths.push(path);
                    }
                }
            }
        }

        paths.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        paths.truncate(query.limit);

        (paths, vec![])
    }

    fn explore(
        &self,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
    ) -> (Vec<ReasoningPath>, Vec<RelatedConcept>) {
        let mut related = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();

        for start in &query.start_concepts {
            self.bfs_explore(start, query, stats, &mut visited, &mut related);
        }

        related.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
        related.truncate(query.limit);

        (vec![], related)
    }

    fn verify(
        &self,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
    ) -> (Vec<ReasoningPath>, Vec<RelatedConcept>) {
        // Verify by finding any path
        let (paths, _) = self.find_paths(query, stats);
        let verified = !paths.is_empty();

        let related = if verified {
            vec![RelatedConcept {
                uri: query.target_concepts.as_ref().unwrap()[0].clone(),
                label: "verified".to_string(),
                relevance: paths.get(0).map(|p| p.confidence).unwrap_or(0.0),
                relation: RelationType::RelatedTo,
                distance: paths.get(0).map(|p| p.length).unwrap_or(0),
            }]
        } else {
            vec![]
        };

        (paths, related)
    }

    fn find_common_ground(
        &self,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
    ) -> (Vec<ReasoningPath>, Vec<RelatedConcept>) {
        let mut ancestors: HashMap<String, HashSet<String>> = HashMap::new();

        // Find ancestors for each starting concept
        for start in &query.start_concepts {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back((start.clone(), 0));

            while let Some((current, depth)) = queue.pop_front() {
                if depth > query.max_depth || visited.contains(&current) {
                    continue;
                }
                visited.insert(current.clone());
                stats.nodes_visited += 1;

                ancestors
                    .entry(start.clone())
                    .or_default()
                    .insert(current.clone());

                // Follow hierarchical relations upward
                for edge in self.graph.get_outgoing_edges(&current) {
                    if edge.relation.is_hierarchical() {
                        queue.push_back((edge.target_id.clone(), depth + 1));
                        stats.edges_traversed += 1;
                    }
                }
            }
        }

        // Find common ancestors
        let mut common: HashSet<String> = ancestors
            .values()
            .next()
            .cloned()
            .unwrap_or_default();

        for ancestor_set in ancestors.values().skip(1) {
            common = common.intersection(ancestor_set).cloned().collect();
        }

        let related: Vec<RelatedConcept> = common
            .into_iter()
            .filter_map(|uri| {
                self.graph.get_node(&uri).map(|node| RelatedConcept {
                    uri: uri.clone(),
                    label: node.label.clone(),
                    relevance: 1.0,
                    relation: RelationType::IsA,
                    distance: 0,
                })
            })
            .take(query.limit)
            .collect();

        (vec![], related)
    }

    fn find_causal_chain(
        &self,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
    ) -> (Vec<ReasoningPath>, Vec<RelatedConcept>) {
        let causal_relations = [
            RelationType::Causes,
            RelationType::HasPrerequisite,
            RelationType::HasSubevent,
            RelationType::HasFirstSubevent,
            RelationType::HasLastSubevent,
            RelationType::MotivatedByGoal,
        ];

        let modified_query = CommonsenseQuery {
            relations: causal_relations.to_vec(),
            ..query.clone()
        };

        self.explore(&modified_query, stats)
    }

    fn find_analogy(
        &self,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
    ) -> (Vec<ReasoningPath>, Vec<RelatedConcept>) {
        // Simple analogy: find concepts with similar relations
        let mut pattern_matches: HashMap<String, f64> = HashMap::new();

        for start in &query.start_concepts {
            let start_edges = self.graph.get_node_edges(start);
            let start_pattern: HashSet<_> = start_edges
                .iter()
                .map(|e| format!("{:?}", e.relation))
                .collect();

            // Find other concepts with similar relation patterns
            for node in self.graph.nodes() {
                if query.start_concepts.contains(&node.uri) {
                    continue;
                }

                let node_edges = self.graph.get_node_edges(&node.uri);
                let node_pattern: HashSet<_> = node_edges
                    .iter()
                    .map(|e| format!("{:?}", e.relation))
                    .collect();

                let intersection = start_pattern.intersection(&node_pattern).count();
                let union = start_pattern.union(&node_pattern).count();
                let jaccard = if union > 0 {
                    intersection as f64 / union as f64
                } else {
                    0.0
                };

                if jaccard > 0.3 {
                    *pattern_matches.entry(node.uri.clone()).or_insert(0.0) += jaccard;
                }

                stats.nodes_visited += 1;
            }
        }

        let related: Vec<RelatedConcept> = pattern_matches
            .into_iter()
            .filter_map(|(uri, relevance)| {
                self.graph.get_node(&uri).map(|node| RelatedConcept {
                    uri,
                    label: node.label.clone(),
                    relevance,
                    relation: RelationType::SimilarTo,
                    distance: 0,
                })
            })
            .take(query.limit)
            .collect();

        (vec![], related)
    }

    fn bfs_path(
        &self,
        start: &str,
        target: &str,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
    ) -> Option<ReasoningPath> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<(String, Option<RelationType>, f64)>)> = VecDeque::new();

        queue.push_back((start.to_string(), vec![(start.to_string(), None, 1.0)]));

        while let Some((current, path)) = queue.pop_front() {
            if path.len() > query.max_depth + 1 {
                continue;
            }

            if current == target {
                return Some(self.path_to_reasoning_path(path));
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            stats.nodes_visited += 1;

            for edge in self.graph.get_node_edges(&current) {
                let next = if edge.source_id == current {
                    &edge.target_id
                } else {
                    &edge.source_id
                };

                if visited.contains(next) {
                    continue;
                }

                // Check relation filter
                if !query.relations.is_empty() && !query.relations.contains(&edge.relation) {
                    continue;
                }

                // Check confidence
                if edge.confidence < query.min_confidence {
                    continue;
                }

                let mut new_path = path.clone();
                new_path.push((next.clone(), Some(edge.relation), edge.weight));
                queue.push_back((next.clone(), new_path));
                stats.edges_traversed += 1;
            }
        }

        None
    }

    fn bfs_explore(
        &self,
        start: &str,
        query: &CommonsenseQuery,
        stats: &mut QueryStats,
        visited: &mut HashSet<String>,
        related: &mut Vec<RelatedConcept>,
    ) {
        let mut queue: VecDeque<(String, usize, f64)> = VecDeque::new();
        queue.push_back((start.to_string(), 0, 1.0));

        while let Some((current, depth, relevance)) = queue.pop_front() {
            if depth > query.max_depth || visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            stats.nodes_visited += 1;

            for edge in self.graph.get_node_edges(&current) {
                let next = if edge.source_id == current {
                    &edge.target_id
                } else {
                    &edge.source_id
                };

                if visited.contains(next) {
                    continue;
                }

                if !query.relations.is_empty() && !query.relations.contains(&edge.relation) {
                    continue;
                }

                if edge.confidence < query.min_confidence {
                    continue;
                }

                let new_relevance = relevance * edge.confidence * 0.8;

                if let Some(node) = self.graph.get_node(next) {
                    related.push(RelatedConcept {
                        uri: next.clone(),
                        label: node.label.clone(),
                        relevance: new_relevance,
                        relation: edge.relation,
                        distance: depth + 1,
                    });
                }

                queue.push_back((next.clone(), depth + 1, new_relevance));
                stats.edges_traversed += 1;
            }
        }
    }

    fn path_to_reasoning_path(
        &self,
        path: Vec<(String, Option<RelationType>, f64)>,
    ) -> ReasoningPath {
        let mut confidence = 1.0;
        let mut nodes = Vec::new();
        let mut explanation_parts = Vec::new();

        for (i, (uri, rel, weight)) in path.iter().enumerate() {
            let label = self
                .graph
                .get_node(uri)
                .map(|n| n.label.clone())
                .unwrap_or_else(|| uri.split('/').last().unwrap_or(uri).to_string());

            nodes.push(PathNode {
                uri: uri.clone(),
                label: label.clone(),
                relation_to_next: if i < path.len() - 1 { path[i + 1].1 } else { None },
                edge_weight: if i > 0 { Some(*weight) } else { None },
            });

            if i > 0 {
                confidence *= weight / 10.0; // Normalize weight
                if let Some(rel) = rel {
                    explanation_parts.push(format!("{:?} {}", rel, label));
                }
            } else {
                explanation_parts.push(label);
            }
        }

        ReasoningPath {
            nodes,
            confidence: confidence.min(1.0),
            length: path.len() - 1,
            explanation: explanation_parts.join(" → "),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder() {
        let query = CommonsenseQuery::new("/c/en/dog")
            .via(RelationType::IsA)
            .to("/c/en/animal")
            .max_depth(3)
            .min_confidence(0.7);

        assert_eq!(query.start_concepts, vec!["/c/en/dog"]);
        assert_eq!(query.target_concepts, Some(vec!["/c/en/animal".to_string()]));
        assert_eq!(query.max_depth, 3);
        assert_eq!(query.min_confidence, 0.7);
    }
}
