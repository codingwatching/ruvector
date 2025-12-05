//! ConceptNet API Types
//!
//! Strongly-typed representations of ConceptNet's JSON-LD responses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A ConceptNet node (concept)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    #[serde(rename = "@id")]
    pub id: String,
    pub label: Option<String>,
    pub language: Option<String>,
    pub term: Option<String>,
    pub sense_label: Option<String>,
}

impl ConceptNode {
    /// Extract the term from the URI (e.g., "/c/en/dog" -> "dog")
    pub fn term_from_uri(&self) -> String {
        self.id
            .split('/')
            .nth(3)
            .unwrap_or(&self.id)
            .replace('_', " ")
    }

    /// Get the language code (e.g., "en", "es", "ja")
    pub fn language_code(&self) -> Option<&str> {
        self.id.split('/').nth(2)
    }
}

/// A ConceptNet relation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    #[serde(rename = "@id")]
    pub id: String,
    pub label: Option<String>,
}

impl Relation {
    /// Get the relation name (e.g., "/r/IsA" -> "IsA")
    pub fn name(&self) -> &str {
        self.id.trim_start_matches("/r/")
    }

    /// Check if this is a symmetric relation
    pub fn is_symmetric(&self) -> bool {
        matches!(
            self.name(),
            "RelatedTo" | "SimilarTo" | "Synonym" | "Antonym" | "DistinctFrom"
        )
    }
}

/// ConceptNet relation types for commonsense reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum RelationType {
    // Taxonomic relations
    IsA,
    InstanceOf,
    DefinedAs,
    MannerOf,

    // Part-whole relations
    PartOf,
    HasA,
    MadeOf,

    // Spatial relations
    AtLocation,
    LocatedNear,

    // Causal relations
    Causes,
    CausesDesire,
    HasFirstSubevent,
    HasLastSubevent,
    HasPrerequisite,
    HasSubevent,

    // Goal/desire relations
    MotivatedByGoal,
    ObstructedBy,
    UsedFor,
    CapableOf,

    // Property relations
    HasProperty,
    SymbolOf,
    ReceivesAction,

    // Lexical relations
    RelatedTo,
    SimilarTo,
    Synonym,
    Antonym,
    DistinctFrom,
    DerivedFrom,
    FormOf,
    EtymologicallyRelatedTo,
    EtymologicallyDerivedFrom,

    // External links
    ExternalURL,
    DbPedia,

    // Catch-all
    #[serde(other)]
    Other,
}

impl RelationType {
    /// Get the semantic weight for reasoning
    pub fn reasoning_weight(&self) -> f32 {
        match self {
            RelationType::IsA => 0.95,
            RelationType::InstanceOf => 0.90,
            RelationType::PartOf => 0.85,
            RelationType::HasA => 0.80,
            RelationType::Causes => 0.90,
            RelationType::HasPrerequisite => 0.88,
            RelationType::UsedFor => 0.82,
            RelationType::CapableOf => 0.78,
            RelationType::HasProperty => 0.75,
            RelationType::RelatedTo => 0.50,
            RelationType::SimilarTo => 0.70,
            RelationType::Synonym => 0.95,
            RelationType::Antonym => 0.85,
            _ => 0.50,
        }
    }

    /// Check if this relation implies inheritance
    pub fn is_hierarchical(&self) -> bool {
        matches!(
            self,
            RelationType::IsA | RelationType::InstanceOf | RelationType::PartOf | RelationType::MannerOf
        )
    }
}

impl From<&str> for RelationType {
    fn from(s: &str) -> Self {
        let name = s.trim_start_matches("/r/");
        match name {
            "IsA" => RelationType::IsA,
            "InstanceOf" => RelationType::InstanceOf,
            "DefinedAs" => RelationType::DefinedAs,
            "MannerOf" => RelationType::MannerOf,
            "PartOf" => RelationType::PartOf,
            "HasA" => RelationType::HasA,
            "MadeOf" => RelationType::MadeOf,
            "AtLocation" => RelationType::AtLocation,
            "LocatedNear" => RelationType::LocatedNear,
            "Causes" => RelationType::Causes,
            "CausesDesire" => RelationType::CausesDesire,
            "HasFirstSubevent" => RelationType::HasFirstSubevent,
            "HasLastSubevent" => RelationType::HasLastSubevent,
            "HasPrerequisite" => RelationType::HasPrerequisite,
            "HasSubevent" => RelationType::HasSubevent,
            "MotivatedByGoal" => RelationType::MotivatedByGoal,
            "ObstructedBy" => RelationType::ObstructedBy,
            "UsedFor" => RelationType::UsedFor,
            "CapableOf" => RelationType::CapableOf,
            "HasProperty" => RelationType::HasProperty,
            "SymbolOf" => RelationType::SymbolOf,
            "ReceivesAction" => RelationType::ReceivesAction,
            "RelatedTo" => RelationType::RelatedTo,
            "SimilarTo" => RelationType::SimilarTo,
            "Synonym" => RelationType::Synonym,
            "Antonym" => RelationType::Antonym,
            "DistinctFrom" => RelationType::DistinctFrom,
            "DerivedFrom" => RelationType::DerivedFrom,
            "FormOf" => RelationType::FormOf,
            "EtymologicallyRelatedTo" => RelationType::EtymologicallyRelatedTo,
            "EtymologicallyDerivedFrom" => RelationType::EtymologicallyDerivedFrom,
            "ExternalURL" => RelationType::ExternalURL,
            "dbpedia" => RelationType::DbPedia,
            _ => RelationType::Other,
        }
    }
}

/// A ConceptNet edge (assertion)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    #[serde(rename = "@id")]
    pub id: String,
    pub start: ConceptNode,
    pub end: ConceptNode,
    pub rel: Relation,
    pub weight: f64,
    #[serde(rename = "surfaceText")]
    pub surface_text: Option<String>,
    pub license: Option<String>,
    pub dataset: Option<String>,
    #[serde(default)]
    pub sources: Vec<Source>,
}

impl Edge {
    /// Get the relation type enum
    pub fn relation_type(&self) -> RelationType {
        RelationType::from(self.rel.id.as_str())
    }

    /// Compute a confidence score combining weight and source quality
    pub fn confidence(&self) -> f64 {
        let base_weight = self.weight.min(10.0) / 10.0;
        let source_bonus = (self.sources.len() as f64 * 0.1).min(0.3);
        (base_weight + source_bonus).min(1.0)
    }

    /// Get a human-readable description
    pub fn description(&self) -> String {
        if let Some(ref text) = self.surface_text {
            text.clone()
        } else {
            format!(
                "{} {} {}",
                self.start.term_from_uri(),
                self.rel.name(),
                self.end.term_from_uri()
            )
        }
    }
}

/// Source attribution for an edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    #[serde(rename = "@id")]
    pub id: Option<String>,
    pub contributor: Option<String>,
    pub activity: Option<String>,
    pub process: Option<String>,
}

/// Pagination view for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct View {
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(rename = "firstPage")]
    pub first_page: Option<String>,
    #[serde(rename = "nextPage")]
    pub next_page: Option<String>,
    #[serde(rename = "previousPage")]
    pub previous_page: Option<String>,
}

/// Response from node lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupResponse {
    #[serde(rename = "@context")]
    pub context: Option<Vec<String>>,
    #[serde(rename = "@id")]
    pub id: String,
    pub label: Option<String>,
    #[serde(default)]
    pub edges: Vec<Edge>,
    pub view: Option<View>,
}

/// Response from search/query endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    #[serde(rename = "@context")]
    pub context: Option<Vec<String>>,
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(default)]
    pub edges: Vec<Edge>,
    pub view: Option<View>,
}

/// Related term with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedTerm {
    #[serde(rename = "@id")]
    pub id: String,
    pub weight: f64,
}

/// Response from /related endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedResponse {
    #[serde(rename = "@context")]
    pub context: Option<Vec<String>>,
    #[serde(rename = "@id")]
    pub id: String,
    #[serde(default)]
    pub related: Vec<RelatedTerm>,
}

/// Response from /relatedness endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatednessResponse {
    #[serde(rename = "@context")]
    pub context: Option<Vec<String>>,
    #[serde(rename = "@id")]
    pub id: String,
    pub value: f64,
}

/// Query parameters for search
#[derive(Debug, Clone, Default)]
pub struct QueryParams {
    pub start: Option<String>,
    pub end: Option<String>,
    pub rel: Option<String>,
    pub node: Option<String>,
    pub other: Option<String>,
    pub sources: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

impl QueryParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_node(mut self, uri: &str) -> Self {
        self.node = Some(uri.to_string());
        self
    }

    pub fn with_rel(mut self, rel: &str) -> Self {
        self.rel = Some(rel.to_string());
        self
    }

    pub fn with_start(mut self, uri: &str) -> Self {
        self.start = Some(uri.to_string());
        self
    }

    pub fn with_end(mut self, uri: &str) -> Self {
        self.end = Some(uri.to_string());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn to_query_string(&self) -> String {
        let mut params = Vec::new();
        if let Some(ref s) = self.start {
            params.push(format!("start={}", urlencoding::encode(s)));
        }
        if let Some(ref e) = self.end {
            params.push(format!("end={}", urlencoding::encode(e)));
        }
        if let Some(ref r) = self.rel {
            params.push(format!("rel={}", urlencoding::encode(r)));
        }
        if let Some(ref n) = self.node {
            params.push(format!("node={}", urlencoding::encode(n)));
        }
        if let Some(ref o) = self.other {
            params.push(format!("other={}", urlencoding::encode(o)));
        }
        if let Some(ref s) = self.sources {
            params.push(format!("sources={}", urlencoding::encode(s)));
        }
        if let Some(l) = self.limit {
            params.push(format!("limit={}", l));
        }
        if let Some(o) = self.offset {
            params.push(format!("offset={}", o));
        }
        params.join("&")
    }
}

/// URL encoding helper
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                ' ' => "%20".to_string(),
                '/' => "%2F".to_string(),
                '?' => "%3F".to_string(),
                '&' => "%26".to_string(),
                '=' => "%3D".to_string(),
                '#' => "%23".to_string(),
                _ if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~' => {
                    c.to_string()
                }
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}

/// Statistics about a concept
#[derive(Debug, Clone, Default)]
pub struct ConceptStats {
    pub total_edges: usize,
    pub incoming_edges: usize,
    pub outgoing_edges: usize,
    pub unique_relations: usize,
    pub unique_connected_concepts: usize,
    pub languages: Vec<String>,
    pub avg_edge_weight: f64,
}

impl ConceptStats {
    pub fn from_edges(edges: &[Edge], concept_uri: &str) -> Self {
        use std::collections::HashSet;

        let mut incoming = 0;
        let mut outgoing = 0;
        let mut relations = HashSet::new();
        let mut concepts = HashSet::new();
        let mut languages = HashSet::new();
        let mut total_weight = 0.0;

        for edge in edges {
            if edge.start.id == concept_uri {
                outgoing += 1;
                concepts.insert(edge.end.id.clone());
                if let Some(lang) = edge.end.language_code() {
                    languages.insert(lang.to_string());
                }
            } else {
                incoming += 1;
                concepts.insert(edge.start.id.clone());
                if let Some(lang) = edge.start.language_code() {
                    languages.insert(lang.to_string());
                }
            }
            relations.insert(edge.rel.id.clone());
            total_weight += edge.weight;
        }

        Self {
            total_edges: edges.len(),
            incoming_edges: incoming,
            outgoing_edges: outgoing,
            unique_relations: relations.len(),
            unique_connected_concepts: concepts.len(),
            languages: languages.into_iter().collect(),
            avg_edge_weight: if edges.is_empty() {
                0.0
            } else {
                total_weight / edges.len() as f64
            },
        }
    }
}
