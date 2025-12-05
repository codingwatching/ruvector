//! ConceptNet API Client Implementation
//!
//! High-performance async client with caching, rate limiting, and retry logic.

use super::cache::ResponseCache;
use super::rate_limiter::{RateLimitResult, RateLimiter};
use super::types::*;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// ConceptNet API errors
#[derive(Error, Debug)]
pub enum ConceptNetError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Rate limited: wait {0:?}")]
    RateLimited(Duration),

    #[error("API returned error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("Invalid URI: {0}")]
    InvalidUri(String),

    #[error("Deserialization error: {0}")]
    DeserializeError(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// Configuration for the ConceptNet client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub base_url: String,
    pub timeout: Duration,
    pub max_retries: usize,
    pub cache_enabled: bool,
    pub cache_ttl_seconds: u64,
    pub cache_max_entries: usize,
    pub rate_limit_enabled: bool,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://api.conceptnet.io".to_string(),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            cache_enabled: true,
            cache_ttl_seconds: 900, // 15 minutes
            cache_max_entries: 1000,
            rate_limit_enabled: true,
        }
    }
}

/// High-performance ConceptNet API client
pub struct ConceptNetClient {
    client: reqwest::Client,
    config: ClientConfig,
    cache: Option<Arc<ResponseCache<String>>>,
    rate_limiter: Option<Arc<RateLimiter>>,
}

impl ConceptNetClient {
    /// Create a new client with default configuration
    pub fn new() -> Self {
        Self::with_config(ClientConfig::default())
    }

    /// Create a new client with custom configuration
    pub fn with_config(config: ClientConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .gzip(true)
            .pool_max_idle_per_host(10)
            .build()
            .expect("Failed to create HTTP client");

        let cache = if config.cache_enabled {
            Some(Arc::new(ResponseCache::new(
                config.cache_max_entries,
                config.cache_ttl_seconds,
            )))
        } else {
            None
        };

        let rate_limiter = if config.rate_limit_enabled {
            Some(Arc::new(RateLimiter::new()))
        } else {
            None
        };

        Self {
            client,
            config,
            cache,
            rate_limiter,
        }
    }

    /// Build a concept URI from language and term
    pub fn concept_uri(language: &str, term: &str) -> String {
        let normalized = term.to_lowercase().replace(' ', "_");
        format!("/c/{}/{}", language, normalized)
    }

    /// Build a relation URI
    pub fn relation_uri(relation: &str) -> String {
        format!("/r/{}", relation)
    }

    /// Lookup a concept node and its edges
    pub async fn lookup(&self, uri: &str) -> Result<LookupResponse, ConceptNetError> {
        self.fetch_with_cache(&format!("{}{}", self.config.base_url, uri), 1)
            .await
    }

    /// Lookup with pagination
    pub async fn lookup_page(&self, uri: &str, limit: usize, offset: usize) -> Result<LookupResponse, ConceptNetError> {
        let url = format!(
            "{}{}?limit={}&offset={}",
            self.config.base_url, uri, limit, offset
        );
        self.fetch_with_cache(&url, 1).await
    }

    /// Query edges with parameters
    pub async fn query(&self, params: QueryParams) -> Result<QueryResponse, ConceptNetError> {
        let url = format!(
            "{}/query?{}",
            self.config.base_url,
            params.to_query_string()
        );
        self.fetch_with_cache(&url, 1).await
    }

    /// Get related terms using Numberbatch embeddings
    pub async fn related(
        &self,
        uri: &str,
        filter_language: Option<&str>,
    ) -> Result<RelatedResponse, ConceptNetError> {
        let mut url = format!("{}/related{}", self.config.base_url, uri);
        if let Some(lang) = filter_language {
            url = format!("{}?filter=/c/{}", url, lang);
        }
        // /related counts as 2 requests
        self.fetch_with_cache(&url, 2).await
    }

    /// Get relatedness score between two concepts
    pub async fn relatedness(
        &self,
        uri1: &str,
        uri2: &str,
    ) -> Result<RelatednessResponse, ConceptNetError> {
        let url = format!(
            "{}/relatedness?node1={}&node2={}",
            self.config.base_url, uri1, uri2
        );
        // /relatedness counts as 2 requests
        self.fetch_with_cache(&url, 2).await
    }

    /// Resolve natural language to URI
    pub async fn resolve_uri(
        &self,
        text: &str,
        language: &str,
    ) -> Result<String, ConceptNetError> {
        let url = format!(
            "{}/uri?text={}&language={}",
            self.config.base_url,
            text.replace(' ', "%20"),
            language
        );
        let response: serde_json::Value = self.fetch_with_cache(&url, 1).await?;
        response
            .get("@id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| ConceptNetError::DeserializeError("Missing @id in response".into()))
    }

    /// Get all edges for a concept (paginated fetch)
    pub async fn get_all_edges(&self, uri: &str, max_edges: usize) -> Result<Vec<Edge>, ConceptNetError> {
        let mut all_edges = Vec::new();
        let page_size = 100;
        let mut offset = 0;

        while all_edges.len() < max_edges {
            let response = self.lookup_page(uri, page_size, offset).await?;
            let edges_count = response.edges.len();
            all_edges.extend(response.edges);

            if edges_count < page_size {
                break;
            }
            offset += page_size;
        }

        all_edges.truncate(max_edges);
        Ok(all_edges)
    }

    /// Get edges filtered by relation type
    pub async fn get_edges_by_relation(
        &self,
        uri: &str,
        relation: RelationType,
    ) -> Result<Vec<Edge>, ConceptNetError> {
        let rel_name = format!("{:?}", relation);
        let params = QueryParams::new()
            .with_node(uri)
            .with_rel(&format!("/r/{}", rel_name));
        let response = self.query(params).await?;
        Ok(response.edges)
    }

    /// Get concept statistics
    pub async fn concept_stats(&self, uri: &str) -> Result<ConceptStats, ConceptNetError> {
        let response = self.lookup(uri).await?;
        Ok(ConceptStats::from_edges(&response.edges, uri))
    }

    /// Batch lookup multiple concepts
    pub async fn batch_lookup(&self, uris: &[&str]) -> Vec<Result<LookupResponse, ConceptNetError>> {
        let futures: Vec<_> = uris.iter().map(|uri| self.lookup(uri)).collect();
        futures::future::join_all(futures).await
    }

    /// Internal fetch with caching and rate limiting
    async fn fetch_with_cache<T: serde::de::DeserializeOwned + serde::Serialize + Clone>(
        &self,
        url: &str,
        cost: usize,
    ) -> Result<T, ConceptNetError> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(url) {
                return serde_json::from_str(&cached)
                    .map_err(|e| ConceptNetError::DeserializeError(e.to_string()));
            }
        }

        // Rate limit check
        if let Some(ref limiter) = self.rate_limiter {
            match limiter.check(cost) {
                RateLimitResult::Allowed { .. } => {}
                RateLimitResult::Limited { wait_duration, .. } => {
                    return Err(ConceptNetError::RateLimited(wait_duration));
                }
            }
        }

        // Fetch with retry
        let mut last_error = None;
        for attempt in 0..self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(100 * 2u64.pow(attempt as u32));
                tokio::time::sleep(delay).await;
            }

            match self.client.get(url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        let text = response.text().await?;

                        // Cache the response
                        if let Some(ref cache) = self.cache {
                            cache.insert(url, text.clone());
                        }

                        return serde_json::from_str(&text)
                            .map_err(|e| ConceptNetError::DeserializeError(e.to_string()));
                    } else {
                        let status = response.status().as_u16();
                        let message = response.text().await.unwrap_or_default();
                        last_error = Some(ConceptNetError::ApiError { status, message });
                    }
                }
                Err(e) => {
                    last_error = Some(ConceptNetError::HttpError(e));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ConceptNetError::ApiError {
                status: 0,
                message: "Unknown error after retries".into(),
            }
        }))
    }

    /// Get rate limit status
    pub fn rate_limit_status(&self) -> Option<super::rate_limiter::RateLimitStatus> {
        self.rate_limiter.as_ref().map(|l| l.status())
    }

    /// Clear the response cache
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear();
        }
    }
}

impl Default for ConceptNetClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_uri() {
        assert_eq!(
            ConceptNetClient::concept_uri("en", "artificial intelligence"),
            "/c/en/artificial_intelligence"
        );
        assert_eq!(
            ConceptNetClient::concept_uri("es", "perro"),
            "/c/es/perro"
        );
    }

    #[test]
    fn test_query_params() {
        let params = QueryParams::new()
            .with_node("/c/en/dog")
            .with_rel("/r/IsA")
            .with_limit(10);

        let qs = params.to_query_string();
        assert!(qs.contains("node="));
        assert!(qs.contains("rel="));
        assert!(qs.contains("limit=10"));
    }
}
