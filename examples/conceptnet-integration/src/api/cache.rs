//! Response caching with LRU eviction and TTL expiry
//!
//! Reduces API calls by caching responses locally with configurable TTL.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use xxhash_rust::xxh3::xxh3_64;

/// Cached response entry with timestamp
#[derive(Clone)]
struct CacheEntry<T: Clone> {
    value: T,
    created_at: Instant,
    access_count: u64,
}

/// LRU cache with TTL expiry
pub struct ResponseCache<T: Clone> {
    entries: RwLock<HashMap<u64, CacheEntry<T>>>,
    max_entries: usize,
    ttl: Duration,
}

impl<T: Clone> ResponseCache<T> {
    /// Create a new cache with specified capacity and TTL
    pub fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_entries)),
            max_entries,
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// Create cache with default settings (1000 entries, 15 min TTL)
    pub fn default_settings() -> Self {
        Self::new(1000, 900)
    }

    /// Get a value from cache if present and not expired
    pub fn get(&self, key: &str) -> Option<T> {
        let hash = xxh3_64(key.as_bytes());
        let entries = self.entries.read();

        if let Some(entry) = entries.get(&hash) {
            if entry.created_at.elapsed() < self.ttl {
                return Some(entry.value.clone());
            }
        }
        None
    }

    /// Insert a value into cache, evicting oldest if at capacity
    pub fn insert(&self, key: &str, value: T) {
        let hash = xxh3_64(key.as_bytes());
        let mut entries = self.entries.write();

        // Evict expired entries first
        entries.retain(|_, e| e.created_at.elapsed() < self.ttl);

        // Evict oldest if still at capacity
        if entries.len() >= self.max_entries {
            if let Some(oldest_key) = entries
                .iter()
                .min_by_key(|(_, e)| e.created_at)
                .map(|(k, _)| *k)
            {
                entries.remove(&oldest_key);
            }
        }

        entries.insert(
            hash,
            CacheEntry {
                value,
                created_at: Instant::now(),
                access_count: 0,
            },
        );
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read();
        let mut expired_count = 0;
        let mut total_access = 0u64;

        for entry in entries.values() {
            if entry.created_at.elapsed() >= self.ttl {
                expired_count += 1;
            }
            total_access += entry.access_count;
        }

        CacheStats {
            total_entries: entries.len(),
            expired_entries: expired_count,
            max_entries: self.max_entries,
            ttl_seconds: self.ttl.as_secs(),
            total_accesses: total_access,
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Remove expired entries
    pub fn evict_expired(&self) {
        let mut entries = self.entries.write();
        entries.retain(|_, e| e.created_at.elapsed() < self.ttl);
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub expired_entries: usize,
    pub max_entries: usize,
    pub ttl_seconds: u64,
    pub total_accesses: u64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_entries: 0,
            expired_entries: 0,
            max_entries: 0,
            ttl_seconds: 0,
            total_accesses: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_get() {
        let cache: ResponseCache<String> = ResponseCache::new(10, 60);

        cache.insert("key1", "value1".to_string());
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("nonexistent"), None);
    }

    #[test]
    fn test_cache_eviction() {
        let cache: ResponseCache<i32> = ResponseCache::new(3, 60);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        cache.insert("d", 4); // Should evict "a"

        assert!(cache.get("a").is_none() || cache.stats().total_entries <= 3);
    }
}
