//! Rate limiter for ConceptNet API compliance
//!
//! ConceptNet allows:
//! - 3600 requests per hour
//! - 120 requests per minute (burst)
//! - /related and /relatedness count as 2 requests

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Token bucket rate limiter with dual limits (burst + sustained)
pub struct RateLimiter {
    minute_window: Mutex<VecDeque<Instant>>,
    hour_window: Mutex<VecDeque<Instant>>,
    minute_limit: usize,
    hour_limit: usize,
}

impl RateLimiter {
    /// Create a new rate limiter with ConceptNet's default limits
    pub fn new() -> Self {
        Self {
            minute_window: Mutex::new(VecDeque::with_capacity(150)),
            hour_window: Mutex::new(VecDeque::with_capacity(4000)),
            minute_limit: 120,
            hour_limit: 3600,
        }
    }

    /// Create with custom limits
    pub fn with_limits(minute_limit: usize, hour_limit: usize) -> Self {
        Self {
            minute_window: Mutex::new(VecDeque::with_capacity(minute_limit + 30)),
            hour_window: Mutex::new(VecDeque::with_capacity(hour_limit + 100)),
            minute_limit,
            hour_limit,
        }
    }

    /// Check if a request can be made, optionally counting as multiple requests
    pub fn check(&self, cost: usize) -> RateLimitResult {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);
        let hour_ago = now - Duration::from_secs(3600);

        let mut minute_window = self.minute_window.lock();
        let mut hour_window = self.hour_window.lock();

        // Clean old entries
        while minute_window.front().map_or(false, |&t| t < minute_ago) {
            minute_window.pop_front();
        }
        while hour_window.front().map_or(false, |&t| t < hour_ago) {
            hour_window.pop_front();
        }

        let minute_remaining = self.minute_limit.saturating_sub(minute_window.len());
        let hour_remaining = self.hour_limit.saturating_sub(hour_window.len());

        if minute_remaining < cost {
            let wait_until = minute_window.front().map_or(now, |&t| t + Duration::from_secs(60));
            return RateLimitResult::Limited {
                wait_duration: wait_until.saturating_duration_since(now),
                reason: RateLimitReason::MinuteLimit,
            };
        }

        if hour_remaining < cost {
            let wait_until = hour_window.front().map_or(now, |&t| t + Duration::from_secs(3600));
            return RateLimitResult::Limited {
                wait_duration: wait_until.saturating_duration_since(now),
                reason: RateLimitReason::HourLimit,
            };
        }

        // Record the requests
        for _ in 0..cost {
            minute_window.push_back(now);
            hour_window.push_back(now);
        }

        RateLimitResult::Allowed {
            minute_remaining: minute_remaining - cost,
            hour_remaining: hour_remaining - cost,
        }
    }

    /// Wait until a request can be made
    pub async fn wait(&self, cost: usize) {
        loop {
            match self.check(cost) {
                RateLimitResult::Allowed { .. } => return,
                RateLimitResult::Limited { wait_duration, .. } => {
                    tokio::time::sleep(wait_duration + Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Get current rate limit status
    pub fn status(&self) -> RateLimitStatus {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);
        let hour_ago = now - Duration::from_secs(3600);

        let minute_window = self.minute_window.lock();
        let hour_window = self.hour_window.lock();

        let minute_used = minute_window.iter().filter(|&&t| t >= minute_ago).count();
        let hour_used = hour_window.iter().filter(|&&t| t >= hour_ago).count();

        RateLimitStatus {
            minute_used,
            minute_limit: self.minute_limit,
            minute_remaining: self.minute_limit.saturating_sub(minute_used),
            hour_used,
            hour_limit: self.hour_limit,
            hour_remaining: self.hour_limit.saturating_sub(hour_used),
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a rate limit check
#[derive(Debug, Clone)]
pub enum RateLimitResult {
    Allowed {
        minute_remaining: usize,
        hour_remaining: usize,
    },
    Limited {
        wait_duration: Duration,
        reason: RateLimitReason,
    },
}

/// Reason for rate limiting
#[derive(Debug, Clone, Copy)]
pub enum RateLimitReason {
    MinuteLimit,
    HourLimit,
}

/// Current rate limit status
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub minute_used: usize,
    pub minute_limit: usize,
    pub minute_remaining: usize,
    pub hour_used: usize,
    pub hour_limit: usize,
    pub hour_remaining: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_allows_requests() {
        let limiter = RateLimiter::with_limits(10, 100);

        for _ in 0..5 {
            match limiter.check(1) {
                RateLimitResult::Allowed { .. } => {}
                RateLimitResult::Limited { .. } => panic!("Should be allowed"),
            }
        }
    }

    #[test]
    fn test_rate_limiter_blocks_excess() {
        let limiter = RateLimiter::with_limits(3, 100);

        // Use all minute quota
        for _ in 0..3 {
            limiter.check(1);
        }

        // Should be blocked
        match limiter.check(1) {
            RateLimitResult::Limited { reason: RateLimitReason::MinuteLimit, .. } => {}
            _ => panic!("Should be limited"),
        }
    }
}
