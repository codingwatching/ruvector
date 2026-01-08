//! Axum middleware layers for security
//!
//! Provides ready-to-use Tower layers for authentication and rate limiting.

use crate::{
    auth::{AuthMiddleware, AuthMode},
    error::SecurityError,
    rate_limit::{OperationType, RateLimiter},
};
use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::net::SocketAddr;
use std::sync::Arc;

/// Security state for middleware
#[derive(Clone)]
pub struct SecurityState {
    /// Authentication middleware
    pub auth: AuthMiddleware,
    /// Rate limiter
    pub rate_limiter: RateLimiter,
}

impl SecurityState {
    /// Create new security state
    pub fn new(auth: AuthMiddleware, rate_limiter: RateLimiter) -> Self {
        Self { auth, rate_limiter }
    }

    /// Create development security state (no auth, disabled rate limiting)
    pub fn development() -> Self {
        Self {
            auth: AuthMiddleware::none(),
            rate_limiter: RateLimiter::disabled(),
        }
    }

    /// Create production security state
    pub fn production(token: &str) -> Self {
        Self {
            auth: AuthMiddleware::bearer(token),
            rate_limiter: RateLimiter::default(),
        }
    }
}

impl Default for SecurityState {
    fn default() -> Self {
        Self::development()
    }
}

/// Authentication middleware layer for axum
///
/// Checks the Authorization header for a valid bearer token.
///
/// # Example
///
/// ```rust,ignore
/// use axum::{Router, routing::get, middleware};
/// use ruvector_security::middleware::{auth_layer, SecurityState};
///
/// let security = SecurityState::production("my_secret_token");
/// let app = Router::new()
///     .route("/api", get(|| async { "protected" }))
///     .layer(middleware::from_fn_with_state(security, auth_layer));
/// ```
pub async fn auth_layer(
    State(security): State<SecurityState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    // Check if localhost bypass is allowed
    let remote_addr = addr.to_string();
    if security.auth.is_localhost_allowed(&remote_addr) {
        return next.run(request).await;
    }

    // Skip auth if mode is None
    if *security.auth.mode() == AuthMode::None {
        return next.run(request).await;
    }

    // Get authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    // Validate token
    match security.auth.validate_header(auth_header) {
        Ok(()) => next.run(request).await,
        Err(e) => {
            let (status, message) = match e {
                SecurityError::AuthenticationRequired => {
                    (StatusCode::UNAUTHORIZED, "Authentication required")
                }
                SecurityError::InvalidToken => (StatusCode::UNAUTHORIZED, "Invalid token"),
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "Authentication error"),
            };
            (status, message).into_response()
        }
    }
}

/// Rate limiting middleware layer for axum
///
/// Applies rate limiting based on operation type and client IP.
///
/// # Example
///
/// ```rust,ignore
/// use axum::{Router, routing::get, middleware};
/// use ruvector_security::middleware::{rate_limit_layer, SecurityState};
///
/// let security = SecurityState::default();
/// let app = Router::new()
///     .route("/api", get(|| async { "limited" }))
///     .layer(middleware::from_fn_with_state(security, rate_limit_layer));
/// ```
pub async fn rate_limit_layer(
    State(security): State<SecurityState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let ip = addr.ip().to_string();

    // Determine operation type from HTTP method
    let op_type = match *request.method() {
        axum::http::Method::GET | axum::http::Method::HEAD => OperationType::Read,
        axum::http::Method::POST | axum::http::Method::PUT | axum::http::Method::DELETE => {
            OperationType::Write
        }
        _ => OperationType::Read,
    };

    // Check rate limit
    match security.rate_limiter.check(op_type, Some(&ip)).await {
        Ok(()) => next.run(request).await,
        Err(SecurityError::RateLimitExceeded { retry_after_secs }) => {
            let mut response = (
                StatusCode::TOO_MANY_REQUESTS,
                format!("Rate limit exceeded. Retry after {} seconds.", retry_after_secs),
            )
                .into_response();

            // Add Retry-After header
            response.headers_mut().insert(
                "Retry-After",
                retry_after_secs.to_string().parse().unwrap(),
            );

            response
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Rate limiting error").into_response(),
    }
}

/// Combined security middleware layer (auth + rate limiting)
///
/// Applies both authentication and rate limiting in a single middleware.
pub async fn security_layer(
    State(security): State<SecurityState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let remote_addr = addr.to_string();
    let ip = addr.ip().to_string();

    // Skip all security for localhost in development mode
    let is_localhost = security.auth.is_localhost_allowed(&remote_addr);
    let is_no_auth = *security.auth.mode() == AuthMode::None;

    if !is_localhost && !is_no_auth {
        // Check authentication first
        let auth_header = request
            .headers()
            .get(AUTHORIZATION)
            .and_then(|h| h.to_str().ok());

        if let Err(e) = security.auth.validate_header(auth_header) {
            let (status, message) = match e {
                SecurityError::AuthenticationRequired => {
                    (StatusCode::UNAUTHORIZED, "Authentication required")
                }
                SecurityError::InvalidToken => (StatusCode::UNAUTHORIZED, "Invalid token"),
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "Authentication error"),
            };
            return (status, message).into_response();
        }
    }

    // Check rate limit (always, even for localhost)
    let op_type = match *request.method() {
        axum::http::Method::GET | axum::http::Method::HEAD => OperationType::Read,
        axum::http::Method::POST | axum::http::Method::PUT | axum::http::Method::DELETE => {
            OperationType::Write
        }
        _ => OperationType::Read,
    };

    if let Err(SecurityError::RateLimitExceeded { retry_after_secs }) =
        security.rate_limiter.check(op_type, Some(&ip)).await
    {
        let mut response = (
            StatusCode::TOO_MANY_REQUESTS,
            format!("Rate limit exceeded. Retry after {} seconds.", retry_after_secs),
        )
            .into_response();

        response.headers_mut().insert(
            "Retry-After",
            retry_after_secs.to_string().parse().unwrap(),
        );

        return response;
    }

    next.run(request).await
}

/// Rate limit headers extractor
///
/// Adds X-RateLimit-* headers to responses
pub struct RateLimitHeaders {
    pub limit: u32,
    pub remaining: u32,
    pub reset: u64,
}

impl RateLimitHeaders {
    /// Apply rate limit headers to a response
    pub fn apply_to_response(&self, mut response: Response) -> Response {
        let headers = response.headers_mut();
        headers.insert("X-RateLimit-Limit", self.limit.into());
        headers.insert("X-RateLimit-Remaining", self.remaining.into());
        headers.insert("X-RateLimit-Reset", self.reset.into());
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limit::RateLimitConfig;

    #[test]
    fn test_security_state_development() {
        let state = SecurityState::development();
        assert_eq!(*state.auth.mode(), AuthMode::None);
    }

    #[test]
    fn test_security_state_production() {
        let state = SecurityState::production("secret_token");
        assert_eq!(*state.auth.mode(), AuthMode::Bearer);
    }
}
