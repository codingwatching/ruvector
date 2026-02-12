//! Query routing and hybrid search.
//!
//! Provides intelligent query routing that selects the optimal search
//! backend (semantic, keyword, temporal, graph, or hybrid) based on
//! query characteristics.

pub mod hybrid;
pub mod router;

pub use hybrid::HybridSearch;
pub use router::{QueryRoute, QueryRouter};
