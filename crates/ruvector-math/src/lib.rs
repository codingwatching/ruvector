//! # RuVector Math
//!
//! Advanced mathematics for next-generation vector search, featuring:
//!
//! - **Optimal Transport**: Wasserstein distances, Sinkhorn algorithm, Sliced Wasserstein
//! - **Information Geometry**: Fisher Information, Natural Gradient, K-FAC
//! - **Product Manifolds**: Mixed-curvature spaces (Euclidean × Hyperbolic × Spherical)
//! - **Spherical Geometry**: Geodesics on the n-sphere for cyclical patterns
//!
//! ## Design Principles
//!
//! 1. **Pure Rust**: No BLAS/LAPACK dependencies for full WASM compatibility
//! 2. **SIMD-Ready**: Hot paths optimized for auto-vectorization
//! 3. **Numerically Stable**: Log-domain arithmetic, clamping, and stable softmax
//! 4. **Modular**: Each component usable independently
//!
//! ## Quick Start
//!
//! ```rust
//! use ruvector_math::optimal_transport::{SlicedWasserstein, SinkhornSolver};
//! use ruvector_math::information_geometry::FisherInformation;
//! use ruvector_math::product_manifold::ProductManifold;
//!
//! // Sliced Wasserstein distance between point clouds
//! let sw = SlicedWasserstein::new(100); // 100 projections
//! let dist = sw.distance(&points_a, &points_b);
//!
//! // Sinkhorn optimal transport
//! let solver = SinkhornSolver::new(0.1, 100); // regularization, max_iters
//! let (transport_plan, cost) = solver.solve(&cost_matrix, &weights_a, &weights_b);
//!
//! // Product manifold operations
//! let manifold = ProductManifold::new(64, 16, 8); // E^64 × H^16 × S^8
//! let dist = manifold.distance(&point_a, &point_b);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod error;
pub mod optimal_transport;
pub mod information_geometry;
pub mod spherical;
pub mod product_manifold;
pub mod utils;

// Re-exports for convenience
pub use error::{MathError, Result};
pub use optimal_transport::{
    SlicedWasserstein, SinkhornSolver, GromovWasserstein,
    TransportPlan, WassersteinConfig,
};
pub use information_geometry::{
    FisherInformation, NaturalGradient, KFACApproximation,
};
pub use spherical::{SphericalSpace, SphericalConfig};
pub use product_manifold::{ProductManifold, ProductManifoldConfig, CurvatureType};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::optimal_transport::*;
    pub use crate::information_geometry::*;
    pub use crate::spherical::*;
    pub use crate::product_manifold::*;
    pub use crate::error::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_version() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
    }
}
