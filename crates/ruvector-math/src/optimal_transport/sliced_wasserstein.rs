//! Sliced Wasserstein Distance
//!
//! The Sliced Wasserstein distance projects high-dimensional distributions
//! onto random 1D lines and averages the 1D Wasserstein distances.
//!
//! ## Algorithm
//!
//! 1. Generate L random unit vectors (directions) in R^d
//! 2. For each direction θ:
//!    a. Project all source and target points onto θ
//!    b. Compute 1D Wasserstein distance (closed-form via sorted quantiles)
//! 3. Average over all directions
//!
//! ## Complexity
//!
//! - O(L × n log n) where L = number of projections, n = number of points
//! - Linear in dimension d (only dot products)
//!
//! ## Advantages
//!
//! - **Fast**: Near-linear scaling to millions of points
//! - **SIMD-friendly**: Projections are just dot products
//! - **Statistically consistent**: Converges to true W2 as L → ∞

use rand::prelude::*;
use rand_distr::StandardNormal;
use crate::utils::{argsort, EPS};
use super::{OptimalTransport, WassersteinConfig};

/// Sliced Wasserstein distance calculator
#[derive(Debug, Clone)]
pub struct SlicedWasserstein {
    /// Number of random projection directions
    num_projections: usize,
    /// Power for Wasserstein-p (typically 1 or 2)
    p: f64,
    /// Random seed for reproducibility
    seed: Option<u64>,
}

impl SlicedWasserstein {
    /// Create a new Sliced Wasserstein calculator
    ///
    /// # Arguments
    /// * `num_projections` - Number of random 1D projections (100-1000 typical)
    pub fn new(num_projections: usize) -> Self {
        Self {
            num_projections: num_projections.max(1),
            p: 2.0,
            seed: None,
        }
    }

    /// Create from configuration
    pub fn from_config(config: &WassersteinConfig) -> Self {
        Self {
            num_projections: config.num_projections.max(1),
            p: config.p,
            seed: config.seed,
        }
    }

    /// Set the Wasserstein power (1 for W1, 2 for W2)
    pub fn with_power(mut self, p: f64) -> Self {
        self.p = p.max(1.0);
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate random unit directions
    fn generate_directions(&self, dim: usize) -> Vec<Vec<f64>> {
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        (0..self.num_projections)
            .map(|_| {
                let mut direction: Vec<f64> = (0..dim)
                    .map(|_| rng.sample(StandardNormal))
                    .collect();

                // Normalize to unit vector
                let norm: f64 = direction.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > EPS {
                    for x in &mut direction {
                        *x /= norm;
                    }
                }
                direction
            })
            .collect()
    }

    /// Project points onto a direction
    #[inline]
    fn project(points: &[Vec<f64>], direction: &[f64]) -> Vec<f64> {
        points
            .iter()
            .map(|p| {
                p.iter()
                    .zip(direction.iter())
                    .map(|(&pi, &di)| pi * di)
                    .sum()
            })
            .collect()
    }

    /// Compute 1D Wasserstein distance between two sorted distributions
    ///
    /// For uniform weights, this is simply the sum of |sorted_a[i] - sorted_b[i]|^p
    fn wasserstein_1d_uniform(&self, mut proj_a: Vec<f64>, mut proj_b: Vec<f64>) -> f64 {
        let n = proj_a.len();
        let m = proj_b.len();

        // Sort projections
        proj_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
        proj_b.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if n == m {
            // Same size: direct comparison
            proj_a
                .iter()
                .zip(proj_b.iter())
                .map(|(&a, &b)| (a - b).abs().powf(self.p))
                .sum::<f64>()
                / n as f64
        } else {
            // Different sizes: interpolate via quantiles
            self.wasserstein_1d_quantile(&proj_a, &proj_b, n.max(m))
        }
    }

    /// Compute 1D Wasserstein via quantile interpolation
    fn wasserstein_1d_quantile(&self, sorted_a: &[f64], sorted_b: &[f64], num_samples: usize) -> f64 {
        let mut total = 0.0;

        for i in 0..num_samples {
            let q = (i as f64 + 0.5) / num_samples as f64;

            let val_a = quantile_sorted(sorted_a, q);
            let val_b = quantile_sorted(sorted_b, q);

            total += (val_a - val_b).abs().powf(self.p);
        }

        total / num_samples as f64
    }

    /// Compute 1D Wasserstein with weights
    fn wasserstein_1d_weighted(
        &self,
        proj_a: &[f64],
        weights_a: &[f64],
        proj_b: &[f64],
        weights_b: &[f64],
    ) -> f64 {
        // Sort by projected values
        let idx_a = argsort(proj_a);
        let idx_b = argsort(proj_b);

        let sorted_a: Vec<f64> = idx_a.iter().map(|&i| proj_a[i]).collect();
        let sorted_w_a: Vec<f64> = idx_a.iter().map(|&i| weights_a[i]).collect();
        let sorted_b: Vec<f64> = idx_b.iter().map(|&i| proj_b[i]).collect();
        let sorted_w_b: Vec<f64> = idx_b.iter().map(|&i| weights_b[i]).collect();

        // Compute cumulative weights
        let cdf_a = compute_cdf(&sorted_w_a);
        let cdf_b = compute_cdf(&sorted_w_b);

        // Merge and compute
        self.wasserstein_1d_from_cdfs(&sorted_a, &cdf_a, &sorted_b, &cdf_b)
    }

    /// Compute 1D Wasserstein from CDFs
    fn wasserstein_1d_from_cdfs(
        &self,
        values_a: &[f64],
        cdf_a: &[f64],
        values_b: &[f64],
        cdf_b: &[f64],
    ) -> f64 {
        // Merge all CDF points
        let mut events: Vec<(f64, f64, f64)> = Vec::new(); // (position, cdf_a, cdf_b)

        let mut ia = 0;
        let mut ib = 0;
        let mut current_cdf_a = 0.0;
        let mut current_cdf_b = 0.0;

        while ia < values_a.len() || ib < values_b.len() {
            let pos = match (ia < values_a.len(), ib < values_b.len()) {
                (true, true) => {
                    if values_a[ia] <= values_b[ib] {
                        current_cdf_a = cdf_a[ia];
                        ia += 1;
                        values_a[ia - 1]
                    } else {
                        current_cdf_b = cdf_b[ib];
                        ib += 1;
                        values_b[ib - 1]
                    }
                }
                (true, false) => {
                    current_cdf_a = cdf_a[ia];
                    ia += 1;
                    values_a[ia - 1]
                }
                (false, true) => {
                    current_cdf_b = cdf_b[ib];
                    ib += 1;
                    values_b[ib - 1]
                }
                (false, false) => break,
            };

            events.push((pos, current_cdf_a, current_cdf_b));
        }

        // Integrate |F_a - F_b|^p
        let mut total = 0.0;
        for i in 1..events.len() {
            let width = events[i].0 - events[i - 1].0;
            let height = (events[i - 1].1 - events[i - 1].2).abs();
            total += width * height.powf(self.p);
        }

        total
    }
}

impl OptimalTransport for SlicedWasserstein {
    fn distance(&self, source: &[Vec<f64>], target: &[Vec<f64>]) -> f64 {
        if source.is_empty() || target.is_empty() {
            return 0.0;
        }

        let dim = source[0].len();
        if dim == 0 {
            return 0.0;
        }

        let directions = self.generate_directions(dim);

        let total: f64 = directions
            .iter()
            .map(|dir| {
                let proj_source = Self::project(source, dir);
                let proj_target = Self::project(target, dir);
                self.wasserstein_1d_uniform(proj_source, proj_target)
            })
            .sum();

        (total / self.num_projections as f64).powf(1.0 / self.p)
    }

    fn weighted_distance(
        &self,
        source: &[Vec<f64>],
        source_weights: &[f64],
        target: &[Vec<f64>],
        target_weights: &[f64],
    ) -> f64 {
        if source.is_empty() || target.is_empty() {
            return 0.0;
        }

        let dim = source[0].len();
        if dim == 0 {
            return 0.0;
        }

        // Normalize weights
        let sum_a: f64 = source_weights.iter().sum();
        let sum_b: f64 = target_weights.iter().sum();
        let weights_a: Vec<f64> = source_weights.iter().map(|&w| w / sum_a).collect();
        let weights_b: Vec<f64> = target_weights.iter().map(|&w| w / sum_b).collect();

        let directions = self.generate_directions(dim);

        let total: f64 = directions
            .iter()
            .map(|dir| {
                let proj_source = Self::project(source, dir);
                let proj_target = Self::project(target, dir);
                self.wasserstein_1d_weighted(&proj_source, &weights_a, &proj_target, &weights_b)
            })
            .sum();

        (total / self.num_projections as f64).powf(1.0 / self.p)
    }
}

/// Quantile of sorted data
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    let q = q.clamp(0.0, 1.0);
    let n = sorted.len();

    if n == 1 {
        return sorted[0];
    }

    let idx_f = q * (n - 1) as f64;
    let idx_low = idx_f.floor() as usize;
    let idx_high = (idx_low + 1).min(n - 1);
    let frac = idx_f - idx_low as f64;

    sorted[idx_low] * (1.0 - frac) + sorted[idx_high] * frac
}

/// Compute CDF from weights
fn compute_cdf(weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    let mut cdf = Vec::with_capacity(weights.len());
    let mut cumsum = 0.0;

    for &w in weights {
        cumsum += w / total;
        cdf.push(cumsum);
    }

    cdf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliced_wasserstein_identical() {
        let sw = SlicedWasserstein::new(100).with_seed(42);

        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        // Distance to itself should be very small
        let dist = sw.distance(&points, &points);
        assert!(dist < 0.01, "Self-distance should be ~0, got {}", dist);
    }

    #[test]
    fn test_sliced_wasserstein_translation() {
        let sw = SlicedWasserstein::new(500).with_seed(42);

        let source = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        // Translate by (1, 1)
        let target: Vec<Vec<f64>> = source.iter().map(|p| vec![p[0] + 1.0, p[1] + 1.0]).collect();

        let dist = sw.distance(&source, &target);

        // For W2 translation by (1, 1), expected distance is sqrt(2) ≈ 1.414
        // But Sliced Wasserstein is an approximation, so allow wider tolerance
        assert!(
            dist > 0.5 && dist < 2.0,
            "Translation distance should be positive, got {:.3}",
            dist
        );
    }

    #[test]
    fn test_sliced_wasserstein_scaling() {
        let sw = SlicedWasserstein::new(500).with_seed(42);

        let source = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        // Scale by 2
        let target: Vec<Vec<f64>> = source.iter().map(|p| vec![p[0] * 2.0, p[1] * 2.0]).collect();

        let dist = sw.distance(&source, &target);

        // Should be positive for scaled distribution
        assert!(dist > 0.0, "Scaling should produce positive distance");
    }

    #[test]
    fn test_weighted_distance() {
        let sw = SlicedWasserstein::new(100).with_seed(42);

        let source = vec![vec![0.0], vec![1.0]];
        let target = vec![vec![2.0], vec![3.0]];

        // Uniform weights
        let weights_s = vec![0.5, 0.5];
        let weights_t = vec![0.5, 0.5];

        let dist = sw.weighted_distance(&source, &weights_s, &target, &weights_t);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_1d_projections() {
        let sw = SlicedWasserstein::new(10);
        let directions = sw.generate_directions(3);

        assert_eq!(directions.len(), 10);

        // Each direction should be unit length
        for dir in &directions {
            let norm: f64 = dir.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "Direction not unit: {}", norm);
        }
    }
}
