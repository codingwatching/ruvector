//! Metal GPU acceleration benchmarks
//!
//! Benchmarks Metal compute shaders for LLM operations.
//! Only runs on macOS with `metal-compute` feature enabled.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(all(target_os = "macos", feature = "metal-compute"))]
use ruvllm_integration::metal::{MetalContext, MetalConfig};
#[cfg(all(target_os = "macos", feature = "metal-compute"))]
use ruvllm_integration::kernels::AttentionConfig;

#[cfg(all(target_os = "macos", feature = "metal-compute"))]
fn bench_flash_attention_metal(c: &mut Criterion) {
    let ctx = match MetalContext::new(MetalConfig::default()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("metal_flash_attention");

    for (seq_len, kv_len) in [(1, 512), (1, 2048), (1, 4096), (4, 512), (4, 2048)] {
        let config = AttentionConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: seq_len,
            causal: true,
            scale: 0.0,
        };

        let query: Vec<f32> = (0..seq_len * config.num_heads * config.head_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let key: Vec<f32> = (0..kv_len * config.num_kv_heads * config.head_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let value: Vec<f32> = (0..kv_len * config.num_kv_heads * config.head_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("metal", format!("seq{}_kv{}", seq_len, kv_len)),
            &(&query, &key, &value, &config),
            |b, (q, k, v, cfg)| {
                b.iter(|| ctx.flash_attention(black_box(*q), black_box(*k), black_box(*v), black_box(*cfg)))
            },
        );
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-compute"))]
fn bench_gemm_metal(c: &mut Criterion) {
    let ctx = match MetalContext::new(MetalConfig::default()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("metal_gemm");

    for size in [128, 256, 512, 1024, 2048] {
        let m = size;
        let n = size;
        let k = size;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("metal_f32", format!("{}x{}", size, size)),
            &(&a, &b, m, n, k),
            |bench, (a, b, m, n, k)| {
                bench.iter(|| ctx.gemm_f32(black_box(*a), black_box(*b), *m, *n, *k))
            },
        );
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-compute"))]
fn bench_rms_norm_metal(c: &mut Criterion) {
    let ctx = match MetalContext::new(MetalConfig::default()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("metal_rms_norm");

    for hidden_size in [1024, 2048, 4096, 8192] {
        let batch_size = 4;
        let mut x: Vec<f32> = (0..batch_size * hidden_size)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let weight: Vec<f32> = vec![1.0; hidden_size];

        group.bench_with_input(
            BenchmarkId::new("metal", format!("hidden{}", hidden_size)),
            &(hidden_size, batch_size),
            |bench, _| {
                bench.iter(|| {
                    let mut x_clone = x.clone();
                    ctx.rms_norm(black_box(&mut x_clone), black_box(&weight), 1e-6)
                })
            },
        );
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-compute"))]
fn bench_rope_metal(c: &mut Criterion) {
    let ctx = match MetalContext::new(MetalConfig::default()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("metal_rope");

    for num_heads in [8, 16, 32] {
        let head_dim = 128;
        let batch_size = 4;
        let mut x: Vec<f32> = (0..batch_size * num_heads * head_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("metal", format!("heads{}", num_heads)),
            &(num_heads, head_dim, batch_size),
            |bench, &(nh, hd, bs)| {
                bench.iter(|| {
                    let mut x_clone = x.clone();
                    ctx.apply_rope(black_box(&mut x_clone), 0, nh, hd, 10000.0)
                })
            },
        );
    }

    group.finish();
}

// CPU baseline comparison
fn bench_cpu_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_gemm");

    for size in [128, 256, 512] {
        let m = size;
        let n = size;
        let k = size;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("naive", format!("{}x{}", size, size)),
            &(&a, &b, m, n, k),
            |bench, (a, b, m, n, k)| {
                bench.iter(|| {
                    let mut c = vec![0.0f32; *m * *n];
                    for i in 0..*m {
                        for j in 0..*n {
                            let mut sum = 0.0f32;
                            for l in 0..*k {
                                sum += a[i * *k + l] * b[l * *n + j];
                            }
                            c[i * *n + j] = sum;
                        }
                    }
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-compute"))]
criterion_group!(
    metal_benches,
    bench_flash_attention_metal,
    bench_gemm_metal,
    bench_rms_norm_metal,
    bench_rope_metal,
    bench_cpu_gemm,
);

#[cfg(not(all(target_os = "macos", feature = "metal-compute")))]
criterion_group!(
    metal_benches,
    bench_cpu_gemm,
);

criterion_main!(metal_benches);
