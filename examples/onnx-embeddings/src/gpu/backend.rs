//! GPU Backend Abstraction Layer
//!
//! Provides a unified interface for different GPU backends:
//! - WebGPU (via wgpu)
//! - CUDA-WASM (optional, via cuda-rust-wasm)
//! - CPU fallback

use crate::{EmbeddingError, Result};
use super::config::{GpuConfig, GpuMode, GpuMemoryStats, PowerPreference};
use std::sync::Arc;

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Device name
    pub name: String,
    /// Vendor name
    pub vendor: String,
    /// Backend type (WebGPU, CUDA-WASM, CPU)
    pub backend: String,
    /// API version
    pub api_version: String,
    /// Driver version
    pub driver_version: String,
    /// Total memory (bytes)
    pub total_memory: u64,
    /// Maximum workgroup size
    pub max_workgroup_size: u32,
    /// Maximum buffer size
    pub max_buffer_size: u64,
    /// Supports compute shaders
    pub supports_compute: bool,
    /// Supports float16
    pub supports_f16: bool,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            vendor: "Unknown".to_string(),
            backend: "CPU".to_string(),
            api_version: "N/A".to_string(),
            driver_version: "N/A".to_string(),
            total_memory: 0,
            max_workgroup_size: 256,
            max_buffer_size: 128 * 1024 * 1024, // 128MB default
            supports_compute: false,
            supports_f16: false,
        }
    }
}

/// GPU buffer handle
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Buffer ID
    pub id: u64,
    /// Size in bytes
    pub size: u64,
    /// Usage flags
    pub usage: BufferUsage,
}

/// Buffer usage flags
#[derive(Debug, Clone, Copy)]
pub enum BufferUsage {
    /// Storage buffer (read-write)
    Storage,
    /// Uniform buffer (read-only)
    Uniform,
    /// Staging buffer (for transfers)
    Staging,
    /// Vertex buffer
    Vertex,
    /// Index buffer
    Index,
}

/// GPU compute pipeline
pub struct ComputePipeline {
    /// Pipeline ID
    pub id: u64,
    /// Shader name
    pub shader_name: String,
    /// Workgroup size
    pub workgroup_size: [u32; 3],
}

/// GPU Backend trait - unified interface for all GPU operations
pub trait GpuBackend: Send + Sync {
    /// Check if GPU is available
    fn is_available(&self) -> bool;

    /// Get device information
    fn device_info(&self) -> GpuInfo;

    /// Get memory statistics
    fn memory_stats(&self) -> GpuMemoryStats;

    /// Create a buffer
    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<GpuBuffer>;

    /// Write data to buffer
    fn write_buffer(&self, buffer: &GpuBuffer, data: &[u8]) -> Result<()>;

    /// Read data from buffer
    fn read_buffer(&self, buffer: &GpuBuffer, size: u64) -> Result<Vec<u8>>;

    /// Create compute pipeline from shader
    fn create_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
        workgroup_size: [u32; 3],
    ) -> Result<ComputePipeline>;

    /// Execute compute pipeline
    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        bindings: &[&GpuBuffer],
        workgroups: [u32; 3],
    ) -> Result<()>;

    /// Synchronize GPU operations
    fn sync(&self) -> Result<()>;

    /// Release buffer
    fn release_buffer(&self, buffer: GpuBuffer) -> Result<()>;

    /// Release pipeline
    fn release_pipeline(&self, pipeline: ComputePipeline) -> Result<()>;
}

/// GPU Device wrapper with lifetime management
pub struct GpuDevice {
    backend: Arc<dyn GpuBackend>,
    config: GpuConfig,
}

impl GpuDevice {
    /// Create new GPU device
    pub fn new(backend: Arc<dyn GpuBackend>, config: GpuConfig) -> Self {
        Self { backend, config }
    }

    /// Get backend reference
    pub fn backend(&self) -> &dyn GpuBackend {
        self.backend.as_ref()
    }

    /// Get config
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
}

// ==================== Backend Implementations ====================

/// CPU fallback backend
pub struct CpuBackend;

impl GpuBackend for CpuBackend {
    fn is_available(&self) -> bool {
        true // CPU always available
    }

    fn device_info(&self) -> GpuInfo {
        GpuInfo {
            name: "CPU Fallback".to_string(),
            vendor: "N/A".to_string(),
            backend: "CPU".to_string(),
            supports_compute: false,
            ..Default::default()
        }
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        GpuMemoryStats::default()
    }

    fn create_buffer(&self, size: u64, _usage: BufferUsage) -> Result<GpuBuffer> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Ok(GpuBuffer {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            size,
            usage: _usage,
        })
    }

    fn write_buffer(&self, _buffer: &GpuBuffer, _data: &[u8]) -> Result<()> {
        Ok(()) // No-op for CPU
    }

    fn read_buffer(&self, _buffer: &GpuBuffer, size: u64) -> Result<Vec<u8>> {
        Ok(vec![0u8; size as usize])
    }

    fn create_pipeline(
        &self,
        _shader_source: &str,
        _entry_point: &str,
        workgroup_size: [u32; 3],
    ) -> Result<ComputePipeline> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Ok(ComputePipeline {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            shader_name: "cpu_fallback".to_string(),
            workgroup_size,
        })
    }

    fn dispatch(
        &self,
        _pipeline: &ComputePipeline,
        _bindings: &[&GpuBuffer],
        _workgroups: [u32; 3],
    ) -> Result<()> {
        Ok(()) // No-op for CPU
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }

    fn release_buffer(&self, _buffer: GpuBuffer) -> Result<()> {
        Ok(())
    }

    fn release_pipeline(&self, _pipeline: ComputePipeline) -> Result<()> {
        Ok(())
    }
}

/// WebGPU backend (via wgpu)
#[cfg(feature = "gpu")]
pub struct WebGpuBackend {
    device: wgpu::Device,
    #[allow(dead_code)]
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

#[cfg(feature = "gpu")]
impl WebGpuBackend {
    /// Create new WebGPU backend
    pub async fn new(config: &GpuConfig) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let power_pref = match config.power_preference {
            PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
            PowerPreference::None => wgpu::PowerPreference::None,
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| EmbeddingError::other("No GPU adapter found"))?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RuVector GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| EmbeddingError::other(format!("Failed to create device: {}", e)))?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }
}

#[cfg(feature = "gpu")]
impl GpuBackend for WebGpuBackend {
    fn is_available(&self) -> bool {
        true
    }

    fn device_info(&self) -> GpuInfo {
        GpuInfo {
            name: self.adapter_info.name.clone(),
            vendor: format!("{:?}", self.adapter_info.vendor),
            backend: format!("{:?}", self.adapter_info.backend),
            api_version: "WebGPU".to_string(),
            driver_version: self.adapter_info.driver.clone(),
            total_memory: 0, // WebGPU doesn't expose this directly
            max_workgroup_size: 256,
            max_buffer_size: self.device.limits().max_storage_buffer_binding_size as u64,
            supports_compute: true,
            supports_f16: self.device.features().contains(wgpu::Features::SHADER_F16),
        }
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        GpuMemoryStats::default() // WebGPU doesn't expose memory stats
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<GpuBuffer> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let wgpu_usage = match usage {
            BufferUsage::Storage => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            BufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Staging => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Vertex => wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Index => wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        };

        let _buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RuVector Buffer"),
            size,
            usage: wgpu_usage,
            mapped_at_creation: false,
        });

        Ok(GpuBuffer {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            size,
            usage,
        })
    }

    fn write_buffer(&self, _buffer: &GpuBuffer, _data: &[u8]) -> Result<()> {
        // In real implementation, would use self.queue.write_buffer()
        Ok(())
    }

    fn read_buffer(&self, _buffer: &GpuBuffer, size: u64) -> Result<Vec<u8>> {
        // In real implementation, would map buffer and read
        Ok(vec![0u8; size as usize])
    }

    fn create_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
        workgroup_size: [u32; 3],
    ) -> Result<ComputePipeline> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let _shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RuVector Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        Ok(ComputePipeline {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            shader_name: entry_point.to_string(),
            workgroup_size,
        })
    }

    fn dispatch(
        &self,
        _pipeline: &ComputePipeline,
        _bindings: &[&GpuBuffer],
        _workgroups: [u32; 3],
    ) -> Result<()> {
        // In real implementation, would create command encoder and dispatch
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn release_buffer(&self, _buffer: GpuBuffer) -> Result<()> {
        Ok(()) // wgpu handles cleanup via Drop
    }

    fn release_pipeline(&self, _pipeline: ComputePipeline) -> Result<()> {
        Ok(()) // wgpu handles cleanup via Drop
    }
}

// ==================== Factory Functions ====================

/// Create appropriate backend based on configuration
pub async fn create_backend(config: &GpuConfig) -> Result<Box<dyn GpuBackend>> {
    match config.mode {
        GpuMode::CpuOnly => {
            Ok(Box::new(CpuBackend))
        }
        #[cfg(feature = "gpu")]
        GpuMode::WebGpu => {
            match WebGpuBackend::new(config).await {
                Ok(backend) => Ok(Box::new(backend)),
                Err(e) if config.fallback_to_cpu => {
                    tracing::warn!("WebGPU not available, falling back to CPU: {}", e);
                    Ok(Box::new(CpuBackend))
                }
                Err(e) => Err(e),
            }
        }
        #[cfg(feature = "cuda-wasm")]
        GpuMode::CudaWasm => {
            // CUDA-WASM implementation would go here
            tracing::warn!("CUDA-WASM backend not yet implemented, using CPU fallback");
            Ok(Box::new(CpuBackend))
        }
        GpuMode::Auto => {
            #[cfg(feature = "gpu")]
            {
                if let Ok(backend) = WebGpuBackend::new(config).await {
                    return Ok(Box::new(backend));
                }
            }
            Ok(Box::new(CpuBackend))
        }
        #[allow(unreachable_patterns)]
        _ => Ok(Box::new(CpuBackend)),
    }
}

/// Probe GPU availability without full initialization
pub async fn probe_gpu() -> bool {
    #[cfg(feature = "gpu")]
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Get GPU info without full backend creation
pub async fn get_device_info() -> Option<GpuInfo> {
    #[cfg(feature = "gpu")]
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;

        let info = adapter.get_info();
        Some(GpuInfo {
            name: info.name,
            vendor: format!("{:?}", info.vendor),
            backend: format!("{:?}", info.backend),
            api_version: "WebGPU".to_string(),
            driver_version: info.driver,
            supports_compute: true,
            ..Default::default()
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}
