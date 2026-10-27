//! Backend SPI for resource construction, pipelines, queries, mapping, memory,
//! and surfaces.
//!
//! These traits are implemented by backend crates. Renderer-facing values and
//! facade methods remain in [`crate::api`].

use crate::{api::*, backend::*};
use std::{future::Future, ops::Range};

/// Backend-native constructors and per-device support facts. No hazard tracker.
pub trait ResourceBackend: Backend {
    fn buffer_support(device: &Self::NativeDevice, desc: &BufferDesc) -> Support<BufferSupport>;
    fn texture_support(device: &Self::NativeDevice, desc: &TextureDesc) -> Support<TextureSupport>;
    fn create_buffer(device: &Self::NativeDevice, desc: &BufferDesc) -> Result<Self::NativeBuffer>;
    fn create_texture(
        device: &Self::NativeDevice,
        desc: &TextureDesc,
    ) -> Result<Self::NativeTexture>;
    fn create_view(
        device: &Self::NativeDevice,
        texture: &Self::NativeTexture,
        desc: &TextureViewDesc,
    ) -> Result<Self::NativeTextureView>;
    fn create_sampler(
        device: &Self::NativeDevice,
        desc: &SamplerDesc,
    ) -> Result<Self::NativeSampler>;
    fn create_bind_group_layout(
        device: &Self::NativeDevice,
        desc: &BindGroupLayoutDesc,
    ) -> Result<Self::NativeBindGroupLayout>;
    fn create_bind_group(
        device: &Self::NativeDevice,
        desc: &BindGroupDesc<'_, Self>,
    ) -> Result<Self::NativeBindGroup>;
    fn create_pipeline_layout(
        device: &Self::NativeDevice,
        desc: &PipelineLayoutDesc<'_, Self>,
    ) -> Result<Self::NativePipelineLayout>;
}

/// Cold shader operations: native direct route first, actual Naga frontend/output
/// pair second, else error. No retry after a selected compiler fails.
pub trait ShaderBackend: Backend {
    type DirectFuture<'a>: Future<Output = Result<Self::NativeShader>> + 'a
    where
        Self: 'a;
    type NagaFuture<'a>: Future<Output = Result<Self::NativeShader>> + 'a
    where
        Self: 'a;
    fn shader_route(device: &Self::NativeDevice, desc: &ShaderDesc) -> ShaderRoute;
    fn compile_direct<'a>(
        device: &'a Self::NativeDevice,
        desc: &'a ShaderDesc,
        target: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a>;
    fn compile_naga<'a>(
        device: &'a Self::NativeDevice,
        desc: &'a ShaderDesc,
        target: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a>;
}

pub trait PipelineBackend: Backend {
    type RasterPipelineFuture<'a>: Future<Output = Result<Self::NativeRasterPipeline>> + 'a
    where
        Self: 'a;
    fn create_raster_pipeline<'a>(
        device: &'a Self::NativeDevice,
        desc: &'a RasterPipelineDesc<Self>,
        dynamic_depth_bias: Option<DynamicDepthBias>,
    ) -> Self::RasterPipelineFuture<'a>;
}

pub trait CacheBackend: Backend {
    fn create_cache(
        device: &Self::NativeDevice,
        data: Option<&[u8]>,
    ) -> Result<Self::NativePipelineCache>;
    fn serialize_cache(cache: &Self::NativePipelineCache) -> Result<Vec<u8>>;
}

/// Optional pass-boundary statistics SPI. A backend exposes this separately
/// from the fixed eleven-field pipeline-statistics query contract.
pub trait PassStatisticsBackend: QueryBackend {
    fn pass_statistics_fields(device: &Self::NativeDevice) -> StatisticsFields;
    fn pass_statistics_invalid_value(device: &Self::NativeDevice) -> Option<u64>;
    fn create_pass_statistics(
        device: &Self::NativeDevice,
        pass: StatisticsPass,
        count: u32,
    ) -> Result<Self::NativeQuerySet>;
    /// Attaches the set to the immediately following matching pass begin.
    fn prepare_pass_statistics(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slot: u32,
        pass: StatisticsPass,
    ) -> Result<()>;
    fn resolve_pass_statistics(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slots: Range<u32>,
        destination: &Self::NativeBuffer,
        offset: u64,
    ) -> Result<()>;
}

pub trait QueryBackend: CommandBackend {
    fn query_supported(device: &Self::NativeDevice, kind: QueryKind) -> bool;
    /// Required ordering for this device and query kind.
    fn query_resolve_mode(_: &Self::NativeDevice, _: QueryKind) -> QueryResolveMode {
        QueryResolveMode::Ordered
    }
    /// Reserved timestamp result value, if the native API reports slot errors.
    fn timestamp_result_format(_device: &Self::NativeDevice) -> TimestampResultFormat {
        TimestampResultFormat::default()
    }
    /// Ticks per second for the specified queue, if known.
    fn timestamp_frequency(_device: &Self::NativeDevice, _lane: LaneId) -> Option<u64> {
        None
    }
    /// Timestamp period and valid counter bits for a queue.
    fn timestamp_clock(device: &Self::NativeDevice, lane: LaneId) -> Option<TimestampClock> {
        let frequency = Self::timestamp_frequency(device, lane).filter(|f| *f != 0)?;
        Some(TimestampClock {
            period_ns: 1_000_000_000.0 / frequency as f64,
            valid_bits: 64,
        })
    }
    fn create_query_set(
        device: &Self::NativeDevice,
        desc: &QueryDesc,
    ) -> Result<Self::NativeQuerySet>;
    fn write_timestamp(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slot: u32,
    ) -> Result<()>;
    fn begin_occlusion(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slot: u32,
    ) -> Result<()>;
    fn reset_queries(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slots: Range<u32>,
    ) -> Result<()>;
    fn end_occlusion(encoder: &mut Self::NativeEncoder) -> Result<()>;
    fn begin_statistics(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slot: u32,
    ) -> Result<()>;
    fn end_statistics(encoder: &mut Self::NativeEncoder) -> Result<()>;
    fn resolve_queries(
        encoder: &mut Self::NativeEncoder,
        set: &Self::NativeQuerySet,
        slots: Range<u32>,
        destination: &Self::NativeBuffer,
        offset: u64,
    ) -> Result<()>;
}

pub trait MappingBackend: Backend {
    type NativeMap;
    type MapFuture<'a>: Future<Output = Result<Self::NativeMap>> + 'a
    where
        Self: 'a;
    fn map<'a>(
        device: &'a Self::NativeDevice,
        buffer: &'a Self::NativeBuffer,
        range: Range<u64>,
        mode: MapMode,
    ) -> Self::MapFuture<'a>;
    fn bytes(map: &Self::NativeMap) -> &[u8];
    fn bytes_mut(map: &mut Self::NativeMap) -> &mut [u8];
    fn flush(map: &mut Self::NativeMap, bytes: Range<u64>) -> Result<()>;
    fn invalidate(map: &mut Self::NativeMap, bytes: Range<u64>) -> Result<()>;
    fn unmap(map: Self::NativeMap);
}

/// Readback is staged at an explicitly encoded position. RHI does not scan
/// resources or choose completion; renderer must await covering checkpoint.
pub trait ReadbackBackend: CommandBackend + CompletionBackend {
    type NativeReadback;
    type ReadbackFuture<'a>: Future<Output = Result<ReadbackData>> + 'a
    where
        Self: 'a;
    fn encode_readback(
        encoder: &mut Self::NativeEncoder,
        src: &Self::NativeBuffer,
        range: Range<u64>,
    ) -> Result<Self::NativeReadback>;
    fn map_readback<'a>(
        device: &'a Self::NativeDevice,
        ticket: Self::NativeReadback,
    ) -> Self::ReadbackFuture<'a>;
}

pub trait MemoryBackend: ResourceBackend {
    fn buffer_requirements(
        device: &Self::NativeDevice,
        desc: &BufferDesc,
    ) -> Result<MemoryRequirements>;
    fn texture_requirements(
        device: &Self::NativeDevice,
        desc: &TextureDesc,
    ) -> Result<MemoryRequirements>;
    type NativeHeap: Clone;
    fn create_heap(device: &Self::NativeDevice, desc: &HeapDesc) -> Result<Self::NativeHeap>;
    fn place_buffer(
        device: &Self::NativeDevice,
        heap: &Self::NativeHeap,
        offset: u64,
        desc: &BufferDesc,
    ) -> Result<Self::NativeBuffer>;
    fn place_texture(
        device: &Self::NativeDevice,
        heap: &Self::NativeHeap,
        offset: u64,
        desc: &TextureDesc,
    ) -> Result<Self::NativeTexture>;
}

pub trait SurfaceBackend: Backend {
    type ConfigureFuture<'a>: Future<Output = Result<()>> + 'a
    where
        Self: 'a;
    type AcquireFuture<'a>: Future<Output = Result<Self::NativeSurfaceImage>> + 'a
    where
        Self: 'a;
    fn generation(surface: &Self::NativeSurface) -> u64;
    fn capabilities(surface: &Self::NativeSurface) -> Result<SurfaceCapabilities>;
    fn supports_config(surface: &Self::NativeSurface, desc: &SurfaceConfig) -> Support<()>;
    /// Configuration and generation change in one native transaction. Failed or
    /// cancelled pending configuration must not publish a half-commit.
    fn configure<'a>(
        surface: &'a mut Self::NativeSurface,
        desc: &'a SurfaceConfig,
    ) -> Self::ConfigureFuture<'a>;
    fn acquire<'a>(surface: &'a mut Self::NativeSurface) -> Self::AcquireFuture<'a>;
}
