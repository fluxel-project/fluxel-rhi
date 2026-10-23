//! Static Core SPI composition shared by the GL backend modules.
//!
//! The module owns backend associated types and backend-error conversion.
//! Instance, resource, and immediate command contracts live in focused sibling
//! modules so no compatibility dispatcher can emerge here.

use super::surface::{GlSurface, GlSurfaceImage};
use super::types::*;
use fluxel_rhi_core::api::{self, Backend, Error};

#[cfg(test)]
std::thread_local! {
    /// Exercises the portable path on a context which also has query buffers.
    pub(crate) static TEST_DISABLE_QUERY_BUFFER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Backend for Gl {
    type NativeDevice = GlDevice;
    type NativeBuffer = GlBuffer;
    type NativeTexture = GlTexture;
    type NativeTextureView = GlTextureView;
    type NativeSampler = GlSampler;
    type NativeBindGroupLayout = GlBindGroupLayout;
    type NativeBindGroup = GlBindGroup;
    type NativePipelineLayout = GlPipelineLayout;
    type NativeShader = GlShader;
    type NativeRasterPipeline = GlRasterPipeline;
    type NativeComputePipeline = GlComputePipeline;
    type NativeMeshPipeline = GlMeshPipeline;
    type NativeRayPipeline = GlRayPipeline;
    type NativePipelineCache = GlPipelineCache;
    type NativeAcceleration = GlAcceleration;
    type NativeSparseAllocation = GlSparseAllocation;
    type NativeQuerySet = GlQuerySet;
    type NativeEncoder = GlEncoder;
    type NativeCompletion = GlCompletion;
    type NativeSurface = GlSurface;
    type NativeSurfaceImage = GlSurfaceImage;
    type NativeUncertainty = ();
    type Compute = super::compute::GlCompute;
    type Mesh = api::Unavailable;
    type Ray = api::Unavailable;
    type Extras = super::extras::GlExtras;
}

impl api::QueryBackend for Gl {
    fn timestamp_result_format(_device: &GlDevice) -> api::TimestampResultFormat {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            api::TimestampResultFormat {
                invalid_value: Some(u64::MAX),
            }
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            api::TimestampResultFormat {
                invalid_value: _device
                    .0
                    .with_native("query timer result format", |provider| {
                        Ok(provider.gles_timestamp_bits().map(|_| u64::MAX))
                    })
                    .ok()
                    .flatten(),
            }
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", feature = "webgl2"),
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl"
        )))]
        {
            api::TimestampResultFormat::default()
        }
    }
    fn query_resolve_mode(_device: &GlDevice, _kind: api::QueryKind) -> api::QueryResolveMode {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            api::QueryResolveMode::AfterCompletion
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            if _device
                .0
                .with_native("query resolve mode", |provider| {
                    Ok(provider.supports_query_buffer())
                })
                .unwrap_or(false)
            {
                api::QueryResolveMode::Ordered
            } else {
                api::QueryResolveMode::AfterCompletion
            }
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", feature = "webgl2"),
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl"
        )))]
        {
            api::QueryResolveMode::AfterCompletion
        }
    }
    fn query_supported(device: &GlDevice, kind: api::QueryKind) -> bool {
        super::query::query_supported(device, kind)
    }

    fn timestamp_clock(device: &GlDevice, lane: api::LaneId) -> Option<api::TimestampClock> {
        if lane.ordinal != 0 {
            return None;
        }
        Some(api::TimestampClock {
            period_ns: 1.0,
            valid_bits: super::query::timestamp_bits(device)?,
        })
    }

    fn timestamp_frequency(device: &GlDevice, lane: api::LaneId) -> Option<u64> {
        (lane.ordinal == 0 && super::query::timestamp_bits(device).is_some())
            .then_some(1_000_000_000)
    }

    fn create_query_set(device: &GlDevice, desc: &api::QueryDesc) -> api::Result<GlQuerySet> {
        super::query::create_query_set(device, desc)
    }

    fn write_timestamp(encoder: &mut GlEncoder, set: &GlQuerySet, slot: u32) -> api::Result<()> {
        super::query::write_timestamp(encoder, set, slot)
    }

    fn begin_occlusion(encoder: &mut GlEncoder, set: &GlQuerySet, slot: u32) -> api::Result<()> {
        super::query::begin_occlusion(encoder, set, slot)
    }

    fn reset_queries(
        encoder: &mut GlEncoder,
        set: &GlQuerySet,
        slots: std::ops::Range<u32>,
    ) -> api::Result<()> {
        super::query::reset(encoder, set, slots)
    }

    fn end_occlusion(encoder: &mut GlEncoder) -> api::Result<()> {
        super::query::end_occlusion(encoder)
    }

    fn begin_statistics(encoder: &mut GlEncoder, set: &GlQuerySet, slot: u32) -> api::Result<()> {
        super::query::begin_statistics(encoder, set, slot)
    }

    fn end_statistics(encoder: &mut GlEncoder) -> api::Result<()> {
        super::query::end_statistics(encoder)
    }

    fn resolve_queries(
        encoder: &mut GlEncoder,
        set: &GlQuerySet,
        slots: std::ops::Range<u32>,
        destination: &GlBuffer,
        offset: u64,
    ) -> api::Result<()> {
        super::query::resolve(encoder, set, slots, destination, offset)
    }
}

/// Converts private platform failures to the public Core error taxonomy.
pub(crate) fn native_error(error: crate::backend::gl::api::GlError) -> Error {
    match error {
        crate::backend::gl::api::GlError::ContextLost { .. } => Error::DeviceLost,
        crate::backend::gl::api::GlError::OutOfMemory { .. } => Error::OutOfMemory,
        crate::backend::gl::api::GlError::Unsupported { reason, .. } => Error::Unsupported(reason),
        crate::backend::gl::api::GlError::Shader { log, .. }
        | crate::backend::gl::api::GlError::Program { log, .. } => Error::Shader(log),
        other => Error::Backend(format!("{other:?}")),
    }
}
