//! Static Core SPI composition shared by the GL backend modules.
//!
//! The module owns backend associated types and backend-error conversion.
//! Instance, resource, and immediate command contracts live in focused sibling
//! modules so no compatibility dispatcher can emerge here.

use super::surface::{GlSurface, GlSurfaceImage};
use super::types::*;
use fluxel_rhi_core::api::{self, Backend, Error};

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
    fn query_supported(device: &GlDevice, kind: api::QueryKind) -> bool {
        super::query::query_supported(device, kind)
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

    fn begin_statistics(_: &mut GlEncoder, _: &GlQuerySet, _: u32) -> api::Result<()> {
        Err(Error::Unsupported("GL pipeline statistics queries"))
    }

    fn end_statistics(_: &mut GlEncoder) -> api::Result<()> {
        Err(Error::Unsupported("GL pipeline statistics queries"))
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
        crate::backend::gl::api::GlError::Shader { log, .. }
        | crate::backend::gl::api::GlError::Program { log, .. } => Error::Shader(log),
        other => Error::Backend(format!("{other:?}")),
    }
}
