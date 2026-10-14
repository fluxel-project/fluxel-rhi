//! Immediate native compute SPI for GL contexts that proved image load/store.
//!
//! A compute pass owns no command list. Pipeline selection, bind-group
//! application, and dispatch each call the adopted context immediately. The
//! static adapter exists on every GL build, while `DeviceCapabilities::compute`
//! keeps GL 4.0--4.2 and WebGL2 from exposing this domain.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{ComputeOps, ComputePipelineDesc, Error, Result};

use super::encoder::GlEncoder;
use super::spi::native_error;
use super::types::{GlComputePipeline, GlComputePipelineInner};
use super::{Gl, GlDevice};

/// Static ComputeOps implementation. Runtime availability is still decided
/// from the exact adopted context's discovery evidence by Core capabilities.
pub struct GlCompute;

impl ComputeOps<Gl> for GlCompute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<Result<GlComputePipeline>>;

    fn create_pipeline<'a>(
        device: &'a GlDevice,
        desc: &'a ComputePipelineDesc<Gl>,
    ) -> Self::PipelineFuture<'a> {
        ready(create_pipeline(device, desc))
    }

    fn begin(encoder: &mut GlEncoder) -> Result<()> {
        // Core serializes passes, but keep the native immediate state honest:
        // a raster pass leaves an active framebuffer that a compute pass may
        // not silently inherit as a second kind of pass.
        if encoder.raster_extent.is_some() {
            return Err(Error::InvalidState);
        }
        encoder.compute_pipeline = None;
        Ok(())
    }

    fn set_pipeline(encoder: &mut GlEncoder, pipeline: &GlComputePipeline) -> Result<()> {
        if !std::rc::Rc::ptr_eq(&encoder.state, &pipeline.state) {
            return Err(Error::InvalidState);
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlComputeDispatchApi as _;
            let result = encoder
                .state
                .with_native("set GL compute pipeline", |provider| {
                    provider.set_compute_program(pipeline.program)
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.compute_pipeline = Some(pipeline.clone());
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            let _ = (encoder, pipeline);
            Err(Error::Unsupported("native GL compute context"))
        }
    }

    fn set_bind_group(
        encoder: &mut GlEncoder,
        slot: u32,
        group: &super::types::GlBindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        let pipeline = encoder
            .compute_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?
            .clone();
        let expected = pipeline
            .layout
            .groups
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        if !crate::backend::gl::backend::raster::layout::matches(expected, &group.layout) {
            return Err(Error::InvalidState);
        }
        crate::backend::gl::backend::raster::binding::validate_dynamic_offsets(group, offsets)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        return crate::backend::gl::backend::raster::binding::bind_native_group(
            encoder,
            &pipeline.layout,
            &pipeline.reflection,
            slot,
            group,
            offsets,
        );
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            let _ = (encoder, pipeline, slot, group, offsets);
            Err(Error::Unsupported("native GL compute context"))
        }
    }

    fn dispatch(encoder: &mut GlEncoder, groups: [u32; 3]) -> Result<()> {
        if encoder.compute_pipeline.is_none() {
            return Err(Error::InvalidState);
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{GlComputeDispatchApi as _, GlDispatchGroups};
            let result = encoder
                .state
                .with_native("dispatch GL compute", |provider| {
                    provider.dispatch(GlDispatchGroups(groups))
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            let _ = (encoder, groups);
            Err(Error::Unsupported("native GL compute context"))
        }
    }

    fn dispatch_indirect(_: &mut GlEncoder, _: &super::types::GlBuffer, _: u64) -> Result<()> {
        // The provider has an indirect verb, but this Core adapter has not yet
        // fixed its argument ABI and visibility edge. Rejecting it preserves
        // the immediate contract instead of guessing one from native GL.
        Err(Error::Unsupported("GL compute indirect dispatch"))
    }

    fn end(encoder: &mut GlEncoder) -> Result<()> {
        encoder.compute_pipeline = None;
        Ok(())
    }

    fn abort(encoder: &mut GlEncoder) {
        encoder.compute_pipeline = None;
    }
}

fn create_pipeline(device: &GlDevice, desc: &ComputePipelineDesc<Gl>) -> Result<GlComputePipeline> {
    if desc.required_subgroup_size.is_some() || desc.cache.is_some() {
        return Err(Error::Unsupported("GL compute pipeline subgroup/cache"));
    }
    // SAFETY: Core keeps both native handles borrowed throughout this
    // synchronous SPI call; only immutable source and ABI metadata is copied.
    let shader = unsafe { desc.compute.module.native_ref() };
    let layout = unsafe { desc.layout.native_ref() }.clone();
    if !std::rc::Rc::ptr_eq(&device.0, &shader.state)
        || !std::rc::Rc::ptr_eq(&device.0, &layout.state)
        || shader.source.stage != crate::backend::gl::api::GlShaderStage::Compute
        || shader.source.entry_point != desc.compute.entry
    {
        return Err(Error::InvalidState);
    }
    let native_layout = super::pipeline::with_naga_binding_names(layout.native.clone(), &[shader])?;
    let descriptor = crate::backend::gl::api::GlProgramDescriptor {
        kind: crate::backend::gl::api::GlProgramKind::Compute {
            shader: shader.source.clone(),
        },
        layout: native_layout,
        debug_name: desc.label.clone(),
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlShaderApi as _;
        return device
            .0
            .with_native("create GL compute pipeline", |provider| {
                provider.create_program(&descriptor)
            })
            .map(|(program, reflection)| {
                GlComputePipeline::from_inner(GlComputePipelineInner {
                    state: device.0.clone(),
                    layout,
                    reflection,
                    program,
                })
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (device, descriptor, layout);
        Err(Error::Unsupported("native GL compute context"))
    }
}
