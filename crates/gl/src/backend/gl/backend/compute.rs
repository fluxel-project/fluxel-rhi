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
        encoder.bound_groups.clear();
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
                let groups = super::raster::binding::compatible_groups(encoder, &pipeline.layout);
                for (slot, group, offsets) in groups {
                    super::raster::binding::bind_native_group(
                        encoder,
                        &pipeline.layout,
                        &pipeline.reflection,
                        pipeline.program,
                        &pipeline.mip_uniforms,
                        slot,
                        &group,
                        &offsets,
                    )?;
                }
                super::push::upload_compute(encoder)?;
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
            pipeline.program,
            &pipeline.mip_uniforms,
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
        super::raster::cube_mirror::refresh(encoder)?;
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

    fn dispatch_indirect(
        encoder: &mut GlEncoder,
        buffer: &super::types::GlBuffer,
        offset: u64,
    ) -> Result<()> {
        if encoder.compute_pipeline.is_none() {
            return Err(Error::InvalidState);
        }
        if !std::rc::Rc::ptr_eq(&encoder.state, &buffer.state) {
            return Err(Error::WrongDevice);
        }
        if offset % 4 != 0
            || offset
                .checked_add(12)
                .is_none_or(|end| end > buffer.desc.size)
        {
            return Err(Error::InvalidRange);
        }
        super::raster::cube_mirror::refresh(encoder)?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::{
                GlBufferRange, GlDispatchIndirectApi as _, GlDispatchIndirectCommand,
            };
            let result = encoder
                .state
                .with_native("dispatch GL indirect", |provider| {
                    provider.dispatch_indirect(GlDispatchIndirectCommand {
                        range: GlBufferRange {
                            buffer: buffer.id,
                            offset,
                            size: 12,
                        },
                        command_offset: 0,
                    })
                })
                .map_err(native_error);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            result
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        Err(Error::Unsupported("native GL compute context"))
    }

    fn set_push_constants(encoder: &mut GlEncoder, offset: u32, bytes: &[u8]) -> Result<()> {
        super::push::write(
            encoder,
            fluxel_rhi_core::api::StageMask::COMPUTE,
            offset,
            bytes,
            true,
        )
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
    if desc.required_subgroup_size.is_some() {
        return Err(Error::Unsupported("GL compute pipeline subgroup size"));
    }
    // SAFETY: Core keeps both native handles borrowed throughout this
    // synchronous SPI call; only immutable source and ABI metadata is copied.
    let shader = super::shader::materialize_stage(
        device,
        unsafe { desc.compute.module.native_ref() },
        &desc.compute.entry,
        fluxel_rhi_core::api::ShaderStage::Compute,
        &desc.compute.constants,
        None,
    )?;
    let shader = &shader;
    let layout = unsafe { desc.layout.native_ref() }.clone();
    if !std::rc::Rc::ptr_eq(&device.0, &shader.state)
        || !std::rc::Rc::ptr_eq(&device.0, &layout.state)
        || shader.source.as_ref().map_or(true, |source| {
            source.stage != crate::backend::gl::api::GlShaderStage::Compute
                || source.entry_point != desc.compute.entry
        })
    {
        return Err(Error::InvalidState);
    }
    super::push::validate_uniforms(&shader.push_uniforms)?;
    super::pipeline::validate_naga_resource_types(&layout, &[shader])?;
    let native_layout = super::pipeline::with_naga_binding_names(layout.native.clone(), &[shader])?;
    let descriptor = crate::backend::gl::api::GlProgramDescriptor {
        kind: crate::backend::gl::api::GlProgramKind::Compute {
            shader: shader.source.clone().ok_or(Error::InvalidState)?,
        },
        layout: native_layout,
        debug_name: desc.label.clone(),
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlShaderApi as _;
        let pipeline = device
            .0
            .with_native("create GL compute pipeline", |provider| {
                // SAFETY: Core validated cache provenance and retains this
                // borrow for synchronous native pipeline creation.
                let cache = desc
                    .cache
                    .as_ref()
                    .map(|cache| unsafe { cache.native_ref() });
                super::cache::create_program(provider, &descriptor, cache)
            })
            .map(|(program, reflection)| {
                GlComputePipeline::from_inner(GlComputePipelineInner {
                    mip_uniforms: shader.mip_uniforms.clone(),
                    push_uniforms: shader.push_uniforms.clone(),
                    state: device.0.clone(),
                    layout,
                    reflection,
                    program,
                })
            })
            .map_err(native_error)?;
        super::push::validate_active(
            &pipeline.push_uniforms,
            &pipeline.layout.push_ranges,
            |name| {
                device
                    .0
                    .with_native("validate GL compute push constants", |provider| {
                        provider.has_uniform(pipeline.program, name)
                    })
                    .map_err(native_error)
            },
        )?;
        return Ok(pipeline);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (device, descriptor, layout);
        Err(Error::Unsupported("native GL compute context"))
    }
}
