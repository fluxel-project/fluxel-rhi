//! Raster-pipeline creation for the immediate GL backend.
//!
//! Linking and VAO allocation happen on the cold creation path. The resulting
//! pipeline retains only native object identities and immutable fixed state;
//! draw calls are still issued directly by the encoder.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{
    self, Error, PipelineBackend, RasterPipelineDesc, Result, VertexFormat, VertexStepMode,
};

use super::spi::native_error;
use super::types::{GlPipelineLayout, GlRasterPipeline, GlRasterPipelineInner};
use super::{Gl, GlDevice};

impl PipelineBackend for Gl {
    type RasterPipelineFuture<'a> = Ready<Result<GlRasterPipeline>>;

    fn create_raster_pipeline<'a>(
        device: &'a GlDevice,
        desc: &'a RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        ready(create_raster_pipeline(device, desc))
    }
}

fn create_raster_pipeline(
    device: &GlDevice,
    desc: &RasterPipelineDesc<Gl>,
) -> Result<GlRasterPipeline> {
    let fragment = desc.fragment.as_ref().ok_or(Error::Unsupported(
        "GL raster pipelines require a fragment stage",
    ))?;
    let (topology, cull_mode, front_face) =
        crate::backend::gl::translate_raster::lower_primitive(desc.primitive)?;
    let depth_stencil = desc.depth_stencil.map(lower_depth_stencil).transpose()?;
    if desc.multiview.is_some() {
        return Err(Error::Unsupported("GL multiview pipeline lowering"));
    }
    let color_targets = desc
        .color_targets
        .iter()
        .map(|target| match target {
            Some(target) => Ok(crate::backend::gl::api::GlColorTargetState {
                write_mask: target.write_mask,
                blend: target
                    .blend
                    .map(|blend| {
                        Ok(crate::backend::gl::api::GlBlendState {
                            color: crate::backend::gl::translate_raster::lower_blend_component(
                                blend.color,
                            )?,
                            alpha: crate::backend::gl::translate_raster::lower_blend_component(
                                blend.alpha,
                            )?,
                        })
                    })
                    .transpose()?,
            }),
            None => Err(Error::Unsupported("GL sparse color targets")),
        })
        .collect::<Result<Vec<_>>>()?;
    let vertex_layout = lower_vertex_layout(&desc.vertex_buffers)?;
    let multisample = crate::backend::gl::translate_raster::lower_multisample(desc.multisample)?;
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    validate_webgl2_fixed_state(&color_targets, depth_stencil, multisample)?;
    // SAFETY: Core verified shader/layout provenance before calling this SPI.
    // We only read immutable backend source/layout descriptors while those
    // public handles are borrowed for this synchronous creation call.
    let vertex = unsafe { desc.vertex.module.native_ref() };
    let fragment = unsafe { fragment.module.native_ref() };
    // SAFETY: Core retains the pipeline-layout handle for synchronous
    // pipeline creation. Copy its immutable ABI before native linking so the
    // returned pipeline can reject incompatible bind groups later.
    let layout = unsafe { desc.layout.native_ref() }.clone();
    let native_layout = with_naga_binding_names(layout.native.clone(), &[vertex, fragment])?;
    let descriptor = crate::backend::gl::api::GlProgramDescriptor {
        kind: crate::backend::gl::api::GlProgramKind::Raster {
            vertex: vertex.source.clone(),
            fragment: fragment.source.clone(),
        },
        layout: native_layout,
        debug_name: desc.label.clone(),
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlShaderApi as _, GlVertexApi as _};
        let result = device
            .0
            .with_native("create GL raster pipeline", |provider| {
                let (program, reflection) = provider.create_program(&descriptor)?;
                match provider.create_vertex_array(&vertex_layout) {
                    Ok(vertex_array) => Ok((program, vertex_array, reflection)),
                    Err(error) => {
                        let _ = provider.destroy_program(program);
                        Err(error)
                    }
                }
            });
        return result
            .map(|(program, vertex_array, reflection)| {
                GlRasterPipeline::from_inner(GlRasterPipelineInner {
                    state: device.0.clone(),
                    layout,
                    program,
                    vertex_array,
                    reflection,
                    vertex_layout: vertex_layout.clone(),
                    raster: crate::backend::gl::api::GlRasterState {
                        topology,
                        cull_mode,
                        front_face,
                        depth_stencil,
                        color_targets,
                        multisample,
                        // The active pass supplies its validated drawable size
                        // at `set_pipeline`; creation cannot guess it.
                        viewport: crate::backend::gl::api::GlViewport {
                            x: 0,
                            y: 0,
                            width: 0,
                            height: 0,
                            min_depth: 0.0f32.to_bits(),
                            max_depth: 1.0f32.to_bits(),
                        },
                        scissor: None,
                        blend_constant: [0; 4],
                    },
                })
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            // WebGL owns program and VAO allocation on the browser thread.
            // They are created here, on the cold path, and are subsequently
            // selected by immediate raster verbs without a command cache.
            let (program, reflection) = device.0.owner.create_raster_program(&descriptor)?;
            let vertex_array = device.0.owner.create_vertex_array()?;
            return Ok(GlRasterPipeline::from_inner(GlRasterPipelineInner {
                state: device.0.clone(),
                layout,
                browser_program: program,
                browser_vertex_array: vertex_array,
                reflection,
                vertex_layout: vertex_layout.clone(),
                raster: crate::backend::gl::api::GlRasterState {
                    topology,
                    cull_mode,
                    front_face,
                    depth_stencil,
                    color_targets,
                    multisample,
                    viewport: crate::backend::gl::api::GlViewport {
                        x: 0,
                        y: 0,
                        width: 0,
                        height: 0,
                        min_depth: 0.0f32.to_bits(),
                        max_depth: 1.0f32.to_bits(),
                    },
                    scissor: None,
                    blend_constant: [0; 4],
                },
            }));
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (
                device,
                descriptor,
                vertex_layout,
                topology,
                cull_mode,
                front_face,
                color_targets,
                multisample,
            );
            Err(Error::Unsupported("GL platform context"))
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
fn validate_webgl2_fixed_state(
    colors: &[crate::backend::gl::api::GlColorTargetState],
    depth: Option<crate::backend::gl::api::GlDepthStencilState>,
    multisample: crate::backend::gl::api::GlMultisampleState,
) -> Result<()> {
    // This surface backend has one color attachment. WebGL2 cannot represent
    // independent blend/write state without an optional extension, so reject
    // the wider Core shape before the program or VAO is allocated.
    if colors.len() != 1 || multisample.sample_mask != u32::MAX {
        return Err(Error::Unsupported("WebGL2 fixed raster state"));
    }
    if let Some(depth) = depth {
        let disabled_face = |face: crate::backend::gl::api::GlStencilFaceState| {
            matches!(
                face.compare,
                crate::backend::gl::api::GlDepthCompareFunction::Always
            ) && matches!(
                face.fail_op,
                crate::backend::gl::api::GlStencilOperation::Keep
            ) && matches!(
                face.depth_fail_op,
                crate::backend::gl::api::GlStencilOperation::Keep
            ) && matches!(
                face.pass_op,
                crate::backend::gl::api::GlStencilOperation::Keep
            ) && face.read_mask == 0
                && face.write_mask == 0
        };
        if !disabled_face(depth.stencil_front)
            || !disabled_face(depth.stencil_back)
            || depth.depth_bias.iter().any(|value| *value != 0)
        {
            return Err(Error::Unsupported("WebGL2 stencil or depth bias state"));
        }
    }
    Ok(())
}

/// Replaces placeholder ABI names with the names Naga actually emitted for
/// this pair of entry points. Native program reflection needs those names;
/// Naga's explicit storage-image unit is carried beside them for bind groups.
pub(crate) fn with_naga_binding_names(
    mut layout: crate::backend::gl::api::GlPipelineLayout,
    shaders: &[&super::types::GlShader],
) -> Result<crate::backend::gl::api::GlPipelineLayout> {
    for binding in &mut layout.bindings {
        let mut names = Vec::new();
        let mut image_units = Vec::new();
        for shader in shaders {
            image_units.extend(
                shader
                    .naga_bindings
                    .iter()
                    .filter(|value| {
                        value.group == binding.location.group
                            && value.binding == binding.location.binding
                    })
                    .filter_map(|value| value.image_unit),
            );
            names.extend(
                shader
                    .naga_bindings
                    .iter()
                    .filter(|value| {
                        value.group == binding.location.group
                            && value.binding == binding.location.binding
                    })
                    .map(|value| value.name.clone()),
            );
        }
        if !names.is_empty() {
            names.sort();
            names.dedup();
            binding.names = names;
        }
        if !image_units.is_empty() {
            image_units.sort_unstable();
            image_units.dedup();
            if image_units.len() != 1 {
                return Err(Error::InvalidState);
            }
            binding.image_unit = image_units.first().copied();
        }
    }
    Ok(layout)
}

fn lower_vertex_layout(
    values: &[api::VertexBufferLayout],
) -> Result<crate::backend::gl::api::GlVertexLayout> {
    let buffers = values
        .iter()
        .enumerate()
        .map(|(slot, value)| {
            Ok(crate::backend::gl::api::GlVertexBufferLayout {
                slot: u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                stride: u32::try_from(value.stride).map_err(|_| Error::InvalidRange)?,
                step_mode: match value.step_mode {
                    VertexStepMode::Vertex => crate::backend::gl::api::GlVertexStepMode::Vertex,
                    VertexStepMode::Instance => crate::backend::gl::api::GlVertexStepMode::Instance,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let attributes = values
        .iter()
        .enumerate()
        .flat_map(|(slot, buffer)| {
            buffer
                .attributes
                .iter()
                .map(move |attribute| (slot, attribute))
        })
        .map(|(slot, attribute)| {
            Ok(crate::backend::gl::api::GlVertexAttribute {
                location: attribute.location,
                buffer_slot: u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                format: lower_vertex_format(attribute.format)?,
                offset: u32::try_from(attribute.offset).map_err(|_| Error::InvalidRange)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(crate::backend::gl::api::GlVertexLayout {
        buffers,
        attributes,
    })
}

/// Lowers depth state without inventing a format conversion. Attachment
/// validation later checks that this declared format matches the pass view.
fn lower_depth_stencil(
    value: api::DepthStencilState,
) -> Result<crate::backend::gl::api::GlDepthStencilState> {
    if !matches!(value.format, api::TextureFormat::Depth32Float)
        || !value.depth_bias_slope_scale.is_finite()
        || !value.depth_bias_clamp.is_finite()
    {
        return Err(Error::Unsupported("GL depth-stencil state"));
    }
    let face = |face: api::StencilFaceState| crate::backend::gl::api::GlStencilFaceState {
        compare: crate::backend::gl::translate_raster::lower_compare(face.compare),
        fail_op: lower_stencil_op(face.fail),
        depth_fail_op: lower_stencil_op(face.depth_fail),
        pass_op: lower_stencil_op(face.pass),
        read_mask: value.stencil_read_mask,
        write_mask: value.stencil_write_mask,
    };
    Ok(crate::backend::gl::api::GlDepthStencilState {
        depth_write_enabled: value.depth_write_enabled,
        depth_compare: crate::backend::gl::translate_raster::lower_compare(value.depth_compare),
        // Shared native/browser ABI: polygon offset factor, then units, then
        // clamp. Units are a float in GL even though Core exposes an integer.
        depth_bias: [
            value.depth_bias_slope_scale.to_bits(),
            (value.depth_bias_constant as f32).to_bits(),
            value.depth_bias_clamp.to_bits(),
        ],
        stencil_front: face(value.stencil_front),
        stencil_back: face(value.stencil_back),
        stencil_reference: 0,
    })
}

fn lower_stencil_op(value: api::StencilOperation) -> crate::backend::gl::api::GlStencilOperation {
    use crate::backend::gl::api::GlStencilOperation as Gl;
    match value {
        api::StencilOperation::Keep => Gl::Keep,
        api::StencilOperation::Zero => Gl::Zero,
        api::StencilOperation::Replace => Gl::Replace,
        api::StencilOperation::IncrementClamp => Gl::IncrementClamp,
        api::StencilOperation::DecrementClamp => Gl::DecrementClamp,
        api::StencilOperation::Invert => Gl::Invert,
        api::StencilOperation::IncrementWrap => Gl::IncrementWrap,
        api::StencilOperation::DecrementWrap => Gl::DecrementWrap,
    }
}

fn lower_vertex_format(value: VertexFormat) -> Result<crate::backend::gl::api::GlVertexFormat> {
    use crate::backend::gl::api::GlVertexFormat as Gl;
    Ok(match value {
        VertexFormat::Float32(1) => Gl::Float32,
        VertexFormat::Float32(2) => Gl::Float32x2,
        VertexFormat::Float32(3) => Gl::Float32x3,
        VertexFormat::Float32(4) => Gl::Float32x4,
        VertexFormat::Uint32(1) => Gl::Uint32,
        VertexFormat::Uint32(2) => Gl::Uint32x2,
        VertexFormat::Uint32(3) => Gl::Uint32x3,
        VertexFormat::Uint32(4) => Gl::Uint32x4,
        VertexFormat::Sint32(1) => Gl::Sint32,
        VertexFormat::Sint32(2) => Gl::Sint32x2,
        VertexFormat::Sint32(3) => Gl::Sint32x3,
        VertexFormat::Sint32(4) => Gl::Sint32x4,
        VertexFormat::Unorm8(2) => Gl::Unorm8x2,
        VertexFormat::Unorm8(4) => Gl::Unorm8x4,
        _ => return Err(Error::Unsupported("GL vertex format")),
    })
}

trait PipelineLayoutNativeRef {
    fn native_ref_safe(&self) -> &GlPipelineLayout;
}
impl PipelineLayoutNativeRef for api::PipelineLayout<Gl> {
    fn native_ref_safe(&self) -> &GlPipelineLayout {
        // SAFETY: this private backend method is called only while Core owns
        // the pipeline-layout handle for a pipeline being created on its device.
        unsafe { self.native_ref() }
    }
}
