//! Metal raster pipeline creation and immutable attachment state.
//!
//! Resource bindings and vertex input occupy disjoint native buffer slots.
//! Pipeline creation preserves color attachment slots, including holes, and
//! command encoding checks the matching render pass signature.

use fluxel_rhi_core::api::{self, PipelineBackend, RasterPipelineDesc};
use std::{future::Future, pin::Pin};

use super::{Metal, MetalDevice, binding_abi};

#[derive(Clone)]
pub struct MetalRasterPipeline {
    #[cfg(target_vendor = "apple")]
    pub(crate) state: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
    #[cfg(target_vendor = "apple")]
    pub(crate) bindings: Vec<super::shader::variant::StageBindings>,
    #[cfg(target_vendor = "apple")]
    pub(crate) layout: super::bindings::MetalPipelineLayout,
    #[cfg(target_vendor = "apple")]
    pub(crate) vertex_buffer_base: usize,
    #[cfg(target_vendor = "apple")]
    pub(crate) view_count: Option<u32>,
    #[cfg(target_vendor = "apple")]
    pub(crate) primitive_type: objc2_metal::MTLPrimitiveType,
    #[cfg(target_vendor = "apple")]
    pub(crate) polygon_mode: api::PolygonMode,
    #[cfg(target_vendor = "apple")]
    pub(crate) unclipped_depth: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) strip_index_format: Option<api::IndexFormat>,
    #[cfg(target_vendor = "apple")]
    pub(crate) cull_mode: api::CullMode,
    #[cfg(target_vendor = "apple")]
    pub(crate) front_face: api::FrontFace,
    #[cfg(target_vendor = "apple")]
    pub(crate) depth_state: Option<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>,
    >,
    #[cfg(target_vendor = "apple")]
    pub(crate) depth_bias: (i32, f32, f32),
    #[cfg(target_vendor = "apple")]
    pub(crate) writes_depth: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) writes_stencil: bool,
    #[cfg(target_vendor = "apple")]
    pub(crate) signature: (
        usize,
        Vec<Option<objc2_metal::MTLPixelFormat>>,
        Option<objc2_metal::MTLPixelFormat>,
    ),
}

#[derive(Clone)]
pub struct MetalComputePipeline {
    #[cfg(target_vendor = "apple")]
    pub(crate) state: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    #[cfg(target_vendor = "apple")]
    pub(crate) bindings: Vec<super::shader::variant::StageBindings>,
    #[cfg(target_vendor = "apple")]
    pub(crate) layout: super::bindings::MetalPipelineLayout,
    #[cfg(target_vendor = "apple")]
    pub(crate) workgroup_size: [u32; 3],
}

impl PipelineBackend for Metal {
    type RasterPipelineFuture<'a> = Pin<Box<dyn Future<Output = api::Result<MetalRasterPipeline>> + 'a>>;

    fn create_raster_pipeline<'a>(
        device: &'a MetalDevice,
        desc: &'a RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        Box::pin(create_raster_pipeline(device, desc))
    }
}

async fn create_raster_pipeline(
    device: &MetalDevice,
    desc: &RasterPipelineDesc<Metal>,
) -> api::Result<MetalRasterPipeline> {
    validate_desc(desc)?;
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::{NSObjectProtocol as _, NSString};
        use objc2_metal::{
            MTLDepthStencilDescriptor, MTLDevice as _, MTLFunction as _, MTLFunctionType,
            MTLRenderPipelineDescriptor, MTLStencilDescriptor,
            MTLVertexDescriptor,
        };
        if desc.depth_stencil.is_some_and(|depth| {
            matches!(
                depth.format,
                api::TextureFormat::Depth32FloatStencil8 | api::TextureFormat::Depth24PlusStencil8
            )
        }) && !super::texture::supports_depth32_stencil(device)
        {
            return Err(api::Error::Unsupported(
                "Metal device family does not support Depth32FloatStencil8",
            ));
        }
        if !device
            ._state
            ._device
            .supportsTextureSampleCount(desc.multisample.count as usize)
        {
            return Err(api::Error::Unsupported("Metal raster sample count"));
        }

        // SAFETY: Core verified shader and layout device identities.
        let layout = unsafe { desc.layout.native_ref().clone() };
        let vertex = unsafe { desc.vertex.module.native_ref() };
        let (vertex_function, vertex_bindings, _) = super::shader::variant::native_function_async(
            device,
            vertex,
            desc.vertex.entry.clone(),
            api::ShaderStage::Vertex,
            desc.vertex.constants.clone(),
            layout.clone(),
            desc.vertex_buffers.len(),
            desc.multiview.is_some(),
        ).await?;
        if vertex_function.functionType() != MTLFunctionType::Vertex {
            return Err(api::Error::Shader(
                "Metal entry is not a vertex function".into(),
            ));
        }
        let mut stage_bindings = vec![vertex_bindings];
        let fragment_function = if let Some(stage) = &desc.fragment {
            // SAFETY: Core verified the fragment shader belongs to this device.
            let shader = unsafe { stage.module.native_ref() };
            let (function, binding, _) = super::shader::variant::native_function_async(
                device,
                shader,
                stage.entry.clone(),
                api::ShaderStage::Fragment,
                stage.constants.clone(),
                layout.clone(),
                0,
                desc.multiview.is_some(),
            ).await?;
            if function.functionType() != MTLFunctionType::Fragment {
                return Err(api::Error::Shader(
                    "Metal entry is not a fragment function".into(),
                ));
            }
            stage_bindings.push(binding);
            Some(function)
        } else {
            None
        };

        let vertex_buffer_base = 31usize
            .checked_sub(desc.vertex_buffers.len())
            .ok_or(api::Error::InvalidRange)?;
        let vertex_descriptor = MTLVertexDescriptor::new();
        let attributes = vertex_descriptor.attributes();
        let layouts = vertex_descriptor.layouts();
        for (buffer_index, layout) in desc.vertex_buffers.iter().enumerate() {
            // Metal's fixed vertex descriptor path is kept within the
            // documented practical per-vertex stride range.
            if layout.stride == 0 || layout.stride > 2048 {
                return Err(api::Error::InvalidRange);
            }
            let native_index = vertex_buffer_base + buffer_index;
            // SAFETY: allocation places vertex buffers within Metal slots 0–30.
            let native_layout = unsafe { layouts.objectAtIndexedSubscript(native_index) };
            // SAFETY: nonzero stride was checked and fits NSUInteger on this target.
            unsafe { native_layout.setStride(layout.stride as usize) };
            native_layout.setStepFunction(match layout.step_mode {
                api::VertexStepMode::Vertex => objc2_metal::MTLVertexStepFunction::PerVertex,
                api::VertexStepMode::Instance => objc2_metal::MTLVertexStepFunction::PerInstance,
            });
            // SAFETY: step rate one is valid for both supported Metal step functions.
            unsafe { native_layout.setStepRate(1) };
            for attribute in &layout.attributes {
                let (format, byte_size) = vertex_format(attribute.format)?;
                let end = attribute
                    .offset
                    .checked_add(byte_size)
                    .ok_or(api::Error::InvalidRange)?;
                if end > layout.stride || attribute.offset > usize::MAX as u64 {
                    return Err(api::Error::InvalidRange);
                }
                // SAFETY: validate_desc bounds locations to Metal's fixed descriptor range.
                let native_attribute =
                    unsafe { attributes.objectAtIndexedSubscript(attribute.location as usize) };
                native_attribute.setFormat(format);
                // SAFETY: offset is checked to fit NSUInteger and lie within the declared stride.
                unsafe { native_attribute.setOffset(attribute.offset as usize) };
                // SAFETY: the public layout slot was translated through the
                // fixed, bounded native vertex-buffer partition above.
                unsafe { native_attribute.setBufferIndex(native_index) };
            }
        }

        let descriptor = MTLRenderPipelineDescriptor::new();
        if desc.multiview.is_some() {
            let native = &device._state._device;
            if !native.respondsToSelector(objc2::sel!(supportsVertexAmplificationCount:))
                || !descriptor.respondsToSelector(objc2::sel!(setMaxVertexAmplificationCount:))
            {
                return Err(api::Error::Unsupported("Metal vertex amplification count"));
            }
            let count = [8usize, 4, 2, 1]
                .into_iter()
                .find(|count| native.supportsVertexAmplificationCount(*count))
                .ok_or(api::Error::Unsupported("Metal vertex amplification count"))?;
            // SAFETY: selector and device support are checked above.
            unsafe { descriptor.setMaxVertexAmplificationCount(count) };
        }
        if device._state.indirect_count_supported {
            descriptor.setSupportIndirectCommandBuffers(true);
        }
        descriptor.setVertexFunction(Some(&vertex_function));
        descriptor.setFragmentFunction(fragment_function.as_deref());
        descriptor.setVertexDescriptor(Some(&vertex_descriptor));
        descriptor.setRasterSampleCount(desc.multisample.count as usize);
        descriptor.setAlphaToCoverageEnabled(desc.multisample.alpha_to_coverage_enabled);
        let depth_format = desc
            .depth_stencil
            .map(|depth| {
                super::texture::metal_format(depth.format)
                    .ok_or(api::Error::Unsupported("Metal depth format"))
            })
            .transpose()?;
        let depth_state = if let Some(depth) = desc.depth_stencil {
            let format = depth_format.ok_or(api::Error::Unsupported("Metal depth format"))?;
            if depth.format != api::TextureFormat::Stencil8 {
                descriptor.setDepthAttachmentPixelFormat(format);
            }
            let native = MTLDepthStencilDescriptor::new();
            native.setDepthCompareFunction(compare_function(depth.depth_compare));
            native.setDepthWriteEnabled(depth.depth_write_enabled);
            if matches!(
                depth.format,
                api::TextureFormat::Depth32FloatStencil8
                    | api::TextureFormat::Depth24PlusStencil8
                    | api::TextureFormat::Stencil8
            ) {
                descriptor.setStencilAttachmentPixelFormat(format);
                let front = MTLStencilDescriptor::new();
                configure_stencil_descriptor(
                    &front,
                    depth.stencil_front,
                    depth.stencil_read_mask,
                    depth.stencil_write_mask,
                );
                let back = MTLStencilDescriptor::new();
                configure_stencil_descriptor(
                    &back,
                    depth.stencil_back,
                    depth.stencil_read_mask,
                    depth.stencil_write_mask,
                );
                native.setFrontFaceStencil(Some(&front));
                native.setBackFaceStencil(Some(&back));
            }
            Some(
                device
                    ._state
                    ._device
                    .newDepthStencilStateWithDescriptor(&native)
                    .ok_or(api::Error::OutOfMemory)?,
            )
        } else {
            None
        };
        let colors = descriptor.colorAttachments();
        let mut color_formats = Vec::with_capacity(desc.color_targets.len());
        for (index, target) in desc.color_targets.iter().enumerate() {
            let Some(target) = target else {
                color_formats.push(None);
                continue;
            };
            // SAFETY: validate_desc limits the color attachment index to Metal's eight slots.
            let color = unsafe { colors.objectAtIndexedSubscript(index) };
            if target.blend.is_some() && !super::texture::is_blendable(target.format) {
                return Err(api::Error::Unsupported(
                    "Metal format does not support blending",
                ));
            }
            let format = super::texture::metal_format(target.format)
                .ok_or(api::Error::Unsupported("Metal color target format"))?;
            color.setPixelFormat(format);
            color.setWriteMask(write_mask(target.write_mask)?);
            if let Some(blend) = target.blend {
                color.setBlendingEnabled(true);
                color.setSourceRGBBlendFactor(blend_factor(blend.color.src_factor, false)?);
                color.setDestinationRGBBlendFactor(blend_factor(blend.color.dst_factor, false)?);
                color.setRgbBlendOperation(blend_operation(blend.color.operation));
                color.setSourceAlphaBlendFactor(blend_factor(blend.alpha.src_factor, true)?);
                color.setDestinationAlphaBlendFactor(blend_factor(blend.alpha.dst_factor, true)?);
                color.setAlphaBlendOperation(blend_operation(blend.alpha.operation));
            } else {
                color.setBlendingEnabled(false);
            }
            color_formats.push(Some(format));
        }
        if let Some(label) = desc.label.as_deref() {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        let archive_lease = if let Some(cache) = &desc.cache {
            // SAFETY: Core verified cache provenance before pipeline creation.
            Some(unsafe { cache.native_ref() }.prepare_render(&descriptor).await?)
        } else {
            None
        };
        let (future, completer) = super::cold::pending();
        let inputs = std::cell::RefCell::new(Some((
            archive_lease,
            descriptor.clone(),
            vertex_function.clone(),
            fragment_function.clone(),
        )));
        let callback = block2::RcBlock::new(move |
            state: *mut objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
            reflection: *mut objc2_metal::MTLRenderPipelineReflection,
            error: *mut objc2_foundation::NSError,
        | {
            let result = if !error.is_null() {
                // SAFETY: Metal supplies a valid NSError for failed compilation.
                Err(api::Error::Backend(format!("Metal raster pipeline: {:?}", unsafe { &*error })))
            } else {
                // SAFETY: successful completion lends valid native objects until
                // this block returns; retain them for the future's consumer.
                let state = unsafe { objc2::rc::Retained::retain(state) }
                    .ok_or(api::Error::InvalidState);
                let reflection = unsafe { objc2::rc::Retained::retain(reflection) }
                    .ok_or(api::Error::InvalidState);
                state.and_then(|state| reflection.map(|reflection| (state, reflection)))
            };
            completer.finish(result);
            drop(inputs.borrow_mut().take());
        });
        // SAFETY: Metal copies the block; `inputs` keeps every descriptor
        // dependency and the cache read lease alive through completion.
        unsafe {
            device._state._device.newRenderPipelineStateWithDescriptor_options_completionHandler(
                &descriptor,
                objc2_metal::MTLPipelineOption::ArgumentInfo | objc2_metal::MTLPipelineOption::BufferTypeInfo,
                block2::RcBlock::as_ptr(&callback).cast(),
            );
        }
        let (state, reflection) = future.await?;
        for stage in &mut stage_bindings {
            match stage.stage {
                api::ShaderStage::Vertex
                    if reflection.respondsToSelector(objc2::sel!(vertexBindings)) =>
                {
                    super::shader::variant::apply_reflected_alignment(
                        stage,
                        &layout,
                        Some(&reflection.vertexBindings()),
                        None,
                    )?
                }
                api::ShaderStage::Fragment
                    if reflection.respondsToSelector(objc2::sel!(fragmentBindings)) =>
                {
                    super::shader::variant::apply_reflected_alignment(
                        stage,
                        &layout,
                        Some(&reflection.fragmentBindings()),
                        None,
                    )?
                }
                api::ShaderStage::Vertex => {
                    #[allow(deprecated)]
                    super::shader::variant::apply_reflected_alignment(
                        stage,
                        &layout,
                        None,
                        reflection.vertexArguments().as_deref(),
                    )?;
                }
                api::ShaderStage::Fragment => {
                    #[allow(deprecated)]
                    super::shader::variant::apply_reflected_alignment(
                        stage,
                        &layout,
                        None,
                        reflection.fragmentArguments().as_deref(),
                    )?;
                }
                _ => return Err(api::Error::InvalidState),
            }
        }
        if stage_bindings.iter().any(|stage| {
            stage.stage == api::ShaderStage::Vertex
                && stage.map.entries.iter().any(|(_, _, slot)| {
                    matches!(
                        slot.kind,
                        binding_abi::SlotKind::Buffer | binding_abi::SlotKind::ArgumentBuffer
                    ) && slot.index as usize >= vertex_buffer_base
                })
        }) {
            return Err(api::Error::Unsupported(
                "Metal native shader and vertex buffer slot collision",
            ));
        }
        Ok(MetalRasterPipeline {
            state,
            bindings: stage_bindings,
            layout,
            vertex_buffer_base,
            view_count: desc.multiview,
            primitive_type: primitive_type(desc.primitive.topology),
            polygon_mode: desc.primitive.polygon_mode,
            unclipped_depth: desc.primitive.unclipped_depth,
            strip_index_format: desc.primitive.strip_index_format,
            cull_mode: desc.primitive.cull_mode,
            front_face: desc.primitive.front_face,
            depth_state,
            writes_depth: desc.depth_stencil.is_some_and(|d| d.depth_write_enabled),
            writes_stencil: desc.depth_stencil.is_some_and(stencil_writes),
            depth_bias: desc.depth_stencil.map_or((0, 0.0, 0.0), |depth| {
                (
                    depth.depth_bias_constant,
                    depth.depth_bias_slope_scale,
                    depth.depth_bias_clamp,
                )
            }),
            signature: (desc.multisample.count as usize, color_formats, depth_format),
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

fn validate_desc(desc: &RasterPipelineDesc<Metal>) -> api::Result<()> {
    if matches!(desc.primitive.polygon_mode, api::PolygonMode::Point)
        || desc.primitive.conservative
        || desc
            .depth_stencil
            .is_some_and(|depth| !supported_depth_state(depth))
        || !supported_sample_count(desc.multisample.count)
        || !full_sample_mask(desc.multisample.mask, desc.multisample.count)
        || desc.color_targets.len() > 8
        || desc
            .color_targets
            .iter()
            .flatten()
            .any(|target| target.write_mask & !0x0f != 0 || !supported_color_format(target.format))
        || desc.vertex_buffers.len() > 31
        || desc
            .vertex_buffers
            .iter()
            .flat_map(|layout| &layout.attributes)
            .any(|attribute| attribute.location >= binding_abi::MAX_VERTEX_ATTRIBUTES)
    {
        return Err(api::Error::Unsupported("Metal raster pipeline descriptor"));
    }
    Ok(())
}

pub(super) fn supported_sample_count(count: u32) -> bool {
    matches!(count, 1 | 2 | 4 | 8)
}

pub(super) fn full_sample_mask(mask: u64, count: u32) -> bool {
    count != 0
        && count <= 64
        && (count == 64 && mask == u64::MAX
            || count < 64 && mask & ((1u64 << count) - 1) == (1u64 << count) - 1)
}

pub(super) fn supported_depth_state(depth: api::DepthStencilState) -> bool {
    matches!(
        depth.format,
        api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32FloatStencil8
            | api::TextureFormat::Stencil8
    ) && (matches!(
        depth.format,
        api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32FloatStencil8
            | api::TextureFormat::Stencil8
    ) || (depth.stencil_read_mask == 0
        && depth.stencil_write_mask == 0
        && inert_stencil_face(depth.stencil_front)
        && inert_stencil_face(depth.stencil_back)))
        && depth.depth_bias_clamp.is_finite()
        && depth.depth_bias_slope_scale.is_finite()
}

pub(super) fn stencil_writes(depth: api::DepthStencilState) -> bool {
    fn writes(face: api::StencilFaceState) -> bool {
        !matches!(face.fail, api::StencilOperation::Keep)
            || !matches!(face.depth_fail, api::StencilOperation::Keep)
            || !matches!(face.pass, api::StencilOperation::Keep)
    }
    depth.stencil_write_mask != 0 && (writes(depth.stencil_front) || writes(depth.stencil_back))
}

fn inert_stencil_face(face: api::StencilFaceState) -> bool {
    matches!(face.compare, api::CompareFunction::Always)
        && matches!(face.fail, api::StencilOperation::Keep)
        && matches!(face.depth_fail, api::StencilOperation::Keep)
        && matches!(face.pass, api::StencilOperation::Keep)
}

#[cfg(target_vendor = "apple")]
pub(super) fn primitive_type(value: api::PrimitiveTopology) -> objc2_metal::MTLPrimitiveType {
    use api::PrimitiveTopology as P;
    use objc2_metal::MTLPrimitiveType as M;
    match value {
        P::PointList => M::Point,
        P::LineList => M::Line,
        P::LineStrip => M::LineStrip,
        P::TriangleList => M::Triangle,
        P::TriangleStrip => M::TriangleStrip,
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn compare_function(value: api::CompareFunction) -> objc2_metal::MTLCompareFunction {
    use api::CompareFunction as F;
    use objc2_metal::MTLCompareFunction as M;
    match value {
        F::Never => M::Never,
        F::Less => M::Less,
        F::Equal => M::Equal,
        F::LessEqual => M::LessEqual,
        F::Greater => M::Greater,
        F::NotEqual => M::NotEqual,
        F::GreaterEqual => M::GreaterEqual,
        F::Always => M::Always,
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn configure_stencil_descriptor(
    descriptor: &objc2_metal::MTLStencilDescriptor,
    face: api::StencilFaceState,
    read_mask: u32,
    write_mask: u32,
) {
    descriptor.setStencilCompareFunction(compare_function(face.compare));
    descriptor.setStencilFailureOperation(stencil_operation(face.fail));
    descriptor.setDepthFailureOperation(stencil_operation(face.depth_fail));
    descriptor.setDepthStencilPassOperation(stencil_operation(face.pass));
    descriptor.setReadMask(read_mask);
    descriptor.setWriteMask(write_mask);
}

#[cfg(target_vendor = "apple")]
fn stencil_operation(value: api::StencilOperation) -> objc2_metal::MTLStencilOperation {
    use api::StencilOperation as S;
    use objc2_metal::MTLStencilOperation as M;
    match value {
        S::Keep => M::Keep,
        S::Zero => M::Zero,
        S::Replace => M::Replace,
        S::Invert => M::Invert,
        S::IncrementClamp => M::IncrementClamp,
        S::DecrementClamp => M::DecrementClamp,
        S::IncrementWrap => M::IncrementWrap,
        S::DecrementWrap => M::DecrementWrap,
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn write_mask(mask: u8) -> api::Result<objc2_metal::MTLColorWriteMask> {
    use objc2_metal::MTLColorWriteMask as M;
    if mask & !0x0f != 0 {
        return Err(api::Error::InvalidRange);
    }
    let mut result = M::None;
    if mask & 0x01 != 0 {
        result |= M::Red;
    }
    if mask & 0x02 != 0 {
        result |= M::Green;
    }
    if mask & 0x04 != 0 {
        result |= M::Blue;
    }
    if mask & 0x08 != 0 {
        result |= M::Alpha;
    }
    Ok(result)
}

#[cfg(target_vendor = "apple")]
pub(super) fn blend_factor(
    value: api::BlendFactor,
    alpha: bool,
) -> api::Result<objc2_metal::MTLBlendFactor> {
    use api::BlendFactor as F;
    use objc2_metal::MTLBlendFactor as M;
    Ok(match value {
        F::Zero => M::Zero,
        F::One => M::One,
        F::SrcColor => M::SourceColor,
        F::OneMinusSrcColor => M::OneMinusSourceColor,
        F::DstColor => M::DestinationColor,
        F::OneMinusDstColor => M::OneMinusDestinationColor,
        F::SrcAlpha => M::SourceAlpha,
        F::OneMinusSrcAlpha => M::OneMinusSourceAlpha,
        F::DstAlpha => M::DestinationAlpha,
        F::OneMinusDstAlpha => M::OneMinusDestinationAlpha,
        F::Constant => {
            if alpha {
                M::BlendAlpha
            } else {
                M::BlendColor
            }
        }
        F::OneMinusConstant => {
            if alpha {
                M::OneMinusBlendAlpha
            } else {
                M::OneMinusBlendColor
            }
        }
        F::SrcAlphaSaturated => M::SourceAlphaSaturated,
        F::Src1 => {
            if alpha {
                M::Source1Alpha
            } else {
                M::Source1Color
            }
        }
        F::OneMinusSrc1 => {
            if alpha {
                M::OneMinusSource1Alpha
            } else {
                M::OneMinusSource1Color
            }
        }
    })
}

#[cfg(target_vendor = "apple")]
pub(super) fn blend_operation(value: api::BlendOperation) -> objc2_metal::MTLBlendOperation {
    use api::BlendOperation as F;
    use objc2_metal::MTLBlendOperation as M;
    match value {
        F::Add => M::Add,
        F::Subtract => M::Subtract,
        F::ReverseSubtract => M::ReverseSubtract,
        F::Min => M::Min,
        F::Max => M::Max,
    }
}

pub(super) fn supported_color_format(format: api::TextureFormat) -> bool {
    super::texture::color_attachment_format(format)
}

#[cfg(target_vendor = "apple")]
fn vertex_format(format: api::VertexFormat) -> api::Result<(objc2_metal::MTLVertexFormat, u64)> {
    use objc2_metal::MTLVertexFormat as Mtl;
    let (format, size) = match format {
        api::VertexFormat::Float32(1) => (Mtl::Float, 4),
        api::VertexFormat::Float32(2) => (Mtl::Float2, 8),
        api::VertexFormat::Float32(3) => (Mtl::Float3, 12),
        api::VertexFormat::Float32(4) => (Mtl::Float4, 16),
        api::VertexFormat::Uint32(1) => (Mtl::UInt, 4),
        api::VertexFormat::Uint32(2) => (Mtl::UInt2, 8),
        api::VertexFormat::Uint32(3) => (Mtl::UInt3, 12),
        api::VertexFormat::Uint32(4) => (Mtl::UInt4, 16),
        api::VertexFormat::Sint32(1) => (Mtl::Int, 4),
        api::VertexFormat::Sint32(2) => (Mtl::Int2, 8),
        api::VertexFormat::Sint32(3) => (Mtl::Int3, 12),
        api::VertexFormat::Sint32(4) => (Mtl::Int4, 16),
        api::VertexFormat::Float16(1) => (Mtl::Half, 2),
        api::VertexFormat::Float16(2) => (Mtl::Half2, 4),
        api::VertexFormat::Float16(3) => (Mtl::Half3, 6),
        api::VertexFormat::Float16(4) => (Mtl::Half4, 8),
        api::VertexFormat::Uint16(1) => (Mtl::UShort, 2),
        api::VertexFormat::Uint16(2) => (Mtl::UShort2, 4),
        api::VertexFormat::Uint16(3) => (Mtl::UShort3, 6),
        api::VertexFormat::Uint16(4) => (Mtl::UShort4, 8),
        api::VertexFormat::Sint16(1) => (Mtl::Short, 2),
        api::VertexFormat::Sint16(2) => (Mtl::Short2, 4),
        api::VertexFormat::Sint16(3) => (Mtl::Short3, 6),
        api::VertexFormat::Sint16(4) => (Mtl::Short4, 8),
        api::VertexFormat::Uint8(1) => (Mtl::UChar, 1),
        api::VertexFormat::Uint8(2) => (Mtl::UChar2, 2),
        api::VertexFormat::Uint8(3) => (Mtl::UChar3, 3),
        api::VertexFormat::Uint8(4) => (Mtl::UChar4, 4),
        api::VertexFormat::Sint8(1) => (Mtl::Char, 1),
        api::VertexFormat::Sint8(2) => (Mtl::Char2, 2),
        api::VertexFormat::Sint8(3) => (Mtl::Char3, 3),
        api::VertexFormat::Sint8(4) => (Mtl::Char4, 4),
        api::VertexFormat::Unorm8(1) => (Mtl::UCharNormalized, 1),
        api::VertexFormat::Unorm8(2) => (Mtl::UChar2Normalized, 2),
        api::VertexFormat::Unorm8(3) => (Mtl::UChar3Normalized, 3),
        api::VertexFormat::Unorm8(4) => (Mtl::UChar4Normalized, 4),
        api::VertexFormat::Snorm8(1) => (Mtl::CharNormalized, 1),
        api::VertexFormat::Snorm8(2) => (Mtl::Char2Normalized, 2),
        api::VertexFormat::Snorm8(3) => (Mtl::Char3Normalized, 3),
        api::VertexFormat::Snorm8(4) => (Mtl::Char4Normalized, 4),
        api::VertexFormat::Unorm16(1) => (Mtl::UShortNormalized, 2),
        api::VertexFormat::Unorm16(2) => (Mtl::UShort2Normalized, 4),
        api::VertexFormat::Unorm16(3) => (Mtl::UShort3Normalized, 6),
        api::VertexFormat::Unorm16(4) => (Mtl::UShort4Normalized, 8),
        api::VertexFormat::Snorm16(1) => (Mtl::ShortNormalized, 2),
        api::VertexFormat::Snorm16(2) => (Mtl::Short2Normalized, 4),
        api::VertexFormat::Snorm16(3) => (Mtl::Short3Normalized, 6),
        api::VertexFormat::Snorm16(4) => (Mtl::Short4Normalized, 8),
        _ => return Err(api::Error::Unsupported("Metal vertex format")),
    };
    Ok((format, size))
}

#[cfg(test)]
mod tests {
    use super::{
        full_sample_mask, supported_color_format, supported_depth_state, supported_sample_count,
    };
    use fluxel_rhi_core::api::{self, TextureFormat};

    #[test]
    fn raster_target_rejects_depth_and_nonimplemented_formats() {
        assert!(supported_color_format(TextureFormat::Rgba8Unorm));
        assert!(supported_color_format(TextureFormat::Bgra8UnormSrgb));
        assert!(!supported_color_format(TextureFormat::Depth32Float));
        assert!(supported_color_format(TextureFormat::Rgba32Float));
    }

    #[test]
    fn depth_pipeline_accepts_depth32_and_combined_stencil_semantics() {
        let face = api::StencilFaceState {
            compare: api::CompareFunction::Always,
            fail: api::StencilOperation::Keep,
            depth_fail: api::StencilOperation::Keep,
            pass: api::StencilOperation::Keep,
        };
        let state = api::DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: api::CompareFunction::LessEqual,
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        };
        assert!(supported_depth_state(state));
        assert!(!supported_depth_state(api::DepthStencilState {
            stencil_write_mask: 1,
            ..state
        }));
        assert!(!supported_depth_state(api::DepthStencilState {
            format: TextureFormat::Rgba8Unorm,
            ..state
        }));
        assert!(supported_depth_state(api::DepthStencilState {
            format: TextureFormat::Depth32FloatStencil8,
            stencil_front: api::StencilFaceState {
                compare: api::CompareFunction::LessEqual,
                fail: api::StencilOperation::Zero,
                depth_fail: api::StencilOperation::IncrementWrap,
                pass: api::StencilOperation::Replace,
            },
            stencil_read_mask: 0x0f,
            stencil_write_mask: 0xf0,
            ..state
        }));
    }

    #[test]
    fn multisample_subset_accepts_only_native_counts_and_full_masks() {
        assert!(supported_sample_count(1));
        assert!(supported_sample_count(4));
        assert!(!supported_sample_count(3));
        assert!(full_sample_mask(u64::from(u32::MAX), 4));
        assert!(full_sample_mask(u64::MAX, 4));
        assert!(!full_sample_mask(0x0f, 8));
    }
}
