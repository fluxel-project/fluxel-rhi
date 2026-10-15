//! Conservative native raster-pipeline creation.
//!
//! Descriptors outside the implemented TriangleList, one color target, and
//! supported attachment/sample subset fail before native creation. Depth32,
//! device-supported 1/2/4/8-sample rasterization with full sample masks, color
//! resolve, blending, color masks, and face culling lower to native state.
//! Combined Depth32/stencil state lowers to Metal. Partial sample masks and
//! dual-source blend remain unsupported.
//! This module creates immutable native state and does not record or schedule
//! work.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{self, PipelineBackend, RasterPipelineDesc};

use super::{Metal, MetalDevice, binding_abi};

#[derive(Clone)]
pub struct MetalRasterPipeline {
    #[cfg(target_vendor = "apple")]
    pub(crate) state: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
    >,
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
    pub(crate) signature: (
        usize,
        objc2_metal::MTLPixelFormat,
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
    pub(crate) workgroup_size: [u32; 3],
}

impl PipelineBackend for Metal {
    type RasterPipelineFuture<'a> = Ready<api::Result<MetalRasterPipeline>>;

    fn create_raster_pipeline<'a>(
        device: &'a MetalDevice,
        desc: &'a RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        ready(create_raster_pipeline(device, desc))
    }
}

fn create_raster_pipeline(
    device: &MetalDevice,
    desc: &RasterPipelineDesc<Metal>,
) -> api::Result<MetalRasterPipeline> {
    validate_desc(desc)?;
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLDepthStencilDescriptor, MTLDevice as _, MTLFunction as _, MTLFunctionType,
            MTLLibrary as _, MTLRenderPipelineDescriptor, MTLStencilDescriptor,
            MTLVertexDescriptor,
        };
        if desc
            .depth_stencil
            .is_some_and(|depth| depth.format == api::TextureFormat::Depth32FloatStencil8)
            && !super::texture::supports_depth32_stencil(device)
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

        // SAFETY: Core verified each shader belongs to this device before the SPI call.
        let vertex = unsafe { desc.vertex.module.native_ref() };
        let vertex_name = NSString::from_str(&desc.vertex.entry);
        let vertex_function = vertex
            ._library
            .newFunctionWithName(&vertex_name)
            .ok_or_else(|| api::Error::Shader("Metal vertex entry not found".into()))?;
        if vertex_function.functionType() != MTLFunctionType::Vertex {
            return Err(api::Error::Shader(
                "Metal entry is not a vertex function".into(),
            ));
        }

        let fragment_function = if let Some(stage) = &desc.fragment {
            // SAFETY: Core verified the fragment shader belongs to this device.
            let shader = unsafe { stage.module.native_ref() };
            let name = NSString::from_str(&stage.entry);
            let function = shader
                ._library
                .newFunctionWithName(&name)
                .ok_or_else(|| api::Error::Shader("Metal fragment entry not found".into()))?;
            if function.functionType() != MTLFunctionType::Fragment {
                return Err(api::Error::Shader(
                    "Metal entry is not a fragment function".into(),
                ));
            }
            Some(function)
        } else {
            None
        };

        let vertex_descriptor = MTLVertexDescriptor::new();
        let attributes = vertex_descriptor.attributes();
        let layouts = vertex_descriptor.layouts();
        for (buffer_index, layout) in desc.vertex_buffers.iter().enumerate() {
            // Metal's fixed vertex descriptor path is kept within the
            // documented practical per-vertex stride range.
            if layout.stride == 0 || layout.stride > 2048 {
                return Err(api::Error::InvalidRange);
            }
            let native_index = binding_abi::vertex_buffer_index(buffer_index as u32)?;
            // SAFETY: `vertex_buffer_index` bounds this to the disjoint native
            // vertex range 28–30 supported by this backend ABI.
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
            descriptor.setDepthAttachmentPixelFormat(format);
            let native = MTLDepthStencilDescriptor::new();
            native.setDepthCompareFunction(compare_function(depth.depth_compare));
            native.setDepthWriteEnabled(depth.depth_write_enabled);
            if depth.format == api::TextureFormat::Depth32FloatStencil8 {
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
        // SAFETY: validate_desc requires exactly one present color target.
        let color = unsafe { colors.objectAtIndexedSubscript(0) };
        let color_format =
            super::texture::metal_format(desc.color_targets[0].as_ref().unwrap().format)
                .ok_or(api::Error::Unsupported("Metal color target format"))?;
        color.setPixelFormat(color_format);
        color.setWriteMask(write_mask(
            desc.color_targets[0].as_ref().unwrap().write_mask,
        )?);
        if let Some(blend) = desc.color_targets[0].as_ref().unwrap().blend {
            color.setBlendingEnabled(true);
            color.setSourceRGBBlendFactor(blend_factor(blend.color.src_factor)?);
            color.setDestinationRGBBlendFactor(blend_factor(blend.color.dst_factor)?);
            color.setRgbBlendOperation(blend_operation(blend.color.operation));
            color.setSourceAlphaBlendFactor(blend_factor(blend.alpha.src_factor)?);
            color.setDestinationAlphaBlendFactor(blend_factor(blend.alpha.dst_factor)?);
            color.setAlphaBlendOperation(blend_operation(blend.alpha.operation));
        } else {
            color.setBlendingEnabled(false);
        }
        if let Some(label) = desc.label.as_deref() {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        let state = device
            ._state
            ._device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| api::Error::Backend(format!("Metal raster pipeline: {error:?}")))?;
        Ok(MetalRasterPipeline {
            state,
            cull_mode: desc.primitive.cull_mode,
            front_face: desc.primitive.front_face,
            depth_state,
            depth_bias: desc.depth_stencil.map_or((0, 0.0, 0.0), |depth| {
                (
                    depth.depth_bias_constant,
                    depth.depth_bias_slope_scale,
                    depth.depth_bias_clamp,
                )
            }),
            signature: (desc.multisample.count as usize, color_format, depth_format),
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

fn validate_desc(desc: &RasterPipelineDesc<Metal>) -> api::Result<()> {
    if desc.fragment.is_none()
        || !desc.vertex.constants.is_empty()
        || desc
            .fragment
            .as_ref()
            .is_some_and(|stage| !stage.constants.is_empty())
        || desc.cache.is_some()
        || !matches!(
            desc.primitive.topology,
            api::PrimitiveTopology::TriangleList
        )
        || desc.primitive.strip_index_format.is_some()
        || !matches!(desc.primitive.polygon_mode, api::PolygonMode::Fill)
        || desc.primitive.unclipped_depth
        || desc.primitive.conservative
        || desc
            .depth_stencil
            .is_some_and(|depth| !supported_depth_state(depth))
        || !supported_sample_count(desc.multisample.count)
        || !full_sample_mask(desc.multisample.mask)
        || desc.multiview.is_some()
        || desc.color_targets.len() != 1
        || desc.color_targets[0].as_ref().is_none_or(|target| {
            target.write_mask & !0x0f != 0 || !supported_color_format(target.format)
        })
        || desc.vertex_buffers.len() > binding_abi::MAX_VERTEX_BUFFERS
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

fn supported_sample_count(count: u32) -> bool {
    matches!(count, 1 | 2 | 4 | 8)
}

fn full_sample_mask(mask: u64) -> bool {
    matches!(mask, u64::MAX | 0x0000_0000_ffff_ffff)
}

fn supported_depth_state(depth: api::DepthStencilState) -> bool {
    matches!(
        depth.format,
        api::TextureFormat::Depth32Float | api::TextureFormat::Depth32FloatStencil8
    ) && (depth.format == api::TextureFormat::Depth32FloatStencil8
        || (depth.stencil_read_mask == 0
            && depth.stencil_write_mask == 0
            && inert_stencil_face(depth.stencil_front)
            && inert_stencil_face(depth.stencil_back)))
        && depth.depth_bias_clamp.is_finite()
        && depth.depth_bias_slope_scale.is_finite()
}

fn inert_stencil_face(face: api::StencilFaceState) -> bool {
    matches!(face.compare, api::CompareFunction::Always)
        && matches!(face.fail, api::StencilOperation::Keep)
        && matches!(face.depth_fail, api::StencilOperation::Keep)
        && matches!(face.pass, api::StencilOperation::Keep)
}

#[cfg(target_vendor = "apple")]
fn compare_function(value: api::CompareFunction) -> objc2_metal::MTLCompareFunction {
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
fn configure_stencil_descriptor(
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
fn write_mask(mask: u8) -> api::Result<objc2_metal::MTLColorWriteMask> {
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
fn blend_factor(value: api::BlendFactor) -> api::Result<objc2_metal::MTLBlendFactor> {
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
        F::Constant => M::BlendColor,
        F::OneMinusConstant => M::OneMinusBlendColor,
        F::SrcAlphaSaturated => M::SourceAlphaSaturated,
        F::Src1 | F::OneMinusSrc1 => {
            return Err(api::Error::Unsupported("Metal dual-source blend"));
        }
    })
}

#[cfg(target_vendor = "apple")]
fn blend_operation(value: api::BlendOperation) -> objc2_metal::MTLBlendOperation {
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

fn supported_color_format(format: api::TextureFormat) -> bool {
    matches!(
        format,
        api::TextureFormat::Rgba8Unorm
            | api::TextureFormat::Rgba8UnormSrgb
            | api::TextureFormat::Bgra8Unorm
            | api::TextureFormat::Bgra8UnormSrgb
            | api::TextureFormat::Rgba16Float
    )
}

#[cfg(target_vendor = "apple")]
fn vertex_format(format: api::VertexFormat) -> api::Result<(objc2_metal::MTLVertexFormat, u64)> {
    use objc2_metal::MTLVertexFormat as Mtl;
    let (format, size) = match format {
        api::VertexFormat::Float32(1) => (Mtl::Float, 4),
        api::VertexFormat::Float32(2) => (Mtl::Float2, 8),
        api::VertexFormat::Float32(3) => (Mtl::Float3, 12),
        api::VertexFormat::Float32(4) => (Mtl::Float4, 16),
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
        assert!(!supported_color_format(TextureFormat::Rgba32Float));
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
        assert!(full_sample_mask(u64::from(u32::MAX)));
        assert!(full_sample_mask(u64::MAX));
        assert!(!full_sample_mask(0x0f));
    }
}
