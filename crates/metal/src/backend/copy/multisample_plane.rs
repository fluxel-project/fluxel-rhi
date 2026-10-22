//! Copy one plane of a multisampled combined depth/stencil texture.
//!
//! Metal blits copy both planes. Render at sample rate into the destination,
//! retaining its other plane. A scratch texture avoids simultaneous source and
//! destination attachment access and supplies the shader-readable usage.

use std::ptr::NonNull;

use fluxel_rhi_core::api::{self, TextureRegion};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCompareFunction,
    MTLDepthStencilDescriptor, MTLDevice as _, MTLLibrary as _, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLResource as _, MTLScissorRect, MTLSize, MTLStencilDescriptor,
    MTLStencilOperation, MTLStorageMode, MTLStoreAction, MTLTexture as _, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

use crate::backend::{Metal, MetalCommandEncoder, MetalTexture};

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Params { int x; int y; uint value; };
struct VertexOutput { float4 position [[position]]; };
struct DepthOutput { float depth [[depth(any)]]; };
vertex VertexOutput copy_vertex(uint v [[vertex_id]]) {
    float2 p = float2((v << 1) & 2, v & 2);
    return { float4(p * float2(2, -2) + float2(-1, 1), 0, 1) };
}
fragment DepthOutput copy_depth(float4 p [[position]], uint sample [[sample_id]],
                          depth2d_ms<float, access::read> src [[texture(0)]],
                          constant Params& a [[buffer(0)]]) {
    return { src.read(uint2(int2(p.xy) + int2(a.x, a.y)), sample) };
}
fragment void copy_stencil(float4 p [[position]], uint sample [[sample_id]],
                           texture2d_ms<uint, access::read> src [[texture(0)]],
                           constant Params& a [[buffer(0)]]) {
    if (src.read(uint2(int2(p.xy) + int2(a.x, a.y)), sample).r != a.value)
        discard_fragment();
}
"#;

pub(super) fn copy(
    encoder: &mut MetalCommandEncoder,
    source: &MetalTexture,
    target: &MetalTexture,
    src: &TextureRegion<'_, Metal>,
    dst: &TextureRegion<'_, Metal>,
) -> api::Result<()> {
    if encoder.render.is_some() || encoder.compute.is_some() {
        return Err(api::Error::InvalidState);
    }
    // The source is blitted, not shader-sampled. The destination is attached
    // with Load/Store to preserve its other plane and pixels outside the copy.
    // Usage::Unknown explicitly permits all usages; a framebuffer-only source
    // or memoryless attachment cannot participate in this operation.
    let target_usage = target.raw.usage();
    if source.raw.isFramebufferOnly()
        || source.raw.storageMode() == MTLStorageMode::Memoryless
        || target.raw.storageMode() == MTLStorageMode::Memoryless
        || (target_usage != MTLTextureUsage::Unknown
            && !target_usage.contains(MTLTextureUsage::RenderTarget))
    {
        return Err(api::Error::Unsupported(
            "Metal multisample plane copy native texture usage",
        ));
    }
    let device = encoder.command.device();
    let depth = src.aspect == api::TextureAspects::DEPTH;
    let format = source.raw.pixelFormat();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SHADER), None)
        .map_err(|e| api::Error::Shader(format!("Metal MSAA plane copy: {e:?}")))?;
    let vertex = library
        .newFunctionWithName(&NSString::from_str("copy_vertex"))
        .ok_or_else(|| api::Error::Shader("Metal MSAA copy vertex missing".into()))?;
    let fragment = library
        .newFunctionWithName(&NSString::from_str(if depth {
            "copy_depth"
        } else {
            "copy_stencil"
        }))
        .ok_or_else(|| api::Error::Shader("Metal MSAA copy fragment missing".into()))?;
    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexFunction(Some(&vertex));
    pipeline_desc.setFragmentFunction(Some(&fragment));
    pipeline_desc.setDepthAttachmentPixelFormat(format);
    pipeline_desc.setStencilAttachmentPixelFormat(format);
    pipeline_desc.setRasterSampleCount(source.desc().samples as usize);
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .map_err(|e| api::Error::Backend(format!("Metal MSAA plane copy pipeline: {e:?}")))?;
    let state_desc = MTLDepthStencilDescriptor::new();
    state_desc.setDepthCompareFunction(MTLCompareFunction::Always);
    state_desc.setDepthWriteEnabled(depth);
    if !depth {
        let stencil = MTLStencilDescriptor::new();
        stencil.setStencilCompareFunction(MTLCompareFunction::Always);
        stencil.setDepthStencilPassOperation(MTLStencilOperation::Replace);
        stencil.setWriteMask(255);
        state_desc.setFrontFaceStencil(Some(&stencil));
        state_desc.setBackFaceStencil(Some(&stencil));
    }
    let state = device
        .newDepthStencilStateWithDescriptor(&state_desc)
        .ok_or(api::Error::OutOfMemory)?;
    let scratch_desc = MTLTextureDescriptor::new();
    scratch_desc.setPixelFormat(format);
    scratch_desc.setTextureType(MTLTextureType::Type2DMultisample);
    // SAFETY: dimensions and sample count were validated at texture creation.
    unsafe {
        scratch_desc.setWidth(source.desc().extent[0] as usize);
        scratch_desc.setHeight(source.desc().extent[1] as usize);
        scratch_desc.setSampleCount(source.desc().samples as usize);
    }
    scratch_desc.setStorageMode(MTLStorageMode::Private);
    scratch_desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
    let scratch = device
        .newTextureWithDescriptor(&scratch_desc)
        .ok_or(api::Error::OutOfMemory)?;
    let stencil_view = if !depth {
        let stencil_format = if format == MTLPixelFormat::Depth24Unorm_Stencil8 {
            MTLPixelFormat::X24_Stencil8
        } else {
            MTLPixelFormat::X32_Stencil8
        };
        Some(
            scratch
                .newTextureViewWithPixelFormat(stencil_format)
                .ok_or(api::Error::Unsupported(
                    "Metal multisample stencil shader view",
                ))?,
        )
    } else {
        None
    };
    let sampled = stencil_view.as_deref().unwrap_or(&scratch);
    let dx = i32::try_from(i64::from(src.origin[0]) - i64::from(dst.origin[0]))
        .map_err(|_| api::Error::InvalidRange)?;
    let dy = i32::try_from(i64::from(src.origin[1]) - i64::from(dst.origin[1]))
        .map_err(|_| api::Error::InvalidRange)?;
    for layer in 0..src.extent[2] {
        let blit = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: scratch matches the source format, size and sample count; the slice was validated.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &source.raw, (src.origin[2] + layer) as usize, src.mip as usize,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: source.desc().extent[0] as usize, height: source.desc().extent[1] as usize, depth: 1 },
                &scratch, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 });
        }
        blit.endEncoding();
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        let d = pass.depthAttachment();
        d.setTexture(Some(&target.raw));
        d.setSlice((dst.origin[2] + layer) as usize);
        d.setLoadAction(MTLLoadAction::Load);
        d.setStoreAction(MTLStoreAction::Store);
        let s = pass.stencilAttachment();
        s.setTexture(Some(&target.raw));
        s.setSlice((dst.origin[2] + layer) as usize);
        s.setLoadAction(MTLLoadAction::Load);
        s.setStoreAction(MTLStoreAction::Store);
        let render = encoder
            .command
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or(api::Error::OutOfMemory)?;
        render.setRenderPipelineState(&pipeline);
        render.setDepthStencilState(Some(&state));
        // SAFETY: the target rectangle and shader texture binding are validated above.
        unsafe {
            render.setFragmentTexture_atIndex(Some(sampled), 0);
            render.setScissorRect(MTLScissorRect {
                x: dst.origin[0] as usize,
                y: dst.origin[1] as usize,
                width: dst.extent[0] as usize,
                height: dst.extent[1] as usize,
            });
            for value in 0..if depth { 1 } else { 256 } {
                let params = [dx as u32, dy as u32, value];
                render.setFragmentBytes_length_atIndex(
                    NonNull::new(params.as_ptr() as *mut _).unwrap(),
                    std::mem::size_of_val(&params),
                    0,
                );
                if !depth {
                    render.setStencilReferenceValue(value);
                }
                render.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
        }
        render.endEncoding();
    }
    Ok(())
}
