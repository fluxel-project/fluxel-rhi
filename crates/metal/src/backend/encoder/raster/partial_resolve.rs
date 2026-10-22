//! Resolve a render-area rectangle while preserving pixels outside it.
//!
//! Metal resolves complete attachments. A private MSAA copy is resolved into a
//! second temporary texture and only the requested rectangle is copied out.
use fluxel_rhi_core::api::{self, Rect};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLCommandBuffer, MTLCommandEncoder as _, MTLDevice as _,
    MTLLoadAction, MTLOrigin, MTLRenderPassDescriptor, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

pub(super) fn resolve_area(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    source: &ProtocolObject<dyn MTLTexture>,
    target: &ProtocolObject<dyn MTLTexture>,
    area: Rect,
    multiview_mask: Option<u32>,
) -> api::Result<()> {
    let mask = multiview_mask.unwrap_or(1);
    if mask == 0 || (multiview_mask.is_some() && target.isFramebufferOnly()) {
        return Err(api::Error::InvalidRange);
    }
    let last = (u32::BITS - mask.leading_zeros()) as usize;
    if last > source.arrayLength() || last > target.arrayLength() {
        return Err(api::Error::InvalidRange);
    }
    let device = command.device();
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(source.pixelFormat());
    desc.setTextureType(MTLTextureType::Type2DMultisample);
    // SAFETY: source is a validated 2D multisampled attachment.
    unsafe {
        desc.setWidth(source.width());
        desc.setHeight(source.height());
        desc.setSampleCount(source.sampleCount());
    }
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::RenderTarget);
    let multisample = device
        .newTextureWithDescriptor(&desc)
        .ok_or(api::Error::OutOfMemory)?;
    desc.setTextureType(MTLTextureType::Type2D);
    // SAFETY: one is a valid sample count for a 2D texture.
    unsafe {
        desc.setSampleCount(1);
    }
    desc.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let resolved = device
        .newTextureWithDescriptor(&desc)
        .ok_or(api::Error::OutOfMemory)?;
    for slice in 0..u32::BITS {
        if mask & (1 << slice) == 0 {
            continue;
        }
        let blit = command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: the scratch texture matches source size, format and sample count.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
        source, slice as usize, 0, MTLOrigin { x: 0, y: 0, z: 0 },
        MTLSize { width: source.width(), height: source.height(), depth: 1 },
        &multisample, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 });
        }
        blit.endEncoding();
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: Metal provides color attachment slot zero.
        let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setTexture(Some(&multisample));
        attachment.setResolveTexture(Some(&resolved));
        attachment.setLoadAction(MTLLoadAction::Load);
        attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
        let render = command
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or(api::Error::OutOfMemory)?;
        render.endEncoding();
        if target.isFramebufferOnly() {
            return draw_to_framebuffer(command, &resolved, target, area);
        }
        let blit = command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: pass validation bounds the rectangle in both matching textures.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
        &resolved, 0, 0, MTLOrigin { x: area.x as usize, y: area.y as usize, z: 0 },
        MTLSize { width: area.width as usize, height: area.height as usize, depth: 1 },
        target, slice as usize, 0, MTLOrigin { x: area.x as usize, y: area.y as usize, z: 0 });
        }
        blit.endEncoding();
    }
    Ok(())
}

fn draw_to_framebuffer(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    source: &ProtocolObject<dyn MTLTexture>,
    target: &ProtocolObject<dyn MTLTexture>,
    area: Rect,
) -> api::Result<()> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLLibrary as _, MTLPrimitiveType, MTLRenderCommandEncoder as _,
        MTLRenderPipelineDescriptor, MTLScissorRect,
    };
    let device = command.device();
    let shader = NSString::from_str(
        r#"#include <metal_stdlib>
using namespace metal;
struct Vertex { float4 position [[position]]; };
vertex Vertex vs(uint id [[vertex_id]]) { float2 p = float2((id << 1) & 2, id & 2); return {float4(p * float2(2,-2) + float2(-1,1),0,1)}; }
fragment float4 fs(Vertex v [[stage_in]], texture2d<float, access::read> t [[texture(0)]]) { return t.read(uint2(v.position.xy)); }
"#,
    );
    let library = device
        .newLibraryWithSource_options_error(&shader, None)
        .map_err(|e| api::Error::Shader(format!("Metal partial resolve shader: {e:?}")))?;
    let pso = MTLRenderPipelineDescriptor::new();
    pso.setVertexFunction(
        library
            .newFunctionWithName(&NSString::from_str("vs"))
            .as_deref(),
    );
    pso.setFragmentFunction(
        library
            .newFunctionWithName(&NSString::from_str("fs"))
            .as_deref(),
    );
    // SAFETY: fixed Metal color attachment array contains slot zero.
    unsafe { pso.colorAttachments().objectAtIndexedSubscript(0) }
        .setPixelFormat(target.pixelFormat());
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pso)
        .map_err(|e| api::Error::Backend(format!("Metal partial resolve pipeline: {e:?}")))?;
    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
    // SAFETY: fixed Metal color attachment array contains slot zero.
    let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
    color.setTexture(Some(target));
    color.setLoadAction(MTLLoadAction::Load);
    color.setStoreAction(MTLStoreAction::Store);
    let render = command
        .renderCommandEncoderWithDescriptor(&pass)
        .ok_or(api::Error::OutOfMemory)?;
    render.setRenderPipelineState(&pipeline);
    // SAFETY: rectangle and texture bounds are validated by the pass caller.
    unsafe {
        render.setFragmentTexture_atIndex(Some(source), 0);
        render.setScissorRect(MTLScissorRect {
            x: area.x as usize,
            y: area.y as usize,
            width: area.width as usize,
            height: area.height as usize,
        });
        render.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
    }
    render.endEncoding();
    Ok(())
}
