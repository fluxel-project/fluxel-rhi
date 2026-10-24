//! Filtered color blits lowered to a small native Metal render pass.
//!
//! `MTLBlitCommandEncoder` only copies equal-sized regions. Scaling therefore
//! uses a fragment shader between private sampled/renderable scratch textures.
//! The source and destination need only their public CopySrc/CopyDst usages.
//! This module records native work directly; it owns neither scheduling nor states.
use super::super::{Metal, MetalCommandEncoder, MetalTexture};
use super::{same_texture, validate_region};
use fluxel_rhi_core::api::{self, FilterMode, TextureRegion};

pub(in crate::backend) fn blit(
    encoder: &mut MetalCommandEncoder,
    src: TextureRegion<'_, Metal>,
    dst: TextureRegion<'_, Metal>,
    filter: FilterMode,
) -> api::Result<()> {
    // SAFETY: Core checked device identity; the borrowed textures outlive direct recording.
    let (source, target) = unsafe { (src.texture.native_ref(), dst.texture.native_ref()) };
    validate_region(source, &src)?;
    validate_region(target, &dst)?;
    let sd = source.desc();
    let td = target.desc();
    if encoder.lane != 0
        || sd.dimension != api::TextureDimension::D2
        || td.dimension != api::TextureDimension::D2
        || sd.samples != 1
        || td.samples != 1
        || sample_kind(sd.format) != sample_kind(td.format)
        || src.aspect != api::TextureAspects::COLOR
        || dst.aspect != api::TextureAspects::COLOR
        || src.extent[2] != dst.extent[2]
        || !sd.usage.contains(api::TextureUsage::COPY_SRC)
        || !td.usage.contains(api::TextureUsage::COPY_DST)
        || !super::super::texture::color_attachment_format(td.format)
        || (same_texture(source, target)
            && src.mip == dst.mip
            && src.origin[2] < dst.origin[2] + dst.extent[2]
            && dst.origin[2] < src.origin[2] + src.extent[2]
            && src.origin[0] < dst.origin[0] + dst.extent[0]
            && dst.origin[0] < src.origin[0] + src.extent[0]
            && src.origin[1] < dst.origin[1] + dst.extent[1]
            && dst.origin[1] < src.origin[1] + src.extent[1])
    {
        return Err(api::Error::Unsupported("Metal color blit shape or usage"));
    }
    let kind = sample_kind(sd.format);
    if kind != SampleKind::Float && filter == FilterMode::Linear {
        return Err(api::Error::Unsupported("Metal linear integer blit"));
    }
    #[cfg(target_vendor = "apple")]
    return native(encoder, source, target, &src, &dst, filter, kind);
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (encoder, source, target, src, dst, filter, kind);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SampleKind {
    Float,
    Uint,
    Sint,
}

fn sample_kind(format: api::TextureFormat) -> SampleKind {
    use api::TextureFormat::*;
    match format {
        R8Uint | Rg8Uint | Rgba8Uint | R16Uint | Rg16Uint | Rgba16Uint | R32Uint | Rg32Uint
        | Rgba32Uint | Rgb10a2Uint | R64Uint | Rg64Uint => SampleKind::Uint,
        R8Sint | Rg8Sint | Rgba8Sint | R16Sint | Rg16Sint | Rgba16Sint | R32Sint | Rg32Sint
        | Rgba32Sint => SampleKind::Sint,
        _ => SampleKind::Float,
    }
}

#[cfg(target_vendor = "apple")]
fn native(
    encoder: &mut MetalCommandEncoder,
    source: &MetalTexture,
    target: &MetalTexture,
    src: &TextureRegion<'_, Metal>,
    dst: &TextureRegion<'_, Metal>,
    filter: FilterMode,
    kind: SampleKind,
) -> api::Result<()> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLDevice as _,
        MTLLibrary as _, MTLLoadAction, MTLOrigin, MTLRenderCommandEncoder as _,
        MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLSamplerDescriptor,
        MTLSamplerMinMagFilter, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture as _,
        MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLViewport,
    };

    if encoder.render.is_some() || encoder.compute.is_some() {
        return Err(api::Error::InvalidState);
    }
    if source.raw.isFramebufferOnly() || target.raw.isFramebufferOnly() {
        return Err(api::Error::Unsupported("Metal blit native texture usage"));
    }
    let device = encoder.command.device();
    if filter == FilterMode::Linear
        && kind == SampleKind::Float
        && !linear_filter_supported(&device, source.desc().format)
    {
        return Err(api::Error::Unsupported("Metal linear blit source format"));
    }
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(shader(kind)), None)
        .map_err(|e| api::Error::Shader(format!("Metal blit shader: {e:?}")))?;
    let vertex = library
        .newFunctionWithName(&NSString::from_str("blit_vertex"))
        .ok_or_else(|| api::Error::Shader("Metal blit vertex missing".into()))?;
    let fragment = library
        .newFunctionWithName(&NSString::from_str("blit_fragment"))
        .ok_or_else(|| api::Error::Shader("Metal blit fragment missing".into()))?;
    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexFunction(Some(&vertex));
    pipeline_desc.setFragmentFunction(Some(&fragment));
    // SAFETY: fixed Metal color attachment slot zero exists.
    unsafe { pipeline_desc.colorAttachments().objectAtIndexedSubscript(0) }
        .setPixelFormat(target.raw.pixelFormat());
    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .map_err(|e| api::Error::Backend(format!("Metal blit pipeline: {e:?}")))?;
    let sampler = if kind == SampleKind::Float {
        let descriptor = MTLSamplerDescriptor::new();
        let native_filter = match filter {
            FilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
            FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
        };
        descriptor.setMinFilter(native_filter);
        descriptor.setMagFilter(native_filter);
        device
            .newSamplerStateWithDescriptor(&descriptor)
            .ok_or(api::Error::OutOfMemory)?
            .into()
    } else {
        None
    };
    let scratch = |format, width, height, usage| {
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setPixelFormat(format);
        descriptor.setTextureType(MTLTextureType::Type2D);
        // SAFETY: validated regions have nonzero dimensions representable as usize.
        unsafe {
            descriptor.setWidth(width as usize);
            descriptor.setHeight(height as usize);
        }
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setUsage(usage);
        device
            .newTextureWithDescriptor(&descriptor)
            .ok_or(api::Error::OutOfMemory)
    };
    let params = [
        0.0,
        0.0,
        src.extent[0] as f32,
        src.extent[1] as f32,
        0.0,
        0.0,
        src.extent[0] as f32 / dst.extent[0] as f32,
        src.extent[1] as f32 / dst.extent[1] as f32,
    ];
    for layer in 0..src.extent[2] {
        let source_scratch = scratch(
            source.raw.pixelFormat(),
            src.extent[0],
            src.extent[1],
            MTLTextureUsage::ShaderRead,
        )?;
        let target_scratch = scratch(
            target.raw.pixelFormat(),
            dst.extent[0],
            dst.extent[1],
            MTLTextureUsage::RenderTarget,
        )?;
        let copy = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: validated source region and matching scratch texture bounds.
        unsafe {
            copy.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &source.raw, (src.origin[2] + layer) as usize, src.mip as usize,
            MTLOrigin { x: src.origin[0] as usize, y: src.origin[1] as usize, z: 0 },
            MTLSize { width: src.extent[0] as usize, height: src.extent[1] as usize, depth: 1 },
            &source_scratch, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 });
        }
        copy.endEncoding();
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: fixed Metal color attachment slot zero exists.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(&target_scratch));
        color.setLoadAction(MTLLoadAction::DontCare);
        color.setStoreAction(MTLStoreAction::Store);
        let render = encoder
            .command
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or(api::Error::OutOfMemory)?;
        render.setRenderPipelineState(&pipeline);
        render.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: target_scratch.width() as f64,
            height: target_scratch.height() as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        // SAFETY: views, parameter bytes, and sampler stay alive through this direct encoding.
        unsafe {
            render.setFragmentTexture_atIndex(Some(&source_scratch), 0);
            if let Some(sampler) = sampler.as_deref() {
                render.setFragmentSamplerState_atIndex(Some(sampler), 0);
            }
            render.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::new(params.as_ptr() as *mut _).unwrap(),
                std::mem::size_of_val(&params),
                0,
            );
            render.setScissorRect(objc2_metal::MTLScissorRect {
                x: 0,
                y: 0,
                width: dst.extent[0] as usize,
                height: dst.extent[1] as usize,
            });
            render.drawPrimitives_vertexStart_vertexCount(
                objc2_metal::MTLPrimitiveType::Triangle,
                0,
                3,
            );
        }
        render.endEncoding();
        let copy = encoder
            .command
            .blitCommandEncoder()
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: target scratch has exactly the destination region extent.
        unsafe {
            copy.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &target_scratch, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: dst.extent[0] as usize, height: dst.extent[1] as usize, depth: 1 },
            &target.raw, (dst.origin[2] + layer) as usize, dst.mip as usize,
            MTLOrigin { x: dst.origin[0] as usize, y: dst.origin[1] as usize, z: 0 });
        }
        copy.endEncoding();
    }
    Ok(())
}

#[cfg(target_vendor = "apple")]
fn linear_filter_supported(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    format: api::TextureFormat,
) -> bool {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::{MTLDevice as _, MTLGPUFamily};
    if super::super::texture::is_filterable(format) {
        return true;
    }
    matches!(
        format,
        api::TextureFormat::R32Float
            | api::TextureFormat::Rg32Float
            | api::TextureFormat::Rgba32Float
    ) && (device.supportsFamily(MTLGPUFamily::Apple9)
        || device.supportsFamily(MTLGPUFamily::Mac2)
        || (device.respondsToSelector(objc2::sel!(supports32BitFloatFiltering))
            && device.supports32BitFloatFiltering()))
}

fn shader(kind: SampleKind) -> &'static str {
    match kind {
        SampleKind::Float => {
            r#"
#include <metal_stdlib>
using namespace metal;
struct Params { float sx,sy,sw,sh,dx,dy,scale_x,scale_y; };
struct Out { float4 position [[position]]; };
vertex Out blit_vertex(uint id [[vertex_id]]) { float2 p=float2((id<<1)&2,id&2); return {float4(p*float2(2,-2)+float2(-1,1),0,1)}; }
fragment float4 blit_fragment(float4 p [[position]], texture2d<float> src [[texture(0)]], sampler s [[sampler(0)]], constant Params& q [[buffer(0)]]) { float2 pixel=float2(q.sx,q.sy)+(p.xy-float2(q.dx,q.dy))*float2(q.scale_x,q.scale_y); return src.sample(s,pixel/float2(q.sw,q.sh)); }
"#
        }
        SampleKind::Uint => integer_shader("uint"),
        SampleKind::Sint => integer_shader("int"),
    }
}

fn integer_shader(scalar: &str) -> &'static str {
    match scalar {
        "uint" => {
            r#"
#include <metal_stdlib>
using namespace metal;
struct Params { float sx,sy,sw,sh,dx,dy,scale_x,scale_y; }; struct Out { float4 position [[position]]; };
vertex Out blit_vertex(uint id [[vertex_id]]) { float2 p=float2((id<<1)&2,id&2); return {float4(p*float2(2,-2)+float2(-1,1),0,1)}; }
fragment uint4 blit_fragment(float4 p [[position]], texture2d<uint, access::read> src [[texture(0)]], constant Params& q [[buffer(0)]]) { return src.read(uint2(floor(float2(q.sx,q.sy)+(p.xy-float2(q.dx,q.dy))*float2(q.scale_x,q.scale_y)))); }
"#
        }
        _ => {
            r#"
#include <metal_stdlib>
using namespace metal;
struct Params { float sx,sy,sw,sh,dx,dy,scale_x,scale_y; }; struct Out { float4 position [[position]]; };
vertex Out blit_vertex(uint id [[vertex_id]]) { float2 p=float2((id<<1)&2,id&2); return {float4(p*float2(2,-2)+float2(-1,1),0,1)}; }
fragment int4 blit_fragment(float4 p [[position]], texture2d<int, access::read> src [[texture(0)]], constant Params& q [[buffer(0)]]) { return src.read(uint2(floor(float2(q.sx,q.sy)+(p.xy-float2(q.dx,q.dy))*float2(q.scale_x,q.scale_y)))); }
"#
        }
    }
}
