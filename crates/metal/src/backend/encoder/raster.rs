//! Immediate Metal raster encoder operations and WebGPU-coordinate lowering.
//!
//! Pass lifetime and draw calls are sent directly to the active native render encoder.

mod pass;

use std::ops::Range;

use fluxel_rhi_core::api::{self, RasterBackend};

use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalRasterPipeline};
use crate::backend::MetalBindGroup;
#[cfg(target_vendor = "apple")]
use crate::backend::binding_abi::vertex_buffer_index;
#[cfg(target_vendor = "apple")]
use crate::backend::bindings::BoundResource;

impl RasterBackend for Metal {
    fn begin_raster(
        encoder: &mut MetalCommandEncoder,
        desc: &api::RasterPassDesc<'_, Self>,
    ) -> api::Result<()> {
        pass::begin_raster(encoder, desc)
    }

    fn set_pipeline(
        encoder: &mut MetalCommandEncoder,
        pipeline: &MetalRasterPipeline,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            if encoder.raster_signature != Some(pipeline.signature) {
                return Err(api::Error::InvalidState);
            }
            render.setRenderPipelineState(&pipeline.state);
            render.setDepthStencilState(pipeline.depth_state.as_deref());
            render.setDepthBias_slopeScale_clamp(
                pipeline.depth_bias.0 as f32,
                pipeline.depth_bias.1,
                pipeline.depth_bias.2,
            );
            render.setCullMode(match pipeline.cull_mode {
                api::CullMode::None => objc2_metal::MTLCullMode::None,
                api::CullMode::Front => objc2_metal::MTLCullMode::Front,
                api::CullMode::Back => objc2_metal::MTLCullMode::Back,
            });
            render.setFrontFacingWinding(match pipeline.front_face {
                api::FrontFace::Cw => objc2_metal::MTLWinding::Clockwise,
                api::FrontFace::Ccw => objc2_metal::MTLWinding::CounterClockwise,
            });
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, pipeline);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn set_bind_group(
        encoder: &mut MetalCommandEncoder,
        group_slot: u32,
        group: &MetalBindGroup,
        offsets: &[u32],
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            if group_slot >= 4 {
                return Err(api::Error::InvalidRange);
            }
            let dynamic_count = group
                .resources
                .iter()
                .filter(|resource| {
                    matches!(
                        resource,
                        BoundResource::Buffer {
                            dynamic_offset: true,
                            ..
                        }
                    )
                })
                .count();
            if offsets.len() != dynamic_count {
                return Err(api::Error::InvalidRange);
            }
            // Validate the complete offset slice before changing native encoder
            // state, so a late bad offset cannot leave a partially rebound group.
            let mut dynamic_index = 0;
            for resource in &group.resources {
                if let BoundResource::Buffer {
                    offset,
                    size,
                    allocation_size,
                    dynamic_offset: true,
                    ..
                } = resource
                {
                    crate::backend::bindings::apply_dynamic_uniform_offset(
                        *offset,
                        *size,
                        *allocation_size,
                        offsets[dynamic_index],
                    )?;
                    dynamic_index += 1;
                }
            }
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            let mut dynamic_index = 0;
            for resource in &group.resources {
                match resource {
                    BoundResource::Buffer {
                        binding,
                        visibility,
                        raw,
                        offset,
                        size,
                        allocation_size,
                        dynamic_offset,
                        ..
                    } => {
                        let index = metal_binding_slot(group_slot, *binding)?;
                        let effective_offset = if *dynamic_offset {
                            let dynamic = offsets[dynamic_index];
                            dynamic_index += 1;
                            crate::backend::bindings::apply_dynamic_uniform_offset(
                                *offset,
                                *size,
                                *allocation_size,
                                dynamic,
                            )?
                        } else {
                            *offset
                        };
                        if visibility.contains(&api::ShaderStage::Vertex) {
                            // SAFETY: index is within the fixed 28-slot ABI;
                            // the effective aligned range was validated and the
                            // retained resource remains alive through completion.
                            unsafe {
                                render.setVertexBuffer_offset_atIndex(
                                    Some(raw),
                                    effective_offset as usize,
                                    index as usize,
                                )
                            };
                        }
                        if visibility.contains(&api::ShaderStage::Fragment) {
                            // SAFETY: index is within the fixed 28-slot ABI;
                            // the effective aligned range was validated and the
                            // retained resource remains alive through completion.
                            unsafe {
                                render.setFragmentBuffer_offset_atIndex(
                                    Some(raw),
                                    effective_offset as usize,
                                    index as usize,
                                )
                            };
                        }
                    }
                    BoundResource::Texture {
                        binding,
                        visibility,
                        raw,
                    } => {
                        let index = metal_binding_slot(group_slot, *binding)? as usize;
                        if visibility.contains(&api::ShaderStage::Vertex) {
                            // SAFETY: index is within the fixed 28-slot ABI and
                            // the texture is retained until command completion.
                            unsafe { render.setVertexTexture_atIndex(Some(raw), index) };
                        }
                        if visibility.contains(&api::ShaderStage::Fragment) {
                            // SAFETY: index is within the fixed 28-slot ABI and
                            // the texture is retained until command completion.
                            unsafe { render.setFragmentTexture_atIndex(Some(raw), index) };
                        }
                    }
                    BoundResource::Sampler {
                        binding,
                        visibility,
                        raw,
                    } => {
                        let index = metal_binding_slot(group_slot, *binding)? as usize;
                        if visibility.contains(&api::ShaderStage::Vertex) {
                            // SAFETY: index is within the fixed 28-slot ABI and
                            // the sampler is retained until command completion.
                            unsafe { render.setVertexSamplerState_atIndex(Some(raw), index) };
                        }
                        if visibility.contains(&api::ShaderStage::Fragment) {
                            // SAFETY: index is within the fixed 28-slot ABI and
                            // the sampler is retained until command completion.
                            unsafe { render.setFragmentSamplerState_atIndex(Some(raw), index) };
                        }
                    }
                }
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, group_slot, group, offsets);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn set_vertex_buffer(
        encoder: &mut MetalCommandEncoder,
        slot: u32,
        buffer: &MetalBuffer,
        offset: u64,
        size: u64,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            let native_index = vertex_buffer_index(slot)?;
            if offset
                .checked_add(size)
                .is_none_or(|end| end > buffer.size())
            {
                return Err(api::Error::InvalidRange);
            }
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            // SAFETY: the public slot maps to a bounded native index outside
            // the resource-binding range, and the byte range fits the buffer.
            unsafe {
                render.setVertexBuffer_offset_atIndex(
                    Some(&buffer.raw),
                    offset as usize,
                    native_index,
                )
            };
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, slot, buffer, offset, size);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn set_index_buffer(
        encoder: &mut MetalCommandEncoder,
        buffer: &MetalBuffer,
        offset: u64,
        size: u64,
        format: api::IndexFormat,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if offset
                .checked_add(size)
                .is_none_or(|end| end > buffer.size())
            {
                return Err(api::Error::InvalidRange);
            }
            encoder.index_buffer = Some((buffer.raw.clone(), offset, format, size));
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, buffer, offset, size, format);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }

    fn viewport(encoder: &mut MetalCommandEncoder, viewport: api::Viewport) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLRenderCommandEncoder as _, MTLViewport};
            if !viewport.x.is_finite()
                || !viewport.y.is_finite()
                || !viewport.width.is_finite()
                || !viewport.height.is_finite()
                || !viewport.min_depth.is_finite()
                || !viewport.max_depth.is_finite()
                || viewport.width < 0.0
                || viewport.height < 0.0
                || viewport.min_depth < 0.0
                || viewport.max_depth > 1.0
                || viewport.min_depth > viewport.max_depth
            {
                return Err(api::Error::InvalidRange);
            }
            encoder
                .render
                .as_ref()
                .ok_or(api::Error::InvalidState)?
                .setViewport(MTLViewport {
                    originX: viewport.x as f64,
                    originY: viewport.y as f64,
                    width: viewport.width as f64,
                    height: viewport.height as f64,
                    znear: viewport.min_depth as f64,
                    zfar: viewport.max_depth as f64,
                });
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, viewport);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn scissor(encoder: &mut MetalCommandEncoder, rect: api::Rect) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLRenderCommandEncoder as _, MTLScissorRect};
            encoder
                .render
                .as_ref()
                .ok_or(api::Error::InvalidState)?
                .setScissorRect(MTLScissorRect {
                    x: rect.x as usize,
                    y: rect.y as usize,
                    width: rect.width as usize,
                    height: rect.height as usize,
                });
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, rect);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn blend_constant(encoder: &mut MetalCommandEncoder, value: [f32; 4]) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            encoder
                .render
                .as_ref()
                .ok_or(api::Error::InvalidState)?
                .setBlendColorRed_green_blue_alpha(value[0], value[1], value[2], value[3]);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, value);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn stencil_reference(encoder: &mut MetalCommandEncoder, value: u32) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLRenderCommandEncoder as _;
            encoder
                .render
                .as_ref()
                .ok_or(api::Error::InvalidState)?
                .setStencilReferenceValue(value);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, value);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn draw(
        encoder: &mut MetalCommandEncoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLPrimitiveType, MTLRenderCommandEncoder as _};
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            // SAFETY: Core validated ordered ranges; this path only exposes the
            // TriangleList pipeline and forwards the declared ranges unchanged.
            unsafe {
                render.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    MTLPrimitiveType::Triangle,
                    vertices.start as usize,
                    (vertices.end - vertices.start) as usize,
                    (instances.end - instances.start) as usize,
                    instances.start as usize,
                )
            };
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, vertices, instances);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn draw_indexed(
        encoder: &mut MetalCommandEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLIndexType, MTLPrimitiveType, MTLRenderCommandEncoder as _};
            let (buffer, base_offset, format, range_size) = encoder
                .index_buffer
                .as_ref()
                .ok_or(api::Error::InvalidState)?;
            let index_size = match format {
                api::IndexFormat::Uint16 => 2u64,
                api::IndexFormat::Uint32 => 4u64,
            };
            let byte_start = u64::from(indices.start)
                .checked_mul(index_size)
                .ok_or(api::Error::InvalidRange)?;
            let byte_count = u64::from(indices.end - indices.start)
                .checked_mul(index_size)
                .ok_or(api::Error::InvalidRange)?;
            if byte_start
                .checked_add(byte_count)
                .is_none_or(|end| end > *range_size)
            {
                return Err(api::Error::InvalidRange);
            }
            let byte_offset = base_offset
                .checked_add(byte_start)
                .ok_or(api::Error::InvalidRange)?;
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            // SAFETY: index range was bounded against the retained buffer range;
            // instance and index counts were checked by Core.
            unsafe {
                render.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    MTLPrimitiveType::Triangle,
                    (indices.end - indices.start) as usize,
                    match format {
                        api::IndexFormat::Uint16 => MTLIndexType::UInt16,
                        api::IndexFormat::Uint32 => MTLIndexType::UInt32,
                    },
                    buffer,
                    byte_offset as usize,
                    (instances.end - instances.start) as usize,
                    base_vertex as isize,
                    instances.start as usize,
                )
            };
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, indices, base_vertex, instances);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn end_raster(encoder: &mut MetalCommandEncoder) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLCommandEncoder as _;
            let render = encoder.render.take().ok_or(api::Error::InvalidState)?;
            render.endEncoding();
            encoder.raster_signature = None;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = encoder;
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn abort_raster(encoder: &mut MetalCommandEncoder) {
        #[cfg(target_vendor = "apple")]
        {
            if let Some(render) = encoder.render.take() {
                use objc2_metal::MTLCommandEncoder as _;
                render.endEncoding();
            }
            encoder.raster_signature = None;
        }
        #[cfg(not(target_vendor = "apple"))]
        let _ = encoder;
    }
}

#[allow(dead_code)] // Used by native Apple lowering; host tests exercise the ABI mapping.
fn metal_binding_slot(group: u32, binding: u32) -> api::Result<u32> {
    if group >= 4 || binding >= 7 {
        return Err(api::Error::InvalidRange);
    }
    Ok(group * 7 + binding)
}

#[cfg(target_vendor = "apple")]
fn load_action(
    load: api::LoadOp<api::ClearColor>,
) -> (
    objc2_metal::MTLLoadAction,
    Option<objc2_metal::MTLClearColor>,
) {
    use objc2_metal::{MTLClearColor, MTLLoadAction};
    match load {
        api::LoadOp::Load => (MTLLoadAction::Load, None),
        api::LoadOp::Discard => (MTLLoadAction::DontCare, None),
        api::LoadOp::Clear(api::ClearColor::Float(value)) => (
            MTLLoadAction::Clear,
            Some(MTLClearColor {
                red: value[0] as f64,
                green: value[1] as f64,
                blue: value[2] as f64,
                alpha: value[3] as f64,
            }),
        ),
        api::LoadOp::Clear(api::ClearColor::Uint(value)) => (
            MTLLoadAction::Clear,
            Some(MTLClearColor {
                red: value[0] as f64,
                green: value[1] as f64,
                blue: value[2] as f64,
                alpha: value[3] as f64,
            }),
        ),
        api::LoadOp::Clear(api::ClearColor::Sint(value)) => (
            MTLLoadAction::Clear,
            Some(MTLClearColor {
                red: value[0] as f64,
                green: value[1] as f64,
                blue: value[2] as f64,
                alpha: value[3] as f64,
            }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::metal_binding_slot;
    use fluxel_rhi_core::api::Error;

    #[test]
    fn binding_slots_use_disjoint_fixed_group_ranges() {
        assert_eq!(metal_binding_slot(0, 0).unwrap(), 0);
        assert_eq!(metal_binding_slot(1, 0).unwrap(), 7);
        assert_eq!(metal_binding_slot(3, 6).unwrap(), 27);
        assert!(matches!(metal_binding_slot(4, 0), Err(Error::InvalidRange)));
        assert!(matches!(metal_binding_slot(0, 7), Err(Error::InvalidRange)));
    }
}
