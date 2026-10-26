//! Immediate Metal raster encoder operations and WebGPU-coordinate lowering.
//!
//! Pass lifetime and draw calls are sent directly to the active native render encoder.

mod bindings;
mod clear_area;
#[cfg(target_vendor = "apple")]
mod partial_resolve;
mod pass;
#[cfg(target_vendor = "apple")]
pub(super) mod resume;
#[cfg(target_vendor = "apple")]
pub(super) use bindings::write_push;

use std::ops::Range;

use fluxel_rhi_core::api::{self};
use fluxel_rhi_core::backend::RasterBackend;

use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalRasterPipeline};
use crate::backend::MetalBindGroup;

#[cfg(target_vendor = "apple")]
pub(super) fn for_view_batches(
    encoder: &MetalCommandEncoder,
    mut draw: impl FnMut(&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>),
) -> api::Result<()> {
    use objc2_metal::{
        MTLCommandBuffer as _, MTLDevice as _, MTLRenderCommandEncoder as _,
        MTLVertexAmplificationViewMapping,
    };
    let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
    let views: Vec<u32> = encoder.raster_view_mask.map_or_else(
        || vec![0],
        |mask| (0..32).filter(|view| mask & (1 << view) != 0).collect(),
    );
    let max = if encoder.raster_view_mask.is_some() {
        let device = encoder.command.device();
        [8, 4, 2, 1]
            .into_iter()
            .find(|count| device.supportsVertexAmplificationCount(*count))
            .ok_or(api::Error::Unsupported(
                "Metal vertex amplification unavailable",
            ))?
    } else {
        1
    };
    for chunk in views.chunks(max) {
        if encoder.raster_view_mask.is_some() {
            let mappings: Vec<_> = chunk
                .iter()
                .map(|view| MTLVertexAmplificationViewMapping {
                    viewportArrayIndexOffset: 0,
                    renderTargetArrayIndexOffset: *view,
                })
                .collect();
            // SAFETY: every batch is bounded by the runtime native amplification limit.
            unsafe {
                render.setVertexAmplificationCount_viewMappings(mappings.len(), mappings.as_ptr())
            };
        }
        for stage in &encoder.raster_bindings {
            if let Some(slot) = stage.map.view_slot {
                let ptr = std::ptr::NonNull::new(chunk.as_ptr() as *mut _)
                    .ok_or(api::Error::InvalidState)?;
                let length = std::mem::size_of_val(chunk);
                // SAFETY: the pipeline reserves this slot and Metal copies the batch mapping.
                unsafe {
                    match stage.stage {
                        api::ShaderStage::Vertex => {
                            render.setVertexBytes_length_atIndex(ptr, length, slot as usize)
                        }
                        api::ShaderStage::Fragment => {
                            render.setFragmentBytes_length_atIndex(ptr, length, slot as usize)
                        }
                        _ => return Err(api::Error::InvalidState),
                    }
                };
            }
        }
        draw(render);
    }
    Ok(())
}

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
            if encoder.raster_signature.as_ref() != Some(&pipeline.signature) {
                return Err(api::Error::InvalidState);
            }
            match (encoder.raster_view_mask, pipeline.view_count) {
                (None, None) => {}
                (Some(mask), Some(count)) if 32 - mask.leading_zeros() <= count => {}
                _ => return Err(api::Error::InvalidState),
            }
            if (encoder.raster_read_only.0 && pipeline.writes_depth)
                || (encoder.raster_read_only.1 && pipeline.writes_stencil)
            {
                return Err(api::Error::InvalidState);
            }
            encoder.mesh_threads = None;
            encoder.raster_bindings = pipeline.bindings.clone();
            encoder.raster_group_count = pipeline.layout.groups.len();
            encoder.raster_vertex_base = pipeline.vertex_buffer_base;
            encoder.raster_push_ranges = pipeline.layout.push_constants.clone();
            encoder.primitive_type = pipeline.primitive_type;
            encoder.strip_index_format = pipeline.strip_index_format;
            render.setTriangleFillMode(match pipeline.polygon_mode {
                api::PolygonMode::Fill => objc2_metal::MTLTriangleFillMode::Fill,
                api::PolygonMode::Line => objc2_metal::MTLTriangleFillMode::Lines,
                _ => return Err(api::Error::Unsupported("Metal polygon mode")),
            });
            render.setDepthClipMode(if pipeline.unclipped_depth {
                objc2_metal::MTLDepthClipMode::Clamp
            } else {
                objc2_metal::MTLDepthClipMode::Clip
            });
            encoder.raster_pipeline_bound = true;
            encoder.raster_pipeline = Some(pipeline.clone());
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
            for (slot, binding) in encoder
                .raster_vertex_buffers
                .clone()
                .into_iter()
                .enumerate()
            {
                if let Some((buffer, offset, size)) = binding {
                    if encoder.raster_vertex_base + slot < 31 {
                        Self::set_vertex_buffer(encoder, slot as u32, &buffer, offset, size)?;
                    }
                }
            }
            bindings::bind_raster_push(encoder)?;
            for (slot, (group, offsets)) in encoder.raster_groups.clone() {
                if (slot as usize) < pipeline.layout.groups.len() {
                    Self::set_bind_group(encoder, slot, &group, &offsets)?;
                }
            }
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
        bindings::set_bind_group(encoder, group_slot, group, offsets)
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
            let slot = usize::try_from(slot).map_err(|_| api::Error::InvalidRange)?;
            if slot >= 31 {
                return Err(api::Error::InvalidRange);
            }
            if offset
                .checked_add(size)
                .is_none_or(|end| end > buffer.size())
            {
                return Err(api::Error::InvalidRange);
            }
            let render = encoder.render.as_ref().ok_or(api::Error::InvalidState)?;
            if encoder.raster_vertex_buffers.len() <= slot {
                encoder.raster_vertex_buffers.resize(slot + 1, None);
            }
            encoder.raster_vertex_buffers[slot] = Some((buffer.clone(), offset, size));
            if encoder.raster_pipeline_bound {
                let native_index = encoder
                    .raster_vertex_base
                    .checked_add(slot)
                    .ok_or(api::Error::InvalidRange)?;
                if native_index >= 31 {
                    return Err(api::Error::InvalidRange);
                }
                // SAFETY: slot and range are bounded and the command buffer retains the buffer.
                unsafe {
                    render.setVertexBuffer_offset_atIndex(
                        Some(&buffer.raw),
                        offset as usize,
                        native_index,
                    )
                };
            }
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
            encoder.raster_viewport = Some(viewport);
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
            let area = encoder.raster_area.ok_or(api::Error::InvalidState)?;
            let x = rect.x.max(area.x).min(area.x.saturating_add(area.width));
            let y = rect.y.max(area.y).min(area.y.saturating_add(area.height));
            let right = rect
                .x
                .saturating_add(rect.width)
                .min(area.x.saturating_add(area.width));
            let bottom = rect
                .y
                .saturating_add(rect.height)
                .min(area.y.saturating_add(area.height));
            encoder
                .render
                .as_ref()
                .ok_or(api::Error::InvalidState)?
                .setScissorRect(MTLScissorRect {
                    x: x as usize,
                    y: y as usize,
                    width: right.saturating_sub(x) as usize,
                    height: bottom.saturating_sub(y) as usize,
                });
            encoder.raster_scissor = Some(rect);
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
            encoder.raster_blend_constant = value;
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
            encoder.raster_stencil_reference = value;
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
            use objc2_metal::MTLRenderCommandEncoder as _;
            // SAFETY: Core validated ordered ranges; this path only exposes the
            // TriangleList pipeline and forwards the declared ranges unchanged.
            for_view_batches(encoder, |render| unsafe {
                render.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    encoder.primitive_type,
                    vertices.start as usize,
                    (vertices.end - vertices.start) as usize,
                    (instances.end - instances.start) as usize,
                    instances.start as usize,
                )
            })?;
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
            use objc2_metal::{MTLIndexType, MTLRenderCommandEncoder as _};
            let (buffer, base_offset, format, range_size) = encoder
                .index_buffer
                .as_ref()
                .ok_or(api::Error::InvalidState)?;
            if encoder.strip_index_format.is_some_and(|expected| {
                std::mem::discriminant(&expected) != std::mem::discriminant(format)
            }) {
                return Err(api::Error::InvalidState);
            }
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
            // SAFETY: index range was bounded against the retained buffer range;
            // instance and index counts were checked by Core.
            for_view_batches(encoder, |render| unsafe {
                render.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    encoder.primitive_type,
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
            })?;
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
            if encoder.occlusion_active {
                return Err(api::Error::InvalidState);
            }
            let render = encoder.render.take().ok_or(api::Error::InvalidState)?;
            render.endEncoding();
            resume::statistics_boundary(encoder)?;
            super::super::query::finish_occlusion_segments(encoder)?;
            for (source, target, area, mask) in std::mem::take(&mut encoder.raster_resolves) {
                partial_resolve::resolve_area(&encoder.command, &source, &target, area, mask)?;
            }
            encoder.occlusion_set = None;
            if let Some((set, slot)) = encoder.raster_end_timestamp.take() {
                use fluxel_rhi_core::backend::QueryBackend as _;
                Metal::write_timestamp(encoder, &set, slot)?;
            }
            encoder.raster_statistics = false;
            encoder.raster_split = false;
            encoder.raster_descriptor = None;
            encoder.raster_pipeline = None;
            encoder.raster_viewport = None;
            encoder.raster_scissor = None;
            encoder.occlusion_slot = None;
            encoder.occlusion_segments.clear();
            encoder.raster_resolves.clear();
            encoder.raster_area = None;
            encoder.raster_view_mask = None;
            encoder.raster_signature = None;
            encoder.raster_pipeline_bound = false;
            encoder.mesh_threads = None;
            encoder.raster_bindings.clear();
            encoder.raster_groups.clear();
            encoder.raster_vertex_buffers.clear();
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
            encoder.raster_statistics = false;
            encoder.raster_split = false;
            encoder.raster_descriptor = None;
            encoder.raster_pipeline = None;
            encoder.raster_viewport = None;
            encoder.raster_scissor = None;
            encoder.occlusion_slot = None;
            encoder.occlusion_segments.clear();
            encoder.raster_resolves.clear();
            encoder.raster_area = None;
            encoder.raster_view_mask = None;
            encoder.raster_signature = None;
            encoder.raster_pipeline_bound = false;
            encoder.mesh_threads = None;
            encoder.occlusion_set = None;
            encoder.occlusion_active = false;
            encoder.raster_end_timestamp = None;
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
#[path = "../../../tests/backend/encoder/raster.rs"]
mod tests;
