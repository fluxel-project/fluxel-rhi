//! Core secondary-raster SPI implementation using D3D12 bundles.
//!
//! A secondary encoder owns a D3D12 bundle until it is closed. A finished
//! bundle contains the resources needed by its recorded native calls, and the
//! primary encoder retains a clone after ExecuteBundle until queue retirement.
//! This module validates compatibility with the active raster pass; it does
//! not replay commands or decide resource transitions.

use crate::backend::{Dx12, Dx12Device};
use fluxel_rhi_core::api::{self, Error, Result};

use super::encoder::{Dx12SecondaryEncoder, Dx12SecondaryWork};

impl api::SecondaryBackend for Dx12 {
    type NativeSecondaryEncoder = Dx12SecondaryEncoder;
    type NativeSecondaryWork = Dx12SecondaryWork;

    fn begin_secondary(
        device: &Dx12Device,
        desc: &api::SecondaryRasterDesc,
    ) -> Result<Self::NativeSecondaryEncoder> {
        Dx12SecondaryEncoder::new(device, desc)
    }

    fn secondary_set_pipeline(
        encoder: &mut Self::NativeSecondaryEncoder,
        pipeline: &Self::NativeRasterPipeline,
    ) -> Result<()> {
        super::encoder::secondary::set_pipeline(encoder, pipeline)
    }

    fn secondary_set_bind_group(
        encoder: &mut Self::NativeSecondaryEncoder,
        slot: u32,
        group: &Self::NativeBindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        super::encoder::secondary::set_bind_group(encoder, slot, group, offsets)
    }

    fn secondary_set_vertex_buffer(
        encoder: &mut Self::NativeSecondaryEncoder,
        slot: u32,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        super::encoder::secondary::set_vertex_buffer(encoder, slot, buffer, offset, size)
    }

    fn secondary_set_index_buffer(
        encoder: &mut Self::NativeSecondaryEncoder,
        buffer: &Self::NativeBuffer,
        offset: u64,
        size: u64,
        format: api::IndexFormat,
    ) -> Result<()> {
        super::encoder::secondary::set_index_buffer(encoder, buffer, offset, size, format)
    }

    fn secondary_viewport(
        encoder: &mut Self::NativeSecondaryEncoder,
        viewport: api::Viewport,
    ) -> Result<()> {
        super::encoder::secondary::set_viewport(encoder, viewport)
    }

    fn secondary_scissor(
        encoder: &mut Self::NativeSecondaryEncoder,
        rect: api::Rect,
    ) -> Result<()> {
        super::encoder::secondary::set_scissor(encoder, rect)
    }

    fn secondary_set_push_constants(
        encoder: &mut Self::NativeSecondaryEncoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()> {
        super::encoder::secondary::set_push_constants(encoder, stages, offset, bytes)
    }

    fn secondary_draw(
        encoder: &mut Self::NativeSecondaryEncoder,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> Result<()> {
        super::encoder::secondary::draw(encoder, vertices, instances)
    }

    fn secondary_draw_indexed(
        encoder: &mut Self::NativeSecondaryEncoder,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> Result<()> {
        super::encoder::secondary::draw_indexed(encoder, indices, base_vertex, instances)
    }

    fn finish_secondary(
        mut encoder: Self::NativeSecondaryEncoder,
    ) -> Result<Self::NativeSecondaryWork> {
        if encoder.closed || encoder.pipeline.is_none() {
            return Err(Error::InvalidState);
        }
        // SAFETY: this encoder exclusively owns an open bundle list and the
        // closed flag prevents a second Close call.
        unsafe { encoder.list.Close() }.map_err(super::native_error)?;
        encoder.closed = true;
        Ok(Dx12SecondaryWork {
            segments: encoder.segments,
            allocator: encoder.allocator,
            list: encoder.list,
            buffers: encoder.buffers,
            groups: encoder.groups,
            pipeline: encoder.pipeline,
            pipelines: encoder.pipelines,
            dynamic_view_tables: encoder.dynamic_view_tables,
            signature: encoder.signature,
            viewport: encoder.viewport,
            scissor: encoder.scissor,
        })
    }

    fn execute_secondary(
        encoder: &mut Self::NativeEncoder,
        work: &Self::NativeSecondaryWork,
    ) -> Result<()> {
        let state = encoder.raster_state.as_ref().ok_or(Error::InvalidState)?;
        for pipeline in &work.pipelines {
            if (state.depth_read_only && pipeline.0.depth_writes)
                || (state.stencil_read_only && pipeline.0.stencil_writes)
            {
                return Err(Error::InvalidState);
            }
        }
        if encoder.raster_formats != work.signature.color_formats
            || encoder.raster_depth_format != work.signature.depth_stencil_format
            || encoder.raster_state.as_ref().and_then(|s| s.multiview_mask)
                != work.signature.multiview_mask
            || encoder
                .raster_samples
                .is_some_and(|samples| samples != work.signature.samples)
        {
            return Err(Error::InvalidState);
        }
        for segment in work.segments.iter().chain(std::iter::once(
            &super::encoder::secondary::SecondarySegment {
                list: work.list.clone(),
                viewport: work.viewport,
                scissor: work.scissor,
            },
        )) {
            if let Some(viewport) = segment.viewport {
                // SAFETY: viewport values are copied immediately into the open
                // primary list; the stack array has no lifetime after this call.
                unsafe {
                    let v = windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT {
                        TopLeftX: viewport.x,
                        TopLeftY: viewport.y,
                        Width: viewport.width,
                        Height: viewport.height,
                        MinDepth: viewport.min_depth,
                        MaxDepth: viewport.max_depth,
                    };
                    if encoder.raster_multiview.is_some() {
                        encoder.list.RSSetViewports(&[v, v]);
                    } else {
                        encoder.list.RSSetViewports(&[v]);
                    }
                }
            }
            if let Some(rect) = segment.scissor {
                let right = rect.x.checked_add(rect.width).ok_or(Error::InvalidRange)?;
                let bottom = rect.y.checked_add(rect.height).ok_or(Error::InvalidRange)?;
                // SAFETY: overflow checks above formed a valid RECT, copied
                // immediately by the open primary command list.
                unsafe {
                    let r = windows::Win32::Foundation::RECT {
                        left: rect.x as i32,
                        top: rect.y as i32,
                        right: right as i32,
                        bottom: bottom as i32,
                    };
                    if encoder.raster_multiview.is_some() {
                        encoder.list.RSSetScissorRects(&[r, r]);
                    } else {
                        encoder.list.RSSetScissorRects(&[r]);
                    }
                }
            }
            // SAFETY: finish_secondary closed this bundle once, compatibility was
            // checked above, and `work` is retained by the primary encoder before
            // submission can retire either native command object.
            unsafe { encoder.list.ExecuteBundle(&segment.list) };
        }
        super::encoder::raster_snapshot::restore(encoder)?;
        encoder.secondary.push(work.clone());
        Ok(())
    }
}
