//! Core secondary-raster SPI implementation using D3D12 bundles.

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
        unsafe { encoder.list.Close() }.map_err(super::native_error)?;
        encoder.closed = true;
        Ok(Dx12SecondaryWork {
            allocator: encoder.allocator,
            list: encoder.list,
            buffers: encoder.buffers,
            groups: encoder.groups,
            pipeline: encoder.pipeline,
            signature: encoder.signature,
            viewport: encoder.viewport,
            scissor: encoder.scissor,
        })
    }

    fn execute_secondary(
        encoder: &mut Self::NativeEncoder,
        work: &Self::NativeSecondaryWork,
    ) -> Result<()> {
        if encoder.raster_format != work.signature.color_formats.first().copied().flatten()
            || encoder.raster_samples != Some(work.signature.samples)
        {
            return Err(Error::InvalidState);
        }
        if let Some(group) = work.groups.first() {
            let heaps = [
                Some(group.0.heaps().view_heap().clone()),
                Some(group.0.heaps().sampler_heap().clone()),
            ];
            unsafe { encoder.list.SetDescriptorHeaps(&heaps) };
        }
        if let Some(viewport) = work.viewport {
            unsafe {
                encoder.list.RSSetViewports(&[
                    windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT {
                        TopLeftX: viewport.x,
                        TopLeftY: viewport.y,
                        Width: viewport.width,
                        Height: viewport.height,
                        MinDepth: viewport.min_depth,
                        MaxDepth: viewport.max_depth,
                    },
                ]);
            }
        }
        if let Some(rect) = work.scissor {
            let right = rect.x.checked_add(rect.width).ok_or(Error::InvalidRange)?;
            let bottom = rect.y.checked_add(rect.height).ok_or(Error::InvalidRange)?;
            unsafe {
                encoder
                    .list
                    .RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                        left: rect.x as i32,
                        top: rect.y as i32,
                        right: right as i32,
                        bottom: bottom as i32,
                    }]);
            }
        }
        unsafe { encoder.list.ExecuteBundle(&work.list) };
        encoder.secondary.push(work.clone());
        Ok(())
    }
}
