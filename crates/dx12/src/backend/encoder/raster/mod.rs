//! Raster-pass lowering to one open D3D12 graphics command list.
//!
//! Core validates command ordering.  This module installs attachments,
//! pipeline state, bindings and draw calls immediately; it does not retain a
//! CPU draw stream or infer cross-pass hazards.  `Dx12Encoder` holds every
//! COM resource and typed wrapper referenced by a recorded call until submit
//! transfers the encoder into fence-retired work.

use super::*;

mod mrt;
#[path = "native.rs"]
mod native;
pub(crate) use mrt::RasterState;

pub(crate) use native::barrier;
pub(crate) fn validate_read_only(
    state: &RasterState,
    pipeline: &Dx12RasterPipeline,
) -> Result<(), Error> {
    if (state.depth_read_only && pipeline.0.depth_writes)
        || (state.stencil_read_only && pipeline.0.stencil_writes)
    {
        Err(Error::InvalidState)
    } else {
        Ok(())
    }
}
pub(super) fn validate_raw_bindings(
    pipeline: &Dx12RasterPipeline,
    groups: &[Option<Dx12BindGroup>],
) -> Result<(), Error> {
    for &(slot, binding) in &pipeline.0.raw_storage_bindings {
        let group = groups
            .get(slot as usize)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidState)?;
        let expected = pipeline
            .0
            .layout
            .groups
            .get(slot as usize)
            .ok_or(Error::InvalidState)?;
        if !std::sync::Arc::ptr_eq(expected, &group.0.layout)
            || !group.0.storage_views_are_raw(binding)
        {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}
pub(super) fn index_format_matches(
    expected: Option<api::IndexFormat>,
    actual: Option<api::IndexFormat>,
) -> bool {
    match expected {
        None => true,
        Some(api::IndexFormat::Uint16) => matches!(actual, Some(api::IndexFormat::Uint16)),
        Some(api::IndexFormat::Uint32) => matches!(actual, Some(api::IndexFormat::Uint32)),
    }
}
#[cfg(test)]
#[path = "../../../tests/backend/encoder/raster.rs"]
mod topology_tests;
impl fluxel_rhi_core::backend::RasterBackend for Dx12 {
    const DYNAMIC_DEPTH_BIAS_AVAILABLE: bool = true;
    const RECT_ATTACHMENT_CLEAR_AVAILABLE: bool = true;
    fn begin_raster(
        encoder: &mut Dx12Encoder,
        desc: &api::RasterPassDesc<'_, Self>,
    ) -> Result<(), Error> {
        let state = mrt::begin(encoder, desc)?;
        encoder.raster_formats = state.formats.clone();
        encoder.raster_depth_format = state.depth_format;
        encoder.raster_samples = state.samples;
        encoder.raster_multiview = state.multiview;
        encoder.raster_state = Some(state);
        Ok(())
    }
    fn set_pipeline(
        encoder: &mut Dx12Encoder,
        pipeline: &super::Dx12RasterPipeline,
    ) -> Result<(), Error> {
        validate_read_only(
            encoder.raster_state.as_ref().ok_or(Error::InvalidState)?,
            pipeline,
        )?;
        if pipeline.0.color_formats != encoder.raster_formats
            || pipeline.0.depth_format != encoder.raster_depth_format
            || encoder
                .raster_samples
                .is_some_and(|samples| samples != pipeline.0.samples)
            || encoder.raster_multiview != pipeline.0.multiview
        {
            return Err(Error::InvalidState);
        }
        if encoder.pipeline.as_ref().map_or(true, |old| {
            !std::sync::Arc::ptr_eq(&old.0.layout, &pipeline.0.layout)
        }) {
            encoder.raster_bound_groups.clear();
            encoder.raster_snapshot.clear_roots();
        }
        // SAFETY: pipeline owns the root signature and PSO COM references, and
        // the encoder retains its clone through command-list execution.
        unsafe {
            encoder.list.SetGraphicsRootSignature(&pipeline.0.root);
            encoder.list.SetPipelineState(&pipeline.0.state);
            encoder.list.IASetPrimitiveTopology(pipeline.0.topology);
        }
        encoder.pipeline = Some(pipeline.clone());
        if !pipeline.0.dynamic_depth_bias {
            encoder.raster_snapshot.depth_bias = None;
        }
        encoder.raster_pipelines.push(pipeline.clone());
        encoder.ray_pipeline = None;
        encoder.mesh_pipeline = None;
        Ok(())
    }
    fn set_bind_group(
        encoder: &mut Dx12Encoder,
        slot: u32,
        group: &super::Dx12BindGroup,
        offsets: &[u32],
    ) -> Result<(), Error> {
        let layout = if let Some(mesh) = &encoder.mesh_pipeline {
            &mesh.0.layout
        } else {
            &encoder
                .pipeline
                .as_ref()
                .ok_or(Error::InvalidState)?
                .0
                .layout
        };
        let expected = layout
            .groups
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        if !std::sync::Arc::ptr_eq(expected, &group.0.layout) {
            return Err(Error::InvalidState);
        }
        let table = layout
            .tables
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        let native = &group.0;
        if native.has_views() != table.views.is_some()
            || native.has_samplers() != table.samplers.is_some()
        {
            return Err(Error::InvalidState);
        }
        if offsets.len() != table.dynamic_bindings.len() {
            return Err(Error::InvalidRange);
        }
        let mut addresses = Vec::with_capacity(offsets.len());
        let mut storage_offsets = Vec::new();
        for (&(binding, class), &offset) in table.dynamic_bindings.iter().zip(offsets) {
            if class != super::super::descriptor::DescriptorClass::Cbv {
                storage_offsets.push((binding, offset));
                continue;
            }
            let root = table
                .dynamic_cbvs
                .iter()
                .find(|root| root.binding == binding)
                .ok_or(Error::InvalidState)?;
            let uniform = native
                .uniform(root.binding)
                .filter(|u| u.dynamic_offset)
                .ok_or(Error::InvalidState)?;
            let offset = u64::from(offset);
            if offset % 256 != 0 {
                return Err(Error::InvalidRange);
            }
            let start = uniform
                .range
                .start
                .checked_add(offset)
                .ok_or(Error::InvalidRange)?;
            let end = uniform
                .range
                .end
                .checked_add(offset)
                .ok_or(Error::InvalidRange)?;
            if end > uniform.buffer.size {
                return Err(Error::InvalidRange);
            }
            // SAFETY: the group retains this same-device buffer.
            let address = unsafe { uniform.buffer.resource.GetGPUVirtualAddress() }
                .checked_add(start)
                .ok_or(Error::InvalidRange)?;
            if address % 256 != 0 {
                return Err(Error::InvalidRange);
            }
            addresses.push((root.root, address));
        }
        let dynamic_views = native.prepare_dynamic_views(&storage_offsets)?;
        // SAFETY: every table and dynamic address has been validated before
        // modifying the list; the group is retained below until completion.
        unsafe {
            if let Some(root) = table.views {
                encoder.list.SetGraphicsRootDescriptorTable(
                    root,
                    dynamic_views
                        .as_ref()
                        .map_or_else(|| native.view_gpu(), |v| v.gpu()),
                );
            }
            if let Some(root) = table.samplers {
                encoder
                    .list
                    .SetGraphicsRootDescriptorTable(root, native.sampler_gpu());
            }
            for &(root, address) in &addresses {
                encoder
                    .list
                    .SetGraphicsRootConstantBufferView(root, address);
            }
        }
        if let Some(root) = table.views {
            encoder.raster_snapshot.roots.insert(
                root,
                super::raster_snapshot::RootValue::Table(
                    dynamic_views
                        .as_ref()
                        .map_or_else(|| native.view_gpu(), |v| v.gpu()),
                ),
            );
        }
        if let Some(root) = table.samplers {
            encoder.raster_snapshot.roots.insert(
                root,
                super::raster_snapshot::RootValue::Table(native.sampler_gpu()),
            );
        }
        for (root, address) in addresses {
            encoder
                .raster_snapshot
                .roots
                .insert(root, super::raster_snapshot::RootValue::Cbv(address));
        }
        encoder.bound_groups.push(group.clone());
        if let Some(table) = dynamic_views {
            encoder.dynamic_view_tables.push(table);
        }
        let slot = slot as usize;
        if encoder.raster_bound_groups.len() <= slot {
            encoder.raster_bound_groups.resize_with(slot + 1, || None);
        }
        encoder.raster_bound_groups[slot] = Some(group.clone());
        Ok(())
    }
    fn set_vertex_buffer(
        encoder: &mut Dx12Encoder,
        slot: u32,
        buffer: &super::Dx12Buffer,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        let pipeline = encoder.pipeline.as_ref().ok_or(Error::InvalidState)?;
        let stride = *pipeline
            .0
            .vertex_strides
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        if offset.checked_add(size).ok_or(Error::InvalidRange)? > buffer.size
            || size > u64::from(u32::MAX)
        {
            return Err(Error::InvalidRange);
        }
        let view = D3D12_VERTEX_BUFFER_VIEW {
            // SAFETY: the buffer is a live same-device COM resource; querying
            // its immutable GPU virtual address has no ownership side effect.
            BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() } + offset,
            SizeInBytes: size as u32,
            StrideInBytes: stride,
        };
        // SAFETY: `view` names a checked byte range of the retained buffer and
        // this encoder exclusively records its open graphics list.
        unsafe {
            encoder
                .list
                .IASetVertexBuffers(slot, Some(std::slice::from_ref(&view)));
        }
        encoder.raster_snapshot.vertices.insert(slot, view);
        encoder.buffers.push(buffer.clone());
        Ok(())
    }
    fn set_index_buffer(
        encoder: &mut Dx12Encoder,
        buffer: &super::Dx12Buffer,
        offset: u64,
        size: u64,
        format: api::IndexFormat,
    ) -> Result<(), Error> {
        if offset.checked_add(size).ok_or(Error::InvalidRange)? > buffer.size
            || size > u64::from(u32::MAX)
        {
            return Err(Error::InvalidRange);
        }
        encoder.raster_index_format = Some(format);
        let format = match format {
            api::IndexFormat::Uint16 => {
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT
            }
            api::IndexFormat::Uint32 => {
                windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT
            }
        };
        let view = D3D12_INDEX_BUFFER_VIEW {
            // SAFETY: as for the vertex buffer, the retained resource exposes
            // a stable GPU address for the checked index range.
            BufferLocation: unsafe { buffer.resource.GetGPUVirtualAddress() } + offset,
            SizeInBytes: size as u32,
            Format: format,
        };
        // SAFETY: the index view is fully initialized from a retained resource.
        unsafe {
            encoder.list.IASetIndexBuffer(Some(&view));
        }
        encoder.raster_snapshot.index = Some(view);
        encoder.buffers.push(buffer.clone());
        Ok(())
    }
    fn viewport(encoder: &mut Dx12Encoder, v: api::Viewport) -> Result<(), Error> {
        let viewport = validate_viewport(v)?;
        encoder.raster_snapshot.viewport = Some(viewport);
        // View instancing selects viewport slots 0 and 1. Reapply both after
        // the caller overrides the pass default, or the second view is clipped.
        // SAFETY: `validate_viewport` checked finite dimensions and this open
        // command list is exclusively owned by the encoder.
        unsafe {
            if encoder.raster_multiview.is_some() {
                encoder.list.RSSetViewports(&[viewport, viewport]);
            } else {
                encoder.list.RSSetViewports(&[viewport]);
            }
        }
        Ok(())
    }
    fn scissor(encoder: &mut Dx12Encoder, r: api::Rect) -> Result<(), Error> {
        let rect = validate_scissor(r)?;
        encoder.raster_snapshot.scissor = Some(rect);
        // Match the two viewport slots selected by the view-instanced PSO.
        // SAFETY: `validate_scissor` checked integer bounds for this list.
        unsafe {
            if encoder.raster_multiview.is_some() {
                encoder.list.RSSetScissorRects(&[rect, rect]);
            } else {
                encoder.list.RSSetScissorRects(&[rect]);
            }
        }
        Ok(())
    }
    fn blend_constant(encoder: &mut Dx12Encoder, value: [f32; 4]) -> Result<(), Error> {
        if value.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidRange);
        }
        // SAFETY: D3D12 copies the values into this encoder's command list.
        unsafe { encoder.list.OMSetBlendFactor(Some(&value)) };
        encoder.raster_snapshot.blend = Some(value);
        Ok(())
    }
    fn stencil_reference(encoder: &mut Dx12Encoder, value: u32) -> Result<(), Error> {
        // SAFETY: this records a scalar value into the open graphics list.
        unsafe { encoder.list.OMSetStencilRef(value & 0xff) };
        encoder.raster_snapshot.stencil = Some(value & 0xff);
        Ok(())
    }
    fn clear_rect(
        encoder: &mut Dx12Encoder,
        rect: api::Rect,
        colors: &[api::ColorAttachmentClear],
        depth: Option<f32>,
        stencil: Option<u32>,
    ) -> Result<(), Error> {
        let rect = validate_scissor(rect)?;
        let (rtvs, dsv, formats, integer_targets, multiview_mask) = {
            let state = encoder.raster_state.as_ref().ok_or(Error::InvalidState)?;
            (
                state.rtvs.clone(),
                state.dsv,
                state.formats.clone(),
                state.integer_clear_targets.clone(),
                state.multiview_mask,
            )
        };
        // D3D12's ordinary clear calls do not carry a view-instance mask.
        // Reject the whole operation rather than clearing only the base RTV
        // or DSV of a multiview pass.
        if multiview_mask.is_some() {
            return Err(Error::Unsupported(
                "DX12 multiview raster attachment rect clear",
            ));
        }
        // SAFETY: Core checked selected attachment slots and depth/stencil
        // writability; the state above retains the CPU descriptors for this
        // open pass. Native clears do not alter graphics pipeline or scissor
        // state.
        unsafe {
            for clear in colors {
                let slot = clear.attachment as usize;
                let format = formats
                    .get(slot)
                    .and_then(|format| *format)
                    .ok_or(Error::InvalidState)?;
                if super::super::texture::integer_color(format) {
                    let target = integer_targets
                        .get(slot)
                        .and_then(Option::as_ref)
                        .ok_or(Error::InvalidState)?;
                    let (signed, bits) =
                        super::super::texture::integer_clear_bits(format, clear.value)
                            .ok_or(Error::InvalidState)?;
                    let pipeline = encoder.device.0.integer_clear_cache.get_or_create(
                        &encoder.device,
                        target.format,
                        target.samples,
                        signed,
                    )?;
                    for (heap, rtv) in &target.views {
                        super::integer_clear::record(
                            &encoder.list,
                            &pipeline,
                            *rtv,
                            target.extent.0 as u32,
                            target.extent.1,
                            rect,
                            bits,
                        )?;
                        if let Some(heap) = heap {
                            encoder.clear_heaps.push(heap.clone());
                        }
                    }
                    encoder.integer_clear_pipelines.push(pipeline);
                    super::restore_after_internal_draw(encoder)?;
                    encoder.list.OMSetRenderTargets(
                        rtvs.len() as u32,
                        Some(rtvs.as_ptr()),
                        false,
                        dsv.as_ref().map(|handle| handle as *const _),
                    );
                    continue;
                }
                let api::ClearColor::Float(value) = clear.value else {
                    return Err(Error::InvalidState);
                };
                let rtv = *rtvs.get(slot).ok_or(Error::InvalidState)?;
                encoder
                    .list
                    .ClearRenderTargetView(rtv, &value, Some(&[rect]));
            }
            if depth.is_some() || stencil.is_some() {
                let dsv = dsv.ok_or(Error::InvalidState)?;
                let mut flags = D3D12_CLEAR_FLAGS(0);
                if depth.is_some() {
                    flags |= D3D12_CLEAR_FLAG_DEPTH;
                }
                if stencil.is_some() {
                    flags |= D3D12_CLEAR_FLAG_STENCIL;
                }
                encoder.list.ClearDepthStencilView(
                    dsv,
                    flags,
                    depth.unwrap_or(0.0),
                    stencil.unwrap_or(0) as u8,
                    Some(&[rect]),
                );
            }
        }
        Ok(())
    }
    fn depth_bias(encoder: &mut Dx12Encoder, value: api::DynamicDepthBias) -> Result<(), Error> {
        if !value.slope_scale.is_finite() || !value.clamp.is_finite() {
            return Err(Error::InvalidRange);
        }
        let list9 = encoder
            .list
            .cast::<windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList9>()
            .map_err(super::super::native_error)?;
        // SAFETY: the active PSO was created with DYNAMIC_DEPTH_BIAS and the
        // call copies finite scalar factors into this open command list.
        unsafe { list9.RSSetDepthBias(value.constant as f32, value.clamp, value.slope_scale) };
        encoder.raster_snapshot.depth_bias = Some(value);
        Ok(())
    }
    fn draw(
        encoder: &mut Dx12Encoder,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> Result<(), Error> {
        if vertices.start > vertices.end || instances.start > instances.end {
            return Err(Error::InvalidRange);
        }
        let pipeline = encoder.pipeline.as_ref().ok_or(Error::InvalidState)?;
        validate_raw_bindings(pipeline, &encoder.raster_bound_groups)?;
        // SAFETY: Core checked pass/pipeline sequencing and range arithmetic;
        // this only appends a draw to the encoder-owned list.
        unsafe {
            if let Some(root) = pipeline.0.vertex_index_root {
                let values = [vertices.start, instances.start, 0];
                encoder
                    .list
                    .SetGraphicsRoot32BitConstants(root, 3, values.as_ptr().cast(), 0);
            }
            encoder.list.DrawInstanced(
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
        Ok(())
    }
    fn draw_indexed(
        encoder: &mut Dx12Encoder,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> Result<(), Error> {
        if indices.start > indices.end || instances.start > instances.end {
            return Err(Error::InvalidRange);
        }
        let pipeline = encoder.pipeline.as_ref().ok_or(Error::InvalidState)?;
        validate_raw_bindings(pipeline, &encoder.raster_bound_groups)?;
        if encoder.raster_index_format.is_none()
            || !index_format_matches(pipeline.0.strip_index_format, encoder.raster_index_format)
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: as above, the checked index/instance ranges are copied into
        // the encoder-owned command list without borrowing caller memory.
        unsafe {
            if let Some(root) = pipeline.0.vertex_index_root {
                let values = [base_vertex as u32, instances.start, 0];
                encoder
                    .list
                    .SetGraphicsRoot32BitConstants(root, 3, values.as_ptr().cast(), 0);
            }
            encoder.list.DrawIndexedInstanced(
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            );
        }
        Ok(())
    }
    fn end_raster(encoder: &mut Dx12Encoder) -> Result<(), Error> {
        let state = encoder.raster_state.take().ok_or(Error::InvalidState)?;
        mrt::end(encoder, state)?;
        encoder.mesh_pipeline = None;
        encoder.pipeline = None;
        encoder.raster_snapshot.clear_roots();
        encoder.raster_bound_groups.clear();
        encoder.raster_formats.clear();
        encoder.raster_depth_format = None;
        encoder.raster_samples = None;
        encoder.raster_multiview = None;
        Ok(())
    }
    fn abort_raster(_: &mut Dx12Encoder) {}
}
