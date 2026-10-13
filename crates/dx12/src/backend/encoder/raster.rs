//! Raster-pass lowering to one open D3D12 graphics command list.
//!
//! Core validates command ordering.  This module installs attachments,
//! pipeline state, bindings and draw calls immediately; it does not retain a
//! CPU draw stream or infer cross-pass hazards.  `Dx12Encoder` holds every
//! COM resource and typed wrapper referenced by a recorded call until submit
//! transfers the encoder into fence-retired work.

use super::*;

#[path = "raster/native.rs"]
mod native;

pub(crate) use native::{barrier, upload_buffer};
impl api::RasterBackend for Dx12 {
    fn begin_raster(
        encoder: &mut Dx12Encoder,
        desc: &api::RasterPassDesc<'_, Self>,
    ) -> Result<(), Error> {
        if desc.colors.len() != 1 || desc.timestamp_writes.is_some() {
            return Err(Error::Unsupported("DX12 raster attachment configuration"));
        }
        let Some(color) = desc.colors[0].as_ref() else {
            return Err(Error::Unsupported("DX12 empty raster pass"));
        };
        let multiview = match desc.multiview_mask {
            None => None,
            Some(0b11) => Some(2),
            Some(_) => return Err(Error::Unsupported("DX12 multiview mask")),
        };
        if matches!(
            color.load,
            api::LoadOp::Clear(api::ClearColor::Uint(_))
                | api::LoadOp::Clear(api::ClearColor::Sint(_))
        ) {
            return Err(Error::Unsupported("DX12 integer color clears"));
        }
        // Hold a native attachment reference in the encoder. A DXGI backbuffer
        // lease or texture view may otherwise disappear before this recorded
        // list reaches the queue.
        let (rtv, target, surface) = match color.view {
            api::AttachmentView::Surface(image) => {
                // SAFETY: Core tied this image lease to the encoder's device;
                // the lease remains borrowed for this call and its resource is cloned.
                let n = unsafe { image.native_ref() };
                (n.0.rtv, n.0.resource.clone(), true)
            }
            api::AttachmentView::Texture(view) => {
                // SAFETY: this typed view belongs to this encoder's device;
                // we retain its texture state before recording native handles.
                let n = unsafe { view.native_ref() };
                if multiview.is_some()
                    && (n.0.desc.dimension != api::TextureViewDimension::D2Array
                        || n.0.desc.layers != (0..2))
                {
                    return Err(Error::InvalidState);
                }
                encoder.textures.push(Dx12Texture(n.0.texture.clone()));
                (
                    n.0.rtv.ok_or(Error::Unsupported("DX12 texture RTV"))?,
                    n.0.texture.resource.clone(),
                    false,
                )
            }
        };
        // SAFETY: `target` is a retained COM resource; GetDesc only reads its
        // immutable native descriptor.
        let native_desc = unsafe { target.GetDesc() };
        let source_samples = native_desc.SampleDesc.Count;
        let attachment_format = match native_desc.Format {
            DXGI_FORMAT_R8G8B8A8_UNORM => api::TextureFormat::Rgba8Unorm,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM => {
                api::TextureFormat::Bgra8Unorm
            }
            _ => return Err(Error::Unsupported("DX12 color attachment format")),
        };
        let source_format = match color.view {
            api::AttachmentView::Texture(view) => {
                // SAFETY: Core supplied a same-device typed view; only its
                // immutable Rust-side descriptor is read here.
                match unsafe { view.native_ref() }.0.texture.desc.format {
                    api::TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
                    api::TextureFormat::Bgra8Unorm => {
                        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM
                    }
                    _ => return Err(Error::Unsupported("DX12 resolve format")),
                }
            }
            api::AttachmentView::Surface(_) => DXGI_FORMAT_R8G8B8A8_UNORM,
        };
        let resolve = match color.resolve_target {
            Some(api::AttachmentView::Surface(image)) if source_samples > 1 => {
                // SAFETY: resolve target is the current same-device surface
                // lease; cloning its resource extends native lifetime.
                let n = unsafe { image.native_ref() };
                barrier(
                    &encoder.list,
                    &n.0.resource,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                );
                Some(n.0.resource.clone())
            }
            Some(_) => return Err(Error::Unsupported("DX12 MSAA resolve target")),
            None => None,
        };
        // Swapchain images are presented between passes. This local boundary
        // is inherent to DXGI WSI; graph-selected texture states remain in
        // `transition`, and no last-use state is recorded here.
        if surface {
            barrier(
                &encoder.list,
                &target,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
        }
        let (dsv, depth_format) = if let Some(depth) = &desc.depth_stencil {
            let api::AttachmentView::Texture(view) = depth.view else {
                return Err(Error::Unsupported("DX12 surface depth attachment"));
            };
            // SAFETY: Core validated device identity for this attachment view;
            // the retained texture keeps its DSV resource alive.
            let n = unsafe { view.native_ref() };
            if n.0.texture.desc.samples != source_samples {
                return Err(Error::InvalidState);
            }
            encoder.textures.push(Dx12Texture(n.0.texture.clone()));
            let handle = n.0.dsv.ok_or(Error::Unsupported("DX12 depth DSV"))?;
            if depth.depth_read_only || depth.stencil_read_only || depth.stencil.is_some() {
                return Err(Error::Unsupported("DX12 depth attachment mode"));
            }
            if let Some((api::LoadOp::Clear(value), _)) = depth.depth {
                // SAFETY: the DSV belongs to the retained depth resource and
                // the rectangle was derived from the validated pass area.
                unsafe {
                    encoder.list.ClearDepthStencilView(
                        handle,
                        windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAG_DEPTH,
                        value,
                        0,
                        Some(&[RECT {
                            left: desc.render_area.x as i32,
                            top: desc.render_area.y as i32,
                            right: desc.render_area.x.saturating_add(desc.render_area.width) as i32,
                            bottom: desc.render_area.y.saturating_add(desc.render_area.height)
                                as i32,
                        }]),
                    )
                };
            }
            encoder.depth_target = Some(n.0.texture.resource.clone());
            (Some(handle), Some(n.0.texture.desc.format))
        } else {
            (None, None)
        };
        super::secondary::validate_inheritance(
            desc.secondary_inheritance.as_ref(),
            attachment_format,
            depth_format,
            source_samples,
            desc.multiview_mask,
        )?;
        // Attachments, viewport and scissor are installed together: D3D12
        // keeps them as command-list state, so splitting them lets a secondary
        // command inherit a half-configured render target. Multiview repeats
        // both viewport slots because the view-instanced PSO addresses 0 and 1.
        // SAFETY: encoder exclusively owns this open list; all referenced RTV,
        // DSV and resources are retained in the encoder through submission.
        unsafe {
            encoder.list.OMSetRenderTargets(
                1,
                Some(&rtv),
                false,
                dsv.as_ref().map(|handle| handle as *const _),
            );
            let viewport = D3D12_VIEWPORT {
                TopLeftX: desc.render_area.x as f32,
                TopLeftY: desc.render_area.y as f32,
                Width: desc.render_area.width as f32,
                Height: desc.render_area.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: desc.render_area.x as i32,
                top: desc.render_area.y as i32,
                right: desc.render_area.x.saturating_add(desc.render_area.width) as i32,
                bottom: desc.render_area.y.saturating_add(desc.render_area.height) as i32,
            };
            if multiview.is_some() {
                // The view-instanced PSO addresses viewport indices 0 and 1.
                // D3D12 requires both slots to be explicitly initialized.
                encoder.list.RSSetViewports(&[viewport, viewport]);
                encoder.list.RSSetScissorRects(&[scissor, scissor]);
                // A PSO with ENABLE_VIEW_INSTANCE_MASKING starts with mask 0,
                // which suppresses every draw until the application sets it.
                let view_list: ID3D12GraphicsCommandList1 =
                    encoder.list.cast().map_err(native_error)?;
                view_list.SetViewInstanceMask(0b11);
            } else {
                encoder.list.RSSetViewports(&[viewport]);
                encoder.list.RSSetScissorRects(&[scissor]);
            }
            match color.load {
                api::LoadOp::Clear(api::ClearColor::Float(value)) => {
                    encoder.list.ClearRenderTargetView(
                        rtv,
                        &value,
                        Some(&[RECT {
                            left: desc.render_area.x as i32,
                            top: desc.render_area.y as i32,
                            right: desc.render_area.x.saturating_add(desc.render_area.width) as i32,
                            bottom: desc.render_area.y.saturating_add(desc.render_area.height)
                                as i32,
                        }]),
                    )
                }
                api::LoadOp::Clear(api::ClearColor::Uint(_))
                | api::LoadOp::Clear(api::ClearColor::Sint(_)) => unreachable!("validated above"),
                api::LoadOp::Load | api::LoadOp::Discard => {}
            }
        }
        if let Some(resolve) = resolve {
            encoder.texture_target = Some(target);
            encoder.resolve_target = Some(resolve);
            encoder.resolve_format = Some(source_format);
        } else if surface {
            encoder.surface_target = Some(target);
        } else {
            encoder.texture_target = Some(target);
        }
        encoder.raster_format = Some(attachment_format);
        encoder.raster_samples = Some(source_samples);
        encoder.raster_multiview = multiview;
        Ok(())
    }
    fn set_pipeline(
        encoder: &mut Dx12Encoder,
        pipeline: &super::Dx12RasterPipeline,
    ) -> Result<(), Error> {
        if encoder.raster_format != Some(pipeline.0.color_format)
            || encoder.raster_samples != Some(pipeline.0.samples)
            || encoder.raster_multiview != pipeline.0.multiview
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: pipeline owns the root signature and PSO COM references, and
        // the encoder retains its clone through command-list execution.
        unsafe {
            encoder.list.SetGraphicsRootSignature(&pipeline.0.root);
            encoder.list.SetPipelineState(&pipeline.0.state);
            encoder
                .list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        encoder.pipeline = Some(pipeline.clone());
        Ok(())
    }
    fn set_bind_group(
        encoder: &mut Dx12Encoder,
        slot: u32,
        group: &super::Dx12BindGroup,
        offsets: &[u32],
    ) -> Result<(), Error> {
        let pipeline = encoder.pipeline.as_ref().ok_or(Error::InvalidState)?;
        let table = pipeline
            .0
            .layout
            .tables
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        let native = &group.0;
        if native.has_views() != table.views.is_some()
            || native.has_samplers() != table.samplers.is_some()
        {
            return Err(Error::InvalidState);
        }
        if offsets.len() != table.dynamic_cbvs.len() {
            return Err(Error::InvalidRange);
        }
        // Binding both descriptor heaps as one operation prevents a later root
        // table from referring to a heap selected by a previous bind group.
        // SAFETY: group and pipeline are same-device validated and retained;
        // dynamic CBV addresses below are range/alignment checked before use.
        unsafe {
            let heaps = [
                Some(native.heaps().view_heap().clone()),
                Some(native.heaps().sampler_heap().clone()),
            ];
            encoder.list.SetDescriptorHeaps(&heaps);
            if let Some(root) = table.views {
                encoder
                    .list
                    .SetGraphicsRootDescriptorTable(root, native.view_gpu());
            }
            if let Some(root) = table.samplers {
                encoder
                    .list
                    .SetGraphicsRootDescriptorTable(root, native.sampler_gpu());
            }
            for (root, &offset) in table.dynamic_cbvs.iter().zip(offsets) {
                let uniform = native
                    .uniform(root.binding)
                    .filter(|uniform| uniform.dynamic_offset)
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
                let end = start
                    .checked_add(root.min_size)
                    .ok_or(Error::InvalidRange)?;
                if end > uniform.buffer.size {
                    return Err(Error::InvalidRange);
                }
                let address = uniform
                    .buffer
                    .resource
                    .GetGPUVirtualAddress()
                    .checked_add(start)
                    .ok_or(Error::InvalidRange)?;
                if address % 256 != 0 {
                    return Err(Error::InvalidRange);
                }
                encoder
                    .list
                    .SetGraphicsRootConstantBufferView(root.root, address);
            }
        }
        encoder.bound_groups.push(group.clone());
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
        encoder.buffers.push(buffer.clone());
        Ok(())
    }
    fn viewport(encoder: &mut Dx12Encoder, v: api::Viewport) -> Result<(), Error> {
        let viewport = validate_viewport(v)?;
        // SAFETY: `validate_viewport` checked finite dimensions and this open
        // command list is exclusively owned by the encoder.
        unsafe {
            encoder.list.RSSetViewports(&[viewport]);
        }
        Ok(())
    }
    fn scissor(encoder: &mut Dx12Encoder, r: api::Rect) -> Result<(), Error> {
        let rect = validate_scissor(r)?;
        // SAFETY: `validate_scissor` checked integer bounds for this list.
        unsafe {
            encoder.list.RSSetScissorRects(&[rect]);
        }
        Ok(())
    }
    fn blend_constant(_: &mut Dx12Encoder, _: [f32; 4]) -> Result<(), Error> {
        Err(Error::Unsupported("DX12 blend constants"))
    }
    fn stencil_reference(_: &mut Dx12Encoder, _: u32) -> Result<(), Error> {
        Err(Error::Unsupported("DX12 stencil"))
    }
    fn draw(
        encoder: &mut Dx12Encoder,
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    ) -> Result<(), Error> {
        if vertices.start > vertices.end || instances.start > instances.end {
            return Err(Error::InvalidRange);
        }
        // SAFETY: Core checked pass/pipeline sequencing and range arithmetic;
        // this only appends a draw to the encoder-owned list.
        unsafe {
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
        // SAFETY: as above, the checked index/instance ranges are copied into
        // the encoder-owned command list without borrowing caller memory.
        unsafe {
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
        if let Some(resolve) = encoder.resolve_target.take() {
            let source = encoder.texture_target.take().ok_or(Error::InvalidState)?;
            let format = encoder.resolve_format.take().ok_or(Error::InvalidState)?;
            barrier(
                &encoder.list,
                &source,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            );
            // Resolve occurs only after both resources were retained and moved
            // into RESOLVE states by the barriers immediately around this call.
            // SAFETY: the same open list owns this ordered resolve operation.
            unsafe {
                encoder
                    .list
                    .ResolveSubresource(&resolve, 0, &source, 0, format);
            }
            barrier(
                &encoder.list,
                &source,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            barrier(
                &encoder.list,
                &resolve,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
                D3D12_RESOURCE_STATE_PRESENT,
            );
        } else if let Some(target) = encoder.surface_target.take() {
            barrier(
                &encoder.list,
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );
        } else if encoder.texture_target.take().is_some() {
        } else {
            return Err(Error::InvalidState);
        }
        encoder.depth_target.take();
        encoder.raster_format = None;
        encoder.raster_samples = None;
        encoder.raster_multiview = None;
        Ok(())
    }
    fn abort_raster(_: &mut Dx12Encoder) {}
}
