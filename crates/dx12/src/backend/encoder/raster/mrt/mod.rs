//! Multiple color attachment recording for a single DX12 render pass.
//!
//! Attachment descriptors are consumed at recording time; resources are
//! retained by the encoder until submission completes. Resolve barriers are
//! local to the pass and restore the caller-selected render attachment state.
use super::*;
mod clear;
mod finish;
use clear::{depth_clear_handles, selected};
pub(super) use finish::end;
use std::collections::HashSet;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
};

struct Resolve {
    resource: ID3D12Resource,
    subresource: u32,
    surface: bool,
    mip_levels: u32,
}

struct Target {
    resource: ID3D12Resource,
    surface: bool,
    resolve: Option<Resolve>,
    subresource: u32,
    format: DXGI_FORMAT,
    layer_count: u32,
    mip_levels: u32,
}

#[derive(Clone)]
pub(crate) struct IntegerClearTarget {
    pub(crate) format: api::TextureFormat,
    pub(crate) samples: u32,
    pub(crate) extent: (u64, u32),
    pub(crate) views: Vec<(Option<ID3D12DescriptorHeap>, D3D12_CPU_DESCRIPTOR_HANDLE)>,
}

pub(crate) struct RasterState {
    targets: Vec<Target>,
    end_timestamp: Option<(super::super::super::Dx12QuerySet, u32)>,
    pub(crate) formats: Vec<Option<api::TextureFormat>>,
    pub(crate) depth_format: Option<api::TextureFormat>,
    pub(crate) samples: Option<u32>,
    pub(crate) multiview: Option<u32>,
    pub(crate) multiview_mask: Option<u32>,
    pub(crate) depth_read_only: bool,
    pub(crate) stencil_read_only: bool,
    /// The bound descriptor sequence includes null RTVs for holes, so it can
    /// be replayed after an internal integer clear draw changes OM state.
    pub(crate) rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(crate) dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(crate) integer_clear_targets: Vec<Option<IntegerClearTarget>>,
}

pub(super) fn begin(
    encoder: &mut Dx12Encoder,
    desc: &api::RasterPassDesc<'_, Dx12>,
) -> Result<RasterState, Error> {
    if encoder.is_compute || desc.colors.len() > 8 {
        return Err(Error::Unsupported("DX12 raster attachment configuration"));
    }
    let multiview = match desc.multiview_mask {
        None => None,
        Some(mask) if mask != 0 && mask <= 15 => Some(32 - mask.leading_zeros()),
        _ => return Err(Error::Unsupported("DX12 multiview mask")),
    };
    let right = desc
        .render_area
        .x
        .checked_add(desc.render_area.width)
        .ok_or(Error::InvalidRange)?;
    let bottom = desc
        .render_area
        .y
        .checked_add(desc.render_area.height)
        .ok_or(Error::InvalidRange)?;
    if desc.render_area.width == 0
        || desc.render_area.height == 0
        || right > i32::MAX as u32
        || bottom > i32::MAX as u32
    {
        return Err(Error::InvalidRange);
    }
    let mut used = HashSet::new();
    let mut pass_extent: Option<(u64, u32)> = None;
    let mut pass_layers: Option<u32> = None;
    let mut formats = Vec::with_capacity(desc.colors.len());
    let mut handles = Vec::with_capacity(desc.colors.len());
    let mut targets = Vec::new();
    let mut samples = None;
    let mut clears = Vec::new();
    let mut integer_clears = Vec::new();
    let mut integer_clear_targets = Vec::with_capacity(desc.colors.len());
    for color in desc.colors {
        let Some(color) = color else {
            formats.push(None);
            handles.push(None);
            integer_clear_targets.push(None);
            continue;
        };
        let clear = match color.load {
            api::LoadOp::Clear(c) => Some(c),
            _ => None,
        };
        let (rtv, resource, surface, format, subresource, extent, layer_count, mip_levels) =
            match color.view {
                api::AttachmentView::Surface(image) => {
                    if multiview.is_some() {
                        return Err(Error::InvalidState);
                    }
                    // SAFETY: Core validates the lease's device and the clone retains the resource.
                    let n = unsafe { image.native_ref() };
                    let resource = n.0.resource.clone();
                    let format = match unsafe { resource.GetDesc() }.Format {
                        DXGI_FORMAT_R8G8B8A8_UNORM => api::TextureFormat::Rgba8Unorm,
                        DXGI_FORMAT_B8G8R8A8_UNORM => api::TextureFormat::Bgra8Unorm,
                        DXGI_FORMAT_R10G10B10A2_UNORM => api::TextureFormat::Rgb10a2Unorm,
                        DXGI_FORMAT_R16G16B16A16_FLOAT => api::TextureFormat::Rgba16Float,
                        _ => return Err(Error::Unsupported("DX12 surface format")),
                    };
                    (
                        n.0.rtv,
                        resource.clone(),
                        true,
                        format,
                        0,
                        {
                            let d = unsafe { resource.GetDesc() };
                            (d.Width, d.Height)
                        },
                        1,
                        1,
                    )
                }
                api::AttachmentView::Texture(view) => {
                    // SAFETY: Core validates device identity and the encoder retains the texture.
                    let n = unsafe { view.native_ref() };
                    if multiview.is_some()
                        && (n.0.desc.dimension != api::TextureViewDimension::D2Array
                            || n.0.desc.layers.end - n.0.desc.layers.start < multiview.unwrap())
                    {
                        return Err(Error::InvalidState);
                    }
                    let resource = n.0.texture.resource.clone();
                    encoder.textures.push(Dx12Texture(n.0.texture.clone()));
                    encoder.view_descriptor_leases.push(
                        n.0.rtv_lease
                            .as_ref()
                            .ok_or(Error::Unsupported("DX12 texture RTV"))?
                            .clone(),
                    );
                    (
                        n.0.rtv.ok_or(Error::Unsupported("DX12 texture RTV"))?,
                        resource,
                        false,
                        n.0.desc.format,
                        n.0.desc.mips.start + n.0.desc.layers.start * n.0.texture.desc.mip_levels,
                        (
                            u64::from((n.0.texture.desc.extent[0] >> n.0.desc.mips.start).max(1)),
                            (n.0.texture.desc.extent[1] >> n.0.desc.mips.start).max(1),
                        ),
                        n.0.desc.layers.end - n.0.desc.layers.start,
                        n.0.texture.desc.mip_levels,
                    )
                }
            };
        if extent.0 < u64::from(right)
            || extent.1 < bottom
            || pass_extent.is_some_and(|e| e != extent)
            || pass_layers.is_some_and(|l| l != layer_count)
        {
            return Err(Error::InvalidRange);
        }
        pass_extent = Some(extent);
        pass_layers = Some(layer_count);
        for layer in 0..layer_count {
            if !used.insert((resource.as_raw() as usize, subresource + layer * mip_levels)) {
                return Err(Error::InvalidState);
            }
        }
        if super::super::super::texture::color_texel_bytes(format).is_none() {
            return Err(Error::Unsupported("DX12 color attachment format"));
        }
        let native_format = super::super::super::texture::dxgi_format(format);
        let active_integer_clear = if super::super::super::texture::integer_color(format) {
            let api::AttachmentView::Texture(view) = color.view else {
                return Err(Error::Unsupported("DX12 integer surface rect clear"));
            };
            // SAFETY: Core validated this view; the encoder retains its descriptor lease.
            let n = unsafe { view.native_ref() };
            let texture = Dx12Texture(n.0.texture.clone());
            let mut views = Vec::new();
            if n.0.texture.desc.dimension == api::TextureDimension::D3 {
                for slice in 0..(n.0.texture.desc.extent[2] >> n.0.desc.mips.start).max(1) {
                    let (heap, handle) = super::super::integer_clear::volume_slice_rtv(
                        &encoder.device,
                        &n.0.texture,
                        n.0.desc.mips.start,
                        slice,
                    )?;
                    views.push((Some(heap), handle));
                }
            } else {
                for layer in n.0.desc.layers.clone() {
                    if !selected(desc.multiview_mask, layer - n.0.desc.layers.start) {
                        continue;
                    }
                    let mut vd = n.0.desc.clone();
                    vd.dimension = if n.0.texture.desc.dimension == api::TextureDimension::D1 {
                        api::TextureViewDimension::D1
                    } else if n.0.texture.desc.array_layers > 1 {
                        api::TextureViewDimension::D2Array
                    } else {
                        api::TextureViewDimension::D2
                    };
                    vd.layers = layer..layer + 1;
                    let view =
                        super::super::super::texture::create_view(&encoder.device, &texture, &vd)?;
                    encoder.view_descriptor_leases.push(
                        view.0
                            .rtv_lease
                            .as_ref()
                            .ok_or(Error::InvalidState)?
                            .clone(),
                    );
                    views.push((None, view.0.rtv.ok_or(Error::InvalidState)?));
                }
            }
            Some(IntegerClearTarget {
                format,
                samples: n.0.texture.desc.samples,
                extent,
                views,
            })
        } else {
            None
        };
        integer_clear_targets.push(active_integer_clear);
        let float_clear = match clear {
            Some(api::ClearColor::Float(c))
                if !super::super::super::texture::integer_color(format) =>
            {
                Some(c)
            }
            Some(c) => {
                let (signed, bits) = super::super::super::texture::integer_clear_bits(format, c)
                    .ok_or(Error::InvalidState)?;
                let api::AttachmentView::Texture(view) = color.view else {
                    return Err(Error::InvalidState);
                };
                let n = unsafe { view.native_ref() };
                let pipeline = encoder.device.0.integer_clear_cache.get_or_create(
                    &encoder.device,
                    format,
                    n.0.texture.desc.samples,
                    signed,
                )?;
                let texture = Dx12Texture(n.0.texture.clone());
                let mut views = Vec::new();
                if n.0.texture.desc.dimension == api::TextureDimension::D3 {
                    for slice in 0..(n.0.texture.desc.extent[2] >> n.0.desc.mips.start).max(1) {
                        let (heap, handle) = super::super::integer_clear::volume_slice_rtv(
                            &encoder.device,
                            &n.0.texture,
                            n.0.desc.mips.start,
                            slice,
                        )?;
                        views.push((Some(heap), handle));
                    }
                } else {
                    for layer in n.0.desc.layers.clone() {
                        if !selected(desc.multiview_mask, layer - n.0.desc.layers.start) {
                            continue;
                        }
                        let mut vd = n.0.desc.clone();
                        vd.dimension = if n.0.texture.desc.dimension == api::TextureDimension::D1 {
                            api::TextureViewDimension::D1
                        } else if n.0.texture.desc.array_layers > 1 {
                            api::TextureViewDimension::D2Array
                        } else {
                            api::TextureViewDimension::D2
                        };
                        vd.layers = layer..layer + 1;
                        let view = super::super::super::texture::create_view(
                            &encoder.device,
                            &texture,
                            &vd,
                        )?;
                        encoder.view_descriptor_leases.push(
                            view.0
                                .rtv_lease
                                .as_ref()
                                .ok_or(Error::InvalidState)?
                                .clone(),
                        );
                        views.push((None, view.0.rtv.ok_or(Error::InvalidState)?));
                    }
                }
                integer_clears.push((pipeline, views, extent, bits));
                None
            }
            None => None,
        };
        // SAFETY: immutable descriptor of the retained native resource.
        let native_desc = unsafe { resource.GetDesc() };
        if samples.is_some_and(|s| s != native_desc.SampleDesc.Count) {
            return Err(Error::InvalidState);
        }
        samples = Some(native_desc.SampleDesc.Count);
        let resolve = if let Some(view) = &color.resolve_target {
            if native_desc.SampleDesc.Count <= 1 || surface {
                return Err(Error::Unsupported("DX12 MSAA resolve source"));
            }
            let (
                resource,
                subresource,
                is_surface,
                dst_format,
                dst_samples,
                dst_extent,
                dst_layers,
                dst_mips,
            ) = match view {
                api::AttachmentView::Surface(image) => {
                    // SAFETY: Core validated this same-device surface lease.
                    let n = unsafe { image.native_ref() };
                    let resource = n.0.resource.clone();
                    let d = unsafe { resource.GetDesc() };
                    (
                        resource,
                        0,
                        true,
                        d.Format,
                        d.SampleDesc.Count,
                        (d.Width, d.Height),
                        1,
                        1,
                    )
                }
                api::AttachmentView::Texture(view) => {
                    // SAFETY: Core validated the texture view's device identity.
                    let n = unsafe { view.native_ref() };
                    if n.0.desc.mips.end != n.0.desc.mips.start + 1
                        || n.0.desc.layers.end <= n.0.desc.layers.start
                        || n.0.rtv.is_none()
                    {
                        return Err(Error::Unsupported("DX12 resolve destination view"));
                    }
                    let d = &n.0.texture.desc;
                    let resource = n.0.texture.resource.clone();
                    let native = unsafe { resource.GetDesc() };
                    encoder.textures.push(Dx12Texture(n.0.texture.clone()));
                    encoder.view_descriptor_leases.push(
                        n.0.rtv_lease
                            .as_ref()
                            .ok_or(Error::Unsupported("DX12 resolve destination view"))?
                            .clone(),
                    );
                    (
                        resource,
                        n.0.desc.mips.start + n.0.desc.layers.start * d.mip_levels,
                        false,
                        native.Format,
                        d.samples,
                        (
                            u64::from((d.extent[0] >> n.0.desc.mips.start).max(1)),
                            (d.extent[1] >> n.0.desc.mips.start).max(1),
                        ),
                        n.0.desc.layers.end - n.0.desc.layers.start,
                        d.mip_levels,
                    )
                }
            };
            if dst_samples != 1
                || dst_format != native_format
                || dst_extent != extent
                || dst_layers != layer_count
            {
                return Err(Error::InvalidState);
            }
            for layer in 0..dst_layers {
                if !used.insert((resource.as_raw() as usize, subresource + layer * dst_mips)) {
                    return Err(Error::InvalidState);
                }
            }
            Some(Resolve {
                resource,
                subresource,
                surface: is_surface,
                mip_levels: dst_mips,
            })
        } else {
            None
        };
        formats.push(Some(format));
        handles.push(Some(rtv));
        clear::color_clear_handles(
            encoder,
            desc.multiview_mask,
            &color.view,
            rtv,
            float_clear,
            &mut clears,
        )?;
        targets.push(Target {
            resource,
            surface,
            resolve,
            subresource,
            format: native_format,
            layer_count,
            mip_levels,
        });
    }
    let (dsv, depth_format, depth_clear, stencil_clear) = if let Some(depth) = &desc.depth_stencil {
        let api::AttachmentView::Texture(view) = depth.view else {
            return Err(Error::Unsupported("DX12 surface depth attachment"));
        };
        // SAFETY: Core validates device identity; the texture is retained.
        let n = unsafe { view.native_ref() };
        if depth.stencil.is_some()
            && !matches!(
                n.0.texture.desc.format,
                api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
            )
        {
            return Err(Error::InvalidState);
        }
        if n.0.desc.mips.end != n.0.desc.mips.start + 1
            || n.0.desc.layers.end <= n.0.desc.layers.start
        {
            return Err(Error::InvalidRange);
        }
        let extent = (
            u64::from((n.0.texture.desc.extent[0] >> n.0.desc.mips.start).max(1)),
            (n.0.texture.desc.extent[1] >> n.0.desc.mips.start).max(1),
        );
        let layers = n.0.desc.layers.end - n.0.desc.layers.start;
        if multiview.is_some_and(|count| {
            n.0.desc.dimension != api::TextureViewDimension::D2Array || layers < count
        }) {
            return Err(Error::InvalidState);
        }
        if extent.0 < u64::from(right)
            || extent.1 < bottom
            || pass_extent.is_some_and(|e| e != extent)
            || pass_layers.is_some_and(|l| l != layers)
        {
            return Err(Error::InvalidRange);
        }
        for layer in n.0.desc.layers.clone() {
            let subresource = n.0.desc.mips.start + layer * n.0.texture.desc.mip_levels;
            if !used.insert((n.0.texture.resource.as_raw() as usize, subresource)) {
                return Err(Error::InvalidState);
            }
        }
        if samples.is_some_and(|s| s != n.0.texture.desc.samples) {
            return Err(Error::InvalidState);
        }
        samples = Some(n.0.texture.desc.samples);
        encoder.textures.push(Dx12Texture(n.0.texture.clone()));
        encoder.view_descriptor_leases.push(
            n.0.dsv_lease
                .as_ref()
                .ok_or(Error::Unsupported("DX12 depth/stencil DSV"))?
                .clone(),
        );
        let has_stencil = matches!(
            n.0.texture.desc.format,
            api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
        );
        let dsv = match (
            depth.depth_read_only,
            depth.stencil_read_only && has_stencil,
        ) {
            (false, false) => n.0.dsv,
            (true, false) => n.0.dsv_read_only_depth,
            (false, true) => n.0.dsv_read_only_stencil,
            (true, true) => n.0.dsv_read_only_both,
        }
        .ok_or(Error::Unsupported("DX12 depth/stencil read-only DSV"))?;
        if (depth.depth_read_only && matches!(depth.depth, Some((api::LoadOp::Clear(_), _))))
            || (depth.stencil_read_only
                && matches!(depth.stencil, Some((api::LoadOp::Clear(_), _))))
        {
            return Err(Error::InvalidState);
        }
        (
            Some(dsv),
            Some(n.0.texture.desc.format),
            depth.depth.and_then(|(load, _)| match load {
                api::LoadOp::Clear(v) => Some(v),
                _ => None,
            }),
            depth.stencil.and_then(|(load, _)| match load {
                api::LoadOp::Clear(v) => Some(v as u8),
                _ => None,
            }),
        )
    } else {
        (None, None, None, None)
    };
    let inherited_samples = samples
        .or_else(|| desc.secondary_inheritance.as_ref().map(|s| s.samples))
        .unwrap_or(1);
    super::super::secondary::validate_inheritance(
        desc.secondary_inheritance.as_ref(),
        &formats,
        depth_format,
        inherited_samples,
        desc.multiview_mask,
    )?;
    let depth_clear_handles = depth_clear_handles(encoder, desc, dsv, depth_clear, stencil_clear)?;
    let null_heap = if handles.iter().any(Option::is_none) {
        // SAFETY: the device is live and the descriptor heap description is complete.
        let heap = unsafe {
            encoder
                .device
                .0
                .native
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .map_err(native_error)?;
        let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let null_desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        // SAFETY: null resource plus valid typed descriptor creates a null RTV.
        unsafe {
            encoder
                .device
                .0
                .native
                .CreateRenderTargetView(None, Some(&null_desc), handle);
        }
        for item in &mut handles {
            if item.is_none() {
                *item = Some(handle);
            }
        }
        Some(heap)
    } else {
        None
    };
    let handles: Vec<_> = handles.into_iter().map(Option::unwrap).collect();
    let rect = RECT {
        left: desc.render_area.x as i32,
        top: desc.render_area.y as i32,
        right: desc.render_area.x.saturating_add(desc.render_area.width) as i32,
        bottom: desc.render_area.y.saturating_add(desc.render_area.height) as i32,
    };
    let viewport = D3D12_VIEWPORT {
        TopLeftX: desc.render_area.x as f32,
        TopLeftY: desc.render_area.y as f32,
        Width: desc.render_area.width as f32,
        Height: desc.render_area.height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let view_list: Option<ID3D12GraphicsCommandList1> = if multiview.is_some() {
        Some(encoder.list.cast().map_err(native_error)?)
    } else {
        None
    };
    let end_timestamp = if let Some(writes) = &desc.timestamp_writes {
        // SAFETY: Core validates ownership and query metadata; the cloned native
        // set is retained by the encoder for every recorded timestamp.
        let set = unsafe { writes.set.native_ref() }.clone();
        if let Some(slot) = writes.begin {
            super::super::query::timestamp(encoder, &set, slot)?;
        }
        writes.end.map(|slot| (set, slot))
    } else {
        None
    };
    for target in &targets {
        encoder.uploads.push(target.resource.clone());
        if let Some(resolve) = &target.resolve {
            encoder.uploads.push(resolve.resource.clone());
        }
        if target.surface {
            barrier(
                &encoder.list,
                &target.resource,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
        }
    }
    let has_integer_clears = !integer_clears.is_empty();
    for (pipeline, views, extent, bits) in integer_clears {
        for (heap, handle) in views {
            super::super::integer_clear::record(
                &encoder.list,
                &pipeline,
                handle,
                extent.0 as u32,
                extent.1,
                rect,
                bits,
            )?;
            if let Some(heap) = heap {
                encoder.clear_heaps.push(heap);
            }
        }
        encoder.integer_clear_pipelines.push(pipeline);
    }
    if has_integer_clears {
        super::super::restore_after_internal_draw(encoder)?;
    }
    encoder.raster_snapshot.viewport = Some(viewport);
    encoder.raster_snapshot.scissor = Some(rect);
    // SAFETY: descriptors reference validated resources and are consumed by these recording calls.
    unsafe {
        encoder.list.OMSetRenderTargets(
            handles.len() as u32,
            if handles.is_empty() {
                None
            } else {
                Some(handles.as_ptr())
            },
            false,
            dsv.as_ref().map(|h| h as *const _),
        );
        if let Some(view_list) = view_list {
            encoder.list.RSSetViewports(&[viewport, viewport]);
            encoder.list.RSSetScissorRects(&[rect, rect]);
            view_list.SetViewInstanceMask(desc.multiview_mask.unwrap());
        } else {
            encoder.list.RSSetViewports(&[viewport]);
            encoder.list.RSSetScissorRects(&[rect]);
        }
        for (handle, color) in clears {
            encoder
                .list
                .ClearRenderTargetView(handle, &color, Some(&[rect]));
        }
        for &handle in &depth_clear_handles {
            if let Some(value) = depth_clear {
                encoder.list.ClearDepthStencilView(
                    handle,
                    D3D12_CLEAR_FLAG_DEPTH,
                    value,
                    0,
                    Some(&[rect]),
                );
            }
            if let Some(value) = stencil_clear {
                encoder.list.ClearDepthStencilView(
                    handle,
                    D3D12_CLEAR_FLAG_STENCIL,
                    0.0,
                    value,
                    Some(&[rect]),
                );
            }
        }
    }
    if let Some(heap) = null_heap {
        encoder.clear_heaps.push(heap);
    }
    Ok(RasterState {
        targets,
        end_timestamp,
        formats,
        depth_format,
        samples,
        multiview,
        multiview_mask: desc.multiview_mask,
        depth_read_only: desc
            .depth_stencil
            .as_ref()
            .is_some_and(|d| d.depth_read_only),
        stencil_read_only: desc
            .depth_stencil
            .as_ref()
            .is_some_and(|d| d.stencil_read_only),
        rtvs: handles,
        dsv,
        integer_clear_targets,
    })
}
