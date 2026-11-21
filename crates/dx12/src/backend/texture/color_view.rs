//! Color texture descriptors and the view owners that retain them.

use super::{
    Dx12Device, Dx12Texture, Dx12TextureView, TextureState, TextureViewState, cube_srv, depth_view,
    dxgi_format, is_depth, sampled_view_support, validated_default_view_usage, view_usage_support,
};
use fluxel_rhi_core::api::{self, Result};
use std::sync::Arc;
use windows::Win32::Graphics::Direct3D12::*;

pub(crate) fn create_view(
    dv: &Dx12Device,
    t: &Dx12Texture,
    v: &api::TextureViewDesc,
) -> Result<Dx12TextureView> {
    let usage = validated_default_view_usage(dv, t, v)?;
    create_native_view(dv, t, v, usage)
}

pub(crate) fn create_view_with_usage(
    dv: &Dx12Device,
    t: &Dx12Texture,
    v: &api::TextureViewDesc,
    usage: api::TextureUsage,
) -> Result<Dx12TextureView> {
    view_usage_support(dv, t, v, usage)?;
    create_native_view(dv, t, v, usage)
}

pub(crate) fn volume_attachment_slice_support(
    dv: &Dx12Device,
    t: &Dx12Texture,
    view: &api::TextureViewDesc,
    depth_slice: u32,
) -> Result<()> {
    if t.0.desc.dimension != api::TextureDimension::D3 {
        return Err(api::Error::Unsupported("DX12 volume color attachment"));
    }
    view_usage_support(dv, t, view, api::TextureUsage::RENDER_ATTACHMENT)?;
    if depth_slice >= (t.0.desc.extent[2] >> view.mips.start).max(1) {
        return Err(api::Error::InvalidRange);
    }
    Ok(())
}

pub(crate) fn create_volume_attachment_slice(
    dv: &Dx12Device,
    t: &Dx12Texture,
    view: &api::TextureViewDesc,
    depth_slice: u32,
) -> Result<Dx12TextureView> {
    volume_attachment_slice_support(dv, t, view, depth_slice)?;
    create_color_view(
        dv,
        t.0.clone(),
        view,
        api::TextureUsage::RENDER_ATTACHMENT,
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Some(depth_slice),
    )
}

fn create_native_view(
    dv: &Dx12Device,
    t: &Dx12Texture,
    v: &api::TextureViewDesc,
    usage: api::TextureUsage,
) -> Result<Dx12TextureView> {
    let texture = t.0.clone();
    let depth = is_depth(texture.desc.format);
    let array = texture.desc.array_layers > 1 || v.dimension == api::TextureViewDimension::D2Array;
    if depth {
        return depth_view::create_depth_view(dv, texture, v, array, usage);
    }
    create_color_view(
        dv,
        texture,
        v,
        usage,
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        None,
    )
}

pub(crate) fn create_sampled_view(
    dv: &Dx12Device,
    t: &Dx12Texture,
    v: &api::TextureViewDesc,
    mapping: api::TextureComponentMapping,
) -> Result<Dx12TextureView> {
    sampled_view_support(dv, t, v)?;
    create_color_view(
        dv,
        t.0.clone(),
        v,
        api::TextureUsage::TEXTURE_BINDING,
        component_mapping(mapping),
        None,
    )
}

fn component_mapping(mapping: api::TextureComponentMapping) -> u32 {
    use api::TextureComponent::*;
    let mut encoded = D3D12_SHADER_COMPONENT_MAPPING_ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES;
    for (index, source) in mapping.0.into_iter().enumerate() {
        let native = match source {
            Red => D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0,
            Green => D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1,
            Blue => D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2,
            Alpha => D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3,
            Zero => D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
            One => D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1,
        };
        encoded |= (native.0 as u32) << (index as u32 * D3D12_SHADER_COMPONENT_MAPPING_SHIFT);
    }
    encoded
}

fn create_color_view(
    dv: &Dx12Device,
    texture: Arc<TextureState>,
    v: &api::TextureViewDesc,
    usage: api::TextureUsage,
    component_mapping: u32,
    attachment_depth_slice: Option<u32>,
) -> Result<Dx12TextureView> {
    let array = texture.desc.array_layers > 1 || v.dimension == api::TextureViewDimension::D2Array;
    let (srv_lease, srv) = if usage.contains(api::TextureUsage::TEXTURE_BINDING) {
        let lease =
            dv.0.cpu_descriptors
                .allocate(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1)?;
        let srv = lease.cpu(0);
        let cube = cube_srv(v);
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: dxgi_format(v.format),
            ViewDimension: if texture.desc.samples > 1
                && v.dimension == api::TextureViewDimension::D2Array
            {
                D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY
            } else if texture.desc.samples > 1 {
                D3D12_SRV_DIMENSION_TEXTURE2DMS
            } else if let Some((dimension, _)) = cube {
                dimension
            } else if v.dimension == api::TextureViewDimension::D1 {
                D3D12_SRV_DIMENSION_TEXTURE1D
            } else if v.dimension == api::TextureViewDimension::D3 {
                D3D12_SRV_DIMENSION_TEXTURE3D
            } else if v.dimension == api::TextureViewDimension::D2Array {
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY
            } else {
                D3D12_SRV_DIMENSION_TEXTURE2D
            },
            Shader4ComponentMapping: component_mapping,
            Anonymous: if texture.desc.samples > 1
                && v.dimension == api::TextureViewDimension::D2Array
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DMSArray: D3D12_TEX2DMS_ARRAY_SRV {
                        FirstArraySlice: v.layers.start,
                        ArraySize: v.layers.end - v.layers.start,
                    },
                }
            } else if texture.desc.samples > 1 {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DMS: D3D12_TEX2DMS_SRV {
                        UnusedField_NothingToDefine: 0,
                    },
                }
            } else if let Some((_, descriptor)) = cube {
                descriptor
            } else if v.dimension == api::TextureViewDimension::D1 {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture1D: D3D12_TEX1D_SRV {
                        MostDetailedMip: v.mips.start,
                        MipLevels: v.mips.end - v.mips.start,
                        ResourceMinLODClamp: 0.0,
                    },
                }
            } else if v.dimension == api::TextureViewDimension::D3 {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_SRV {
                        MostDetailedMip: v.mips.start,
                        MipLevels: v.mips.end - v.mips.start,
                        ResourceMinLODClamp: 0.0,
                    },
                }
            } else if v.dimension == api::TextureViewDimension::D2Array {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: v.mips.start,
                        MipLevels: v.mips.end - v.mips.start,
                        FirstArraySlice: v.layers.start,
                        ArraySize: v.layers.end - v.layers.start,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                }
            } else {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: v.mips.start,
                        MipLevels: v.mips.end - v.mips.start,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                }
            },
        };
        // SAFETY: the validated SRV range belongs to retained texture storage
        // and the destination CPU handle belongs to the retained heap.
        unsafe {
            dv.0.native
                .CreateShaderResourceView(&texture.resource, Some(&desc), srv)
        };
        (Some(lease), Some(srv))
    } else {
        (None, None)
    };
    let (uav_lease, uav) = if usage.contains(api::TextureUsage::STORAGE_BINDING) {
        let lease =
            dv.0.cpu_descriptors
                .allocate(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1)?;
        let uav = lease.cpu(0);
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: dxgi_format(v.format),
            ViewDimension: if texture.desc.dimension == api::TextureDimension::D1 {
                D3D12_UAV_DIMENSION_TEXTURE1D
            } else if texture.desc.dimension == api::TextureDimension::D3 {
                D3D12_UAV_DIMENSION_TEXTURE3D
            } else if array {
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY
            } else {
                D3D12_UAV_DIMENSION_TEXTURE2D
            },
            Anonymous: if texture.desc.dimension == api::TextureDimension::D1 {
                D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture1D: D3D12_TEX1D_UAV {
                        MipSlice: v.mips.start,
                    },
                }
            } else if texture.desc.dimension == api::TextureDimension::D3 {
                D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_UAV {
                        MipSlice: v.mips.start,
                        FirstWSlice: 0,
                        WSize: (texture.desc.extent[2] >> v.mips.start).max(1),
                    },
                }
            } else if array {
                D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                        MipSlice: v.mips.start,
                        FirstArraySlice: v.layers.start,
                        ArraySize: v.layers.end - v.layers.start,
                        PlaneSlice: 0,
                    },
                }
            } else {
                D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: v.mips.start,
                        PlaneSlice: 0,
                    },
                }
            },
        };
        // SAFETY: validated UAV format/mip and retained resource/heap make
        // this descriptor write independent of caller lifetimes.
        unsafe {
            dv.0.native
                .CreateUnorderedAccessView(&texture.resource, None, Some(&desc), uav)
        };
        (Some(lease), Some(uav))
    } else {
        (None, None)
    };
    let (rtv_lease, rtv) = if usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
        let lease =
            dv.0.cpu_descriptors
                .allocate(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1)?;
        let rtv = lease.cpu(0);
        let desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: dxgi_format(v.format),
            ViewDimension: if texture.desc.dimension == api::TextureDimension::D1 {
                D3D12_RTV_DIMENSION_TEXTURE1D
            } else if texture.desc.dimension == api::TextureDimension::D3 {
                D3D12_RTV_DIMENSION_TEXTURE3D
            } else if texture.desc.samples == 1
                && (texture.desc.array_layers > 1
                    || v.dimension == api::TextureViewDimension::D2Array)
            {
                D3D12_RTV_DIMENSION_TEXTURE2DARRAY
            } else if texture.desc.samples == 1 {
                D3D12_RTV_DIMENSION_TEXTURE2D
            } else if texture.desc.array_layers > 1
                || v.dimension == api::TextureViewDimension::D2Array
            {
                D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY
            } else {
                D3D12_RTV_DIMENSION_TEXTURE2DMS
            },
            Anonymous: if texture.desc.dimension == api::TextureDimension::D1 {
                D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture1D: D3D12_TEX1D_RTV {
                        MipSlice: v.mips.start,
                    },
                }
            } else if texture.desc.dimension == api::TextureDimension::D3 {
                D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_RTV {
                        MipSlice: v.mips.start,
                        FirstWSlice: attachment_depth_slice.unwrap_or(0),
                        WSize: attachment_depth_slice
                            .map_or_else(|| (texture.desc.extent[2] >> v.mips.start).max(1), |_| 1),
                    },
                }
            } else if texture.desc.samples == 1
                && (texture.desc.array_layers > 1
                    || v.dimension == api::TextureViewDimension::D2Array)
            {
                D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                        MipSlice: v.mips.start,
                        FirstArraySlice: v.layers.start,
                        ArraySize: v.layers.end - v.layers.start,
                        PlaneSlice: 0,
                    },
                }
            } else if texture.desc.samples == 1 {
                D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_RTV {
                        MipSlice: v.mips.start,
                        PlaneSlice: 0,
                    },
                }
            } else if texture.desc.array_layers > 1
                || v.dimension == api::TextureViewDimension::D2Array
            {
                D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2DMSArray: D3D12_TEX2DMS_ARRAY_RTV {
                        FirstArraySlice: v.layers.start,
                        ArraySize: v.layers.end - v.layers.start,
                    },
                }
            } else {
                D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2DMS: D3D12_TEX2DMS_RTV {
                        UnusedField_NothingToDefine: 0,
                    },
                }
            },
        };
        // SAFETY: the validated RTV descriptor refers to retained texture
        // storage and writes only into this view's retained descriptor heap.
        unsafe {
            dv.0.native
                .CreateRenderTargetView(&texture.resource, Some(&desc), rtv)
        };
        (Some(lease), Some(rtv))
    } else {
        (None, None)
    };
    Ok(Dx12TextureView(Arc::new(TextureViewState {
        texture,
        srv_lease,
        srv,
        uav_lease,
        uav,
        rtv_lease,
        rtv,
        dsv_lease: None,
        dsv: None,
        dsv_read_only_depth: None,
        dsv_read_only_stencil: None,
        dsv_read_only_both: None,
        desc: v.clone(),
        attachment_depth_slice,
    })))
}
