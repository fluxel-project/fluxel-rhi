//! Texture allocation and view-descriptor ownership for the DX12 backend.
//!
//! A `TextureState` owns its D3D12 resource; every `TextureViewState` retains
//! that state and any private descriptor heap needed by its view.  Descriptor
//! heaps therefore outlive all command lists retaining the view.  This module
//! validates format, dimensions and native allocation flags, but leaves
//! cross-operation layout transitions to `encoder::transition` and makes no
//! render-graph lifetime or aliasing decisions.

use super::{Dx12Device, Dx12Texture, Dx12TextureView};
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::Arc;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(crate) struct TextureState {
    pub(super) resource: ID3D12Resource,
    pub(super) desc: api::TextureDesc,
    pub(super) sparse: Option<super::sparse::SparseTextureState>,
}
pub(crate) struct TextureViewState {
    pub(super) texture: Arc<TextureState>,
    pub(super) srv_heap: Option<ID3D12DescriptorHeap>,
    pub(super) srv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) uav_heap: Option<ID3D12DescriptorHeap>,
    pub(super) uav: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) rtv_heap: Option<ID3D12DescriptorHeap>,
    pub(super) rtv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) dsv_heap: Option<ID3D12DescriptorHeap>,
    pub(super) dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) desc: api::TextureViewDesc,
}

pub(crate) fn support(dv: &Dx12Device, d: &api::TextureDesc) -> api::Support<api::TextureSupport> {
    match valid(d) {
        Ok(()) => {
            let depth = d.format == api::TextureFormat::Depth32Float;
            api::Support::Supported(api::TextureSupport {
                max_extent: [16384, 16384, 1],
                max_array_layers: 2_048,
                max_mips: max_mips(d.extent[0], d.extent[1]),
                sample_counts: supported_sample_counts(dv, d),
                compatible_view_formats: vec![d.format],
                filterable: !depth,
                blendable: !depth && d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT),
                storage: !depth && d.usage.contains(api::TextureUsage::STORAGE_BINDING),
            })
        }
        Err(x) => api::Support::Unsupported(x),
    }
}

pub(crate) fn create(dv: &Dx12Device, d: &api::TextureDesc) -> Result<Dx12Texture> {
    valid(d).map_err(Error::Unsupported)?;
    if !supported_sample_counts(dv, d).contains(&d.samples) {
        return Err(Error::Unsupported(
            "DX12 texture sample count/format combination",
        ));
    }
    let depth = d.format == api::TextureFormat::Depth32Float;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let flags = if depth {
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
    } else {
        let mut flags = D3D12_RESOURCE_FLAG_NONE;
        if d.usage.contains(api::TextureUsage::STORAGE_BINDING) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        // The native mip lowering renders each destination mip with an RTV.
        // A public COPY_DST|COPY_SRC mip chain therefore needs the same D3D12
        // allocation capability even when it was not itself declared as a
        // raster attachment.
        if d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || (d.mip_levels > 1
                && d.usage.contains(api::TextureUsage::COPY_DST)
                && d.usage.contains(api::TextureUsage::COPY_SRC))
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }
        flags
    };
    let rd = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: u64::from(d.extent[0]),
        Height: d.extent[1],
        DepthOrArraySize: u16::try_from(d.array_layers).map_err(|_| Error::InvalidRange)?,
        MipLevels: u16::try_from(d.mip_levels).map_err(|_| Error::InvalidRange)?,
        Format: dxgi_format(d.format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: d.samples,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    };
    let clear = if depth {
        Some(D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 1.0,
                    Stencil: 0,
                },
            },
        })
    } else if d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT) {
        Some(D3D12_CLEAR_VALUE {
            Format: dxgi_format(d.format),
            Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
        })
    } else {
        None
    };
    let mut resource = None;
    // SAFETY: heap/resource descriptors are fully initialized local values;
    // D3D12 writes the created COM reference to the valid `resource` output.
    unsafe {
        dv.0.native.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &rd,
            if depth {
                D3D12_RESOURCE_STATE_DEPTH_WRITE
            } else {
                D3D12_RESOURCE_STATE_COMMON
            },
            clear.as_ref().map(|value| value as *const _),
            &mut resource,
        )
    }
    .map_err(err)?;
    Ok(Dx12Texture(Arc::new(TextureState {
        resource: resource.ok_or(Error::OutOfMemory)?,
        desc: d.clone(),
        sparse: None,
    })))
}

pub(crate) fn create_view(
    dv: &Dx12Device,
    t: &Dx12Texture,
    v: &api::TextureViewDesc,
) -> Result<Dx12TextureView> {
    let texture = t.0.clone();
    let depth = texture.desc.format == api::TextureFormat::Depth32Float;
    let array = texture.desc.array_layers > 1;
    if v.format != texture.desc.format
        || !matches!(
            v.dimension,
            api::TextureViewDimension::D2 | api::TextureViewDimension::D2Array
        )
        || (v.dimension == api::TextureViewDimension::D2 && v.layers.end != v.layers.start + 1)
        || (v.dimension == api::TextureViewDimension::D2Array && !array)
        || v.mips.start >= v.mips.end
        || v.mips.end > texture.desc.mip_levels
        || v.layers.start >= v.layers.end
        || v.layers.end > texture.desc.array_layers
        || v.aspects
            != if depth {
                api::TextureAspects::DEPTH
            } else {
                api::TextureAspects::COLOR
            }
    {
        return Err(Error::Unsupported("DX12 D2/D2Array texture view"));
    }
    if depth {
        // A view owns its private one-entry heap. This avoids a global mutable
        // CPU descriptor allocator and keeps the descriptor alive with the view.
        // SAFETY: the descriptor heap descriptor is fully initialized.
        let heap = unsafe {
            dv.0.native
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .map_err(err)?;
        // SAFETY: the newly created one-entry heap is non-empty and retained
        // in the returned view state.
        let dsv = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = depth_dsv_desc(v, texture.desc.samples)?;
        // SAFETY: texture and heap are retained by the returned view and the
        // validated DSV descriptor names only the requested subresources.
        unsafe {
            dv.0.native
                .CreateDepthStencilView(&texture.resource, Some(&desc), dsv)
        };
        return Ok(Dx12TextureView(Arc::new(TextureViewState {
            texture,
            srv_heap: None,
            srv: None,
            uav_heap: None,
            uav: None,
            rtv_heap: None,
            rtv: None,
            dsv_heap: Some(heap),
            dsv: Some(dsv),
            desc: v.clone(),
        })));
    }
    let shader_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    let (srv_heap, srv) = if texture
        .desc
        .usage
        .contains(api::TextureUsage::TEXTURE_BINDING)
    {
        // SAFETY: the initialized one-entry SRV heap is owned by this view.
        let heap = unsafe {
            dv.0.native
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&shader_heap_desc)
        }
        .map_err(err)?;
        // SAFETY: `heap` is non-empty and remains owned by the view.
        let srv = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: dxgi_format(texture.desc.format),
            ViewDimension: if v.dimension == api::TextureViewDimension::D2Array {
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY
            } else {
                D3D12_SRV_DIMENSION_TEXTURE2D
            },
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: if v.dimension == api::TextureViewDimension::D2Array {
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
        (Some(heap), Some(srv))
    } else {
        (None, None)
    };
    let (uav_heap, uav) = if texture
        .desc
        .usage
        .contains(api::TextureUsage::STORAGE_BINDING)
    {
        // SAFETY: the initialized one-entry UAV heap is owned by this view.
        let heap = unsafe {
            dv.0.native
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&shader_heap_desc)
        }
        .map_err(err)?;
        // SAFETY: `heap` is non-empty and remains owned by the view.
        let uav = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: dxgi_format(texture.desc.format),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: v.mips.start,
                    PlaneSlice: 0,
                },
            },
        };
        // SAFETY: validated UAV format/mip and retained resource/heap make
        // this descriptor write independent of caller lifetimes.
        unsafe {
            dv.0.native
                .CreateUnorderedAccessView(&texture.resource, None, Some(&desc), uav)
        };
        (Some(heap), Some(uav))
    } else {
        (None, None)
    };
    let (rtv_heap, rtv) = if texture
        .desc
        .usage
        .contains(api::TextureUsage::RENDER_ATTACHMENT)
    {
        if v.mips.end != v.mips.start + 1 {
            return Err(Error::Unsupported("DX12 RTV view requires one mip"));
        }
        // SAFETY: the initialized one-entry RTV heap is owned by this view.
        let heap = unsafe {
            dv.0.native
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .map_err(err)?;
        // SAFETY: `heap` is non-empty and remains owned by the view.
        let rtv = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: dxgi_format(texture.desc.format),
            ViewDimension: if texture.desc.samples == 1
                && v.dimension == api::TextureViewDimension::D2Array
            {
                D3D12_RTV_DIMENSION_TEXTURE2DARRAY
            } else if texture.desc.samples == 1 {
                D3D12_RTV_DIMENSION_TEXTURE2D
            } else {
                D3D12_RTV_DIMENSION_TEXTURE2DMS
            },
            Anonymous: if texture.desc.samples == 1
                && v.dimension == api::TextureViewDimension::D2Array
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
        (Some(heap), Some(rtv))
    } else {
        (None, None)
    };
    Ok(Dx12TextureView(Arc::new(TextureViewState {
        texture,
        srv_heap,
        srv,
        uav_heap,
        uav,
        rtv_heap,
        rtv,
        dsv_heap: None,
        dsv: None,
        desc: v.clone(),
    })))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DsvLayers {
    start: u32,
    count: u32,
}

fn dsv_layers(view: &api::TextureViewDesc) -> Result<DsvLayers> {
    let count = view
        .layers
        .end
        .checked_sub(view.layers.start)
        .ok_or(Error::InvalidRange)?;
    if count == 0 {
        return Err(Error::InvalidRange);
    }
    Ok(DsvLayers {
        start: view.layers.start,
        count,
    })
}

fn depth_dsv_desc(
    view: &api::TextureViewDesc,
    samples: u32,
) -> Result<D3D12_DEPTH_STENCIL_VIEW_DESC> {
    if view.mips != (0..1) {
        return Err(Error::Unsupported("DX12 depth DSV mip"));
    }
    let layers = dsv_layers(view)?;
    Ok(D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        ViewDimension: if samples == 1 && view.dimension == api::TextureViewDimension::D2Array {
            D3D12_DSV_DIMENSION_TEXTURE2DARRAY
        } else if samples == 1 {
            D3D12_DSV_DIMENSION_TEXTURE2D
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        },
        Flags: D3D12_DSV_FLAG_NONE,
        Anonymous: if samples == 1 && view.dimension == api::TextureViewDimension::D2Array {
            D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: 0,
                    FirstArraySlice: layers.start,
                    ArraySize: layers.count,
                },
            }
        } else if samples == 1 {
            D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
            }
        } else {
            D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DMS: D3D12_TEX2DMS_DSV {
                    UnusedField_NothingToDefine: 0,
                },
            }
        },
    })
}

fn valid(d: &api::TextureDesc) -> std::result::Result<(), &'static str> {
    let depth = d.format == api::TextureFormat::Depth32Float;
    if d.dimension != api::TextureDimension::D2
        || !matches!(
            d.format,
            api::TextureFormat::Rgba8Unorm
                | api::TextureFormat::Bgra8Unorm
                | api::TextureFormat::Depth32Float
        )
        || d.extent[0] == 0
        || d.extent[1] == 0
        || d.extent[0] > 16_384
        || d.extent[1] > 16_384
        || d.extent[2] != 1
        || d.array_layers == 0
        || d.array_layers > 2_048
        || !matches!(d.samples, 1 | 2 | 4 | 8)
    {
        return Err("DX12 RGBA8/Depth32Float D2 texture subset");
    }
    if (d.samples > 1 && d.array_layers != 1)
        || (d.samples > 1
            && (d.mip_levels != 1
                || d.usage.contains(api::TextureUsage::TEXTURE_BINDING)
                || d.usage.contains(api::TextureUsage::STORAGE_BINDING)
                || d.usage.contains(api::TextureUsage::COPY_DST)
                || d.usage.contains(api::TextureUsage::COPY_SRC)))
    {
        return Err("DX12 multisample texture usage");
    }
    if depth
        && (d.mip_levels != 1
            || !d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || d.usage.contains(api::TextureUsage::TEXTURE_BINDING)
            || d.usage.contains(api::TextureUsage::STORAGE_BINDING)
            // The command path does not yet expose a depth-aspect copy
            // transition. Rejecting these usages here keeps support facts in
            // sync with the executable DX12 path.
            || d.usage.contains(api::TextureUsage::COPY_SRC)
            || d.usage.contains(api::TextureUsage::COPY_DST))
    {
        return Err("DX12 depth texture usage");
    }
    if d.mip_levels == 0 || d.mip_levels > max_mips(d.extent[0], d.extent[1]) {
        return Err("DX12 mip count");
    }
    if d.usage.contains(api::TextureUsage::STORAGE_BINDING) && d.mip_levels != 1 {
        return Err("DX12 storage texture mip chain");
    }
    Ok(())
}
fn supported_sample_counts(dv: &Dx12Device, d: &api::TextureDesc) -> Vec<u32> {
    let format = dxgi_format(d.format);
    [1, 2, 4, 8]
        .into_iter()
        .filter(|&count| {
            if count == 1 {
                return true;
            }
            let mut quality = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
                Format: format,
                SampleCount: count,
                Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
                NumQualityLevels: 0,
            };
            // SAFETY: `quality` is the exact feature struct required by this
            // query and stays mutable/alive for the synchronous D3D12 call.
            unsafe {
                dv.0.native.CheckFeatureSupport(
                    D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                    &mut quality as *mut _ as *mut _,
                    std::mem::size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32,
                )
            }
            .is_ok()
                && quality.NumQualityLevels > 0
        })
        .collect()
}
fn dxgi_format(format: api::TextureFormat) -> DXGI_FORMAT {
    match format {
        api::TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        api::TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        api::TextureFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        _ => unreachable!("validated DX12 texture format"),
    }
}
fn max_mips(width: u32, height: u32) -> u32 {
    1 + width.max(height).ilog2()
}
fn err(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv_array_range_preserves_the_selected_layers() {
        let view = api::TextureViewDesc {
            format: api::TextureFormat::Depth32Float,
            dimension: api::TextureViewDimension::D2Array,
            aspects: api::TextureAspects::DEPTH,
            mips: 0..1,
            layers: 3..5,
        };
        assert_eq!(dsv_layers(&view), Ok(DsvLayers { start: 3, count: 2 }));
    }

    #[test]
    fn dsv_array_range_rejects_an_empty_range() {
        let view = api::TextureViewDesc {
            format: api::TextureFormat::Depth32Float,
            dimension: api::TextureViewDimension::D2Array,
            aspects: api::TextureAspects::DEPTH,
            mips: 0..1,
            layers: 2..2,
        };
        assert!(matches!(dsv_layers(&view), Err(Error::InvalidRange)));
    }

    #[test]
    fn array_depth_view_lowers_its_selected_layers() {
        let view = api::TextureViewDesc {
            format: api::TextureFormat::Depth32Float,
            dimension: api::TextureViewDimension::D2Array,
            aspects: api::TextureAspects::DEPTH,
            mips: 0..1,
            layers: 3..5,
        };
        let desc = depth_dsv_desc(&view, 1).unwrap();
        assert_eq!(desc.ViewDimension, D3D12_DSV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: `depth_dsv_desc` initialized the union's Texture2DArray arm
        // for the asserted view dimension immediately above.
        let array = unsafe { desc.Anonymous.Texture2DArray };
        assert_eq!(array.FirstArraySlice, 3);
        assert_eq!(array.ArraySize, 2);
    }
}
