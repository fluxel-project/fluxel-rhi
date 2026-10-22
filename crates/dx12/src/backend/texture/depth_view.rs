//! Native depth and stencil view descriptors.
use super::*;

pub(super) fn create_depth_view(
    dv: &Dx12Device,
    texture: Arc<TextureState>,
    v: &api::TextureViewDesc,
    array: bool,
) -> Result<Dx12TextureView> {
    let (heap, dsv, read_depth, read_stencil, read_both) = if v.aspects
        == depth_aspects(texture.desc.format)
        && texture
            .desc
            .usage
            .contains(api::TextureUsage::RENDER_ATTACHMENT)
    {
        // A view owns its private heap. This avoids a global mutable
        // CPU descriptor allocator and keeps the descriptor alive with the view.
        // SAFETY: the descriptor heap descriptor is fully initialized.
        let heap = unsafe {
            dv.0.native
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: if depth_aspects(texture.desc.format)
                        == api::TextureAspects::DEPTH
                    {
                        2
                    } else {
                        4
                    },
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .map_err(err)?;
        // SAFETY: the newly created heap is non-empty and retained
        // in the returned view state.
        let dsv = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = depth_dsv_desc(
            v,
            texture.desc.samples,
            texture.desc.array_layers > 1 || v.dimension == api::TextureViewDimension::D2Array,
        )?;
        // SAFETY: texture and heap are retained by the returned view and the
        // validated DSV descriptor names only the requested subresources.
        unsafe {
            dv.0.native
                .CreateDepthStencilView(&texture.resource, Some(&desc), dsv)
        };
        // SAFETY: the device remains alive for this synchronous query.
        let stride = unsafe {
            dv.0.native
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV)
        } as usize;
        let create_read_only = |index: usize, flags| {
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: dsv.ptr + index * stride,
            };
            let mut readonly = desc;
            readonly.Flags = flags;
            // SAFETY: each index is inside this retained heap; the descriptor
            // describes the same validated resource and subresources as `desc`.
            unsafe {
                dv.0.native
                    .CreateDepthStencilView(&texture.resource, Some(&readonly), handle)
            };
            handle
        };
        let read_depth = create_read_only(1, D3D12_DSV_FLAG_READ_ONLY_DEPTH);
        let (read_stencil, read_both) =
            if depth_aspects(texture.desc.format) == api::TextureAspects::DEPTH {
                (None, None)
            } else {
                (
                    Some(create_read_only(2, D3D12_DSV_FLAG_READ_ONLY_STENCIL)),
                    Some(create_read_only(
                        3,
                        D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL,
                    )),
                )
            };
        (
            Some(heap),
            Some(dsv),
            Some(read_depth),
            read_stencil,
            read_both,
        )
    } else {
        (None, None, None, None, None)
    };
    let (srv_heap, srv) = if texture
        .desc
        .usage
        .contains(api::TextureUsage::TEXTURE_BINDING)
        && v.aspects != (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
    {
        // SAFETY: the descriptor heap specification is fully initialized.
        let srv_heap = unsafe {
            dv.0.native
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .map_err(err)?;
        // SAFETY: the heap has one descriptor and is retained by the view.
        let srv = unsafe { srv_heap.GetCPUDescriptorHandleForHeapStart() };
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: match (texture.desc.format, v.aspects) {
                (api::TextureFormat::Depth16Unorm, _) => DXGI_FORMAT_R16_UNORM,
                (api::TextureFormat::Depth24PlusStencil8, api::TextureAspects::DEPTH) => {
                    DXGI_FORMAT_R24_UNORM_X8_TYPELESS
                }
                (api::TextureFormat::Depth24PlusStencil8, api::TextureAspects::STENCIL) => {
                    DXGI_FORMAT_X24_TYPELESS_G8_UINT
                }
                (api::TextureFormat::Depth32FloatStencil8, api::TextureAspects::DEPTH) => {
                    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
                }
                (api::TextureFormat::Depth32FloatStencil8, api::TextureAspects::STENCIL) => {
                    DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
                }
                _ => DXGI_FORMAT_R32_FLOAT,
            },
            ViewDimension: if texture.desc.samples > 1 && array {
                D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY
            } else if texture.desc.samples > 1 {
                D3D12_SRV_DIMENSION_TEXTURE2DMS
            } else if texture.desc.dimension == api::TextureDimension::D1 {
                D3D12_SRV_DIMENSION_TEXTURE1D
            } else if array {
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY
            } else {
                D3D12_SRV_DIMENSION_TEXTURE2D
            },
            Shader4ComponentMapping: if v.aspects == api::TextureAspects::STENCIL {
                0x1689
            } else {
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
            },
            Anonymous: if texture.desc.samples > 1 && array {
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
            } else if texture.desc.dimension == api::TextureDimension::D1 {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture1D: D3D12_TEX1D_SRV {
                        MostDetailedMip: v.mips.start,
                        MipLevels: v.mips.end - v.mips.start,
                        ResourceMinLODClamp: 0.0,
                    },
                }
            } else if array {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: v.mips.start,
                        MipLevels: v.mips.end - v.mips.start,
                        FirstArraySlice: v.layers.start,
                        ArraySize: v.layers.end - v.layers.start,
                        PlaneSlice: u32::from(v.aspects == api::TextureAspects::STENCIL),
                        ResourceMinLODClamp: 0.0,
                    },
                }
            } else {
                D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: v.mips.start,
                        MipLevels: v.mips.end - v.mips.start,
                        PlaneSlice: u32::from(v.aspects == api::TextureAspects::STENCIL),
                        ResourceMinLODClamp: 0.0,
                    },
                }
            },
        };
        // SAFETY: the typeless resource and validated SRV describe the same retained depth plane.
        unsafe {
            dv.0.native
                .CreateShaderResourceView(&texture.resource, Some(&srv_desc), srv)
        };
        (Some(srv_heap), Some(srv))
    } else {
        (None, None)
    };
    return Ok(Dx12TextureView(Arc::new(TextureViewState {
        texture,
        srv_heap,
        srv,
        uav_heap: None,
        uav: None,
        rtv_heap: None,
        rtv: None,
        dsv_heap: heap,
        dsv,
        dsv_read_only_depth: read_depth,
        dsv_read_only_stencil: read_stencil,
        dsv_read_only_both: read_both,
        desc: v.clone(),
    })));
}
