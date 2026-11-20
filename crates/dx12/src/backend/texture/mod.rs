//! Texture allocation and view-descriptor ownership for the DX12 backend.
//!
//! A `TextureState` owns its D3D12 resource; every `TextureViewState` retains
//! that state and any leased CPU descriptor range needed by its view.  Encoder
//! work clones a lease before recording raw CPU handles, so the range outlives
//! all command lists retaining it.  This module
//! validates format, dimensions and native allocation flags, but leaves
//! cross-operation layout transitions to `encoder::transition` and makes no
//! render-graph lifetime or aliasing decisions.

mod color_view;
mod depth_view;
mod format;
mod validate;
pub(super) use color_view::{create_sampled_view, create_view};
pub(super) use format::{
    bc_block_bytes, color_texel_bytes, depth_aspects, dxgi_format, format_has_support,
    format_supports_texture_atomic_add_native, format_supports_texture_atomic_bitwise_native,
    format_supports_texture_atomic_signed_min_native,
    format_supports_texture_atomic_unsigned_min_native, integer_clear_bits, integer_color,
    is_depth,
};
use format::{format_available, sample_count_supported, supported_sample_counts};
use validate::{max_array_layers, max_extent, max_mips, valid};

use super::{Dx12Device, Dx12Texture, Dx12TextureView};
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::Arc;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

pub(crate) struct TextureState {
    pub(super) resource: ID3D12Resource,
    /// Pool lease follows the resource field so native destruction precedes
    /// range return when the last texture/view/recording owner is dropped.
    #[allow(dead_code)] // Drop ownership returns the range; it is not otherwise queried.
    pub(super) default_allocation: Option<super::memory::DefaultAllocationLease>,
    /// A placed resource retains the heap backing it until all texture/view
    /// and recorded-work references have retired.
    pub(super) backing_heap: Option<ID3D12Heap>,
    pub(super) desc: api::TextureDesc,
    pub(super) sparse: Option<super::sparse::SparseTextureState>,
}
pub(crate) struct TextureViewState {
    pub(super) texture: Arc<TextureState>,
    pub(super) srv_lease: Option<super::descriptor::CpuDescriptorLease>,
    pub(super) srv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) uav_lease: Option<super::descriptor::CpuDescriptorLease>,
    pub(super) uav: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) rtv_lease: Option<super::descriptor::CpuDescriptorLease>,
    pub(super) rtv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) dsv_lease: Option<super::descriptor::CpuDescriptorLease>,
    pub(super) dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) dsv_read_only_depth: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) dsv_read_only_stencil: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) dsv_read_only_both: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) desc: api::TextureViewDesc,
}

pub(crate) fn support(dv: &Dx12Device, d: &api::TextureDesc) -> api::Support<api::TextureSupport> {
    match validate_allocation(dv, d) {
        Ok(()) => {
            let depth = is_depth(d.format);
            api::Support::Supported(api::TextureSupport {
                max_extent: max_extent(d.dimension),
                max_array_layers: max_array_layers(d),
                max_mips: max_mips(d.extent[0], d.extent[1].max(d.extent[2])),
                sample_counts: if d.dimension != api::TextureDimension::D2 {
                    vec![1]
                } else {
                    supported_sample_counts(dv, d)
                },
                compatible_view_formats: permitted_view_formats(dv, d),
                filterable: !depth
                    && format_has_support(dv, d.format, D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE),
                blendable: !depth
                    && format_has_support(
                        dv,
                        d.format,
                        D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_BLENDABLE,
                    ),
                storage: !depth && d.usage.contains(api::TextureUsage::STORAGE_BINDING),
            })
        }
        Err(x) => api::Support::Unsupported(x),
    }
}

fn validate_allocation(
    dv: &Dx12Device,
    d: &api::TextureDesc,
) -> std::result::Result<(), &'static str> {
    valid(d)?;
    if !format_available(dv, d) {
        return Err("DX12 texture format support");
    }
    if !sample_count_supported(dv, d.format, d.samples) {
        return Err("DX12 texture sample count/format combination");
    }
    Ok(())
}

// Share reinterpretation admission between capability reporting and view creation.
fn view_format_available(
    dv: &Dx12Device,
    desc: &api::TextureDesc,
    view: api::TextureFormat,
    usage: api::TextureUsage,
) -> bool {
    if view == desc.format {
        return true;
    }
    desc.view_formats.contains(&view)
        && (!usage.contains(api::TextureUsage::TEXTURE_BINDING)
            || format_has_support(dv, view, D3D12_FORMAT_SUPPORT1_SHADER_LOAD))
        && (!(desc.samples > 1 && usage.contains(api::TextureUsage::TEXTURE_BINDING))
            || format_has_support(dv, view, D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD))
        && (!usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || format_has_support(dv, view, D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
}

fn permitted_view_formats(dv: &Dx12Device, desc: &api::TextureDesc) -> Vec<api::TextureFormat> {
    let mut formats = vec![desc.format];
    for &view in &desc.view_formats {
        if !formats.contains(&view) && view_format_available(dv, desc, view, desc.usage) {
            formats.push(view);
        }
    }
    formats
}

/// Native description and clear metadata used identically for committed and
/// placed texture allocations.
pub(crate) struct AllocationDesc {
    pub(crate) resource: D3D12_RESOURCE_DESC,
    pub(crate) clear: Option<D3D12_CLEAR_VALUE>,
    pub(crate) initial_state: D3D12_RESOURCE_STATES,
}

pub(crate) fn allocation_desc(dv: &Dx12Device, d: &api::TextureDesc) -> Result<AllocationDesc> {
    validate_allocation(dv, d).map_err(Error::Unsupported)?;
    let depth = is_depth(d.format);
    let flags = if depth {
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
    } else {
        let mut flags = D3D12_RESOURCE_FLAG_NONE;
        if d.usage.contains(api::TextureUsage::STORAGE_BINDING) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        // The blit lowering renders into COPY_DST color textures. Allocate
        // the native RTV capability for its supported formats when available.
        if d.samples > 1
            || d.usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || (d.dimension == api::TextureDimension::D2
                && d.samples == 1
                && matches!(
                    d.format,
                    api::TextureFormat::Rgba8Unorm | api::TextureFormat::Bgra8Unorm
                )
                && d.usage.contains(api::TextureUsage::COPY_DST)
                && format_has_support(dv, d.format, D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
        {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }
        flags
    };
    let rd = D3D12_RESOURCE_DESC {
        Dimension: if d.dimension == api::TextureDimension::D1 {
            D3D12_RESOURCE_DIMENSION_TEXTURE1D
        } else if d.dimension == api::TextureDimension::D3 {
            D3D12_RESOURCE_DIMENSION_TEXTURE3D
        } else {
            D3D12_RESOURCE_DIMENSION_TEXTURE2D
        },
        Alignment: 0,
        Width: u64::from(d.extent[0]),
        Height: d.extent[1],
        DepthOrArraySize: u16::try_from(if d.dimension == api::TextureDimension::D3 {
            d.extent[2]
        } else {
            d.array_layers
        })
        .map_err(|_| Error::InvalidRange)?,
        MipLevels: u16::try_from(d.mip_levels).map_err(|_| Error::InvalidRange)?,
        Format: if d.view_formats.iter().any(|&view| view != d.format) {
            format::typeless_color(d.format).ok_or(Error::InvalidState)?
        } else if is_depth(d.format) && d.usage.contains(api::TextureUsage::TEXTURE_BINDING) {
            match d.format {
                api::TextureFormat::Depth16Unorm => DXGI_FORMAT_R16_TYPELESS,
                api::TextureFormat::Depth24PlusStencil8 => DXGI_FORMAT_R24G8_TYPELESS,
                api::TextureFormat::Depth32FloatStencil8 => DXGI_FORMAT_R32G8X24_TYPELESS,
                _ => DXGI_FORMAT_R32_TYPELESS,
            }
        } else {
            dxgi_format(d.format)
        },
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: d.samples,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    };
    let clear = if depth {
        Some(D3D12_CLEAR_VALUE {
            Format: dxgi_format(d.format),
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
    Ok(AllocationDesc {
        resource: rd,
        clear,
        initial_state: if depth {
            D3D12_RESOURCE_STATE_DEPTH_WRITE
        } else {
            D3D12_RESOURCE_STATE_COMMON
        },
    })
}

pub(crate) fn create(dv: &Dx12Device, d: &api::TextureDesc) -> Result<Dx12Texture> {
    let allocation = allocation_desc(dv, d)?;
    let requirements = super::memory::requirements(
        dv,
        &allocation.resource,
        super::memory::texture_class_bits(dv, &allocation.resource),
    )?;
    let class = (0..64)
        .find(|bit| requirements.native_type_bits & (1u64 << bit) != 0)
        .ok_or(Error::Unsupported("DX12 texture heap class"))?;
    let pooled =
        dv.0.default_resources
            .allocate(class, requirements.size, requirements.alignment)?;
    let (resource, lease, backing_heap) = if let Some(pool) = pooled {
        let mut resource = None;
        // SAFETY: pool ranges use GetResourceAllocationInfo size/alignment and
        // a heap class admitted by the same native placement contract.
        let placed = unsafe {
            dv.0.native.CreatePlacedResource(
                &pool.heap,
                pool.offset,
                &allocation.resource,
                allocation.initial_state,
                allocation.clear.as_ref().map(|value| value as *const _),
                &mut resource,
            )
        };
        match placed {
            Ok(()) => (
                resource.ok_or(Error::OutOfMemory)?,
                Some(pool.lease),
                Some(pool.heap),
            ),
            Err(error) if super::memory::is_out_of_memory(&error) => {
                drop(pool);
                let mut resource = None;
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                    MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                    CreationNodeMask: 1,
                    VisibleNodeMask: 1,
                };
                // SAFETY: the valid resource descriptor is retried through the
                // committed allocation fallback after pool placement OOM.
                unsafe {
                    dv.0.native.CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &allocation.resource,
                        allocation.initial_state,
                        allocation.clear.as_ref().map(|value| value as *const _),
                        &mut resource,
                    )
                }
                .map_err(err)?;
                (resource.ok_or(Error::OutOfMemory)?, None, None)
            }
            Err(error) => return Err(err(error)),
        }
    } else {
        let mut resource = None;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        // SAFETY: heap/resource descriptors are fully initialized local values;
        // D3D12 writes the created COM reference to the valid `resource` output.
        unsafe {
            dv.0.native.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &allocation.resource,
                allocation.initial_state,
                allocation.clear.as_ref().map(|value| value as *const _),
                &mut resource,
            )
        }
        .map_err(err)?;
        (resource.ok_or(Error::OutOfMemory)?, None, None)
    };
    Ok(Dx12Texture(Arc::new(TextureState {
        resource,
        default_allocation: lease,
        backing_heap,
        desc: d.clone(),
        sparse: None,
    })))
}

pub(crate) fn view_support(
    dv: &Dx12Device,
    t: &Dx12Texture,
    v: &api::TextureViewDesc,
) -> Result<()> {
    validate::view(dv, &t.0.desc, v, t.0.desc.usage)
}

pub(super) fn sampled_view_support(
    dv: &Dx12Device,
    t: &Dx12Texture,
    v: &api::TextureViewDesc,
) -> Result<()> {
    let usage = api::TextureUsage::TEXTURE_BINDING;
    if !t.0.desc.usage.contains(usage) || is_depth(t.0.desc.format) {
        return Err(Error::Unsupported("DX12 sampled color texture view"));
    }
    validate::view(dv, &t.0.desc, v, usage)
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
    is_array: bool,
) -> Result<D3D12_DEPTH_STENCIL_VIEW_DESC> {
    if view.mips.start.checked_add(1) != Some(view.mips.end)
        || (samples > 1 && view.mips.start != 0)
    {
        return Err(Error::Unsupported("DX12 depth DSV mip"));
    }
    let layers = dsv_layers(view)?;
    Ok(D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: dxgi_format(view.format),
        ViewDimension: if view.dimension == api::TextureViewDimension::D1 {
            D3D12_DSV_DIMENSION_TEXTURE1D
        } else if samples == 1 && is_array {
            D3D12_DSV_DIMENSION_TEXTURE2DARRAY
        } else if samples == 1 {
            D3D12_DSV_DIMENSION_TEXTURE2D
        } else if is_array {
            D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        },
        Flags: D3D12_DSV_FLAG_NONE,
        Anonymous: if view.dimension == api::TextureViewDimension::D1 {
            D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture1D: D3D12_TEX1D_DSV {
                    MipSlice: view.mips.start,
                },
            }
        } else if samples == 1 && is_array {
            D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: view.mips.start,
                    FirstArraySlice: layers.start,
                    ArraySize: layers.count,
                },
            }
        } else if samples == 1 {
            D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_DSV {
                    MipSlice: view.mips.start,
                },
            }
        } else if is_array {
            D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_DSV {
                    FirstArraySlice: layers.start,
                    ArraySize: layers.count,
                },
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

fn err(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}

#[cfg(test)]
#[path = "../../tests/backend/texture.rs"]
mod tests;
