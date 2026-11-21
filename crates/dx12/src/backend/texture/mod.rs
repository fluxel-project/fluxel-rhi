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
pub(super) use color_view::{
    create_sampled_view, create_view, create_view_with_usage, create_volume_attachment_slice,
    volume_attachment_slice_support,
};
pub(super) use format::sample_count_supported;
use format::{available_format_support, native_format_has_support, supported_sample_counts};
pub(super) use format::{
    bc_block_bytes, color_texel_bytes, depth_aspects, dxgi_format, format_has_support,
    format_supports_texture_atomic_add_native, format_supports_texture_atomic_bitwise_native,
    format_supports_texture_atomic_signed_min_native,
    format_supports_texture_atomic_unsigned_min_native, integer_clear_bits, integer_color,
    is_depth, storage_view_format_available, typed_storage_access,
};
pub(super) use format::{native_format_support, sample_load_requirement};
use validate::valid;
pub(super) use validate::{max_array_layers, max_extent, max_mips};

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
    /// True only for a locally committed default allocation.
    pub(super) owns_residency: bool,
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
    /// A view-local W slice. Resource transitions still address the full mip.
    pub(super) attachment_depth_slice: Option<u32>,
}

impl TextureViewState {
    pub(in crate::backend) fn depth_stencil_handle(
        &self,
        depth_read_only: bool,
        stencil_read_only: bool,
    ) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        use api::TextureAspects as A;
        match (
            depth_read_only && self.desc.aspects.0 & A::DEPTH.0 != 0,
            stencil_read_only && self.desc.aspects.0 & A::STENCIL.0 != 0,
        ) {
            (false, false) => self.dsv,
            (true, false) => self.dsv_read_only_depth,
            (false, true) => self.dsv_read_only_stencil,
            (true, true) => self.dsv_read_only_both,
        }
    }
}

pub(crate) fn support(dv: &Dx12Device, d: &api::TextureDesc) -> api::Support<api::TextureSupport> {
    match validate_allocation(dv, d) {
        Ok(format_support) => {
            let depth = is_depth(d.format);
            let mip_ceiling = max_mips(d);
            let max_mips = if d.mip_levels > 1
                || (mip_ceiling > 1 && supports_mip_chain(dv, d, format_support))
            {
                mip_ceiling
            } else {
                1
            };
            api::Support::Supported(api::TextureSupport {
                max_extent: max_extent(d.dimension),
                max_array_layers: max_array_layers(d),
                max_mips,
                sample_counts: if d.dimension != api::TextureDimension::D2 {
                    vec![1]
                } else {
                    supported_sample_counts(dv, d)
                },
                compatible_view_formats: permitted_view_formats(dv, d),
                filterable: filterable_color(d.format, d.samples, format_support),
                blendable: !depth
                    && format_support.contains(
                        D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_BLENDABLE,
                    ),
                storage: !depth && d.usage.contains(api::TextureUsage::STORAGE_BINDING),
            })
        }
        Err(x) => api::Support::Unsupported(x),
    }
}

pub(super) fn filterable_color(
    format: api::TextureFormat,
    samples: u32,
    support: D3D12_FORMAT_SUPPORT1,
) -> bool {
    samples == 1
        && !is_depth(format)
        && !integer_color(format)
        && support.contains(D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)
}

fn validate_allocation(
    dv: &Dx12Device,
    d: &api::TextureDesc,
) -> std::result::Result<D3D12_FORMAT_SUPPORT1, &'static str> {
    valid(d)?;
    let format_support = available_format_support(dv, d).ok_or("DX12 texture format support")?;
    if d.mip_levels > 1 && !supports_mip_chain(dv, d, format_support) {
        return Err("DX12 texture format has no mip chain support");
    }
    if !sample_count_supported(dv, d.format, d.samples) {
        return Err("DX12 texture sample count/format combination");
    }
    Ok(format_support)
}

fn supports_mip_chain(
    dv: &Dx12Device,
    desc: &api::TextureDesc,
    format_support: D3D12_FORMAT_SUPPORT1,
) -> bool {
    allocation_format(desc).is_ok_and(|native| {
        if native == dxgi_format(desc.format) {
            format_support.contains(D3D12_FORMAT_SUPPORT1_MIP)
        } else {
            native_format_has_support(dv, native, D3D12_FORMAT_SUPPORT1_MIP)
        }
    })
}

fn allocation_format(desc: &api::TextureDesc) -> Result<DXGI_FORMAT> {
    if desc.view_formats.iter().any(|&view| view != desc.format) {
        format::typeless_color(desc.format).ok_or(Error::InvalidState)
    } else if is_depth(desc.format) && desc.usage.contains(api::TextureUsage::TEXTURE_BINDING) {
        Ok(match desc.format {
            api::TextureFormat::Depth16Unorm => DXGI_FORMAT_R16_TYPELESS,
            api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Stencil8 => {
                DXGI_FORMAT_R24G8_TYPELESS
            }
            api::TextureFormat::Depth32FloatStencil8 => DXGI_FORMAT_R32G8X24_TYPELESS,
            _ => DXGI_FORMAT_R32_TYPELESS,
        })
    } else {
        Ok(dxgi_format(desc.format))
    }
}

pub(super) fn sampled_srv_format(
    format: api::TextureFormat,
    aspect: api::TextureAspects,
) -> DXGI_FORMAT {
    if is_depth(format) {
        depth_view::srv_format(format, aspect)
    } else {
        dxgi_format(format)
    }
}

// Share reinterpretation admission between capability reporting and view creation.
fn view_format_available(
    dv: &Dx12Device,
    desc: &api::TextureDesc,
    view: api::TextureFormat,
    usage: api::TextureUsage,
) -> bool {
    if usage.contains(api::TextureUsage::STORAGE_BINDING)
        && !storage_view_format_available(dv, view, desc.dimension, None)
    {
        return false;
    }
    if view == desc.format {
        return true;
    }
    desc.view_formats.contains(&view)
        && (!usage.contains(api::TextureUsage::TEXTURE_BINDING)
            || format_has_support(dv, view, sample_load_requirement(desc.samples)))
        && (!usage.contains(api::TextureUsage::RENDER_ATTACHMENT)
            || (format_has_support(
                dv,
                view,
                if desc.samples > 1 {
                    D3D12_FORMAT_SUPPORT1_RENDER_TARGET
                        | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET
                } else {
                    D3D12_FORMAT_SUPPORT1_RENDER_TARGET
                },
            ) && sample_count_supported(dv, view, desc.samples)))
}

fn permitted_view_formats(dv: &Dx12Device, desc: &api::TextureDesc) -> Vec<api::TextureFormat> {
    let mut formats = Vec::new();
    for view in std::iter::once(desc.format).chain(desc.view_formats.iter().copied()) {
        if !formats.contains(&view)
            && default_view_usage(desc, view)
                .is_ok_and(|usage| view_format_available(dv, desc, view, usage))
        {
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
        Format: allocation_format(d)?,
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
    create_with_route(dv, d, super::memory::AllocationRoute::Automatic)
}

pub(super) fn create_with_route(
    dv: &Dx12Device,
    d: &api::TextureDesc,
    route: super::memory::AllocationRoute,
) -> Result<Dx12Texture> {
    let allocation = allocation_desc(dv, d)?;
    let requirements = super::memory::requirements(
        dv,
        &allocation.resource,
        super::memory::texture_class_bits(dv, &allocation.resource),
    )?;
    let pooled = if route == super::memory::AllocationRoute::Committed {
        None
    } else {
        let class = (0..64)
            .find(|bit| requirements.native_type_bits & (1u64 << bit) != 0)
            .ok_or(Error::Unsupported("DX12 texture heap class"))?;
        dv.0.default_resources
            .allocate(class, requirements.size, requirements.alignment)?
    };
    let (resource, lease, backing_heap): (ID3D12Resource, _, _) = if let Some(pool) = pooled {
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
    super::set_debug_name(&resource, d.label.as_deref());
    Ok(Dx12Texture(Arc::new(TextureState {
        resource,
        default_allocation: lease,
        owns_residency: backing_heap.is_none(),
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
    validated_default_view_usage(dv, t, v).map(drop)
}

fn default_view_usage(
    desc: &api::TextureDesc,
    format: api::TextureFormat,
) -> Result<api::TextureUsage> {
    use api::TextureUsage as U;
    let mut usage = desc.usage;
    if usage.contains(U::STORAGE_BINDING)
        && matches!(
            format,
            api::TextureFormat::Rgba8UnormSrgb | api::TextureFormat::Bgra8UnormSrgb
        )
    {
        // Legacy sRGB views provide SRV/RTV roles when available. A separate
        // linear view supplies the UAV for the same typeless allocation.
        usage.0 &= !U::STORAGE_BINDING.0;
        if usage.0 & (U::TEXTURE_BINDING | U::RENDER_ATTACHMENT).0 == 0 {
            return Err(Error::Unsupported("DX12 sRGB view has no descriptor role"));
        }
    }
    Ok(usage)
}

fn validated_default_view_usage(
    dv: &Dx12Device,
    t: &Dx12Texture,
    view: &api::TextureViewDesc,
) -> Result<api::TextureUsage> {
    let usage = default_view_usage(&t.0.desc, view.format)?;
    validate::view(dv, &t.0.desc, view, usage)?;
    Ok(usage)
}

fn cube_srv(
    view: &api::TextureViewDesc,
) -> Option<(D3D12_SRV_DIMENSION, D3D12_SHADER_RESOURCE_VIEW_DESC_0)> {
    let mip_levels = view.mips.end - view.mips.start;
    match view.dimension {
        api::TextureViewDimension::Cube => Some((
            D3D12_SRV_DIMENSION_TEXTURECUBE,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV {
                    MostDetailedMip: view.mips.start,
                    MipLevels: mip_levels,
                    ResourceMinLODClamp: 0.0,
                },
            },
        )),
        api::TextureViewDimension::CubeArray => Some((
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
            D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV {
                    MostDetailedMip: view.mips.start,
                    MipLevels: mip_levels,
                    First2DArrayFace: view.layers.start,
                    NumCubes: (view.layers.end - view.layers.start) / 6,
                    ResourceMinLODClamp: 0.0,
                },
            },
        )),
        _ => None,
    }
}

pub(super) fn view_usage_support(
    dv: &Dx12Device,
    t: &Dx12Texture,
    view: &api::TextureViewDesc,
    usage: api::TextureUsage,
) -> Result<()> {
    use api::TextureUsage as U;
    let texture = &t.0.desc;
    if !texture.usage.contains(usage) {
        return Err(Error::InvalidState);
    }
    validate::view(dv, texture, view, usage)?;
    if is_depth(texture.format) {
        let all_aspects = depth_aspects(texture.format);
        if usage.contains(U::RENDER_ATTACHMENT) && view.aspects != all_aspects {
            return Err(Error::Unsupported("DX12 depth attachment view aspects"));
        }
        if usage.contains(U::TEXTURE_BINDING)
            && view.aspects == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
        {
            return Err(Error::Unsupported(
                "DX12 sampled depth-stencil view needs one aspect",
            ));
        }
    }
    Ok(())
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
    crate::backend::native_error(e)
}

#[cfg(test)]
#[path = "../../tests/backend/texture.rs"]
mod tests;
