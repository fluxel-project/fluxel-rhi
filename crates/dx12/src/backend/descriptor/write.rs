//! Bind-group resource validation used while writing descriptor-table ranges.
//!
//! This module validates only the resource-to-layout correspondence and never
//! allocates heaps or writes descriptors.  Keeping it separate makes the
//! parent descriptor module own heap ranges and bind-group lifetime, while the
//! writer path owns the per-entry runtime-array and view-dimension contract.

use fluxel_rhi_core::api::{self, Error, Result};

use super::{BindingSlot, DescriptorClass, Dx12, Dx12Device};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT_R32_TYPELESS};

/// Initializes an inaccessible array element without retaining a resource.
/// Null views use a valid raw-buffer description; shaders must not access holes.
pub(super) fn write_hole(
    device: &Dx12Device,
    class: DescriptorClass,
    view: D3D12_CPU_DESCRIPTOR_HANDLE,
    sampler: u32,
) {
    // SAFETY: the caller reserved these destination heap entries. Null views
    // have a complete valid description and cannot reference a dead resource.
    unsafe {
        match class {
            DescriptorClass::Cbv => device.0.native.CreateConstantBufferView(None, view),
            DescriptorClass::Srv => {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: 1,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                device.0.native.CreateShaderResourceView(
                    None::<&ID3D12Resource>,
                    Some(&desc),
                    view,
                );
            }
            DescriptorClass::Uav => {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: 1,
                            StructureByteStride: 0,
                            CounterOffsetInBytes: 0,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        },
                    },
                };
                device.0.native.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&desc),
                    view,
                );
            }
            DescriptorClass::Sampler => {
                let desc = D3D12_SAMPLER_DESC {
                    Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                    MipLODBias: 0.0,
                    MaxAnisotropy: 1,
                    ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                    BorderColor: [0.0; 4],
                    MinLOD: 0.0,
                    MaxLOD: f32::MAX,
                };
                device
                    .0
                    .native
                    .CreateSampler(&desc, device.0.descriptors.samplers.cpu(sampler));
            }
        }
    }
}

pub(super) fn resource_descriptor_count(
    slot: &BindingSlot,
    resource: &api::BindingResource<'_, Dx12>,
) -> Result<u32> {
    if !slot.runtime_sized {
        return Ok(slot.descriptor_count);
    }
    let api::BindingResource::Array(values) = resource else {
        return Err(Error::InvalidState);
    };
    let count = u32::try_from(values.len()).map_err(|_| Error::InvalidRange)?;
    if count == 0 || count > slot.descriptor_count {
        return Err(Error::InvalidRange);
    }
    // Reserve the whole declared range: lower D3D12 binding tiers require
    // every covered CBV/UAV descriptor to be initialized.
    Ok(slot.descriptor_count)
}

pub(super) fn validate_texture_view_dimension(
    device: &Dx12Device,
    slot: &BindingSlot,
    resource: &api::BindingResource<'_, Dx12>,
) -> Result<()> {
    let expected = slot.view_dimension.ok_or(Error::InvalidState)?;
    match resource {
        api::BindingResource::TextureView(view) => {
            validate_sampled_view(device, slot, view, expected)
        }
        api::BindingResource::Array(views) => {
            for view in views.iter().flatten() {
                let api::BindingResource::TextureView(view) = view else {
                    return Err(Error::Unsupported("DX12 sampled texture array resource"));
                };
                validate_sampled_view(device, slot, view, expected)?;
            }
            Ok(())
        }
        _ => Err(Error::Unsupported("DX12 sampled texture binding resource")),
    }
}

fn validate_sampled_view(
    device: &Dx12Device,
    slot: &BindingSlot,
    view: &api::TextureView<Dx12>,
    expected: api::TextureViewDimension,
) -> Result<()> {
    // SAFETY: Core checked the view belongs to this device.
    let native = unsafe { view.native_ref() };
    if native.0.desc.dimension != expected
        || (native.0.texture.desc.samples > 1) != slot.multisampled
        || slot
            .texture_format
            .is_some_and(|format| format != native.0.desc.format)
    {
        return Err(Error::InvalidState);
    }
    if let Some(sample) = slot.texture_sample {
        let format = native.0.desc.format;
        use api::TextureFormat as F;
        let integer_signed = matches!(
            format,
            F::R8Sint
                | F::Rg8Sint
                | F::Rgba8Sint
                | F::R16Sint
                | F::Rg16Sint
                | F::Rgba16Sint
                | F::R32Sint
                | F::Rg32Sint
                | F::Rgba32Sint
        );
        let stencil_plane = native.0.desc.aspects == api::TextureAspects::STENCIL;
        let depth_plane = native.0.desc.aspects == api::TextureAspects::DEPTH;
        let integer_unsigned = stencil_plane
            || matches!(
                format,
                F::R8Uint
                    | F::Rg8Uint
                    | F::Rgba8Uint
                    | F::R16Uint
                    | F::Rg16Uint
                    | F::Rgba16Uint
                    | F::R32Uint
                    | F::Rg32Uint
                    | F::Rgba32Uint
                    | F::Rgb10a2Uint
                    | F::R64Uint
                    | F::Rg64Uint
            );
        let depth_format = super::super::texture::is_depth(format);
        let depth = depth_format && depth_plane;
        if depth_format && !depth_plane && !stencil_plane {
            return Err(Error::InvalidState);
        }
        let float32 = matches!(format, F::R32Float | F::Rg32Float | F::Rgba32Float);
        let compatible = match sample {
            api::TextureSampleType::Float => {
                !integer_signed && !integer_unsigned && !depth_format && !float32
            }
            api::TextureSampleType::UnfilterableFloat => {
                !integer_signed && !integer_unsigned && !depth_format
            }
            api::TextureSampleType::Depth => depth,
            api::TextureSampleType::Sint => integer_signed,
            api::TextureSampleType::Uint => integer_unsigned,
        };
        if !compatible {
            return Err(Error::InvalidState);
        }
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: match format {
                F::Depth16Unorm => windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UNORM,
                F::Depth32Float | F::Depth24Plus => {
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT
                }
                F::Depth24PlusStencil8 if depth_plane => {
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R24_UNORM_X8_TYPELESS
                }
                F::Depth24PlusStencil8 => {
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_X24_TYPELESS_G8_UINT
                }
                F::Depth32FloatStencil8 if depth_plane => {
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
                }
                F::Depth32FloatStencil8 => {
                    windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
                }
                _ => super::super::texture::dxgi_format(format),
            },
            ..Default::default()
        };
        // SAFETY: initialized matching feature structure and live device.
        let ok = unsafe {
            device.0.native.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                &mut support as *mut _ as *mut _,
                std::mem::size_of_val(&support) as u32,
            )
        }
        .is_ok();
        let required = if sample == api::TextureSampleType::Float {
            D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE
        } else {
            D3D12_FORMAT_SUPPORT1_SHADER_LOAD
        };
        if !ok || !support.Support1.contains(required) {
            return Err(Error::Unsupported("DX12 sampled texture format support"));
        }
    }
    Ok(())
}
