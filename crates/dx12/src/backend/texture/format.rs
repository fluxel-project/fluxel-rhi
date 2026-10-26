//! DXGI texture format mapping and block footprint metadata.
use super::Dx12Device;
use fluxel_rhi_core::api;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub(crate) fn dxgi_format(format: api::TextureFormat) -> DXGI_FORMAT {
    match format {
        api::TextureFormat::R8Unorm => DXGI_FORMAT_R8_UNORM,
        api::TextureFormat::R8Snorm => DXGI_FORMAT_R8_SNORM,
        api::TextureFormat::R8Uint => DXGI_FORMAT_R8_UINT,
        api::TextureFormat::R8Sint => DXGI_FORMAT_R8_SINT,
        api::TextureFormat::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        api::TextureFormat::Rg8Snorm => DXGI_FORMAT_R8G8_SNORM,
        api::TextureFormat::Rg8Uint => DXGI_FORMAT_R8G8_UINT,
        api::TextureFormat::Rg8Sint => DXGI_FORMAT_R8G8_SINT,
        api::TextureFormat::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        api::TextureFormat::Rgba8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
        api::TextureFormat::Rgba8Uint => DXGI_FORMAT_R8G8B8A8_UINT,
        api::TextureFormat::Rgba8Sint => DXGI_FORMAT_R8G8B8A8_SINT,
        api::TextureFormat::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        api::TextureFormat::R16Unorm => DXGI_FORMAT_R16_UNORM,
        api::TextureFormat::R16Snorm => DXGI_FORMAT_R16_SNORM,
        api::TextureFormat::R16Uint => DXGI_FORMAT_R16_UINT,
        api::TextureFormat::R16Sint => DXGI_FORMAT_R16_SINT,
        api::TextureFormat::R16Float => DXGI_FORMAT_R16_FLOAT,
        api::TextureFormat::Rg16Unorm => DXGI_FORMAT_R16G16_UNORM,
        api::TextureFormat::Rg16Snorm => DXGI_FORMAT_R16G16_SNORM,
        api::TextureFormat::Rg16Uint => DXGI_FORMAT_R16G16_UINT,
        api::TextureFormat::Rg16Sint => DXGI_FORMAT_R16G16_SINT,
        api::TextureFormat::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        api::TextureFormat::Rgba16Unorm => DXGI_FORMAT_R16G16B16A16_UNORM,
        api::TextureFormat::Rgba16Snorm => DXGI_FORMAT_R16G16B16A16_SNORM,
        api::TextureFormat::Rgba16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
        api::TextureFormat::Rgba16Sint => DXGI_FORMAT_R16G16B16A16_SINT,
        api::TextureFormat::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        api::TextureFormat::R32Uint => DXGI_FORMAT_R32_UINT,
        api::TextureFormat::R32Sint => DXGI_FORMAT_R32_SINT,
        api::TextureFormat::R32Float => DXGI_FORMAT_R32_FLOAT,
        api::TextureFormat::Rg32Uint => DXGI_FORMAT_R32G32_UINT,
        api::TextureFormat::Rg32Sint => DXGI_FORMAT_R32G32_SINT,
        api::TextureFormat::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        api::TextureFormat::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        api::TextureFormat::Rgba32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
        api::TextureFormat::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        api::TextureFormat::Rgb10a2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
        api::TextureFormat::Rgb10a2Uint => DXGI_FORMAT_R10G10B10A2_UINT,
        api::TextureFormat::Rg11b10Ufloat => DXGI_FORMAT_R11G11B10_FLOAT,
        api::TextureFormat::Rgb9e5Ufloat => DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
        api::TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        api::TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        api::TextureFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        api::TextureFormat::Depth24Plus => DXGI_FORMAT_D32_FLOAT,
        api::TextureFormat::Depth16Unorm => DXGI_FORMAT_D16_UNORM,
        api::TextureFormat::Depth24PlusStencil8 => DXGI_FORMAT_D24_UNORM_S8_UINT,
        api::TextureFormat::Depth32FloatStencil8 => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        api::TextureFormat::Bc1RgbaUnorm => DXGI_FORMAT_BC1_UNORM,
        api::TextureFormat::Bc1RgbaUnormSrgb => DXGI_FORMAT_BC1_UNORM_SRGB,
        api::TextureFormat::Bc2RgbaUnorm => DXGI_FORMAT_BC2_UNORM,
        api::TextureFormat::Bc2RgbaUnormSrgb => DXGI_FORMAT_BC2_UNORM_SRGB,
        api::TextureFormat::Bc3RgbaUnorm => DXGI_FORMAT_BC3_UNORM,
        api::TextureFormat::Bc3RgbaUnormSrgb => DXGI_FORMAT_BC3_UNORM_SRGB,
        api::TextureFormat::Bc4RUnorm => DXGI_FORMAT_BC4_UNORM,
        api::TextureFormat::Bc4RSnorm => DXGI_FORMAT_BC4_SNORM,
        api::TextureFormat::Bc5RgUnorm => DXGI_FORMAT_BC5_UNORM,
        api::TextureFormat::Bc5RgSnorm => DXGI_FORMAT_BC5_SNORM,
        api::TextureFormat::Bc6hRgbUfloat => DXGI_FORMAT_BC6H_UF16,
        api::TextureFormat::Bc6hRgbFloat => DXGI_FORMAT_BC6H_SF16,
        api::TextureFormat::Bc7RgbaUnorm => DXGI_FORMAT_BC7_UNORM,
        api::TextureFormat::Bc7RgbaUnormSrgb => DXGI_FORMAT_BC7_UNORM_SRGB,
        _ => unreachable!("validated DX12 texture format"),
    }
}

pub(super) fn compatible_color_view(base: api::TextureFormat, view: api::TextureFormat) -> bool {
    use api::TextureFormat::*;
    base == view
        || matches!(
            (base, view),
            (Rgba8Unorm, Rgba8UnormSrgb)
                | (Rgba8UnormSrgb, Rgba8Unorm)
                | (Bgra8Unorm, Bgra8UnormSrgb)
                | (Bgra8UnormSrgb, Bgra8Unorm)
        )
}

pub(super) fn typeless_color(format: api::TextureFormat) -> Option<DXGI_FORMAT> {
    use api::TextureFormat::*;
    match format {
        Rgba8Unorm | Rgba8UnormSrgb => Some(DXGI_FORMAT_R8G8B8A8_TYPELESS),
        Bgra8Unorm | Bgra8UnormSrgb => Some(DXGI_FORMAT_B8G8R8A8_TYPELESS),
        _ => None,
    }
}
pub(crate) fn bc_block_bytes(format: api::TextureFormat) -> Option<u32> {
    use api::TextureFormat::*;
    match format {
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb | Bc4RUnorm | Bc4RSnorm => Some(8),
        Bc2RgbaUnorm | Bc2RgbaUnormSrgb | Bc3RgbaUnorm | Bc3RgbaUnormSrgb | Bc5RgUnorm
        | Bc5RgSnorm | Bc6hRgbUfloat | Bc6hRgbFloat | Bc7RgbaUnorm | Bc7RgbaUnormSrgb => Some(16),
        _ => None,
    }
}

/// Color texel size in bytes; BC formats use `bc_block_bytes` instead.
pub(crate) fn color_texel_bytes(format: api::TextureFormat) -> Option<u32> {
    use api::TextureFormat::*;
    Some(match format {
        R8Unorm | R8Snorm | R8Uint | R8Sint => 1,
        Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint | R16Unorm | R16Snorm | R16Uint | R16Sint
        | R16Float => 2,
        Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Bgra8Unorm
        | Bgra8UnormSrgb | Rg16Unorm | Rg16Snorm | Rg16Uint | Rg16Sint | Rg16Float | R32Uint
        | R32Sint | R32Float | Rgb10a2Unorm | Rgb10a2Uint | Rg11b10Ufloat | Rgb9e5Ufloat => 4,
        Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint | Rgba16Float | Rg32Uint | Rg32Sint
        | Rg32Float => 8,
        Rgba32Uint | Rgba32Sint | Rgba32Float => 16,
        _ => return None,
    })
}

pub(crate) fn integer_color(format: api::TextureFormat) -> bool {
    use api::TextureFormat::*;
    matches!(
        format,
        R8Uint
            | R8Sint
            | Rg8Uint
            | Rg8Sint
            | Rgba8Uint
            | Rgba8Sint
            | R16Uint
            | R16Sint
            | Rg16Uint
            | Rg16Sint
            | Rgba16Uint
            | Rgba16Sint
            | R32Uint
            | R32Sint
            | Rg32Uint
            | Rg32Sint
            | Rgba32Uint
            | Rgba32Sint
            | Rgb10a2Uint
    )
}

pub(crate) fn integer_clear_bits(
    format: api::TextureFormat,
    value: api::ClearColor,
) -> Option<(bool, [u32; 4])> {
    use api::TextureFormat::*;
    match (format, value) {
        (
            R8Uint | Rg8Uint | Rgba8Uint | R16Uint | Rg16Uint | Rgba16Uint | R32Uint | Rg32Uint
            | Rgba32Uint | Rgb10a2Uint,
            api::ClearColor::Uint(v),
        ) => Some((false, v)),
        (
            R8Sint | Rg8Sint | Rgba8Sint | R16Sint | Rg16Sint | Rgba16Sint | R32Sint | Rg32Sint
            | Rgba32Sint,
            api::ClearColor::Sint(v),
        ) => Some((true, v.map(|x| x as u32))),
        _ => None,
    }
}

pub(super) fn format_available(device: &Dx12Device, desc: &api::TextureDesc) -> bool {
    if bc_block_bytes(desc.format).is_some() && (desc.extent[0] % 4 != 0 || desc.extent[1] % 4 != 0)
    {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS8::default();
        // SAFETY: the query writes only the matching, initialized options struct.
        let result = unsafe {
            device.0.native.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS8,
                &mut options as *mut _ as *mut _,
                std::mem::size_of_val(&options) as u32,
            )
        };
        if result.is_err() || !options.UnalignedBlockTexturesSupported.as_bool() {
            return false;
        }
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: dxgi_format(desc.format),
        ..Default::default()
    };
    // SAFETY: the feature query receives the matching initialized native structure.
    unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &mut support as *mut _ as *mut _,
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support
            .Support1
            .contains(if desc.dimension == api::TextureDimension::D1 {
                D3D12_FORMAT_SUPPORT1_TEXTURE1D
            } else if desc.dimension == api::TextureDimension::D3 {
                D3D12_FORMAT_SUPPORT1_TEXTURE3D
            } else {
                D3D12_FORMAT_SUPPORT1_TEXTURE2D
            })
        && (!is_depth(desc.format)
            || support
                .Support1
                .contains(D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL))
        && (!desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
            || (matches!(
                desc.format,
                api::TextureFormat::Depth32Float
                    | api::TextureFormat::Depth16Unorm
                    | api::TextureFormat::Depth24Plus
                    | api::TextureFormat::Depth24PlusStencil8
                    | api::TextureFormat::Depth32FloatStencil8
            ) && format_has_support(
                device,
                if desc.format == api::TextureFormat::Depth16Unorm {
                    api::TextureFormat::R16Unorm
                } else {
                    api::TextureFormat::R32Float
                },
                D3D12_FORMAT_SUPPORT1_SHADER_LOAD,
            ) && depth_planes_available(device, desc.format, desc.samples > 1))
            || (!is_depth(desc.format)
                && support.Support1.contains(D3D12_FORMAT_SUPPORT1_SHADER_LOAD)))
        && (!(desc.samples > 1 || desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT))
            || is_depth(desc.format)
            || support
                .Support1
                .contains(D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
        && (desc.samples == 1
            || !desc.usage.contains(api::TextureUsage::TEXTURE_BINDING)
            || if is_depth(desc.format) {
                format_has_support(
                    device,
                    if desc.format == api::TextureFormat::Depth16Unorm {
                        api::TextureFormat::R16Unorm
                    } else {
                        api::TextureFormat::R32Float
                    },
                    D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD,
                )
            } else {
                support
                    .Support1
                    .contains(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD)
            })
        && (!desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
            || storage_available(device, desc))
}
fn storage_available(device: &Dx12Device, desc: &api::TextureDesc) -> bool {
    let format = match desc.format {
        api::TextureFormat::Rgba8UnormSrgb
            if desc.view_formats.contains(&api::TextureFormat::Rgba8Unorm) =>
        {
            api::TextureFormat::Rgba8Unorm
        }
        api::TextureFormat::Bgra8UnormSrgb
            if desc.view_formats.contains(&api::TextureFormat::Bgra8Unorm) =>
        {
            api::TextureFormat::Bgra8Unorm
        }
        value => value,
    };
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: dxgi_format(format),
        ..Default::default()
    };
    // SAFETY: the feature query receives its matching initialized structure.
    unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &mut support as *mut _ as *mut _,
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support
            .Support1
            .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
        && support
            .Support2
            .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
        && support
            .Support2
            .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)
}
fn depth_planes_available(
    device: &Dx12Device,
    format: api::TextureFormat,
    multisampled: bool,
) -> bool {
    let formats: &[DXGI_FORMAT] = match format {
        api::TextureFormat::Depth24PlusStencil8 => &[
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            DXGI_FORMAT_X24_TYPELESS_G8_UINT,
        ],
        api::TextureFormat::Depth32FloatStencil8 => &[
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
        ],
        _ => return true,
    };
    formats.iter().all(|&format| {
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: format,
            ..Default::default()
        };
        // SAFETY: the query writes to the matching initialized support structure.
        unsafe {
            device.0.native.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                &mut support as *mut _ as *mut _,
                std::mem::size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.contains(D3D12_FORMAT_SUPPORT1_SHADER_LOAD)
            && (!multisampled
                || support
                    .Support1
                    .contains(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD))
    })
}
pub(crate) fn format_has_support(
    device: &Dx12Device,
    format: api::TextureFormat,
    flag: D3D12_FORMAT_SUPPORT1,
) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: dxgi_format(format),
        ..Default::default()
    };
    // SAFETY: the feature query writes into the matching initialized struct.
    unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &mut support as *mut _ as *mut _,
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support.Support1.contains(flag)
}

pub(super) fn supported_sample_counts(dv: &Dx12Device, d: &api::TextureDesc) -> Vec<u32> {
    [1, 2, 4, 8, 16, 32]
        .into_iter()
        .filter(|&count| sample_count_supported(dv, d.format, count))
        .collect()
}

/// Check only the sample count needed by resource creation. The full supported
/// list is appropriate for capability queries, but asking the driver about
/// every multisample count for a single-sample texture repeats unrelated work.
pub(super) fn sample_count_supported(
    dv: &Dx12Device,
    format: api::TextureFormat,
    count: u32,
) -> bool {
    if count == 1 {
        return true;
    }
    if ![2, 4, 8, 16, 32].contains(&count) {
        return false;
    }
    let mut quality = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
        Format: dxgi_format(format),
        SampleCount: count,
        Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
        NumQualityLevels: 0,
    };
    // SAFETY: `quality` is the exact feature struct required by this query
    // and stays mutable/alive for the synchronous D3D12 call.
    unsafe {
        dv.0.native.CheckFeatureSupport(
            D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
            &mut quality as *mut _ as *mut _,
            std::mem::size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32,
        )
    }
    .is_ok()
        && quality.NumQualityLevels > 0
}

pub(crate) fn is_depth(format: api::TextureFormat) -> bool {
    matches!(
        format,
        api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32FloatStencil8
    )
}
pub(crate) fn depth_aspects(format: api::TextureFormat) -> api::TextureAspects {
    if matches!(
        format,
        api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
    ) {
        api::TextureAspects::DEPTH
    } else {
        api::TextureAspects::DEPTH | api::TextureAspects::STENCIL
    }
}
