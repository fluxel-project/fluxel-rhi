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
        api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Stencil8 => {
            DXGI_FORMAT_D24_UNORM_S8_UINT
        }
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
    base == view || typeless_color(base).is_some_and(|format| typeless_color(view) == Some(format))
}

pub(super) fn typeless_color(format: api::TextureFormat) -> Option<DXGI_FORMAT> {
    use api::TextureFormat::*;
    match format {
        Rgba8Unorm | Rgba8UnormSrgb => Some(DXGI_FORMAT_R8G8B8A8_TYPELESS),
        Bgra8Unorm | Bgra8UnormSrgb => Some(DXGI_FORMAT_B8G8R8A8_TYPELESS),
        Bc1RgbaUnorm | Bc1RgbaUnormSrgb => Some(DXGI_FORMAT_BC1_TYPELESS),
        Bc2RgbaUnorm | Bc2RgbaUnormSrgb => Some(DXGI_FORMAT_BC2_TYPELESS),
        Bc3RgbaUnorm | Bc3RgbaUnormSrgb => Some(DXGI_FORMAT_BC3_TYPELESS),
        Bc7RgbaUnorm | Bc7RgbaUnormSrgb => Some(DXGI_FORMAT_BC7_TYPELESS),
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

pub(super) fn available_format_support(
    device: &Dx12Device,
    desc: &api::TextureDesc,
) -> Option<D3D12_FORMAT_SUPPORT1> {
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
            return None;
        }
    }
    let format = dxgi_format(desc.format);
    let support = native_format_support(device, format)?.Support1;
    let dimensions = texture_dimension_support(desc.dimension);
    let allocation = super::allocation_format(desc).ok()?;
    let depth = is_depth(desc.format);
    let sampled = desc.usage.contains(api::TextureUsage::TEXTURE_BINDING);
    let render = desc.usage.contains(api::TextureUsage::RENDER_ATTACHMENT);
    let multisampled = desc.samples > 1;
    let available = support.contains(dimensions)
        && (allocation == format || native_format_has_support(device, allocation, dimensions))
        && (!depth || support.contains(D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL))
        && (!sampled
            || if depth {
                depth_planes_available(device, desc.format, desc.samples)
            } else {
                support.contains(sample_load_requirement(desc.samples))
            })
        && (!(multisampled || render)
            || depth
            || support.contains(D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
        && (!multisampled
            || depth
            || support.contains(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET))
        && (!desc.usage.contains(api::TextureUsage::STORAGE_BINDING)
            || storage_available(device, desc));
    available.then_some(support)
}

fn texture_dimension_support(dimension: api::TextureDimension) -> D3D12_FORMAT_SUPPORT1 {
    match dimension {
        api::TextureDimension::D1 => D3D12_FORMAT_SUPPORT1_TEXTURE1D,
        api::TextureDimension::D2 => D3D12_FORMAT_SUPPORT1_TEXTURE2D,
        api::TextureDimension::D3 => D3D12_FORMAT_SUPPORT1_TEXTURE3D,
    }
}

pub(in crate::backend) fn sample_load_requirement(samples: u32) -> D3D12_FORMAT_SUPPORT1 {
    if samples > 1 {
        D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD
    } else {
        D3D12_FORMAT_SUPPORT1_SHADER_LOAD
    }
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
    storage_view_format_available(device, format, desc.dimension, None)
}

/// Validates an exact layout access, or any typed access at allocation time.
pub(crate) fn storage_view_format_available(
    device: &Dx12Device,
    format: api::TextureFormat,
    dimension: api::TextureDimension,
    access: Option<api::StorageAccess>,
) -> bool {
    if color_texel_bytes(format).is_none()
        || matches!(
            format,
            api::TextureFormat::Rgba8UnormSrgb | api::TextureFormat::Bgra8UnormSrgb
        )
    {
        return false;
    }
    native_format_support(device, dxgi_format(format)).is_some_and(|support| {
        support
            .Support1
            .contains(texture_dimension_support(dimension))
            && typed_storage_access(&support, access)
    })
}

pub(crate) fn typed_storage_access(
    support: &D3D12_FEATURE_DATA_FORMAT_SUPPORT,
    access: Option<api::StorageAccess>,
) -> bool {
    if !support
        .Support1
        .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
    {
        return false;
    }
    let load = support
        .Support2
        .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD);
    let store = support
        .Support2
        .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
    match access {
        None => load || store,
        Some(api::StorageAccess::Read) => load,
        Some(api::StorageAccess::Write) => store,
        // Atomic operation support is checked separately for the format and stage.
        Some(api::StorageAccess::ReadWrite | api::StorageAccess::Atomic) => load && store,
    }
}
fn depth_planes_available(device: &Dx12Device, format: api::TextureFormat, samples: u32) -> bool {
    [api::TextureAspects::DEPTH, api::TextureAspects::STENCIL]
        .into_iter()
        .filter(|aspect| depth_aspects(format).0 & aspect.0 != 0)
        .all(|aspect| {
            native_format_has_support(
                device,
                super::sampled_srv_format(format, aspect),
                sample_load_requirement(samples),
            )
        })
}

pub(crate) fn format_has_support(
    device: &Dx12Device,
    format: api::TextureFormat,
    flag: D3D12_FORMAT_SUPPORT1,
) -> bool {
    native_format_has_support(device, dxgi_format(format), flag)
}

pub(super) fn native_format_has_support(
    device: &Dx12Device,
    format: DXGI_FORMAT,
    flag: D3D12_FORMAT_SUPPORT1,
) -> bool {
    native_format_support(device, format).is_some_and(|facts| facts.Support1.contains(flag))
}

pub(in crate::backend) fn native_format_support(
    device: &Dx12Device,
    format: DXGI_FORMAT,
) -> Option<D3D12_FEATURE_DATA_FORMAT_SUPPORT> {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: format,
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
    .then_some(support)
}

pub(crate) fn format_supports_texture_atomic_add_native(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    format: api::TextureFormat,
) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: dxgi_format(format),
        ..Default::default()
    };
    // SAFETY: the feature query writes into the matching initialized struct.
    unsafe {
        device.CheckFeatureSupport(
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
            .contains(D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
}

pub(crate) fn format_supports_texture_atomic_unsigned_min_native(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    format: api::TextureFormat,
) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: dxgi_format(format),
        ..Default::default()
    };
    // SAFETY: the initialized structure has the exact type and size requested
    // by this synchronous format-support query.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &mut support as *mut _ as *mut _,
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support
            .Support1
            .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
        && support.Support2.contains(
            D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX
                | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE,
        )
}

pub(crate) fn format_supports_texture_atomic_signed_min_native(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    format: api::TextureFormat,
) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: dxgi_format(format),
        ..Default::default()
    };
    // SAFETY: the initialized structure has the exact type and size requested
    // by this synchronous format-support query.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &mut support as *mut _ as *mut _,
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support
            .Support1
            .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
        && support.Support2.contains(
            D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX
                | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE,
        )
}

pub(crate) fn format_supports_texture_atomic_bitwise_native(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    format: api::TextureFormat,
) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: dxgi_format(format),
        ..Default::default()
    };
    // SAFETY: the initialized structure has the exact type and size requested
    // by this synchronous format-support query.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &mut support as *mut _ as *mut _,
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support
            .Support1
            .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
        && support.Support2.contains(
            D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE,
        )
}

pub(super) fn supported_sample_counts(dv: &Dx12Device, d: &api::TextureDesc) -> Vec<u32> {
    let mut candidate = d.clone();
    let mut counts = Vec::new();
    candidate.samples = 1;
    if d.samples == 1
        || (super::valid(&candidate).is_ok() && available_format_support(dv, &candidate).is_some())
    {
        counts.push(1);
    }
    // The structural and format gates are shared by all multisampled
    // counts. The native quality-level check then qualifies each count.
    candidate.samples = 2;
    if d.samples > 1
        || (super::valid(&candidate).is_ok() && available_format_support(dv, &candidate).is_some())
    {
        counts.extend(
            [2, 4, 8, 16, 32]
                .into_iter()
                .filter(|&count| count == d.samples || sample_count_supported(dv, d.format, count)),
        );
    }
    counts
}

/// Check only the sample count needed by resource creation. The full supported
/// list is appropriate for capability queries, but asking the driver about
/// every multisample count for a single-sample texture repeats unrelated work.
pub(in crate::backend) fn sample_count_supported(
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
            | api::TextureFormat::Stencil8
    )
}
pub(crate) fn depth_aspects(format: api::TextureFormat) -> api::TextureAspects {
    if format == api::TextureFormat::Stencil8 {
        api::TextureAspects::STENCIL
    } else if matches!(
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
