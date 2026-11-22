//! Native shading-rate commands, rate-image admission and pass-end reset.

use super::{Dx12Device, Dx12Texture, encoder::Dx12Encoder};
use fluxel_rhi_core::api::{
    Error, Result, ShadingRate, ShadingRateCapabilities, ShadingRateCombiner,
    ShadingRateImageCapabilities,
};
use windows::{
    Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT_R8_UINT},
    core::Interface,
};

pub(super) fn probe(device: &ID3D12Device) -> Option<ShadingRateCapabilities> {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS6::default();
    // SAFETY: the selector and initialized output structure agree. The
    // query writes the result synchronously into this local value.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS6,
            std::ptr::from_mut(&mut options).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .ok()?;
    let image = if options.VariableShadingRateTier == D3D12_VARIABLE_SHADING_RATE_TIER_2
        && options.ShadingRateImageTileSize != 0
    {
        let mut combiners = D3D12_FEATURE_DATA_D3D12_OPTIONS10::default();
        // SAFETY: OPTIONS10 matches this writable local record.
        let sum_combiner = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS10,
                std::ptr::from_mut(&mut combiners).cast(),
                std::mem::size_of_val(&combiners) as u32,
            )
        }
        .is_ok()
            && combiners.VariableRateShadingSumCombinerSupported.as_bool();
        Some(ShadingRateImageCapabilities {
            tile_size: options.ShadingRateImageTileSize,
            sum_combiner,
        })
    } else {
        None
    };
    (options.VariableShadingRateTier != D3D12_VARIABLE_SHADING_RATE_TIER_NOT_SUPPORTED).then_some(
        ShadingRateCapabilities {
            additional_rates: options.AdditionalShadingRatesSupported.as_bool(),
            image,
        },
    )
}

pub(super) fn set(encoder: &mut Dx12Encoder, rate: ShadingRate) -> Result<()> {
    program(encoder, rate, None, None)
}

pub(super) fn set_state(
    encoder: &mut Dx12Encoder,
    rate: ShadingRate,
    combiners: [ShadingRateCombiner; 2],
    image: Option<&Dx12Texture>,
) -> Result<()> {
    let caps = image_caps(&encoder.device)?;
    if !caps.sum_combiner && combiners.contains(&ShadingRateCombiner::Sum) {
        return Err(Error::Unsupported("DX12 sum shading rate combiner"));
    }
    if let Some(image) = image {
        validate_image(&encoder.device, image)?;
    }
    program(encoder, rate, Some(combiners.map(native_combiner)), image)
}

fn image_caps(device: &Dx12Device) -> Result<ShadingRateImageCapabilities> {
    device
        .0
        .shading_rate
        .and_then(|rate| rate.image)
        .ok_or(Error::Unsupported("DX12 shading rate image and combiners"))
}

pub(super) fn validate_image(device: &Dx12Device, image: &Dx12Texture) -> Result<()> {
    image_caps(device)?;
    // SAFETY: the retained native image is queried synchronously.
    let desc = unsafe { image.0.resource.GetDesc() };
    let incompatible = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
        | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
        | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.Format != DXGI_FORMAT_R8_UINT
        || desc.DepthOrArraySize != 1
        || desc.MipLevels != 1
        || desc.SampleDesc.Count != 1
        || desc.SampleDesc.Quality != 0
        || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN
        || desc.Flags.0 & incompatible.0 != 0
    {
        return Err(Error::Unsupported("DX12 shading rate image descriptor"));
    }
    Ok(())
}

fn program(
    encoder: &mut Dx12Encoder,
    rate: ShadingRate,
    combiners: Option<[D3D12_SHADING_RATE_COMBINER; 2]>,
    image: Option<&Dx12Texture>,
) -> Result<()> {
    let caps = encoder
        .device
        .0
        .shading_rate
        .ok_or(Error::Unsupported("DX12 shading rate"))?;
    let samples = encoder.raster_samples.ok_or(Error::Unsupported(
        "DX12 shading rate without raster attachments",
    ))?;
    if !caps.supports(rate, samples) {
        return Err(Error::Unsupported(
            "DX12 shading rate for the pass sample count",
        ));
    }
    let list = if let Some(list) = &encoder.shading_rate_list {
        list
    } else {
        let list = encoder
            .list
            .cast::<ID3D12GraphicsCommandList5>()
            .map_err(|_| Error::Unsupported("DX12 shading rate command list"))?;
        encoder.shading_rate_list.insert(list)
    };
    // SAFETY: this is the open raster command list and rate is qualified for
    // the device and current sample count. Optional Tier 2 parameters were
    // validated before recording and the caller provides the image state.
    unsafe {
        list.RSSetShadingRate(
            native_rate(rate),
            combiners.as_ref().map(|value| value.as_ptr()),
        );
        if image.is_some() || encoder.shading_rate_image_bound {
            list.RSSetShadingRateImage(image.map(|image| &image.0.resource));
        }
    }
    encoder.shading_rate_image_bound = image.is_some();
    if let Some(image) = image {
        encoder.textures.push(image.clone());
    }
    Ok(())
}

pub(super) fn reset(encoder: &mut Dx12Encoder) {
    if let Some(list) = encoder.shading_rate_list.take() {
        // SAFETY: the same list is still open and fine shading is always valid.
        unsafe {
            list.RSSetShadingRate(D3D12_SHADING_RATE_1X1, None);
            if std::mem::take(&mut encoder.shading_rate_image_bound) {
                list.RSSetShadingRateImage(None::<&ID3D12Resource>);
            }
        }
    }
}

pub(super) fn native_rate(rate: ShadingRate) -> D3D12_SHADING_RATE {
    match rate {
        ShadingRate::X1Y1 => D3D12_SHADING_RATE_1X1,
        ShadingRate::X1Y2 => D3D12_SHADING_RATE_1X2,
        ShadingRate::X2Y1 => D3D12_SHADING_RATE_2X1,
        ShadingRate::X2Y2 => D3D12_SHADING_RATE_2X2,
        ShadingRate::X2Y4 => D3D12_SHADING_RATE_2X4,
        ShadingRate::X4Y2 => D3D12_SHADING_RATE_4X2,
        ShadingRate::X4Y4 => D3D12_SHADING_RATE_4X4,
    }
}

fn native_combiner(combiner: ShadingRateCombiner) -> D3D12_SHADING_RATE_COMBINER {
    match combiner {
        ShadingRateCombiner::Passthrough => D3D12_SHADING_RATE_COMBINER_PASSTHROUGH,
        ShadingRateCombiner::Override => D3D12_SHADING_RATE_COMBINER_OVERRIDE,
        ShadingRateCombiner::Min => D3D12_SHADING_RATE_COMBINER_MIN,
        ShadingRateCombiner::Max => D3D12_SHADING_RATE_COMBINER_MAX,
        ShadingRateCombiner::Sum => D3D12_SHADING_RATE_COMBINER_SUM,
    }
}
