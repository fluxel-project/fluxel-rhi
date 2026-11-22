//! Native dynamic per-draw shading rate and pass-end state restoration.

use super::encoder::Dx12Encoder;
use fluxel_rhi_core::api::{Error, Result, ShadingRate, ShadingRateCapabilities};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

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
    (options.VariableShadingRateTier != D3D12_VARIABLE_SHADING_RATE_TIER_NOT_SUPPORTED).then_some(
        ShadingRateCapabilities {
            additional_rates: options.AdditionalShadingRatesSupported.as_bool(),
        },
    )
}

pub(super) fn set(encoder: &mut Dx12Encoder, rate: ShadingRate) -> Result<()> {
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
    // the device and the current attachment sample count. No combiners are set.
    unsafe { list.RSSetShadingRate(native_rate(rate), None) };
    Ok(())
}

pub(super) fn reset(encoder: &mut Dx12Encoder) {
    if let Some(list) = encoder.shading_rate_list.take() {
        // SAFETY: the same list is still open and fine shading is always valid.
        unsafe { list.RSSetShadingRate(D3D12_SHADING_RATE_1X1, None) };
    }
}

fn native_rate(rate: ShadingRate) -> D3D12_SHADING_RATE {
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
