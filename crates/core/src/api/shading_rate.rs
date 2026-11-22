//! Explicit raster shading frequency, selected by the caller.

use super::{Device, Error, RasterPass, Result};
use crate::backend::RasterBackend;

/// Number of pixels covered by a coarse shading invocation, as width × height.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadingRate {
    X1Y1,
    X1Y2,
    X2Y1,
    X2Y2,
    X2Y4,
    X4Y2,
    X4Y4,
}

/// Device limits for explicitly requested coarse pixel shading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadingRateCapabilities {
    /// Support for requests containing a four-pixel axis.
    pub additional_rates: bool,
}

impl ShadingRateCapabilities {
    /// Checks a rate request against the raster pass's sample count.
    pub fn supports(self, rate: ShadingRate, samples: u32) -> bool {
        match rate {
            ShadingRate::X1Y1 => samples.is_power_of_two(),
            ShadingRate::X1Y2 | ShadingRate::X2Y1 | ShadingRate::X2Y2 => {
                matches!(samples, 1 | 2 | 4)
            }
            ShadingRate::X2Y4 => self.additional_rates && matches!(samples, 1 | 2),
            ShadingRate::X4Y2 | ShadingRate::X4Y4 => self.additional_rates && samples == 1,
        }
    }
}

/// Opt-in operations for the device's raster shading rate.
pub struct ShadingRateDomain<'a, B: RasterBackend> {
    device: &'a Device<B>,
}

impl<B: RasterBackend> Device<B> {
    pub fn shading_rate(&self) -> Option<ShadingRateDomain<'_, B>> {
        self.caps
            .shading_rate
            .map(|_| ShadingRateDomain { device: self })
    }
}

impl<B: RasterBackend> ShadingRateDomain<'_, B> {
    /// Changes the requested rate for subsequent draws in this pass.
    /// The pass begins at `X1Y1`. Shader and pipeline rules can select finer shading.
    pub fn set_rate(&self, pass: &mut RasterPass<'_, B>, rate: ShadingRate) -> Result<()> {
        if pass.encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        if !pass.active || pass.encoder.poisoned {
            return Err(Error::InvalidState);
        }
        B::set_shading_rate(
            pass.encoder.native.as_mut().ok_or(Error::InvalidState)?,
            rate,
        )
    }
}
