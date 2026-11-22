//! Explicit raster shading frequency, selected by the caller.

use super::{Device, Error, RasterPass, Result, Texture};
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
    /// Availability of image-based shading and explicit rate combiners.
    pub image: Option<ShadingRateImageCapabilities>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadingRateImageCapabilities {
    /// Width and height, in framebuffer pixels, covered by one image texel.
    pub tile_size: u32,
    /// Allows combining coarse-pixel axes with `Sum`.
    pub sum_combiner: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadingRateCombiner {
    Passthrough,
    Override,
    Min,
    Max,
    Sum,
}

impl ShadingRateCapabilities {
    /// Checks a rate request against the raster pass's sample count.
    pub fn supports(self, rate: ShadingRate, samples: u32) -> bool {
        match rate {
            ShadingRate::X1Y1 => matches!(samples, 1 | 2 | 4 | 8 | 16),
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
    /// Encodes a rate for one `R8Uint` image texel on this backend.
    /// The caller selects rates valid for the intended raster sample count.
    pub fn image_texel(&self, rate: ShadingRate) -> Result<u8> {
        let caps = self
            .device
            .caps
            .shading_rate
            .ok_or(Error::Unsupported("shading rate image"))?;
        if caps.image.is_none() || !caps.supports(rate, 1) {
            return Err(Error::Unsupported("shading rate image texel"));
        }
        B::shading_rate_texel(rate)
    }

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

    /// Combines the base rate with the primitive rate, then that result with
    /// the image rate. `None` unbinds the image. The caller transitions a
    /// supplied image to `ImageLayout::ShadingRateSource` before this call.
    /// Image texels outside the extent supply `X1Y1`.
    pub fn set_state(
        &self,
        pass: &mut RasterPass<'_, B>,
        rate: ShadingRate,
        combiners: [ShadingRateCombiner; 2],
        image: Option<&Texture<B>>,
    ) -> Result<()> {
        let caps = self
            .device
            .caps
            .shading_rate
            .and_then(|v| v.image)
            .ok_or(Error::Unsupported("shading rate image and combiners"))?;
        if pass.encoder.device != self.device.id
            || image.is_some_and(|image| image.device != self.device.id)
        {
            return Err(Error::WrongDevice);
        }
        if !caps.sum_combiner && combiners.contains(&ShadingRateCombiner::Sum) {
            return Err(Error::Unsupported("sum shading rate combiner"));
        }
        if !pass.active || pass.encoder.poisoned {
            return Err(Error::InvalidState);
        }
        B::set_shading_rate_state(
            pass.encoder.native.as_mut().ok_or(Error::InvalidState)?,
            rate,
            combiners,
            image.map(|image| &image.native),
        )
    }
}
