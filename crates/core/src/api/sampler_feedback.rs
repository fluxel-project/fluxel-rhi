//! Explicit sampler feedback maps and conversion to readable texels.

use super::{
    CommandEncoder, Device, DeviceId, Error, Result, Texture, TextureDimension, TextureFormat,
    TextureUsage,
};
use crate::backend::{Backend, CommandBackend};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerFeedbackTier {
    Tier09,
    Tier10,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerFeedbackMode {
    /// Decodes the most detailed requested mip. A cleared texel is 255.
    MinMip,
    /// Decodes each requested mip region. A cleared texel is zero.
    MipRegionUsed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplerFeedbackDesc {
    pub mode: SamplerFeedbackMode,
    /// Width and height in texels of the base mip of the sampled texture.
    pub mip_region: [u32; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerFeedbackState {
    UnorderedAccess,
    ResolveSource,
}

/// Minimum base dimensions and exact subresource counts of an R8Uint decode target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplerFeedbackDecodeLayout {
    pub minimum_extent: [u32; 2],
    pub mip_levels: u32,
    pub array_layers: u32,
}

/// Opaque map associated with exactly one sampled texture.
pub struct SamplerFeedbackMap<B: Backend> {
    pub(crate) device: DeviceId,
    target: Texture<B>,
    native: B::NativeTexture,
    desc: SamplerFeedbackDesc,
    decode: SamplerFeedbackDecodeLayout,
}

impl<B: Backend> SamplerFeedbackMap<B> {
    pub fn descriptor(&self) -> SamplerFeedbackDesc {
        self.desc
    }

    pub fn decode_layout(&self) -> SamplerFeedbackDecodeLayout {
        self.decode
    }
}

/// Backend SPI for dedicated opaque maps. The `NativeTexture` used for a map
/// is private to this domain and must not be treated as an ordinary texture.
pub trait SamplerFeedbackBackend: CommandBackend {
    fn create_feedback_map(
        device: &Self::NativeDevice,
        target: &Self::NativeTexture,
        desc: SamplerFeedbackDesc,
    ) -> Result<Self::NativeTexture>;
    fn clear_feedback_map(
        encoder: &mut Self::NativeEncoder,
        target: &Self::NativeTexture,
        map: &Self::NativeTexture,
    ) -> Result<()>;
    fn transition_feedback_map(
        encoder: &mut Self::NativeEncoder,
        map: &Self::NativeTexture,
        before: SamplerFeedbackState,
        after: SamplerFeedbackState,
    ) -> Result<()>;
    fn decode_feedback_map(
        encoder: &mut Self::NativeEncoder,
        map: &Self::NativeTexture,
        destination: &Self::NativeTexture,
    ) -> Result<()>;
}

pub struct SamplerFeedbackDomain<'a, B: SamplerFeedbackBackend> {
    device: &'a Device<B>,
}

impl<B: SamplerFeedbackBackend> Device<B> {
    pub fn sampler_feedback(&self) -> Option<SamplerFeedbackDomain<'_, B>> {
        self.caps
            .sampler_feedback
            .map(|_| SamplerFeedbackDomain { device: self })
    }
}

impl<B: SamplerFeedbackBackend> SamplerFeedbackDomain<'_, B> {
    /// Creates an opaque map in unordered-access state. Clear it before use.
    pub fn create_map(
        &self,
        texture: &Texture<B>,
        desc: SamplerFeedbackDesc,
    ) -> Result<SamplerFeedbackMap<B>> {
        if texture.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let target = texture.descriptor().ok_or(Error::InvalidState)?;
        if target.dimension != TextureDimension::D2
            || target.samples != 1
            || !target.usage.contains(TextureUsage::TEXTURE_BINDING)
        {
            return Err(Error::Unsupported("sampler feedback target texture"));
        }
        let [width, height] = desc.mip_region;
        if [width, height]
            .into_iter()
            .zip(target.extent)
            .any(|(region, extent)| !region.is_power_of_two() || region < 4 || region > extent / 2)
        {
            return Err(Error::InvalidRange);
        }
        let decode = SamplerFeedbackDecodeLayout {
            minimum_extent: [
                target.extent[0].div_ceil(width),
                target.extent[1].div_ceil(height),
            ],
            mip_levels: match desc.mode {
                SamplerFeedbackMode::MinMip => 1,
                SamplerFeedbackMode::MipRegionUsed => target.mip_levels,
            },
            array_layers: target.array_layers,
        };
        let native = B::create_feedback_map(&self.device.native, &texture.native, desc)?;
        Ok(SamplerFeedbackMap {
            device: self.device.id,
            target: texture.clone(),
            native,
            desc,
            decode,
        })
    }

    /// Clears all feedback. The caller keeps the map in unordered-access state.
    pub fn clear(
        &self,
        encoder: &mut CommandEncoder<B>,
        map: &SamplerFeedbackMap<B>,
    ) -> Result<()> {
        self.check_device(encoder, map)?;
        encoder.call(|encoder| B::clear_feedback_map(encoder, &map.target.native, &map.native))
    }

    /// Records a caller-selected transition. Unordered access to unordered
    /// access orders consecutive accesses to this map.
    pub fn transition(
        &self,
        encoder: &mut CommandEncoder<B>,
        map: &SamplerFeedbackMap<B>,
        before: SamplerFeedbackState,
        after: SamplerFeedbackState,
    ) -> Result<()> {
        self.check_device(encoder, map)?;
        encoder.call(|encoder| B::transition_feedback_map(encoder, &map.native, before, after))
    }

    /// Decodes the complete map into an R8Uint texture. The map must be in
    /// resolve-source state and the destination in `ImageLayout::ResolveDst`.
    pub fn decode(
        &self,
        encoder: &mut CommandEncoder<B>,
        map: &SamplerFeedbackMap<B>,
        destination: &Texture<B>,
    ) -> Result<()> {
        self.check_device(encoder, map)?;
        if destination.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let desc = destination.descriptor().ok_or(Error::InvalidState)?;
        let layout = map.decode;
        if desc.dimension != TextureDimension::D2
            || desc.format != TextureFormat::R8Uint
            || desc.samples != 1
            || desc.mip_levels != layout.mip_levels
            || desc.array_layers != layout.array_layers
            || desc.extent[0] < layout.minimum_extent[0]
            || desc.extent[1] < layout.minimum_extent[1]
            || !desc.usage.contains(TextureUsage::COPY_DST)
        {
            return Err(Error::Unsupported("sampler feedback decode destination"));
        }
        encoder.call(|encoder| B::decode_feedback_map(encoder, &map.native, &destination.native))
    }

    fn check_device(&self, encoder: &CommandEncoder<B>, map: &SamplerFeedbackMap<B>) -> Result<()> {
        if encoder.device != self.device.id || map.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        Ok(())
    }
}
