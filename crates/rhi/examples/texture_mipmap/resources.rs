//! Explicit mip-chain texture allocation and generation helpers.
//!
//! This module describes only the generated texture and its transitions; the
//! tunnel pipeline and outer frame lifecycle live in sibling modules.

use super::MipmapBackend;
use fluxel_rhi::api::*;

pub struct MipChain<B: MipmapBackend> {
    lane: LaneId,
    texture: Texture<B>,
    pub(super) width: u32,
    pub(super) height: u32,
    levels: u32,
}
impl<B: MipmapBackend> MipChain<B> {
    pub fn new(d: &Device<B>, lane: LaneId, width: u32, height: u32) -> Result<Self> {
        let levels = 1 + width.max(height).ilog2();
        let texture = d.create_texture(&TextureDesc {
            label: Some("generated mip chain".into()),
            dimension: TextureDimension::D2,
            extent: [width, height, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: levels,
            samples: 1,
            usage: TextureUsage(
                TextureUsage::COPY_DST.0
                    | TextureUsage::COPY_SRC.0
                    | TextureUsage::TEXTURE_BINDING.0,
            ),
            view_formats: vec![],
        })?;
        Ok(Self {
            lane,
            texture,
            width,
            height,
            levels,
        })
    }
    pub fn texture(&self) -> &Texture<B> {
        &self.texture
    }
    pub fn levels(&self) -> u32 {
        self.levels
    }
    /// Uploads the source KTX level and produces all remaining levels on the GPU.
    pub fn record_generation(&self, d: &Device<B>, base: &[u8]) -> Result<RecordedWork<B>> {
        let mut e = d.encoder(self.lane)?;
        self.encode_generation(&mut e, base)?;
        e.finish()
    }
    pub fn encode_generation(&self, e: &mut CommandEncoder<B>, base: &[u8]) -> Result<()> {
        if base.len() != self.width as usize * self.height as usize * 4 {
            return Err(Error::InvalidRange);
        }
        e.write_texture(
            TextureRegion {
                texture: &self.texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [self.width, self.height, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: self.width * 4,
                rows_per_image: self.height,
            },
            base,
        )?;
        // `write_texture` returns mip 0 to the documented Common boundary.
        // Each subsequent transition names the immediately preceding use, so
        // a Vulkan implementation can lower the chain without tracking image
        // state behind the RHI API.
        self.transition(
            e,
            0,
            ImageLayout::Common,
            ImageLayout::CopySrc,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        )?;
        for mip in 1..self.levels {
            self.transition(
                e,
                mip,
                ImageLayout::Undefined,
                ImageLayout::CopyDst,
                TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            )?;
            let sw = (self.width >> (mip - 1)).max(1);
            let sh = (self.height >> (mip - 1)).max(1);
            let dw = (self.width >> mip).max(1);
            let dh = (self.height >> mip).max(1);
            e.blit(
                TextureRegion {
                    texture: &self.texture,
                    mip: mip - 1,
                    origin: [0, 0, 0],
                    extent: [sw, sh, 1],
                    aspect: TextureAspects::COLOR,
                },
                TextureRegion {
                    texture: &self.texture,
                    mip,
                    origin: [0, 0, 0],
                    extent: [dw, dh, 1],
                    aspect: TextureAspects::COLOR,
                },
                FilterMode::Linear,
            )?;
            if mip + 1 < self.levels {
                self.transition(
                    e,
                    mip,
                    ImageLayout::CopyDst,
                    ImageLayout::CopySrc,
                    TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                )?;
            }
        }
        for mip in 0..self.levels {
            let (layout, accesses) = if mip + 1 == self.levels {
                (ImageLayout::CopyDst, AccessMask::COPY_WRITE)
            } else {
                (ImageLayout::CopySrc, AccessMask::COPY_READ)
            };
            self.transition(
                e,
                mip,
                layout,
                ImageLayout::ShaderReadOnly,
                TextureUses {
                    stages: StageMask::COPY,
                    accesses,
                },
                TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            )?;
        }
        Ok(())
    }

    fn transition(
        &self,
        encoder: &mut CommandEncoder<B>,
        mip: u32,
        from_layout: ImageLayout,
        to_layout: ImageLayout,
        from: TextureUses,
        to: TextureUses,
    ) -> Result<()> {
        encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &self.texture,
                aspects: TextureAspects::COLOR,
                mips: mip..mip + 1,
                layers: 0..1,
            },
            usage: StateTransition { from, to },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })])
    }
}
pub(super) struct Ktx {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixels: Vec<u8>,
}
pub(super) fn parse_ktx(bytes: &[u8]) -> Result<Ktx> {
    const MAGIC: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";
    if bytes.len() < 68 || &bytes[..12] != MAGIC || &bytes[12..16] != b"\x01\x02\x03\x04" {
        return Err(Error::Unsupported("09 requires little-endian KTX1"));
    }
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    if word(24) != 0x1908 || word(28) != 0x8058 {
        return Err(Error::Unsupported("09 requires RGBA8 KTX"));
    }
    let (width, height) = (word(36), word(40));
    let image = 64usize
        .checked_add(word(60) as usize)
        .ok_or(Error::InvalidRange)?;
    let size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::InvalidRange)?;
    let end = image
        .checked_add(4)
        .and_then(|n| n.checked_add(size))
        .ok_or(Error::InvalidRange)?;
    if width == 0 || height == 0 || end > bytes.len() || word(image) as usize != size {
        return Err(Error::InvalidRange);
    }
    Ok(Ktx {
        width,
        height,
        pixels: bytes[image + 4..end].to_vec(),
    })
}
