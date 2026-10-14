//! glTF/KTX asset loading, texture resources, and instance data for example 03.
use super::super::gltf;
use super::*;

pub(crate) const ROCK: &[u8] = include_bytes!("../assets/models/rock01.gltf");
pub(crate) const PLANET: &[u8] = include_bytes!("../assets/models/lavaplanet.gltf");
pub(crate) const ROCK_TEXTURE: &[u8] =
    include_bytes!("../assets/textures/texturearray_rocks_rgba.ktx");
pub(crate) const PLANET_TEXTURE: &[u8] = include_bytes!("../assets/textures/lavaplanet_rgba.ktx");
pub(crate) const INSTANCES: u32 = 8192;

pub(crate) struct Ktx {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: u32,
    pub(crate) mips: u32,
    pub(crate) data: Vec<Vec<Vec<u8>>>,
}
impl Ktx {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 68
            || bytes[..12]
                != [
                    0xab, b'K', b'T', b'X', b' ', b'1', b'1', 0xbb, 13, 10, 26, 10,
                ]
        {
            return Err(Error::Unsupported("03 expects KTX1"));
        }
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if word(12) != 0x04030201 || word(16) != 0x1401 || word(24) != 0x1908 || word(28) != 0x8058
        {
            return Err(Error::Unsupported("03 expects RGBA8 KTX"));
        }
        let (width, height, layers, mips) = (word(36), word(40), word(48).max(1), word(56));
        if width == 0 || height == 0 || mips == 0 {
            return Err(Error::InvalidRange);
        }
        let mut at = 64 + word(60) as usize;
        let mut data = Vec::with_capacity(mips as usize);
        for mip in 0..mips {
            let image_size_at = at.checked_add(4).ok_or(Error::InvalidRange)?;
            if image_size_at > bytes.len() {
                return Err(Error::InvalidRange);
            }
            let size = word(at) as usize;
            at = image_size_at;
            let width = (width >> mip).max(1) as usize;
            let height = (height >> mip).max(1) as usize;
            let layer_size = width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(4))
                .ok_or(Error::InvalidRange)?;
            let expected = layer_size
                .checked_mul(layers as usize)
                .ok_or(Error::InvalidRange)?;
            if size != expected {
                return Err(Error::Unsupported("03 KTX array mip layout"));
            }
            let mut level = Vec::with_capacity(layers as usize);
            for _ in 0..layers {
                let end = at.checked_add(layer_size).ok_or(Error::InvalidRange)?;
                if end > bytes.len() {
                    return Err(Error::InvalidRange);
                }
                level.push(bytes[at..end].to_vec());
                at = (end + 3) & !3;
            }
            data.push(level);
        }
        Ok(Self {
            width,
            height,
            layers,
            mips,
            data,
        })
    }
}

pub(crate) struct SceneData {
    pub(crate) rock_vertices: Vec<u8>,
    pub(crate) rock_indices: Vec<u8>,
    pub(crate) planet_vertices: Vec<u8>,
    pub(crate) planet_indices: Vec<u8>,
    pub(crate) rock_pixels: Ktx,
    pub(crate) planet_pixels: Vec<u8>,
    pub(crate) planet_extent: [u32; 2],
    pub(crate) instances: Vec<u8>,
}
pub(crate) fn load_scene() -> Result<SceneData> {
    let rock = gltf::load_embedded_model(ROCK, gltf::LoadOptions::CPP_PORT)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let planet = gltf::load_embedded_model(PLANET, gltf::LoadOptions::CPP_PORT)
        .map_err(|error| Error::Backend(error.to_string()))?;
    let rock_pixels = Ktx::parse(ROCK_TEXTURE)?;
    let planet_ktx = Ktx::parse(PLANET_TEXTURE)?;
    if rock_pixels.layers != 5 || rock_pixels.mips != 10 || planet_ktx.layers != 1 {
        return Err(Error::Unsupported(
            "03 requires five-layer ten-mip rock and single-layer planet KTX",
        ));
    }
    let planet_pixels = planet_ktx
        .data
        .first()
        .and_then(|level| level.first())
        .cloned()
        .ok_or(Error::InvalidRange)?;
    Ok(SceneData {
        rock_vertices: rock.vertices,
        rock_indices: rock.indices,
        planet_vertices: planet.vertices,
        planet_indices: planet.indices,
        planet_extent: [planet_ktx.width, planet_ktx.height],
        instances: instances(rock_pixels.layers),
        rock_pixels,
        planet_pixels,
    })
}

pub(crate) fn texture<B: InstancingBackend>(
    device: &Device<B>,
    label: &str,
    width: u32,
    height: u32,
) -> Result<Texture<B>> {
    let desc = TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [width, height, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING.union(TextureUsage::COPY_DST),
        view_formats: vec![],
    };
    match device.texture_support(&desc) {
        Support::Supported(support) if support.filterable => device.create_texture(&desc),
        Support::Supported(_) => Err(Error::Unsupported("03 needs filterable RGBA8")),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}
pub(crate) fn array_texture<B: InstancingBackend>(
    device: &Device<B>,
    ktx: &Ktx,
) -> Result<Texture<B>> {
    let desc = TextureDesc {
        label: Some("03 rock texture array".into()),
        dimension: TextureDimension::D2,
        extent: [ktx.width, ktx.height, 1],
        array_layers: ktx.layers,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: ktx.mips,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING.union(TextureUsage::COPY_DST),
        view_formats: vec![],
    };
    match device.texture_support(&desc) {
        Support::Supported(support) if support.filterable => device.create_texture(&desc),
        Support::Supported(_) => Err(Error::Unsupported(
            "03 needs filterable RGBA8 texture arrays",
        )),
        Support::Unsupported(reason) => Err(Error::Unsupported(reason)),
    }
}
pub(crate) fn depth<B: InstancingBackend>(
    device: &Device<B>,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("03 depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((texture, view))
}
pub(crate) fn color_view() -> TextureViewDesc {
    TextureViewDesc {
        format: TextureFormat::Rgba8Unorm,
        dimension: TextureViewDimension::D2,
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
    }
}
pub(crate) fn array_view(ktx: &Ktx) -> TextureViewDesc {
    TextureViewDesc {
        format: TextureFormat::Rgba8Unorm,
        dimension: TextureViewDimension::D2Array,
        aspects: TextureAspects::COLOR,
        mips: 0..ktx.mips,
        layers: 0..ktx.layers,
    }
}
pub(crate) fn upload_base<B: InstancingBackend>(
    encoder: &mut CommandEncoder<B>,
    texture: &Texture<B>,
    pixels: &[u8],
    extent: [u32; 2],
) -> Result<()> {
    encoder.write_texture(
        TextureRegion {
            texture,
            mip: 0,
            origin: [0; 3],
            extent: [extent[0], extent[1], 1],
            aspect: TextureAspects::COLOR,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: extent[0] * 4,
            rows_per_image: extent[1],
        },
        pixels,
    )
}
pub(crate) fn upload_ktx<B: InstancingBackend>(
    encoder: &mut CommandEncoder<B>,
    texture: &Texture<B>,
    ktx: &Ktx,
) -> Result<()> {
    for (mip, levels) in ktx.data.iter().enumerate() {
        let width = (ktx.width >> mip).max(1);
        let height = (ktx.height >> mip).max(1);
        for (layer, pixels) in levels.iter().enumerate() {
            encoder.write_texture(
                TextureRegion {
                    texture,
                    mip: mip as u32,
                    origin: [0, 0, layer as u32],
                    extent: [width, height, 1],
                    aspect: TextureAspects::COLOR,
                },
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: width * 4,
                    rows_per_image: height,
                },
                pixels,
            )?;
        }
    }
    Ok(())
}
fn instances(layers: u32) -> Vec<u8> {
    // Keep RHI captures repeatable. The ordinary C++ demo seeds its standard
    // random engine from wall time, so individual rock positions will differ.
    let mut random = Rng(0);
    let mut output = Vec::with_capacity(INSTANCES as usize * 32);
    for index in 0..INSTANCES {
        let range = if index < INSTANCES / 2 {
            [7., 11.]
        } else {
            [14., 18.]
        };
        let radius =
            ((range[1] * range[1] - range[0] * range[0]) * random.f() + range[0] * range[0]).sqrt();
        let angle = 6.2831853 * random.f();
        let layer = ((random.f() * layers as f32) as u32) % layers;
        for value in [
            radius * angle.cos(),
            random.f() * 0.5 - 0.25,
            radius * angle.sin(),
            3.14159 * random.f(),
            3.14159 * random.f(),
            3.14159 * random.f(),
            (1.5 + random.f() - random.f()) * 0.75,
            layer as f32,
        ] {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    output
}
struct Rng(u32);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / 16777216.
    }
}
