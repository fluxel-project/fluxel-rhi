//! KTX 1 RGBA8 loading for the compute input texture.
use fluxel_rhi::api::{Error, Result};

pub(super) struct Ktx {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixels: Vec<u8>,
}

pub(super) fn load_rgba8() -> Result<Ktx> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/textures/vulkan_11_rgba.ktx"
    );
    let bytes = std::fs::read(path).map_err(|error| Error::Backend(format!("{path}: {error}")))?;
    parse_rgba8(&bytes)
}

fn parse_rgba8(bytes: &[u8]) -> Result<Ktx> {
    const MAGIC: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";
    if bytes.len() < 68 || &bytes[..12] != MAGIC {
        return Err(Error::Shader("06_compute expects KTX 1".into()));
    }
    let word = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    if word(12) != 0x0403_0201
        || word(16) != 0x1401
        || word(20) != 1
        || word(24) != 0x1908
        || word(28) != 0x8058
        || word(32) != 0x1908
    {
        return Err(Error::Unsupported(
            "06_compute requires little-endian RGBA8 KTX",
        ));
    }
    let (width, height) = (word(36), word(40));
    let image = 64usize
        .checked_add(word(60) as usize)
        .ok_or(Error::InvalidRange)?;
    let prefix_end = image.checked_add(4).ok_or(Error::InvalidRange)?;
    if prefix_end > bytes.len() {
        return Err(Error::InvalidRange);
    }
    let length = word(image) as usize;
    let end = prefix_end.checked_add(length).ok_or(Error::InvalidRange)?;
    if width == 0
        || height == 0
        || end > bytes.len()
        || length != width as usize * height as usize * 4
    {
        return Err(Error::InvalidRange);
    }
    Ok(Ktx {
        width,
        height,
        pixels: bytes[prefix_end..end].to_vec(),
    })
}
