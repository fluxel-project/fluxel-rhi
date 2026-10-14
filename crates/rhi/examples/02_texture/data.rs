//! Texture asset decoding and transform-byte construction.
//!
//! This module owns CPU-side data preparation only; GPU resource creation and
//! command encoding remain in the scene module.
use fluxel_rhi::api::*;

pub(super) fn quad() -> [u8; 128] {
    // Core's framebuffer and sampled-texture Y both point down. The C++
    // positive-height Vulkan viewport put its +Y vertices on the lower half;
    // with Core's WebGPU-style viewport the upper vertices need texture v=0.
    let x: [f32; 32] = [
        1., 1., 0., 1., 0., 0., 0., 1., -1., 1., 0., 0., 0., 0., 0., 1., -1., -1., 0., 0., 1., 0.,
        0., 1., 1., -1., 0., 1., 1., 0., 0., 1.,
    ];
    let mut b = [0; 128];
    for (i, v) in x.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
pub(super) fn texture_transform(extent: [u32; 2]) -> [u8; 256] {
    // Match the source's right-handed look-at camera: z=-2.5, a 15° yaw and
    // a 60° perspective. WebGPU/DX12 still use a [0, 1] depth range; only
    // the example's world/view convention is right-handed.
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / 30f32.to_radians().tan();
    let projection = [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        256. / -255.9,
        -1.,
        0.,
        0.,
        -25.6 / 255.9,
        0.,
    ];
    let angle = 15f32.to_radians();
    let (s, c) = angle.sin_cos();
    let rotation = [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.];
    let view = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., -2.5, 1.,
    ];
    let model = multiply(view, rotation);
    let mut bytes = [0; 256];
    write_matrix(&mut bytes, 0, projection);
    write_matrix(&mut bytes, 64, model);
    // `Camera::viewPos` from the source camera at (0, 0, -2.5).
    for (index, value) in [0.0_f32, 0.0, 2.5, 0.0, 0.0].into_iter().enumerate() {
        bytes[128 + index * 4..132 + index * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut output = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    output
}

fn write_matrix(bytes: &mut [u8], offset: usize, matrix: [f32; 16]) {
    for (index, value) in matrix.into_iter().enumerate() {
        bytes[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(windows)]
pub(super) fn load_ktx() -> Result<(Vec<u8>, u32, u32)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/textures/metalplate01_rgba.ktx"
    );
    let bytes = std::fs::read(path).map_err(|error| Error::Backend(format!("{path}: {error}")))?;
    parse_ktx(&bytes)
}

#[cfg(target_os = "linux")]
pub(super) fn load_ktx() -> Result<(Vec<u8>, u32, u32)> {
    parse_ktx(include_bytes!("../assets/textures/metalplate01_rgba.ktx"))
}

#[cfg(target_arch = "wasm32")]
pub(super) fn load_ktx() -> Result<(Vec<u8>, u32, u32)> {
    parse_ktx(include_bytes!("../assets/textures/metalplate01_rgba.ktx"))
}

fn parse_ktx(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    const MAGIC: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";
    if bytes.len() < 68 || &bytes[..12] != MAGIC || &bytes[12..16] != b"\x01\x02\x03\x04" {
        return Err(Error::Unsupported("KTX1 little-endian RGBA8 texture"));
    }
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let (width, height) = (word(36), word(40));
    if word(24) != 0x1908 || word(28) != 0x8058 || width == 0 || height == 0 {
        return Err(Error::Unsupported("KTX1 RGBA8 texture"));
    }
    let offset = 64usize
        .checked_add(word(60) as usize)
        .ok_or(Error::InvalidRange)?;
    let size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::InvalidRange)?;
    if offset
        .checked_add(4 + size)
        .is_none_or(|end| end > bytes.len())
        || word(offset) as usize != size
    {
        return Err(Error::InvalidRange);
    }
    Ok((bytes[offset + 4..offset + 4 + size].to_vec(), width, height))
}
