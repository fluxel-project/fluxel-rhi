//! Immediate buffer fills and standalone texture subresource clears.
//!
//! Each chunk is uploaded before returning. The staging bytes contain data,
//! never commands, and no fill work is deferred to submission.

use super::*;

pub(super) fn texture(
    encoder: &mut GlEncoder,
    texture: &GlTexture,
    mips: std::ops::Range<u32>,
    layers: std::ops::Range<u32>,
    value: api::TextureClearValue,
) -> Result<()> {
    use crate::backend::gl::api::{GlFormat as F, GlTextureDimension as D};
    if !std::rc::Rc::ptr_eq(&encoder.state, &texture.state) {
        return Err(Error::WrongDevice);
    }
    let desc = texture.desc;
    if mips.start >= mips.end
        || mips.end > desc.mip_level_count
        || layers.start >= layers.end
        || layers.end
            > if desc.dimension == D::D3 {
                1
            } else {
                desc.extent.depth_or_layers
            }
    {
        return Err(Error::InvalidRange);
    }
    if !matches!(desc.dimension, D::D1 | D::D2 | D::D2Array | D::D3) {
        return Err(Error::Unsupported("GL clear texture dimension"));
    }
    let valid = match (desc.format, value) {
        (
            F::Depth16Unorm
            | F::Depth24Unorm
            | F::Depth24PlusStencil8
            | F::Depth32Float
            | F::Depth32FloatStencil8,
            api::TextureClearValue::Depth(v),
        ) => v.is_finite() && (0.0..=1.0).contains(&v),
        (
            F::R8Uint
            | F::Rg8Uint
            | F::Rgba8Uint
            | F::R16Uint
            | F::Rg16Uint
            | F::Rgba16Uint
            | F::Rgb10a2Uint
            | F::R32Uint
            | F::Rg32Uint
            | F::Rgba32Uint,
            api::TextureClearValue::Color(api::ClearColor::Uint(_)),
        ) => true,
        (
            F::R8Sint
            | F::Rg8Sint
            | F::Rgba8Sint
            | F::R16Sint
            | F::Rg16Sint
            | F::Rgba16Sint
            | F::R32Sint
            | F::Rg32Sint
            | F::Rgba32Sint,
            api::TextureClearValue::Color(api::ClearColor::Sint(_)),
        ) => true,
        (
            F::R8Unorm
            | F::R8Snorm
            | F::Rg8Unorm
            | F::Rg8Snorm
            | F::Rgba8Unorm
            | F::Rgba8Srgb
            | F::Rgba8Snorm
            | F::Bgra8Unorm
            | F::Bgra8Srgb
            | F::R16Float
            | F::R16Unorm
            | F::R16Snorm
            | F::Rg16Float
            | F::Rg16Unorm
            | F::Rg16Snorm
            | F::Rgba16Float
            | F::Rgba16Unorm
            | F::Rgba16Snorm
            | F::Rgba32Float
            | F::Rgb9e5Ufloat
            | F::Rgb10a2Unorm
            | F::Rg11b10Ufloat
            | F::R32Float
            | F::Rg32Float,
            api::TextureClearValue::Color(api::ClearColor::Float(_)),
        ) => true,
        _ => false,
    };
    if !valid {
        return Err(Error::Unsupported("GL texture clear format or value"));
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    let result = encoder
        .state
        .with_native("clear GL texture", |provider| {
            provider.clear_texture_subresources(texture.id, mips, layers, value)
        })
        .map_err(native_error);
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    let result = encoder
        .browser
        .as_mut()
        .ok_or(Error::InvalidState)?
        .clear_texture_subresources(&texture.browser, mips, layers, value);
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    let result: Result<()> = Err(Error::Unsupported("GL platform context"));
    if result.is_ok() {
        encoder.issued = true;
    } else {
        encoder.indeterminate = true;
    }
    result
}

pub(super) fn buffer(
    encoder: &mut GlEncoder,
    destination: &GlBuffer,
    range: std::ops::Range<u64>,
    value: u32,
) -> Result<()> {
    use crate::backend::gl::api::GlBufferUsage;
    if !std::rc::Rc::ptr_eq(&encoder.state, &destination.state) {
        return Err(Error::WrongDevice);
    }
    if range.start > range.end
        || range.end > destination.desc.size
        || range.start % 4 != 0
        || range.end % 4 != 0
        || !destination
            .desc
            .usage
            .contains(GlBufferUsage::COPY_DESTINATION)
    {
        return Err(Error::InvalidRange);
    }
    if range.is_empty() {
        return Ok(());
    }
    let mut bytes = [0u8; 16 * 1024];
    for word in bytes.chunks_exact_mut(4) {
        word.copy_from_slice(&value.to_le_bytes());
    }
    let mut offset = range.start;
    while offset < range.end {
        let count = (range.end - offset).min(bytes.len() as u64) as usize;
        transfer::write_buffer(encoder, destination, offset, &bytes[..count])?;
        offset += count as u64;
    }
    Ok(())
}
