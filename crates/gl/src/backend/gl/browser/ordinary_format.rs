//! WebGL2 ordinary texel storage and exact client transfer encodings.
use crate::backend::gl::api::GlFormat as F;
use fluxel_rhi_core::api::{Error, Result};
use web_sys::WebGl2RenderingContext as G;

#[derive(Clone, Copy)]
pub(super) struct Encoding {
    pub internal: u32,
    pub pixel: u32,
    pub ty: u32,
    pub bytes: u8,
    pub integer: bool,
    pub float32: bool,
}

pub(super) fn encoding(f: F) -> Option<Encoding> {
    let (internal, pixel, ty, bytes, integer, float32) = match f {
        F::R8Unorm => (G::R8, G::RED, G::UNSIGNED_BYTE, 1, false, false),
        F::R8Snorm => (G::R8_SNORM, G::RED, G::BYTE, 1, false, false),
        F::R8Uint => (G::R8UI, G::RED_INTEGER, G::UNSIGNED_BYTE, 1, true, false),
        F::R8Sint => (G::R8I, G::RED_INTEGER, G::BYTE, 1, true, false),
        F::Rg8Unorm => (G::RG8, G::RG, G::UNSIGNED_BYTE, 2, false, false),
        F::Rg8Snorm => (G::RG8_SNORM, G::RG, G::BYTE, 2, false, false),
        F::Rg8Uint => (G::RG8UI, G::RG_INTEGER, G::UNSIGNED_BYTE, 2, true, false),
        F::Rg8Sint => (G::RG8I, G::RG_INTEGER, G::BYTE, 2, true, false),
        F::Rgba8Unorm => (G::RGBA8, G::RGBA, G::UNSIGNED_BYTE, 4, false, false),
        F::Rgba8Srgb => (G::SRGB8_ALPHA8, G::RGBA, G::UNSIGNED_BYTE, 4, false, false),
        F::Rgba8Snorm => (G::RGBA8_SNORM, G::RGBA, G::BYTE, 4, false, false),
        F::Rgba8Uint => (
            G::RGBA8UI,
            G::RGBA_INTEGER,
            G::UNSIGNED_BYTE,
            4,
            true,
            false,
        ),
        F::Rgba8Sint => (G::RGBA8I, G::RGBA_INTEGER, G::BYTE, 4, true, false),
        F::R16Unorm => (0x822A, G::RED, G::UNSIGNED_SHORT, 2, false, false),
        F::R16Snorm => (0x8F98, G::RED, G::SHORT, 2, false, false),
        F::Rg16Unorm => (0x822C, G::RG, G::UNSIGNED_SHORT, 4, false, false),
        F::Rg16Snorm => (0x8F99, G::RG, G::SHORT, 4, false, false),
        F::Rgba16Unorm => (0x805B, G::RGBA, G::UNSIGNED_SHORT, 8, false, false),
        F::Rgba16Snorm => (0x8F9B, G::RGBA, G::SHORT, 8, false, false),
        F::R16Uint => (G::R16UI, G::RED_INTEGER, G::UNSIGNED_SHORT, 2, true, false),
        F::R16Sint => (G::R16I, G::RED_INTEGER, G::SHORT, 2, true, false),
        F::R16Float => (G::R16F, G::RED, G::HALF_FLOAT, 2, false, false),
        F::Rg16Uint => (G::RG16UI, G::RG_INTEGER, G::UNSIGNED_SHORT, 4, true, false),
        F::Rg16Sint => (G::RG16I, G::RG_INTEGER, G::SHORT, 4, true, false),
        F::Rg16Float => (G::RG16F, G::RG, G::HALF_FLOAT, 4, false, false),
        F::Rgba16Uint => (
            G::RGBA16UI,
            G::RGBA_INTEGER,
            G::UNSIGNED_SHORT,
            8,
            true,
            false,
        ),
        F::Rgba16Sint => (G::RGBA16I, G::RGBA_INTEGER, G::SHORT, 8, true, false),
        F::Rgba16Float => (G::RGBA16F, G::RGBA, G::HALF_FLOAT, 8, false, false),
        F::R32Uint => (G::R32UI, G::RED_INTEGER, G::UNSIGNED_INT, 4, true, false),
        F::R32Sint => (G::R32I, G::RED_INTEGER, G::INT, 4, true, false),
        F::R32Float => (G::R32F, G::RED, G::FLOAT, 4, false, true),
        F::Rg32Uint => (G::RG32UI, G::RG_INTEGER, G::UNSIGNED_INT, 8, true, false),
        F::Rg32Sint => (G::RG32I, G::RG_INTEGER, G::INT, 8, true, false),
        F::Rg32Float => (G::RG32F, G::RG, G::FLOAT, 8, false, true),
        F::Rgba32Uint => (
            G::RGBA32UI,
            G::RGBA_INTEGER,
            G::UNSIGNED_INT,
            16,
            true,
            false,
        ),
        F::Rgba32Sint => (G::RGBA32I, G::RGBA_INTEGER, G::INT, 16, true, false),
        F::Rgba32Float => (G::RGBA32F, G::RGBA, G::FLOAT, 16, false, true),
        F::Rgb9e5Ufloat => (
            G::RGB9_E5,
            G::RGB,
            G::UNSIGNED_INT_5_9_9_9_REV,
            4,
            false,
            false,
        ),
        F::Rgb10a2Uint => (
            G::RGB10_A2UI,
            G::RGBA_INTEGER,
            G::UNSIGNED_INT_2_10_10_10_REV,
            4,
            true,
            false,
        ),
        F::Rgb10a2Unorm => (
            G::RGB10_A2,
            G::RGBA,
            G::UNSIGNED_INT_2_10_10_10_REV,
            4,
            false,
            false,
        ),
        F::Rg11b10Ufloat => (
            G::R11F_G11F_B10F,
            G::RGB,
            G::UNSIGNED_INT_10F_11F_11F_REV,
            4,
            false,
            false,
        ),
        _ => return None,
    };
    Some(Encoding {
        internal,
        pixel,
        ty,
        bytes,
        integer,
        float32,
    })
}

pub(super) fn type_bytes(ty: u32) -> usize {
    match ty {
        G::UNSIGNED_BYTE | G::BYTE => 1,
        G::UNSIGNED_SHORT | G::SHORT | G::HALF_FLOAT => 2,
        _ => 4,
    }
}

pub(super) fn typed_pixels(bytes: &[u8], ty: u32) -> Result<js_sys::Object> {
    if bytes.len() % type_bytes(ty) != 0 {
        return Err(Error::InvalidRange);
    }
    let buffer = js_sys::Uint8Array::from(bytes).buffer();
    Ok(match ty {
        G::UNSIGNED_BYTE => js_sys::Uint8Array::new(&buffer).into(),
        G::BYTE => js_sys::Int8Array::new(&buffer).into(),
        G::UNSIGNED_SHORT | G::HALF_FLOAT => js_sys::Uint16Array::new(&buffer).into(),
        G::SHORT => js_sys::Int16Array::new(&buffer).into(),
        G::INT => js_sys::Int32Array::new(&buffer).into(),
        G::FLOAT => js_sys::Float32Array::new(&buffer).into(),
        _ => js_sys::Uint32Array::new(&buffer).into(),
    })
}

/// Probe the exact source format and type into a scratch pixel-pack buffer.
/// A renderable format is insufficient: WebGL readPixels can require a wider
/// type or different channel layout, which would change Core's raw bytes.
pub(super) fn exact_readback_supported(gl: &G, format: F) -> bool {
    let Some(e) = encoding(format) else {
        return false;
    };
    // Integer transfer preserves the represented bits. Float and SNORM
    // readPixels conversions may canonicalize NaNs or collapse distinct
    // encodings of -1, so an accepted pair does not prove a raw byte copy.
    if !e.integer {
        return false;
    }

    let Some(texture) = gl.create_texture() else {
        return false;
    };
    let Some(framebuffer) = gl.create_framebuffer() else {
        gl.delete_texture(Some(&texture));
        return false;
    };
    let Some(buffer) = gl.create_buffer() else {
        gl.delete_framebuffer(Some(&framebuffer));
        gl.delete_texture(Some(&texture));
        return false;
    };
    gl.bind_texture(G::TEXTURE_2D, Some(&texture));
    gl.tex_storage_2d(G::TEXTURE_2D, 1, e.internal, 1, 1);
    gl.bind_framebuffer(G::READ_FRAMEBUFFER, Some(&framebuffer));
    gl.framebuffer_texture_2d(
        G::READ_FRAMEBUFFER,
        G::COLOR_ATTACHMENT0,
        G::TEXTURE_2D,
        Some(&texture),
        0,
    );
    gl.read_buffer(G::COLOR_ATTACHMENT0);
    gl.bind_buffer(G::PIXEL_PACK_BUFFER, Some(&buffer));
    gl.buffer_data_with_i32(G::PIXEL_PACK_BUFFER, e.bytes as i32, G::STREAM_READ);
    gl.pixel_storei(G::PACK_ALIGNMENT, 1);
    let complete = gl.check_framebuffer_status(G::READ_FRAMEBUFFER) == G::FRAMEBUFFER_COMPLETE;
    let called = complete
        && gl
            .read_pixels_with_f64(0, 0, 1, 1, e.pixel, e.ty, 0.0)
            .is_ok();
    let ok = called && gl.get_error() == G::NO_ERROR && !gl.is_context_lost();
    gl.pixel_storei(G::PACK_ALIGNMENT, 4);
    gl.bind_buffer(G::PIXEL_PACK_BUFFER, None);
    gl.bind_framebuffer(G::READ_FRAMEBUFFER, None);
    gl.bind_texture(G::TEXTURE_2D, None);
    gl.delete_buffer(Some(&buffer));
    gl.delete_framebuffer(Some(&framebuffer));
    gl.delete_texture(Some(&texture));
    let _ = gl.get_error();
    ok
}
