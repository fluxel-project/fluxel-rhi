//! WebGL2 ordinary texel storage and exact client transfer encodings.
use crate::backend::api::GlFormat as F;
use fluxel_rhi_core::api::{Error, Result};
use wasm_bindgen::JsCast;
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

/// Returns the byte count selected from an RGBA8 pixel-pack read for formats
/// whose direct RED/RG `readPixels` pair is not accepted by some WebGL2
/// implementations. The encoder copies these leading channels out of a small
/// GPU scratch PBO, so no normalized value crosses the CPU boundary.
pub(super) fn widened_unorm_readback_bytes(format: F) -> Option<u8> {
    match format {
        F::R8Unorm => Some(1),
        F::Rg8Unorm => Some(2),
        _ => None,
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

/// Reads an ArrayBufferView's backing bytes without converting its elements.
fn typed_view_bytes(view: &js_sys::Object) -> Option<Vec<u8>> {
    let key = |name| wasm_bindgen::JsValue::from_str(name);
    let buffer = js_sys::Reflect::get(view, &key("buffer"))
        .ok()?
        .dyn_into::<js_sys::ArrayBuffer>()
        .ok()?;
    let offset = js_sys::Reflect::get(view, &key("byteOffset"))
        .ok()?
        .as_f64()
        .and_then(|value| u32::try_from(value as u64).ok())?;
    let length = js_sys::Reflect::get(view, &key("byteLength"))
        .ok()?
        .as_f64()
        .and_then(|value| u32::try_from(value as u64).ok())?;
    Some(js_sys::Uint8Array::new_with_byte_offset_and_length(&buffer, offset, length).to_vec())
}

/// Probe an exact ordinary-color readback with known endpoint and interior
/// values. A renderable format is insufficient: WebGL `readPixels` can require
/// a wider type or different channel layout, which would change Core bytes.
pub(super) fn exact_readback_supported(gl: &G, format: F) -> bool {
    let Some(e) = encoding(format) else {
        return false;
    };
    // Float and SNORM reads can canonicalize NaNs or collapse distinct -1
    // encodings. This probe therefore only publishes integer and UNORM pairs.
    let unorm = matches!(
        format,
        F::R8Unorm
            | F::Rg8Unorm
            | F::Rgba8Unorm
            | F::R16Unorm
            | F::Rg16Unorm
            | F::Rgba16Unorm
            | F::Rgb10a2Unorm
    );
    if !e.integer && !unorm {
        return false;
    }
    if !matches!(
        e.ty,
        G::UNSIGNED_BYTE
            | G::BYTE
            | G::UNSIGNED_SHORT
            | G::SHORT
            | G::UNSIGNED_INT
            | G::INT
            | G::UNSIGNED_INT_2_10_10_10_REV
    ) {
        return false;
    }
    let widened_bytes = widened_unorm_readback_bytes(format);
    let mut expected = vec![0; usize::from(e.bytes) * 4];
    match e.ty {
        G::UNSIGNED_BYTE | G::BYTE => {
            for (index, byte) in expected.iter_mut().enumerate() {
                *byte = [0, 255, 1, 128, 17, 239][index % 6];
            }
        }
        G::UNSIGNED_SHORT | G::SHORT => {
            for (index, chunk) in expected.chunks_exact_mut(2).enumerate() {
                chunk.copy_from_slice(
                    &[0u16, u16::MAX, 1, 0x8000, 0x11ff, 0xee00][index % 6].to_ne_bytes(),
                );
            }
        }
        G::UNSIGNED_INT | G::INT | G::UNSIGNED_INT_2_10_10_10_REV => {
            for (index, chunk) in expected.chunks_exact_mut(4).enumerate() {
                chunk.copy_from_slice(
                    &[0u32, u32::MAX, 0x8000_0001, 0x3555_55aa][index % 4].to_ne_bytes(),
                );
            }
        }
        _ => unreachable!(),
    }

    let Some(texture) = gl.create_texture() else {
        return false;
    };
    let Some(framebuffer) = gl.create_framebuffer() else {
        gl.delete_texture(Some(&texture));
        return false;
    };
    gl.bind_texture(G::TEXTURE_2D, Some(&texture));
    gl.tex_storage_2d(G::TEXTURE_2D, 1, e.internal, 2, 2);
    // The probe's two-texel rows are tightly packed even for R8/R16.
    gl.pixel_storei(G::UNPACK_ALIGNMENT, 1);
    let upload = typed_pixels(&expected, e.ty).ok();
    let uploaded = upload.is_some_and(|pixels| {
        gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
            G::TEXTURE_2D,
            0,
            0,
            0,
            2,
            2,
            e.pixel,
            e.ty,
            Some(&pixels),
        )
        .is_ok()
    });
    gl.bind_framebuffer(G::READ_FRAMEBUFFER, Some(&framebuffer));
    gl.framebuffer_texture_2d(
        G::READ_FRAMEBUFFER,
        G::COLOR_ATTACHMENT0,
        G::TEXTURE_2D,
        Some(&texture),
        0,
    );
    gl.read_buffer(G::COLOR_ATTACHMENT0);
    gl.pixel_storei(G::PACK_ALIGNMENT, 1);
    let complete = gl.check_framebuffer_status(G::READ_FRAMEBUFFER) == G::FRAMEBUFFER_COMPLETE;
    let (read_pixel, read_ty, expected_readback) = if let Some(channel_bytes) = widened_bytes {
        let mut rgba = Vec::with_capacity(4 * 4);
        for texel in expected.chunks_exact(usize::from(e.bytes)) {
            rgba.extend_from_slice(&texel[..usize::from(channel_bytes)]);
            rgba.extend_from_slice(&[0; 2][..3 - usize::from(channel_bytes)]);
            rgba.push(u8::MAX);
        }
        (G::RGBA, G::UNSIGNED_BYTE, rgba)
    } else {
        (e.pixel, e.ty, expected.clone())
    };
    let pixels = typed_pixels(&vec![0; expected_readback.len()], read_ty).ok();
    let called = uploaded
        && complete
        && pixels.as_ref().is_some_and(|pixels| {
            gl.read_pixels_with_opt_array_buffer_view(0, 0, 2, 2, read_pixel, read_ty, Some(pixels))
                .is_ok()
        });
    let exact = pixels.is_some_and(|pixels| typed_view_bytes(&pixels) == Some(expected_readback));
    let ok = called && exact && gl.get_error() == G::NO_ERROR && !gl.is_context_lost();
    gl.pixel_storei(G::PACK_ALIGNMENT, 4);
    gl.pixel_storei(G::UNPACK_ALIGNMENT, 4);
    gl.bind_framebuffer(G::READ_FRAMEBUFFER, None);
    gl.bind_texture(G::TEXTURE_2D, None);
    gl.delete_framebuffer(Some(&framebuffer));
    gl.delete_texture(Some(&texture));
    let _ = gl.get_error();
    ok
}
