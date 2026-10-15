//! Immediate buffer/texture uploads and readback copies.
use super::*;
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    feature = "native-gles-egl",
    all(target_arch = "wasm32", feature = "webgl2")
))]
use std::rc::Rc;

pub(crate) fn buffer_to_texture(
    _: &mut GlEncoder,
    _: &GlBuffer,
    _: TexelCopyBufferLayout,
    _: TextureRegion<'_, Gl>,
) -> Result<()> {
    Err(Error::Unsupported("GL buffer-to-texture copy"))
}

pub(crate) fn texture_to_buffer(
    encoder: &mut GlEncoder,
    src: TextureRegion<'_, Gl>,
    dst: &GlBuffer,
    layout: TexelCopyBufferLayout,
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{
            GlBufferRange, GlBufferUsage, GlCopyDomainApi as _, GlTextureUsage,
        };

        // SAFETY: Core keeps the texture region borrowed for this direct
        // operation; the native reference contributes immutable allocation
        // metadata and is never retained beyond the current GL call.
        let source = unsafe { src.texture.native_ref() };
        if !Rc::ptr_eq(&encoder.state, &source.state)
            || !Rc::ptr_eq(&encoder.state, &dst.state)
            || !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !dst.desc.usage.contains(GlBufferUsage::COPY_DESTINATION)
        {
            return Err(Error::InvalidRange);
        }
        let source_region = lower_region(src)?;
        let pixel_layout =
            crate::backend::gl::translate::lower_texel_copy_layout(layout, source.desc.format)?;
        let required = pixel_layout
            .required_bytes(source_region)
            .map_err(|_| Error::InvalidRange)?;
        let write_size = required
            .checked_sub(layout.offset)
            .ok_or(Error::InvalidRange)?;
        let write_end = layout
            .offset
            .checked_add(write_size)
            .ok_or(Error::InvalidRange)?;
        if write_end > dst.desc.size {
            return Err(Error::InvalidRange);
        }

        let result = encoder
            .state
            .with_native("read GL texture into buffer", |provider| {
                // `read_texture` performs the historical GL readPixels
                // operation synchronously; upload the returned top-left
                // bytes before this command call returns instead of
                // retaining a CPU copy operation for later replay.
                let readback = provider.read_texture(source_region, pixel_layout)?;
                let start = usize::try_from(layout.offset).map_err(|_| {
                    crate::backend::gl::api::GlError::OutOfMemory {
                        operation: "read-texture-to-buffer",
                    }
                })?;
                let end = usize::try_from(required).map_err(|_| {
                    crate::backend::gl::api::GlError::OutOfMemory {
                        operation: "read-texture-to-buffer",
                    }
                })?;
                let bytes = readback.bytes.get(start..end).ok_or_else(|| {
                    crate::backend::gl::api::GlError::Validation {
                        operation: "read-texture-to-buffer",
                        message: "native readback returned a short pixel buffer".into(),
                    }
                })?;
                provider.upload_buffer(
                    GlBufferRange {
                        buffer: dst.id,
                        offset: layout.offset,
                        size: write_size,
                    },
                    bytes,
                )
            })
            .map_err(native_error);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            // The pixel read or destination upload may have reached GL
            // before a later native error was observed.
            encoder.indeterminate = true;
        }
        return result;
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        // SAFETY: Core retains `src.texture` for the complete synchronous
        // immediate verb. The browser wrapper is borrowed only while
        // `readPixels` and the destination `bufferSubData` execute.
        let source = unsafe { src.texture.native_ref() };
        use crate::backend::gl::api::{
            GlBufferUsage, GlFormat, GlTextureDimension, GlTextureUsage,
        };
        if !Rc::ptr_eq(&encoder.state, &source.state)
            || !Rc::ptr_eq(&encoder.state, &dst.state)
            || !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !dst.desc.usage.contains(GlBufferUsage::COPY_DESTINATION)
            || !matches!(
                source.desc.format,
                GlFormat::Rgba8Unorm | GlFormat::Rgba8Srgb
            )
            || source.desc.dimension != GlTextureDimension::D2
            || src.aspect != api::TextureAspects::COLOR
        {
            return Err(Error::Unsupported(
                "WebGL2 texture-to-buffer requires an RGBA8 2D color rectangle",
            ));
        }
        if src.origin[2] != 0 || src.extent[2] != 1 {
            return Err(Error::InvalidRange);
        }
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = browser.copy_rgba8_texture_to_buffer(
            &source.browser,
            src.mip,
            [src.origin[0], src.origin[1]],
            [src.extent[0], src.extent[1]],
            &dst.browser,
            layout.offset,
            layout.bytes_per_row,
            layout.rows_per_image,
        );
        if result.is_ok() {
            encoder.issued = true;
        } else {
            // readPixels may have reached the context before a following
            // buffer upload or error observation failed.
            encoder.indeterminate = true;
        }
        return result;
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (encoder, src, dst, layout);
        Err(Error::Unsupported("GL texture-to-buffer platform"))
    }
}

pub(crate) fn write_buffer(
    encoder: &mut GlEncoder,
    dst: &GlBuffer,
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
        let size = u64::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?;
        let result = encoder
            .state
            .with_native("upload GL buffer", |provider| {
                provider.upload_buffer(
                    GlBufferRange {
                        buffer: dst.id,
                        offset,
                        size,
                    },
                    bytes,
                )
            })
            .map_err(native_error);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            encoder.indeterminate = true;
        }
        return result;
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = browser.write_browser_buffer(&dst.browser, offset, bytes);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            // A browser error can arrive after BufferSubData reached GL.
            encoder.indeterminate = true;
        }
        result
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    Err(Error::Unsupported("GL platform context"))
}

pub(crate) fn write_texture(
    encoder: &mut GlEncoder,
    dst: TextureRegion<'_, Gl>,
    layout: TexelCopyBufferLayout,
    bytes: &[u8],
) -> Result<()> {
    // SAFETY: Core holds the texture borrow for this synchronous command;
    // its descriptor is immutable allocation metadata used only to choose
    // the validated native pixel-layout lowering.
    let texture = unsafe { dst.texture.native_ref() };
    let layout =
        crate::backend::gl::translate::lower_texel_copy_layout(layout, texture.desc.format)?;
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        let destination = lower_region(dst)?;
        use crate::backend::gl::api::GlCopyDomainApi as _;
        let result = encoder
            .state
            .with_native("upload GL texture", |provider| {
                provider.upload_texture(destination, layout, bytes)
            })
            .map_err(native_error);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            encoder.indeterminate = true;
        }
        return result;
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = browser.write_texture_region(
                &texture.browser,
                dst.mip,
                dst.origin,
                dst.extent,
                layout.bytes_per_row,
                layout.rows_per_image,
                bytes,
            );
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            return result;
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (encoder, dst, layout, bytes);
            Err(Error::Unsupported("GL platform context"))
        }
    }
}
