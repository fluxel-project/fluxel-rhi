//! Immediate buffer/texture uploads and readback copies.
use super::*;
use crate::backend::gl::api::{GlFormat, GlTextureAspect};

pub(crate) fn buffer_to_texture(
    encoder: &mut GlEncoder,
    source: &GlBuffer,
    layout: TexelCopyBufferLayout,
    destination: TextureRegion<'_, Gl>,
) -> Result<()> {
    // SAFETY: Core holds the texture borrow for this synchronous transfer.
    let texture = unsafe { destination.texture.native_ref() };
    let depth_upload = destination.aspect == api::TextureAspects::DEPTH
        && matches!(
            texture.desc.format,
            GlFormat::Depth16Unorm | GlFormat::Depth32Float
        );
    let color_upload = destination.aspect == api::TextureAspects::COLOR
        && (texture.desc.format.compressed_info().is_some()
            || texture
                .desc
                .format
                .supports_blit_aspect(GlTextureAspect::Color));
    if !(color_upload || depth_upload) {
        return Err(Error::Unsupported(
            "GL buffer-to-texture supports only Depth16/Depth32Float depth uploads",
        ));
    }
    if !std::rc::Rc::ptr_eq(&encoder.state, &source.state)
        || !std::rc::Rc::ptr_eq(&encoder.state, &texture.state)
    {
        return Err(Error::WrongDevice);
    }
    if !source
        .desc
        .usage
        .contains(crate::backend::gl::api::GlBufferUsage::COPY_SOURCE)
        || !texture
            .desc
            .usage
            .contains(crate::backend::gl::api::GlTextureUsage::COPY_DESTINATION)
    {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
        let destination = lower_region(destination)?;
        let layout =
            crate::backend::gl::translate::lower_texel_copy_layout(layout, texture.desc.format)?;
        let required = layout
            .required_bytes(destination)
            .map_err(|_| Error::InvalidRange)?;
        if required > source.desc.size {
            return Err(Error::InvalidRange);
        }
        let result = encoder
            .state
            .with_native("copy GL buffer to texture", |provider| {
                provider.buffer_to_texture(
                    GlBufferRange {
                        buffer: source.id,
                        offset: 0,
                        size: required,
                    },
                    destination,
                    layout,
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
        let result = if texture.desc.format.compressed_info().is_some() {
            browser.copy_compressed_buffer_to_texture(
                &source.browser,
                layout.offset,
                layout.bytes_per_row,
                layout.rows_per_image,
                &texture.browser,
                destination.mip,
                destination.origin,
                destination.extent,
            )
        } else {
            browser.copy_buffer_to_texture(
                &source.browser,
                layout.offset,
                layout.bytes_per_row,
                layout.rows_per_image,
                &texture.browser,
                destination.mip,
                destination.origin,
                destination.extent,
            )
        };
        if result.is_ok() {
            encoder.issued = true;
        } else {
            encoder.indeterminate = true;
        }
        return result;
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    Err(Error::Unsupported("GL buffer-to-texture platform"))
}

pub(crate) fn texture_to_buffer(
    encoder: &mut GlEncoder,
    src: TextureRegion<'_, Gl>,
    dst: &GlBuffer,
    layout: TexelCopyBufferLayout,
) -> Result<()> {
    // Desktop GL exposes depth readback through a depth-only read framebuffer.
    // Keep native depth readback format-specific: Depth16Unorm uses unsigned
    // short and Depth32Float uses float. Stencil packing differs by API.
    let source = unsafe { src.texture.native_ref() };
    let desktop_depth = src.aspect == api::TextureAspects::DEPTH
        && matches!(
            source.desc.format,
            crate::backend::gl::api::GlFormat::Depth16Unorm
                | crate::backend::gl::api::GlFormat::Depth32Float
        );
    if src.aspect != api::TextureAspects::COLOR && !desktop_depth {
        return Err(Error::Unsupported(
            "GL depth/stencil texture-to-buffer transfer",
        ));
    }
    // SAFETY: the region borrows its texture for this immediate operation.
    let source = unsafe { src.texture.native_ref() };
    if !std::rc::Rc::ptr_eq(&encoder.state, &source.state)
        || !std::rc::Rc::ptr_eq(&encoder.state, &dst.state)
    {
        return Err(Error::WrongDevice);
    }
    if !source
        .desc
        .usage
        .contains(crate::backend::gl::api::GlTextureUsage::COPY_SOURCE)
        || !dst
            .desc
            .usage
            .contains(crate::backend::gl::api::GlBufferUsage::COPY_DESTINATION)
    {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};

        // SAFETY: Core keeps the texture region borrowed for this direct
        // operation; the native reference contributes immutable allocation
        // metadata and is never retained beyond the current GL call.
        let source = unsafe { src.texture.native_ref() };
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
                provider.copy_texture_to_buffer(
                    source_region,
                    GlBufferRange {
                        buffer: dst.id,
                        offset: 0,
                        size: required,
                    },
                    pixel_layout,
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
        use crate::backend::gl::api::GlTextureDimension;
        if src.aspect != api::TextureAspects::COLOR
            || !matches!(
                source.desc.dimension,
                GlTextureDimension::D2 | GlTextureDimension::D2Array | GlTextureDimension::D3
            )
            || src.aspect != api::TextureAspects::COLOR
        {
            return Err(Error::Unsupported(
                "WebGL2 texture-to-buffer requires a supported color region",
            ));
        }
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = browser.copy_texture_to_buffer(
            &source.browser,
            src.mip,
            src.origin,
            src.extent,
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
    if !std::rc::Rc::ptr_eq(&encoder.state, &dst.state) {
        return Err(Error::WrongDevice);
    }
    if !dst
        .desc
        .usage
        .contains(crate::backend::gl::api::GlBufferUsage::COPY_DESTINATION)
    {
        return Err(Error::InvalidState);
    }
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
    if dst.aspect != api::TextureAspects::COLOR {
        return Err(Error::Unsupported("GL depth/stencil texture byte upload"));
    }
    // SAFETY: Core holds the texture borrow for this synchronous command;
    // its descriptor is immutable allocation metadata used only to choose
    // the validated native pixel-layout lowering.
    let texture = unsafe { dst.texture.native_ref() };
    if !std::rc::Rc::ptr_eq(&encoder.state, &texture.state) {
        return Err(Error::WrongDevice);
    }
    if !texture
        .desc
        .usage
        .contains(crate::backend::gl::api::GlTextureUsage::COPY_DESTINATION)
    {
        return Err(Error::InvalidState);
    }
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
            let result = if texture.desc.format.compressed_info().is_some() {
                browser.write_compressed_texture(
                    &texture.browser,
                    dst.mip,
                    dst.origin,
                    dst.extent,
                    layout.offset,
                    layout.bytes_per_row,
                    layout.rows_per_image,
                    bytes,
                )
            } else {
                let start = usize::try_from(layout.offset).map_err(|_| Error::InvalidRange)?;
                let payload = bytes.get(start..).ok_or(Error::InvalidRange)?;
                browser.write_texture_region(
                    &texture.browser,
                    dst.mip,
                    dst.origin,
                    dst.extent,
                    layout.bytes_per_row,
                    layout.rows_per_image,
                    payload,
                )
            };
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
