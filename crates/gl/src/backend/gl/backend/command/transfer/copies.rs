//! Direct buffer copies, texture copies, and image blits.
use super::*;

pub(crate) fn copy_buffer(
    encoder: &mut GlEncoder,
    src: &GlBuffer,
    src_offset: u64,
    dst: &GlBuffer,
    dst_offset: u64,
    bytes: u64,
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlBufferRange, GlCopyDomainApi as _};
        let result = encoder
            .state
            .with_native("copy GL buffer", |provider| {
                provider.copy_buffer_range(
                    GlBufferRange {
                        buffer: src.id,
                        offset: src_offset,
                        size: bytes,
                    },
                    GlBufferRange {
                        buffer: dst.id,
                        offset: dst_offset,
                        size: bytes,
                    },
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
        let result =
            browser.copy_browser_buffer(&src.browser, src_offset, &dst.browser, dst_offset, bytes);
        if result.is_ok() {
            encoder.issued = true;
        } else {
            // The browser may report an error after it accepted a GL call.
            // Core must never classify that prefix as a clean rejection.
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

pub(crate) fn copy_texture(
    encoder: &mut GlEncoder,
    src: TextureRegion<'_, Gl>,
    dst: TextureRegion<'_, Gl>,
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlCopyDomainApi as _;
        let src = lower_region(src)?;
        let dst = lower_region(dst)?;
        let result = encoder
            .state
            .with_native("copy GL texture", |provider| {
                provider.copy_texture_region(src, dst)
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
        // WebGL2 has no texture-copy entry point. Its owner implements this
        // exact-size copy with short-lived FBOs and a nearest framebuffer
        // blit. Keep the operation inside this synchronous encoder call.
        // SAFETY: Core retains both Texture borrows for the duration of
        // this function; each browser wrapper is only inspected/used now.
        let source = unsafe { src.texture.native_ref() };
        // SAFETY: same Core borrow guarantee as for `source`.
        let destination = unsafe { dst.texture.native_ref() };
        if src.aspect != api::TextureAspects::COLOR || dst.aspect != api::TextureAspects::COLOR {
            return Err(Error::Unsupported(
                "WebGL2 texture copy supports color aspects only",
            ));
        }
        if src.origin[2] != 0
            || dst.origin[2] != 0
            || src.extent[2] != 1
            || dst.extent[2] != 1
            || src.extent != dst.extent
        {
            return Err(Error::InvalidRange);
        }
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = browser.blit_texture_regions(
            &source.browser,
            src.mip,
            [src.origin[0], src.origin[1]],
            [src.extent[0], src.extent[1]],
            &destination.browser,
            dst.mip,
            [dst.origin[0], dst.origin[1]],
            [dst.extent[0], dst.extent[1]],
            crate::backend::gl::api::GlFilterMode::Nearest,
        );
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
    Err(Error::Unsupported("GL platform context"))
}

pub(crate) fn blit(
    encoder: &mut GlEncoder,
    src: TextureRegion<'_, Gl>,
    dst: TextureRegion<'_, Gl>,
    filter: api::FilterMode,
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlFramebufferApi as _, GlTextureUsage};

        // SAFETY: Core keeps both texture handles borrowed for this
        // synchronous command; the descriptors and IDs are immutable.
        let source = unsafe { src.texture.native_ref() };
        // SAFETY: same synchronous borrow guarantee as for `source`.
        let destination = unsafe { dst.texture.native_ref() };
        let source_desc = source.desc;
        let destination_desc = destination.desc;
        let source_region =
            crate::backend::gl::translate::lower_texture_region(source.id, source_desc, &src)?;
        let destination_region = crate::backend::gl::translate::lower_texture_region(
            destination.id,
            destination_desc,
            &dst,
        )?;
        if source_desc.format != destination_desc.format
            || source_desc.sample_count != 1
            || destination_desc.sample_count != 1
            || source_desc.dimension != crate::backend::gl::api::GlTextureDimension::D2
            || destination_desc.dimension != crate::backend::gl::api::GlTextureDimension::D2
            || !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !destination_desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
            || source_region.subresource.aspect != crate::backend::gl::api::GlTextureAspect::Color
            || destination_region.subresource.aspect
                != crate::backend::gl::api::GlTextureAspect::Color
            || source_region.subresource.base_layer != 0
            || destination_region.subresource.base_layer != 0
            || source_region.subresource.layer_count != 1
            || destination_region.subresource.layer_count != 1
            || source_region.extent.depth_or_layers != 1
            || destination_region.extent.depth_or_layers != 1
        {
            return Err(Error::Unsupported("GL color texture blit shape or usage"));
        }
        let source_mip = source_desc
            .mip_extent(source_region.subresource.mip_level)
            .ok_or(Error::InvalidRange)?;
        let destination_mip = destination_desc
            .mip_extent(destination_region.subresource.mip_level)
            .ok_or(Error::InvalidRange)?;
        // Core rectangles use top-left coordinates. The two mip heights
        // differ, so convert each endpoint with its own allocation facts.
        let source_y = source_mip
            .height
            .checked_sub(
                source_region.origin[1]
                    .checked_add(source_region.extent.height)
                    .ok_or(Error::InvalidRange)?,
            )
            .ok_or(Error::InvalidRange)?;
        let destination_y = destination_mip
            .height
            .checked_sub(
                destination_region.origin[1]
                    .checked_add(destination_region.extent.height)
                    .ok_or(Error::InvalidRange)?,
            )
            .ok_or(Error::InvalidRange)?;
        let source_view = crate::backend::gl::api::GlTextureView {
            target: crate::backend::gl::api::GlAttachmentTarget::Texture(source.id),
            format: source_desc.format,
            mip_level: source_region.subresource.mip_level,
            array_layer: 0,
            layer_count: 1,
            width: source_mip.width,
            height: source_mip.height,
            sample_count: 1,
        };
        let destination_view = crate::backend::gl::api::GlTextureView {
            target: crate::backend::gl::api::GlAttachmentTarget::Texture(destination.id),
            format: destination_desc.format,
            mip_level: destination_region.subresource.mip_level,
            array_layer: 0,
            layer_count: 1,
            width: destination_mip.width,
            height: destination_mip.height,
            sample_count: 1,
        };
        let source_framebuffer = crate::backend::gl::api::GlFramebufferDescriptor {
            color_attachments: vec![source_view],
            depth_stencil_attachment: None,
            draw_buffers: vec![0],
        };
        let destination_framebuffer = crate::backend::gl::api::GlFramebufferDescriptor {
            color_attachments: vec![destination_view],
            depth_stencil_attachment: None,
            draw_buffers: vec![0],
        };
        let region = crate::backend::gl::api::GlBlitRegion {
            src_offset: [source_region.origin[0], source_y],
            src_extent: [source_region.extent.width, source_region.extent.height],
            dst_offset: [destination_region.origin[0], destination_y],
            dst_extent: [
                destination_region.extent.width,
                destination_region.extent.height,
            ],
        };
        let filter = match filter {
            api::FilterMode::Nearest => crate::backend::gl::api::GlFilterMode::Nearest,
            api::FilterMode::Linear => crate::backend::gl::api::GlFilterMode::Linear,
        };
        let result = encoder
            .state
            .with_native("blit GL texture mip", |provider| {
                let source_fbo = provider.create_framebuffer(&source_framebuffer)?;
                let destination_fbo = match provider.create_framebuffer(&destination_framebuffer) {
                    Ok(framebuffer) => framebuffer,
                    Err(error) => {
                        let _ = provider.destroy_framebuffer(source_fbo);
                        return Err(error);
                    }
                };
                let blit = provider.blit_framebuffer(
                    source_fbo,
                    destination_fbo,
                    region,
                    filter,
                    crate::backend::gl::api::GlBlitMask {
                        color: true,
                        depth: false,
                        stencil: false,
                    },
                );
                let drop_destination = provider.destroy_framebuffer(destination_fbo);
                let drop_source = provider.destroy_framebuffer(source_fbo);
                blit.and(drop_destination).and(drop_source)
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
        // SAFETY: Core holds both texture borrows through this synchronous
        // blit. The browser wrappers are used immediately and never retained.
        let source = unsafe { src.texture.native_ref() };
        let destination = unsafe { dst.texture.native_ref() };
        if src.aspect != api::TextureAspects::COLOR || dst.aspect != api::TextureAspects::COLOR {
            return Err(Error::Unsupported(
                "WebGL2 texture blit supports color aspects only",
            ));
        }
        if src.origin[2] != 0 || dst.origin[2] != 0 || src.extent[2] != 1 || dst.extent[2] != 1 {
            return Err(Error::InvalidRange);
        }
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = browser.blit_texture_regions(
            &source.browser,
            src.mip,
            [src.origin[0], src.origin[1]],
            [src.extent[0], src.extent[1]],
            &destination.browser,
            dst.mip,
            [dst.origin[0], dst.origin[1]],
            [dst.extent[0], dst.extent[1]],
            match filter {
                api::FilterMode::Nearest => crate::backend::gl::api::GlFilterMode::Nearest,
                api::FilterMode::Linear => crate::backend::gl::api::GlFilterMode::Linear,
            },
        );
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
    {
        let _ = (encoder, src, dst, filter);
        Err(Error::Unsupported("GL texture blit platform"))
    }
}
