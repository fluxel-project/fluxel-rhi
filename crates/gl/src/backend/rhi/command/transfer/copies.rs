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
    if !std::rc::Rc::ptr_eq(&encoder.state, &src.state)
        || !std::rc::Rc::ptr_eq(&encoder.state, &dst.state)
    {
        return Err(Error::WrongDevice);
    }
    if !src
        .desc
        .usage
        .contains(crate::backend::api::GlBufferUsage::COPY_SOURCE)
        || !dst
            .desc
            .usage
            .contains(crate::backend::api::GlBufferUsage::COPY_DESTINATION)
    {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::{GlBufferRange, GlCopyDomainApi as _};
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
    // SAFETY: both Core textures remain borrowed throughout this call.
    let source = unsafe { src.texture.native_ref() };
    let destination = unsafe { dst.texture.native_ref() };
    if !std::rc::Rc::ptr_eq(&encoder.state, &source.state)
        || !std::rc::Rc::ptr_eq(&encoder.state, &destination.state)
    {
        return Err(Error::WrongDevice);
    }
    if !source
        .desc
        .usage
        .contains(crate::backend::api::GlTextureUsage::COPY_SOURCE)
        || !destination
            .desc
            .usage
            .contains(crate::backend::api::GlTextureUsage::COPY_DESTINATION)
    {
        return Err(Error::InvalidState);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::GlCopyDomainApi as _;
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
        let aspect = browser_blit_aspect(src.aspect, dst.aspect)?;
        if source.desc.format != destination.desc.format || src.extent != dst.extent {
            return Err(Error::InvalidRange);
        }
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = browser.blit_texture_regions(
            &source.browser,
            src.mip,
            src.origin[2],
            [src.origin[0], src.origin[1]],
            [src.extent[0], src.extent[1]],
            &destination.browser,
            dst.mip,
            dst.origin[2],
            [dst.origin[0], dst.origin[1]],
            [dst.extent[0], dst.extent[1]],
            src.extent[2],
            aspect,
            crate::backend::api::GlFilterMode::Nearest,
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
        use crate::backend::api::{GlFramebufferApi as _, GlTextureUsage};

        // SAFETY: Core keeps both texture handles borrowed for this
        // synchronous command; the descriptors and IDs are immutable.
        let source = unsafe { src.texture.native_ref() };
        // SAFETY: same synchronous borrow guarantee as for `source`.
        let destination = unsafe { dst.texture.native_ref() };
        let source_desc = source.desc;
        let destination_desc = destination.desc;
        let source_region =
            crate::backend::translate::lower_texture_region(source.id, source_desc, &src)?;
        let destination_region = crate::backend::translate::lower_texture_region(
            destination.id,
            destination_desc,
            &dst,
        )?;
        // ES 3.0/3.1 and older desktop GL require fixed and floating-point
        // framebuffer classes to match even though newer profiles relax this.
        let (separate_float_class, fixed_source_requires_fixed) = encoder
            .state
            .with_native("query blit format rules", |provider| {
                use crate::backend::api::GlFamilyApi as _;
                Ok(match provider.profile() {
                    crate::backend::api::GlFamilyProfile::Desktop { major, minor } => {
                        (major < 4 || (major == 4 && minor < 3), false)
                    }
                    crate::backend::api::GlFamilyProfile::Embedded { major, minor } => {
                        (false, major < 3 || (major == 3 && minor < 2))
                    }
                    crate::backend::api::GlFamilyProfile::WebGl2 => (false, true),
                })
            })
            .map_err(native_error)?;
        if source_desc.sample_count != 1
            || destination_desc.sample_count != 1
            || !matches!(
                source_desc.dimension,
                crate::backend::api::GlTextureDimension::D2
                    | crate::backend::api::GlTextureDimension::D2Array
                    | crate::backend::api::GlTextureDimension::D3
            )
            || !matches!(
                destination_desc.dimension,
                crate::backend::api::GlTextureDimension::D2
                    | crate::backend::api::GlTextureDimension::D2Array
                    | crate::backend::api::GlTextureDimension::D3
            )
            || !source_desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            || !destination_desc
                .usage
                .contains(GlTextureUsage::COPY_DESTINATION)
            || source_region.subresource.aspect != destination_region.subresource.aspect
            || !source_desc
                .format
                .blit_compatible(destination_desc.format, source_region.subresource.aspect)
            || (separate_float_class
                && source_region.subresource.aspect == crate::backend::api::GlTextureAspect::Color
                && source_desc.format.is_float_color() != destination_desc.format.is_float_color())
            || (fixed_source_requires_fixed
                && source_region.subresource.aspect == crate::backend::api::GlTextureAspect::Color
                && !source_desc.format.is_integer_color()
                && !source_desc.format.is_float_color()
                && destination_desc.format.is_float_color())
            || (filter == api::FilterMode::Linear
                && (source_desc.format.is_integer_color()
                    || source_region.subresource.aspect
                        != crate::backend::api::GlTextureAspect::Color))
            || source_region.extent.depth_or_layers != destination_region.extent.depth_or_layers
        {
            return Err(Error::Unsupported(
                "GL texture blit format, aspect, filter, shape or usage",
            ));
        }
        let source_layer = if source_desc.dimension == crate::backend::api::GlTextureDimension::D3 {
            source_region.origin[2]
        } else {
            source_region.subresource.base_layer
        };
        let destination_layer =
            if destination_desc.dimension == crate::backend::api::GlTextureDimension::D3 {
                destination_region.origin[2]
            } else {
                destination_region.subresource.base_layer
            };
        if source.id == destination.id
            && source_region.subresource.mip_level == destination_region.subresource.mip_level
            && source_layer < destination_layer + destination_region.extent.depth_or_layers
            && destination_layer < source_layer + source_region.extent.depth_or_layers
        {
            return Err(Error::InvalidRange);
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
        let mut source_view = crate::backend::api::GlTextureView {
            aspect: source_region.subresource.aspect,
            target: crate::backend::api::GlAttachmentTarget::Texture(source.id),
            format: source_desc.format,
            mip_level: source_region.subresource.mip_level,
            array_layer: if source_desc.dimension == crate::backend::api::GlTextureDimension::D3 {
                source_region.origin[2]
            } else {
                source_region.subresource.base_layer
            },
            layer_count: 1,
            width: source_mip.width,
            height: source_mip.height,
            sample_count: 1,
        };
        let mut destination_view = crate::backend::api::GlTextureView {
            aspect: destination_region.subresource.aspect,
            target: crate::backend::api::GlAttachmentTarget::Texture(destination.id),
            format: destination_desc.format,
            mip_level: destination_region.subresource.mip_level,
            array_layer: if destination_desc.dimension
                == crate::backend::api::GlTextureDimension::D3
            {
                destination_region.origin[2]
            } else {
                destination_region.subresource.base_layer
            },
            layer_count: 1,
            width: destination_mip.width,
            height: destination_mip.height,
            sample_count: 1,
        };
        let region = crate::backend::api::GlBlitRegion {
            src_offset: [source_region.origin[0], source_y],
            src_extent: [source_region.extent.width, source_region.extent.height],
            dst_offset: [destination_region.origin[0], destination_y],
            dst_extent: [
                destination_region.extent.width,
                destination_region.extent.height,
            ],
        };
        let filter = match filter {
            api::FilterMode::Nearest => crate::backend::api::GlFilterMode::Nearest,
            api::FilterMode::Linear => crate::backend::api::GlFilterMode::Linear,
        };
        let result = encoder
            .state
            .with_native("blit GL texture mip", |provider| {
                for _ in 0..source_region.extent.depth_or_layers {
                    let color = source_region.subresource.aspect
                        == crate::backend::api::GlTextureAspect::Color;
                    let framebuffer = |view| crate::backend::api::GlFramebufferDescriptor {
                        color_attachments: if color { vec![Some(view)] } else { vec![] },
                        depth_stencil_attachment: if color { None } else { Some(view) },
                        draw_buffers: if color { vec![0] } else { vec![] },
                    };
                    let source_framebuffer = framebuffer(source_view);
                    let destination_framebuffer = framebuffer(destination_view);
                    let source_fbo = provider.create_framebuffer(&source_framebuffer)?;
                    let destination_fbo =
                        match provider.create_framebuffer(&destination_framebuffer) {
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
                        crate::backend::api::GlBlitMask {
                            color,
                            depth: source_region.subresource.aspect
                                == crate::backend::api::GlTextureAspect::DepthOnly,
                            stencil: source_region.subresource.aspect
                                == crate::backend::api::GlTextureAspect::StencilOnly,
                        },
                    );
                    let drop_destination = provider.destroy_framebuffer(destination_fbo);
                    let drop_source = provider.destroy_framebuffer(source_fbo);
                    blit.and(drop_destination).and(drop_source)?;
                    source_view.array_layer += 1;
                    destination_view.array_layer += 1;
                }
                Ok(())
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
        let aspect = browser_blit_aspect(src.aspect, dst.aspect)?;
        if src.extent[2] != dst.extent[2] {
            return Err(Error::InvalidRange);
        }
        let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
        let result = browser.blit_texture_regions(
            &source.browser,
            src.mip,
            src.origin[2],
            [src.origin[0], src.origin[1]],
            [src.extent[0], src.extent[1]],
            &destination.browser,
            dst.mip,
            dst.origin[2],
            [dst.origin[0], dst.origin[1]],
            [dst.extent[0], dst.extent[1]],
            src.extent[2],
            aspect,
            match filter {
                api::FilterMode::Nearest => crate::backend::api::GlFilterMode::Nearest,
                api::FilterMode::Linear => crate::backend::api::GlFilterMode::Linear,
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

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
fn browser_blit_aspect(
    src: api::TextureAspects,
    dst: api::TextureAspects,
) -> Result<crate::backend::api::GlTextureAspect> {
    use crate::backend::api::GlTextureAspect;
    if src != dst {
        return Err(Error::InvalidRange);
    }
    match src.0 {
        1 => Ok(GlTextureAspect::Color),
        2 => Ok(GlTextureAspect::DepthOnly),
        4 => Ok(GlTextureAspect::StencilOnly),
        _ => Err(Error::Unsupported("WebGL2 texture blit aspect")),
    }
}
