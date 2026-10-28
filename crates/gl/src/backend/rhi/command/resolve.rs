//! Independent multisample resolves issued immediately through temporary FBOs.
//!
//! This is deliberately separate from raster-pass attachment resolves: Core's
//! `CommandEncoder::resolve` is a standalone transfer and must not depend on a
//! pass being open.  The FBOs exist only for the duration of the native call;
//! the encoder retains no command payload or replay state.

use super::*;

pub(super) fn texture(
    encoder: &mut GlEncoder,
    src: TextureRegion<'_, Gl>,
    dst: TextureRegion<'_, Gl>,
) -> Result<()> {
    // SAFETY: Core retains both textures throughout this synchronous command.
    // Their allocation identity and descriptor are immutable while borrowed.
    let source = unsafe { src.texture.native_ref() };
    // SAFETY: same Core borrow guarantee as `source`.
    let destination = unsafe { dst.texture.native_ref() };
    if !std::rc::Rc::ptr_eq(&encoder.state, &source.state)
        || !std::rc::Rc::ptr_eq(&encoder.state, &destination.state)
    {
        return Err(Error::WrongDevice);
    }

    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        return native(encoder, source, destination, src, dst);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        // BrowserOwner uses a multisample renderbuffer rather than a texture.
        // Its dedicated resolve helper is kept at that owner boundary, where
        // the JS handles and WebGL state are private.
        return browser(encoder, source, destination, src, dst);
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (encoder, source, destination, src, dst);
        Err(Error::Unsupported("GL resolve platform"))
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn native(
    encoder: &mut GlEncoder,
    source: &GlTexture,
    destination: &GlTexture,
    src: TextureRegion<'_, Gl>,
    dst: TextureRegion<'_, Gl>,
) -> Result<()> {
    use crate::backend::api::{
        GlAttachmentTarget, GlBlitMask, GlBlitRegion, GlExtent3d, GlFamilyApi as _,
        GlFamilyProfile, GlFilterMode, GlFramebufferApi as _, GlFramebufferDescriptor,
        GlResourceApi as _, GlTextureAspect, GlTextureDesc, GlTextureDimension, GlTextureUsage,
        GlTextureView,
    };

    let source_region =
        crate::backend::translate::lower_texture_region(source.id, source.desc, &src)?;
    let destination_region =
        crate::backend::translate::lower_texture_region(destination.id, destination.desc, &dst)?;
    // GL's multisample blit is only the Core color resolve contract.  Keep
    // depth/stencil out rather than promising backend-specific resolve rules.
    if source.desc.sample_count <= 1
        || destination.desc.sample_count != 1
        || source.desc.format != destination.desc.format
        || !matches!(
            source.desc.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        )
        || !matches!(
            destination.desc.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array
        )
        || source_region.subresource.aspect != GlTextureAspect::Color
        || destination_region.subresource.aspect != GlTextureAspect::Color
        || source_region.extent != destination_region.extent
        || !source.desc.usage.contains(GlTextureUsage::COPY_SOURCE)
        || !destination
            .desc
            .usage
            .contains(GlTextureUsage::COPY_DESTINATION)
        || source.id == destination.id
    {
        return Err(Error::InvalidRange);
    }

    let source_mip = source
        .desc
        .mip_extent(source_region.subresource.mip_level)
        .ok_or(Error::InvalidRange)?;
    let destination_mip = destination
        .desc
        .mip_extent(destination_region.subresource.mip_level)
        .ok_or(Error::InvalidRange)?;
    // Offscreen GL storage uses Core's logical row order directly.  Surface
    // publication performs its separate window-axis conversion; applying it
    // here would invert an ordinary texture-to-texture resolve.
    let mut source_view = GlTextureView {
        aspect: GlTextureAspect::Color,
        target: GlAttachmentTarget::Texture(source.id),
        format: source.desc.format,
        mip_level: source_region.subresource.mip_level,
        array_layer: source_region.subresource.base_layer,
        layer_count: 1,
        width: source_mip.width,
        height: source_mip.height,
        sample_count: source.desc.sample_count,
    };
    let mut destination_view = GlTextureView {
        aspect: GlTextureAspect::Color,
        target: GlAttachmentTarget::Texture(destination.id),
        format: destination.desc.format,
        mip_level: destination_region.subresource.mip_level,
        array_layer: destination_region.subresource.base_layer,
        layer_count: 1,
        width: destination_mip.width,
        height: destination_mip.height,
        sample_count: 1,
    };
    let region = GlBlitRegion {
        src_offset: [source_region.origin[0], source_region.origin[1]],
        src_extent: [source_region.extent.width, source_region.extent.height],
        dst_offset: [destination_region.origin[0], destination_region.origin[1]],
        dst_extent: [
            destination_region.extent.width,
            destination_region.extent.height,
        ],
    };
    let result = encoder
        .state
        .with_native("resolve GL texture", |provider| {
            let framebuffer = |view| GlFramebufferDescriptor {
                color_attachments: vec![Some(view)],
                depth_stencil_attachment: None,
                draw_buffers: vec![0],
            };
            let mask = GlBlitMask {
                color: true,
                depth: false,
                stencil: false,
            };
            // ES 3 requires identical source and destination rectangles for a
            // multisample blit.  Resolve into a same-coordinate scratch image
            // first, then use the unrestricted single-sample blit for the
            // caller's destination offset. Desktop GL retains its direct path.
            let es_offset_workaround =
                matches!(provider.profile(), GlFamilyProfile::Embedded { .. })
                    && region.src_offset != region.dst_offset;
            if es_offset_workaround {
                let scratch = provider.create_texture_resource(GlTextureDesc {
                    dimension: GlTextureDimension::D2,
                    extent: GlExtent3d {
                        width: source_mip.width,
                        height: source_mip.height,
                        depth_or_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    format: source.desc.format,
                    usage: GlTextureUsage::RENDER_ATTACHMENT,
                })?;
                let scratch_view = GlTextureView {
                    aspect: GlTextureAspect::Color,
                    target: GlAttachmentTarget::Texture(scratch),
                    format: source.desc.format,
                    mip_level: 0,
                    array_layer: 0,
                    layer_count: 1,
                    width: source_mip.width,
                    height: source_mip.height,
                    sample_count: 1,
                };
                let scratch_fbo = match provider.create_framebuffer(&framebuffer(scratch_view)) {
                    Ok(fbo) => fbo,
                    Err(error) => {
                        let _ = provider.destroy_texture_resource(scratch);
                        return Err(error);
                    }
                };
                let same_coordinates = GlBlitRegion {
                    src_offset: region.src_offset,
                    src_extent: region.src_extent,
                    dst_offset: region.src_offset,
                    dst_extent: region.src_extent,
                };
                let work = (|| {
                    for _ in 0..source_region.extent.depth_or_layers {
                        let source_fbo = provider.create_framebuffer(&framebuffer(source_view))?;
                        let destination_fbo =
                            match provider.create_framebuffer(&framebuffer(destination_view)) {
                                Ok(fbo) => fbo,
                                Err(error) => {
                                    let _ = provider.destroy_framebuffer(source_fbo);
                                    return Err(error);
                                }
                            };
                        let resolve = provider.blit_framebuffer(
                            source_fbo,
                            scratch_fbo,
                            same_coordinates,
                            GlFilterMode::Nearest,
                            mask,
                        );
                        let copy = resolve.and_then(|()| {
                            provider.blit_framebuffer(
                                scratch_fbo,
                                destination_fbo,
                                region,
                                GlFilterMode::Nearest,
                                mask,
                            )
                        });
                        let drop_destination = provider.destroy_framebuffer(destination_fbo);
                        let drop_source = provider.destroy_framebuffer(source_fbo);
                        copy.and(drop_destination).and(drop_source)?;
                        source_view.array_layer += 1;
                        destination_view.array_layer += 1;
                    }
                    Ok(())
                })();
                let drop_fbo = provider.destroy_framebuffer(scratch_fbo);
                let drop_texture = provider.destroy_texture_resource(scratch);
                return work.and(drop_fbo).and(drop_texture);
            }
            for _ in 0..source_region.extent.depth_or_layers {
                let source_fbo = provider.create_framebuffer(&framebuffer(source_view))?;
                let destination_fbo =
                    match provider.create_framebuffer(&framebuffer(destination_view)) {
                        Ok(fbo) => fbo,
                        Err(error) => {
                            let _ = provider.destroy_framebuffer(source_fbo);
                            return Err(error);
                        }
                    };
                let resolve = provider.blit_framebuffer(
                    source_fbo,
                    destination_fbo,
                    region,
                    GlFilterMode::Nearest,
                    mask,
                );
                let drop_destination = provider.destroy_framebuffer(destination_fbo);
                let drop_source = provider.destroy_framebuffer(source_fbo);
                resolve.and(drop_destination).and(drop_source)?;
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
    result
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
fn browser(
    encoder: &mut GlEncoder,
    source: &GlTexture,
    destination: &GlTexture,
    src: TextureRegion<'_, Gl>,
    dst: TextureRegion<'_, Gl>,
) -> Result<()> {
    let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
    let result = browser.resolve_texture_regions(
        &source.browser,
        src.mip,
        src.origin,
        src.extent,
        &destination.browser,
        dst.mip,
        dst.origin,
        dst.extent,
        src.aspect,
        dst.aspect,
    );
    if result.is_ok() {
        encoder.issued = true;
    } else {
        encoder.indeterminate = true;
    }
    result
}
