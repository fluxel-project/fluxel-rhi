//! Texture view validation and native alias ownership.
use super::*;
use crate::backend::gl::api::{GlAttachmentTarget, GlTextureDimension};

pub(super) fn create_view(
    device: &GlDevice,
    texture: &GlTexture,
    desc: &api::TextureViewDesc,
) -> Result<GlTextureView> {
    if !std::rc::Rc::ptr_eq(&device.0, &texture.state) {
        return Err(Error::WrongDevice);
    }
    let allocated = texture.desc;
    let layers = match allocated.dimension {
        GlTextureDimension::D1 | GlTextureDimension::D2 | GlTextureDimension::D3 => 1,
        GlTextureDimension::D2Array => allocated.extent.depth_or_layers,
        _ => return Err(Error::Unsupported("GL texture view dimension")),
    };
    if desc.mips.start >= desc.mips.end
        || desc.mips.end > allocated.mip_level_count
        || desc.layers.start >= desc.layers.end
        || desc.layers.end > layers
    {
        return Err(Error::InvalidRange);
    }
    let aspects = match allocated.format {
        crate::backend::gl::api::GlFormat::Depth16Unorm
        | crate::backend::gl::api::GlFormat::Depth24Unorm
        | crate::backend::gl::api::GlFormat::Depth32Float => api::TextureAspects::DEPTH,
        crate::backend::gl::api::GlFormat::Depth24PlusStencil8
        | crate::backend::gl::api::GlFormat::Depth32FloatStencil8 => {
            api::TextureAspects::DEPTH | api::TextureAspects::STENCIL
        }
        crate::backend::gl::api::GlFormat::Stencil8 => api::TextureAspects::STENCIL,
        _ => api::TextureAspects::COLOR,
    };
    if desc.aspects != aspects
        && !(aspects == (api::TextureAspects::DEPTH | api::TextureAspects::STENCIL)
            && (desc.aspects == api::TextureAspects::DEPTH
                || desc.aspects == api::TextureAspects::STENCIL))
    {
        return Err(Error::InvalidRange);
    }
    let requested = crate::backend::gl::translate::lower_texture_format(desc.format)?;
    if requested != allocated.format {
        if !texture.view_formats.contains(&desc.format) {
            return Err(Error::InvalidRange);
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if !crate::backend::gl::native::texture_view::compatible_formats(
            allocated.format,
            requested,
        ) {
            return Err(Error::Unsupported("incompatible GL texture view formats"));
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        return Err(Error::Unsupported(
            "GL texture view format reinterpretation",
        ));
    }
    let dimension = match desc.dimension {
        api::TextureViewDimension::D1 => GlTextureDimension::D1,
        api::TextureViewDimension::D2 => GlTextureDimension::D2,
        api::TextureViewDimension::D2Array => GlTextureDimension::D2Array,
        api::TextureViewDimension::D3 => GlTextureDimension::D3,
        api::TextureViewDimension::Cube => GlTextureDimension::Cube,
        api::TextureViewDimension::CubeArray => GlTextureDimension::CubeArray,
        _ => return Err(Error::Unsupported("GL texture view dimension")),
    };
    let count = desc.layers.end - desc.layers.start;
    let compatible = matches!(
        (allocated.dimension, dimension),
        (GlTextureDimension::D1, GlTextureDimension::D1)
            | (GlTextureDimension::D2, GlTextureDimension::D2)
            | (GlTextureDimension::D2, GlTextureDimension::D2Array)
            | (GlTextureDimension::D2Array, GlTextureDimension::D2Array)
            | (GlTextureDimension::D2Array, GlTextureDimension::D2)
            | (GlTextureDimension::D2Array, GlTextureDimension::Cube)
            | (GlTextureDimension::D2Array, GlTextureDimension::CubeArray)
            | (GlTextureDimension::D3, GlTextureDimension::D3)
    );
    if !compatible
        || (dimension == GlTextureDimension::D2 && count != 1)
        || (dimension == GlTextureDimension::Cube && count != 6)
        || (dimension == GlTextureDimension::CubeArray && count % 6 != 0)
    {
        return Err(Error::Unsupported(
            "GL texture view dimension or layer count",
        ));
    }
    let extent = allocated
        .mip_extent(desc.mips.start)
        .ok_or(Error::InvalidRange)?;
    if matches!(
        dimension,
        GlTextureDimension::Cube | GlTextureDimension::CubeArray
    ) && extent.width != extent.height
    {
        return Err(Error::InvalidRange);
    }
    let full = requested == allocated.format
        && dimension == allocated.dimension
        && desc.mips == (0..allocated.mip_level_count)
        && desc.layers == (0..layers);
    // FBO attachments address the parent texture's mip and layer range
    // directly. OVR multiview can attach a consecutive array interval without
    // a texture-view object; sampling still requires an actual alias below.
    let attachment = requested == allocated.format
        && desc.mips.end - desc.mips.start == 1
        && (count == 1
            && matches!(
                allocated.dimension,
                GlTextureDimension::D2 | GlTextureDimension::D2Array
            )
            || dimension == GlTextureDimension::D2Array
                && allocated.dimension == GlTextureDimension::D2Array);
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        let mut native = crate::backend::gl::api::GlTextureView {
            target: GlAttachmentTarget::Texture(texture.id),
            format: allocated.format,
            aspect: match desc.aspects {
                api::TextureAspects::DEPTH if aspects != desc.aspects => {
                    crate::backend::gl::api::GlTextureAspect::DepthOnly
                }
                api::TextureAspects::STENCIL if aspects != desc.aspects => {
                    crate::backend::gl::api::GlTextureAspect::StencilOnly
                }
                _ => crate::backend::gl::api::GlTextureAspect::All,
            },
            mip_level: desc.mips.start,
            array_layer: desc.layers.start,
            layer_count: count,
            width: extent.width,
            height: extent.height,
            sample_count: allocated.sample_count,
        };
        let mut view_texture = None;
        let mut cube_mirror = None;
        let mut stencil_mirror = None;
        let stencil_sampling = desc.aspects == api::TextureAspects::STENCIL
            && aspects != desc.aspects
            && allocated
                .usage
                .contains(crate::backend::gl::api::GlTextureUsage::SAMPLED);
        let mut stencil_sampling_ready = false;
        // Binding one image mip does not require a texture-view name when
        // every array layer (or every 3D slice) remains visible.
        let storage_full_layers = allocated
            .usage
            .contains(crate::backend::gl::api::GlTextureUsage::STORAGE_BINDING)
            && requested == allocated.format
            && dimension == allocated.dimension
            && desc.mips.end - desc.mips.start == 1
            && desc.layers == (0..layers);
        if !full || stencil_sampling {
            let mut view_desc = allocated;
            view_desc.dimension = dimension;
            view_desc.format = requested;
            if requested != allocated.format {
                let mut format_check = allocated;
                format_check.format = requested;
                if let Support::Unsupported(reason) = super::texture_support(device, format_check) {
                    return Err(Error::Unsupported(reason));
                }
            }
            view_desc.extent = crate::backend::gl::api::GlExtent3d {
                width: extent.width,
                height: if dimension == GlTextureDimension::D1 {
                    1
                } else {
                    extent.height
                },
                depth_or_layers: match dimension {
                    GlTextureDimension::D2 => 1,
                    GlTextureDimension::D3 => extent.depth_or_layers,
                    _ => count,
                },
            };
            view_desc.mip_level_count = desc.mips.end - desc.mips.start;
            let alias = device.0.with_native("create texture view", |provider| {
                provider.create_texture_view_resource(
                    texture.id,
                    view_desc,
                    desc.mips.start,
                    desc.layers.start,
                )
            });
            match alias {
                Ok(id) => {
                    native.target = GlAttachmentTarget::Texture(id);
                    native.format = requested;
                    native.mip_level = 0;
                    native.array_layer = 0;
                    view_texture = Some(GlTexture::from_inner(GlTextureInner {
                        state: device.0.clone(),
                        id,
                        desc: view_desc,
                        view_formats: texture.view_formats.clone(),
                    }));
                    if stencil_sampling {
                        match device.0.with_native("select stencil view", |provider| {
                            provider.set_depth_stencil_texture_mode(id, true)
                        }) {
                            Ok(()) => stencil_sampling_ready = true,
                            Err(crate::backend::gl::api::GlError::Unsupported { .. }) => {}
                            Err(error) => return Err(native_error(error)),
                        }
                    }
                }
                Err(crate::backend::gl::api::GlError::Unsupported { .. })
                    if (attachment && requested == allocated.format)
                        || storage_full_layers
                        || stencil_sampling => {}
                Err(crate::backend::gl::api::GlError::Unsupported { .. })
                    if dimension == GlTextureDimension::Cube
                        && requested == allocated.format
                        && matches!(
                            allocated.format,
                            crate::backend::gl::api::GlFormat::Rgba8Unorm
                                | crate::backend::gl::api::GlFormat::Rgba8Uint
                                | crate::backend::gl::api::GlFormat::Rgba8Sint
                        )
                        && allocated.sample_count == 1
                        && allocated
                            .usage
                            .contains(crate::backend::gl::api::GlTextureUsage::SAMPLED) =>
                {
                    let id = device
                        .0
                        .with_native("create cube mirror", |provider| {
                            provider.create_cube_mirror(texture.id, view_desc)
                        })
                        .map_err(native_error)?;
                    cube_mirror = Some(GlTexture::from_inner(GlTextureInner {
                        state: device.0.clone(),
                        id,
                        desc: view_desc,
                        view_formats: Vec::new(),
                    }));
                }
                Err(error) => return Err(native_error(error)),
            }
        }
        if stencil_sampling && !stencil_sampling_ready && view_texture.is_none() {
            stencil_sampling_ready = device
                .0
                .with_native("query stencil texturing", |provider| {
                    Ok(provider.stencil_texturing_supported())
                })
                .map_err(native_error)?;
        }
        if stencil_sampling_ready
            && view_texture.is_none()
            && matches!(
                allocated.dimension,
                GlTextureDimension::D2 | GlTextureDimension::D2Array
            )
            && matches!(
                dimension,
                GlTextureDimension::D2 | GlTextureDimension::D2Array
            )
            && allocated.sample_count == 1
        {
            use crate::backend::gl::api::GlResourceApi as _;
            let mut mirror_desc = allocated;
            mirror_desc.extent = extent;
            mirror_desc.dimension = dimension;
            mirror_desc.extent.depth_or_layers = if dimension == GlTextureDimension::D2 {
                1
            } else {
                count
            };
            mirror_desc.mip_level_count = desc.mips.end - desc.mips.start;
            if allocated.dimension == GlTextureDimension::D2Array
                && matches!(
                    dimension,
                    GlTextureDimension::D2 | GlTextureDimension::D2Array
                )
                && device
                    .0
                    .with_native("query stencil array layout", |provider| {
                        Ok(provider.requires_stencil_extract())
                    })
                    .map_err(native_error)?
            {
                mirror_desc.format = crate::backend::gl::api::GlFormat::R8Uint;
                mirror_desc.usage = crate::backend::gl::api::GlTextureUsage::SAMPLED
                    | crate::backend::gl::api::GlTextureUsage::RENDER_ATTACHMENT;
            }
            let id = device
                .0
                .with_native("create stencil mirror", |provider| {
                    provider.create_texture_resource(mirror_desc)
                })
                .map_err(native_error)?;
            stencil_mirror = Some(GlTexture::from_inner(GlTextureInner {
                state: device.0.clone(),
                id,
                desc: mirror_desc,
                view_formats: Vec::new(),
            }));
        }
        return Ok(GlTextureView {
            texture: texture.clone(),
            aspects: desc.aspects,
            full_sampling_view: (full
                || view_texture.is_some()
                || cube_mirror.is_some()
                || stencil_mirror.is_some())
                && (!stencil_sampling || stencil_sampling_ready),
            cube_mirror,
            stencil_mirror,
            source_layer: desc.layers.start,
            dimension: desc.dimension,
            mip_count: desc.mips.end - desc.mips.start,
            source_mip: desc.mips.start,
            view_texture,
            native,
        });
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        let cube_mirror = if dimension == GlTextureDimension::Cube
            && requested == allocated.format
            && matches!(
                allocated.format,
                crate::backend::gl::api::GlFormat::Rgba8Unorm
                    | crate::backend::gl::api::GlFormat::Rgba8Uint
                    | crate::backend::gl::api::GlFormat::Rgba8Sint
            )
            && allocated.sample_count == 1
            && allocated
                .usage
                .contains(crate::backend::gl::api::GlTextureUsage::SAMPLED)
        {
            Some(texture.browser.create_cube_mirror(
                desc.mips.start,
                desc.mips.end - desc.mips.start,
                desc.layers.start,
            )?)
        } else {
            None
        };
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        let has_mirror = cube_mirror.is_some();
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        let has_mirror = false;
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        let stencil_sampling_supported =
            if desc.aspects == api::TextureAspects::STENCIL && aspects != desc.aspects {
                device.0.owner.stencil_texturing_supported()?
            } else {
                true
            };
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        let stencil_sampling_supported = false;
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        let stencil_mirror = if stencil_sampling_supported
            && desc.aspects == api::TextureAspects::STENCIL
            && aspects != desc.aspects
            && matches!(
                allocated.dimension,
                GlTextureDimension::D2 | GlTextureDimension::D2Array
            )
            && matches!(
                dimension,
                GlTextureDimension::D2 | GlTextureDimension::D2Array
            )
            && allocated.sample_count == 1
            && allocated
                .usage
                .contains(crate::backend::gl::api::GlTextureUsage::SAMPLED)
        {
            Some(texture.browser.create_stencil_mirror(
                desc.mips.start,
                desc.mips.end - desc.mips.start,
                dimension,
                count,
            )?)
        } else {
            None
        };
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        let has_stencil_mirror = stencil_mirror.is_some();
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        let has_stencil_mirror = false;
        if !full && !attachment && !has_mirror && !has_stencil_mirror {
            return Err(Error::Unsupported("GL texture view subrange"));
        }
        Ok(GlTextureView {
            texture: texture.clone(),
            aspects: desc.aspects,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            cube_mirror,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            stencil_mirror,
            source_layer: desc.layers.start,
            full_sampling_view: (full || has_mirror || has_stencil_mirror)
                && stencil_sampling_supported,
            dimension: desc.dimension,
            mip_count: desc.mips.end - desc.mips.start,
            source_mip: desc.mips.start,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            mip_level: desc.mips.start,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            array_layer: desc.layers.start,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            layer_count: count,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            width: extent.width,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            height: extent.height,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            sample_count: allocated.sample_count,
        })
    }
}
