//! Pure texture admission from the owning context's exact discovery facts.
//! No allocation, command recording, or resource-use planning occurs here.
use super::*;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    feature = "native-gles-egl",
    all(target_arch = "wasm32", feature = "webgl2")
))]
pub(super) fn texture_support_from_discovery(
    discovery: &crate::backend::gl::api::GlDiscoverySnapshot,
    desc: crate::backend::gl::api::GlTextureDesc,
    browser: bool,
) -> Support<TextureSupport> {
    use crate::backend::gl::api::{
        GlFormat, GlFormatResourceKind, GlTextureDimension, GlTextureUsage,
    };
    let desktop_1d = desc.dimension == GlTextureDimension::D1
        && matches!(
            discovery.context().profile(),
            crate::backend::gl::api::GlFamilyProfile::Desktop { .. }
        );
    if !desktop_1d
        && !matches!(
            desc.dimension,
            GlTextureDimension::D2 | GlTextureDimension::D2Array | GlTextureDimension::D3
        )
    {
        return Support::Unsupported("GL texture dimension");
    }
    if desktop_1d && (desc.extent.height != 1 || desc.extent.depth_or_layers != 1) {
        return Support::Unsupported("GL 1D texture extent");
    }
    if desc.dimension == GlTextureDimension::D3
        && (desc
            .format
            .supports_blit_aspect(crate::backend::gl::api::GlTextureAspect::DepthOnly)
            || desc
                .format
                .supports_blit_aspect(crate::backend::gl::api::GlTextureAspect::StencilOnly))
    {
        return Support::Unsupported("GL depth/stencil formats do not permit 3D texture storage");
    }
    let renderbuffer = browser && (desc.sample_count > 1 || desc.format == GlFormat::Stencil8);
    let standalone_stencil_copy =
        renderbuffer && desc.format == GlFormat::Stencil8 && desc.sample_count == 1;
    let renderbuffer_usage = GlTextureUsage::RENDER_ATTACHMENT
        | GlTextureUsage::COPY_SOURCE
        | GlTextureUsage::COPY_DESTINATION;
    // A multisample color attachment cannot use the generic GL copy/readback
    // paths, whose executors reject multisample storage.  It can nevertheless
    // be a COPY_SOURCE for the one supported operation: an immediate color
    // resolve.  Keep this admission tied to an actually renderable color
    // attachment; discovery's `copy_source` flag remains the proof for every
    // generic copy/readback route.
    let multisample_resolve_source = desc.sample_count > 1
        && desc.usage.contains(GlTextureUsage::RENDER_ATTACHMENT)
        && desc.usage.contains(GlTextureUsage::COPY_SOURCE)
        && desc
            .format
            .supports_blit_aspect(crate::backend::gl::api::GlTextureAspect::Color);
    let resolve_usage = GlTextureUsage::RENDER_ATTACHMENT | GlTextureUsage::COPY_SOURCE;
    if renderbuffer
        && (desc.dimension != GlTextureDimension::D2
            || desc.extent.depth_or_layers != 1
            || desc.mip_level_count != 1
            || if standalone_stencil_copy {
                desc.usage.is_empty() || !renderbuffer_usage.contains(desc.usage)
            } else {
                desc.usage.is_empty() || !resolve_usage.contains(desc.usage)
            })
    {
        return Support::Unsupported(
            "WebGL2 renderbuffer requires single-layer attachment storage",
        );
    }
    if desc.format.compressed_info().is_some() {
        if desc.dimension == GlTextureDimension::D3
            && !desc
                .format
                .supports_compressed_3d(discovery.context().profile(), discovery.extensions())
        {
            return Support::Unsupported(
                "GL compressed 3D target requires format-specific capability evidence",
            );
        }
        if browser
            && desc.dimension == GlTextureDimension::D2Array
            && !matches!(
                desc.format,
                GlFormat::Etc2Rgb8Unorm
                    | GlFormat::Etc2Rgb8Srgb
                    | GlFormat::Etc2Rgb8A1Unorm
                    | GlFormat::Etc2Rgb8A1Srgb
                    | GlFormat::Etc2Rgba8Unorm
                    | GlFormat::Etc2Rgba8Srgb
                    | GlFormat::EacR11Unorm
                    | GlFormat::EacR11Snorm
                    | GlFormat::EacRg11Unorm
                    | GlFormat::EacRg11Snorm
            )
        {
            return Support::Unsupported("WebGL compressed array target lacks extension support");
        }
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if !browser && !crate::backend::gl::native::native_texture_format_supported(desc.format) {
        return Support::Unsupported("GL native texture storage mapping");
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    if browser && !crate::backend::gl::browser::supports_texture_storage(desc.format) {
        return Support::Unsupported("WebGL2 texture storage mapping");
    }
    let limits = discovery.limits();
    let spatial_limit = if desc.dimension == GlTextureDimension::D3 {
        limits.max_3d_texture_size
    } else {
        limits.max_texture_size
    };
    let depth_limit = if desc.dimension == GlTextureDimension::D3 {
        limits.max_3d_texture_size
    } else if desc.dimension == GlTextureDimension::D1 {
        1
    } else {
        limits.max_array_texture_layers
    };
    if desc.extent.width > spatial_limit
        || (desc.dimension != GlTextureDimension::D1 && desc.extent.height > spatial_limit)
        || desc.extent.depth_or_layers > depth_limit
        || desc.extent.width > i32::MAX as u32
        || desc.extent.height > i32::MAX as u32
        || desc.extent.depth_or_layers > i32::MAX as u32
        || desc.mip_level_count > i32::MAX as u32
        || (renderbuffer && desc.sample_count > limits.max_samples.max(1))
    {
        return Support::Unsupported("GL texture size or sample count");
    }
    let max_axis = desc
        .extent
        .width
        .max(if desc.dimension == GlTextureDimension::D1 {
            1
        } else {
            desc.extent.height
        });
    let max_axis = if desc.dimension == GlTextureDimension::D3 {
        max_axis.max(desc.extent.depth_or_layers)
    } else {
        max_axis
    };
    let max_mips = u32::BITS - max_axis.leading_zeros();
    if desc.mip_level_count > max_mips || (desc.sample_count > 1 && desc.mip_level_count != 1) {
        return Support::Unsupported("GL texture mip count");
    }
    let kind = if renderbuffer {
        GlFormatResourceKind::Renderbuffer
    } else if desc.dimension == GlTextureDimension::D2Array && desc.sample_count > 1 {
        GlFormatResourceKind::MultisampleArray
    } else {
        GlFormatResourceKind::Texture
    };
    let Some(facts) = discovery
        .formats()
        .get_for(kind, desc.format, desc.sample_count)
    else {
        return Support::Unsupported("GL texture format/sample evidence");
    };
    if (standalone_stencil_copy || multisample_resolve_source) && !facts.renderable {
        return Support::Unsupported(
            "WebGL2 framebuffer transfers require a renderable attachment",
        );
    }
    if (desc.usage.contains(GlTextureUsage::SAMPLED) && !facts.sampled)
        || (desc.usage.contains(GlTextureUsage::RENDER_ATTACHMENT) && !facts.renderable)
        // WebGL2's format table does not claim generic renderbuffer copy
        // support. The sole admitted exception is S8 single-sample
        // private-FBO blit below; it is typed, non-sampled, and still
        // requires actual renderability from this context. Public usage may
        // be copy-only because this backend owns those transient attachments.
        || (desc.usage.contains(GlTextureUsage::COPY_SOURCE)
            && !facts.copy_source
            && !standalone_stencil_copy
            && !multisample_resolve_source)
        || (desc.usage.contains(GlTextureUsage::COPY_DESTINATION)
            && !facts.copy_destination
            && !standalone_stencil_copy)
        || (desc.usage.contains(GlTextureUsage::STORAGE_BINDING)
            && !(facts.storage_read || facts.storage_write))
    {
        return Support::Unsupported("GL texture format usage evidence");
    }
    Support::Supported(TextureSupport {
        max_extent: [
            spatial_limit,
            if desc.dimension == GlTextureDimension::D1 {
                1
            } else {
                spatial_limit
            },
            if desc.dimension == GlTextureDimension::D3 {
                spatial_limit
            } else {
                1
            },
        ],
        max_array_layers: if desc.dimension == GlTextureDimension::D3 {
            1
        } else {
            limits.max_array_texture_layers
        },
        max_mips,
        sample_counts: vec![desc.sample_count],
        compatible_view_formats: Vec::new(),
        filterable: facts.filterable,
        blendable: facts.blendable,
        storage: facts.storage_read || facts.storage_write,
    })
}
