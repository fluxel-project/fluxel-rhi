//! Native GL implementation-limit discovery.

use super::formats::{nonnegative, to_u32};
use super::*;

pub(super) fn limits(
    query: &impl NativeGlQuery,
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
    report: &ProbeReport,
) -> Result<GlLimits, NativeDiscoveryError> {
    let u = |token, name| nonnegative(query.integer(token), name);
    let pair = query
        .integer_pair(glow_const::MAX_VIEWPORT_DIMS)
        .ok_or(NativeDiscoveryError::QueryFailed("GL_MAX_VIEWPORT_DIMS"))?;
    let optional_routes = optional_limit_routes(profile, extensions);
    // A ceiling is recorded only where the allocation it bounds exists: below
    // the storage floor the value stays at its one-sample fail-closed floor
    // instead of advertising a multisample texture the context cannot make.
    let texture_multisample = supports_multisample_texture_storage(profile);
    // Optional domains must not turn a context that lacks one optional limit
    // token into a failed baseline discovery.  A legal route is only permission
    // to ask, not proof the driver can answer; a refused query therefore
    // records the fail-closed zero fact.  This also contains EGL stacks which
    // report a newer ES version string for an ES 3.0 request but reject the
    // ES3.1-only storage/compute tokens in that actual context.
    let optional = |enabled, token, _name| {
        if enabled {
            Ok(query
                .integer(token)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(0))
        } else {
            Ok(0)
        }
    };
    let indexed = |enabled, token, index, _name| {
        if enabled {
            // See `optional`: an unavailable optional indexed limit is a
            // closed capability fact, not a reason to reject ES3 baseline.
            Ok(query
                .indexed_integer(token, index)
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(0))
        } else {
            Ok(0)
        }
    };
    let anisotropy = extensions
        .is_acquired(GlKnownExtension::ExtTextureFilterAnisotropic)
        .then(|| query.float(glow_const::MAX_TEXTURE_MAX_ANISOTROPY_EXT))
        .flatten()
        .and_then(GlFiniteF32::new);
    Ok(GlLimits {
        max_texture_size: u(glow_const::MAX_TEXTURE_SIZE, "GL_MAX_TEXTURE_SIZE")?,
        max_3d_texture_size: u(glow_const::MAX_3D_TEXTURE_SIZE, "GL_MAX_3D_TEXTURE_SIZE")?,
        max_array_texture_layers: u(
            glow_const::MAX_ARRAY_TEXTURE_LAYERS,
            "GL_MAX_ARRAY_TEXTURE_LAYERS",
        )?,
        max_cube_map_texture_size: u(
            glow_const::MAX_CUBE_MAP_TEXTURE_SIZE,
            "GL_MAX_CUBE_MAP_TEXTURE_SIZE",
        )?,
        max_renderbuffer_size: u(
            glow_const::MAX_RENDERBUFFER_SIZE,
            "GL_MAX_RENDERBUFFER_SIZE",
        )?,
        max_color_attachments: u(
            glow_const::MAX_COLOR_ATTACHMENTS,
            "GL_MAX_COLOR_ATTACHMENTS",
        )?,
        max_draw_buffers: u(glow_const::MAX_DRAW_BUFFERS, "GL_MAX_DRAW_BUFFERS")?,
        max_vertex_attributes: u(glow_const::MAX_VERTEX_ATTRIBS, "GL_MAX_VERTEX_ATTRIBS")?,
        max_viewport_dimensions: [
            to_u32(pair[0], "GL_MAX_VIEWPORT_DIMS")?,
            to_u32(pair[1], "GL_MAX_VIEWPORT_DIMS")?,
        ],
        max_viewports: optional(
            matches!(profile, GlFamilyProfile::Desktop { major: 4, minor } if minor >= 1),
            glow_const::MAX_VIEWPORTS,
            "GL_MAX_VIEWPORTS",
        )?,
        max_vertex_texture_image_units: u(
            glow_const::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
            "GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS",
        )?,
        max_fragment_texture_image_units: u(
            glow_const::MAX_TEXTURE_IMAGE_UNITS,
            "GL_MAX_TEXTURE_IMAGE_UNITS",
        )?,
        max_combined_texture_image_units: u(
            glow_const::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS",
        )?,
        max_uniform_buffer_bindings: u(
            glow_const::MAX_UNIFORM_BUFFER_BINDINGS,
            "GL_MAX_UNIFORM_BUFFER_BINDINGS",
        )?,
        max_uniform_block_size: u64::from(u(
            glow_const::MAX_UNIFORM_BLOCK_SIZE,
            "GL_MAX_UNIFORM_BLOCK_SIZE",
        )?),
        uniform_buffer_offset_alignment: u64::from(u(
            glow_const::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
            "GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT",
        )?),
        max_vertex_uniform_blocks: u(
            glow_const::MAX_VERTEX_UNIFORM_BLOCKS,
            "GL_MAX_VERTEX_UNIFORM_BLOCKS",
        )?,
        max_fragment_uniform_blocks: u(
            glow_const::MAX_FRAGMENT_UNIFORM_BLOCKS,
            "GL_MAX_FRAGMENT_UNIFORM_BLOCKS",
        )?,
        max_compute_uniform_blocks: optional(
            optional_routes.compute,
            glow_const::MAX_COMPUTE_UNIFORM_BLOCKS,
            "GL_MAX_COMPUTE_UNIFORM_BLOCKS",
        )?,
        max_combined_uniform_blocks: u(
            glow_const::MAX_COMBINED_UNIFORM_BLOCKS,
            "GL_MAX_COMBINED_UNIFORM_BLOCKS",
        )?,
        max_storage_buffer_bindings: optional(
            optional_routes.storage,
            glow_const::MAX_SHADER_STORAGE_BUFFER_BINDINGS,
            "GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS",
        )?,
        max_storage_block_size: u64::from(optional(
            optional_routes.storage,
            glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE,
            "GL_MAX_SHADER_STORAGE_BLOCK_SIZE",
        )?),
        storage_buffer_offset_alignment: u64::from(optional(
            optional_routes.storage,
            glow_const::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT,
            "GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT",
        )?),
        max_vertex_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_VERTEX_SHADER_STORAGE_BLOCKS,
            "GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS",
        )?,
        max_fragment_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
            "GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS",
        )?,
        max_compute_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS",
        )?,
        max_combined_storage_blocks: optional(
            optional_routes.storage,
            glow_const::MAX_COMBINED_SHADER_STORAGE_BLOCKS,
            "GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS",
        )?,
        max_image_units: optional(
            optional_routes.image,
            glow_const::MAX_IMAGE_UNITS,
            "GL_MAX_IMAGE_UNITS",
        )?,
        max_combined_image_units: optional(
            optional_routes.image,
            glow_const::MAX_COMBINED_IMAGE_UNIFORMS,
            "GL_MAX_COMBINED_IMAGE_UNIFORMS",
        )?,
        max_samples: u(glow_const::MAX_SAMPLES, "GL_MAX_SAMPLES")?,
        max_color_texture_samples: optional(
            texture_multisample,
            glow_const::MAX_COLOR_TEXTURE_SAMPLES,
            "GL_MAX_COLOR_TEXTURE_SAMPLES",
        )?
        .max(1),
        max_depth_texture_samples: optional(
            texture_multisample,
            glow_const::MAX_DEPTH_TEXTURE_SAMPLES,
            "GL_MAX_DEPTH_TEXTURE_SAMPLES",
        )?
        .max(1),
        max_integer_samples: optional(
            texture_multisample,
            glow_const::MAX_INTEGER_SAMPLES,
            "GL_MAX_INTEGER_SAMPLES",
        )?
        .max(1),
        max_compute_work_group_count: [
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[0]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[1]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_COUNT,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_COUNT[2]",
            )?,
        ],
        max_compute_work_group_size: [
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                0,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[0]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                1,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[1]",
            )?,
            indexed(
                optional_routes.compute,
                glow_const::MAX_COMPUTE_WORK_GROUP_SIZE,
                2,
                "GL_MAX_COMPUTE_WORK_GROUP_SIZE[2]",
            )?,
        ],
        max_compute_work_group_invocations: optional(
            optional_routes.compute,
            glow_const::MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
            "GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS",
        )?,
        // No GL family exposes a portable multi-draw-indirect count limit query,
        // so the fact is `None` here -- and this is the row that decides the
        // domain, not a note beside it: `GlLimits::supports_multi_draw_indirect`
        // is exactly "a queried count of at least one", so with no query to read
        // one from, the limit half answers false on every real context and the
        // capability cannot enable however the route is resolved. The route and
        // the probe are false here too and independently: `GlowProbes` has no
        // multi-draw-indirect probe because glow 0.18 does not bind the entry
        // point, and no executor implements the verb (plan P2-15, narrowly
        // resolved as a fixture-only domain rather than left to be discovered a
        // fourth time).
        max_multi_draw_indirect_count: None,
        // The view count is only queryable on a context that acquired the
        // multiview extension; without it the fact is 0 and the multiview floor
        // cannot be met by any route.
        max_multiview_view_count: optional(
            extensions.is_acquired(GlKnownExtension::OvrMultiview2),
            glow_const::MAX_MULTIVIEW_VIEWS,
            "GL_MAX_VIEWS_OVR",
        )?,
        query_counter_bits: report.query_counter_bits.unwrap_or(0),
        max_texture_anisotropy: anisotropy,
    })
}

/// The resolved legal route for every optional limit family.
///
/// GLES does not inherit desktop `GL_ARB_*` alternatives.  In particular an
/// ES 3.0 context may report vendor aggregate extensions whose names describe
/// ES 3.1 functionality, but it has no legal SSBO/image/compute limit tokens.
/// Keep the route decision before every query so a rejected token cannot be
/// mistaken for a missing device capability after the fact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OptionalLimitRoutes {
    compute: bool,
    storage: bool,
    image: bool,
}

fn optional_limit_routes(
    profile: GlFamilyProfile,
    extensions: &GlExtensionSet,
) -> OptionalLimitRoutes {
    let desktop_extension = |extension: GlKnownExtension| {
        matches!(profile, GlFamilyProfile::Desktop { .. })
            && extension.is_legal_for(profile)
            && extensions.is_acquired(extension)
    };
    OptionalLimitRoutes {
        compute: profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
            || desktop_extension(GlKnownExtension::ArbComputeShader),
        storage: profile.meets(Some(GlVersion::new(4, 3)), Some(GlVersion::new(3, 1)))
            || desktop_extension(GlKnownExtension::ArbShaderStorageBufferObject),
        image: profile.meets(Some(GlVersion::new(4, 2)), Some(GlVersion::new(3, 1)))
            || desktop_extension(GlKnownExtension::ArbShaderImageLoadStore),
    }
}

#[cfg(test)]
mod optional_limit_route_tests {
    use super::optional_limit_routes;
    use crate::backend::gl::api::{GlExtensionSet, GlFamilyProfile};

    #[test]
    fn gles30_has_no_es31_limit_route() {
        let routes = optional_limit_routes(
            GlFamilyProfile::Embedded { major: 3, minor: 0 },
            &GlExtensionSet::default(),
        );
        assert!(!routes.compute);
        assert!(!routes.storage);
        assert!(!routes.image);
    }

    #[test]
    fn gles31_enables_its_core_limit_routes() {
        let routes = optional_limit_routes(
            GlFamilyProfile::Embedded { major: 3, minor: 1 },
            &GlExtensionSet::default(),
        );
        assert!(routes.compute);
        assert!(routes.storage);
        assert!(routes.image);
    }
}
