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
        max_compute_texture_image_units: optional(
            optional_routes.compute,
            glow::MAX_COMPUTE_TEXTURE_IMAGE_UNITS,
            "GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS",
        )?,
        max_combined_texture_image_units: u(
            glow_const::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS",
        )?,
        max_default_uniform_vectors: if matches!(profile, GlFamilyProfile::Desktop { .. }) {
            [
                u(0x8B4A, "GL_MAX_VERTEX_UNIFORM_COMPONENTS")? / 4,
                u(0x8B49, "GL_MAX_FRAGMENT_UNIFORM_COMPONENTS")? / 4,
                optional(
                    optional_routes.compute,
                    0x8263,
                    "GL_MAX_COMPUTE_UNIFORM_COMPONENTS",
                )? / 4,
            ]
        } else {
            [
                u(0x8DFB, "GL_MAX_VERTEX_UNIFORM_VECTORS")?,
                u(0x8DFD, "GL_MAX_FRAGMENT_UNIFORM_VECTORS")?,
                optional(
                    optional_routes.compute,
                    0x8263,
                    "GL_MAX_COMPUTE_UNIFORM_COMPONENTS",
                )? / 4,
            ]
        },
        max_uniform_locations: if matches!(profile, GlFamilyProfile::Desktop { major, minor } if major > 4 || major == 4 && minor >= 3)
        {
            Some(
                query
                    .integer(0x826E)
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or(0),
            )
        } else {
            None
        },
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
        max_storage_block_size: if optional_routes.storage {
            query
                .integer64(glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE)
                .and_then(|value| u64::try_from(value).ok())
                .unwrap_or(0)
        } else {
            0
        },
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
        max_vertex_image_uniforms: optional(
            optional_routes.image,
            glow::MAX_VERTEX_IMAGE_UNIFORMS,
            "GL_MAX_VERTEX_IMAGE_UNIFORMS",
        )?,
        max_fragment_image_uniforms: optional(
            optional_routes.image,
            glow::MAX_FRAGMENT_IMAGE_UNIFORMS,
            "GL_MAX_FRAGMENT_IMAGE_UNIFORMS",
        )?,
        max_compute_image_uniforms: optional(
            optional_routes.image && optional_routes.compute,
            glow::MAX_COMPUTE_IMAGE_UNIFORMS,
            "GL_MAX_COMPUTE_IMAGE_UNIFORMS",
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
        // Multi-draw has no hardware-limit query. The count parameter is a
        // `GLsizei`, so this is the largest ABI-representable request, not an
        // observation or claim about a driver's hardware batching capacity.
        max_multi_draw_indirect_count: Some(i32::MAX as u32),
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
    use super::{NativeGlQuery, ProbeReport, glow_const, limits, optional_limit_routes};
    use crate::backend::gl::api::{GlExtensionSet, GlFamilyProfile};
    use std::cell::RefCell;

    #[derive(Default)]
    struct Query(RefCell<Vec<u32>>, RefCell<Vec<u32>>);
    impl NativeGlQuery for Query {
        fn take_error(&self) -> bool {
            false
        }
        fn string(&self, _: u32) -> Option<String> {
            None
        }
        fn integer(&self, token: u32) -> Option<i64> {
            self.0.borrow_mut().push(token);
            Some(match token {
                glow::MAX_VERTEX_IMAGE_UNIFORMS => 3,
                glow::MAX_FRAGMENT_IMAGE_UNIFORMS => 5,
                glow::MAX_COMPUTE_IMAGE_UNIFORMS => 7,
                glow::MAX_COMPUTE_TEXTURE_IMAGE_UNITS => 13,
                glow::MAX_COMBINED_IMAGE_UNIFORMS => 11,
                _ => 1,
            })
        }
        fn integer64(&self, token: u32) -> Option<i64> {
            self.1.borrow_mut().push(token);
            Some(if token == glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE {
                5_i64 * 1024 * 1024 * 1024
            } else {
                1
            })
        }
        fn integer_pair(&self, _: u32) -> Option<[i64; 2]> {
            Some([1, 1])
        }
        fn indexed_integer(&self, _: u32, _: u32) -> Option<i64> {
            Some(1)
        }
        fn float(&self, _: u32) -> Option<f32> {
            None
        }
        fn indexed_string(&self, _: u32, _: u32) -> Option<String> {
            None
        }
        fn drawable_attachment(&self, _: u32, _: u32) -> Option<i64> {
            None
        }
    }

    #[test]
    fn image_limits_query_per_stage_only_when_legal() {
        for (profile, expected) in [
            (
                GlFamilyProfile::Embedded { major: 3, minor: 0 },
                [0, 0, 0, 0],
            ),
            (
                GlFamilyProfile::Embedded { major: 3, minor: 1 },
                [3, 5, 7, 11],
            ),
            (
                GlFamilyProfile::Desktop { major: 4, minor: 2 },
                [3, 5, 0, 11],
            ),
            (
                GlFamilyProfile::Desktop { major: 4, minor: 3 },
                [3, 5, 7, 11],
            ),
        ] {
            let query = Query::default();
            let result = limits(
                &query,
                profile,
                &GlExtensionSet::default(),
                &ProbeReport::default(),
            )
            .unwrap();
            assert_eq!(
                [
                    result.max_vertex_image_uniforms,
                    result.max_fragment_image_uniforms,
                    result.max_compute_image_uniforms,
                    result.max_combined_image_units
                ],
                expected
            );
            let queries = query.0.borrow();
            let has_compute = matches!(
                profile,
                GlFamilyProfile::Embedded {
                    major: 3,
                    minor: 1..
                } | GlFamilyProfile::Desktop {
                    major: 4,
                    minor: 3..
                }
            );
            assert_eq!(
                result.max_compute_texture_image_units,
                if has_compute { 13 } else { 0 }
            );
            assert_eq!(
                queries.contains(&glow::MAX_COMPUTE_TEXTURE_IMAGE_UNITS),
                has_compute
            );
            for (token, value) in [
                glow::MAX_VERTEX_IMAGE_UNIFORMS,
                glow::MAX_FRAGMENT_IMAGE_UNIFORMS,
                glow::MAX_COMPUTE_IMAGE_UNIFORMS,
                glow::MAX_COMBINED_IMAGE_UNIFORMS,
            ]
            .into_iter()
            .zip(expected)
            {
                assert_eq!(queries.contains(&token), value != 0);
            }
        }
    }

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

    #[test]
    fn storage_block_limit_uses_glint64_query() {
        let query = Query::default();
        let result = limits(
            &query,
            GlFamilyProfile::Desktop { major: 4, minor: 3 },
            &GlExtensionSet::default(),
            &ProbeReport::default(),
        )
        .unwrap();
        assert_eq!(result.max_storage_block_size, 5 * 1024 * 1024 * 1024);
        assert!(
            query
                .1
                .borrow()
                .contains(&glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE)
        );
        assert!(
            !query
                .0
                .borrow()
                .contains(&glow_const::MAX_SHADER_STORAGE_BLOCK_SIZE)
        );
    }
}
