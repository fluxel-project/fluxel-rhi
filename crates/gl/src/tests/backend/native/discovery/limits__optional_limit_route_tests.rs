use super::{NativeGlQuery, ProbeReport, glow_const, limits, optional_limit_routes};
use crate::backend::api::{GlExtensionSet, GlFamilyProfile};
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
