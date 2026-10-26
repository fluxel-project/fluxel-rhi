//! Unit tests for the corresponding implementation module.

use super::*;
#[test]
fn triangle_proof_rejects_unprovable_source_forms() {
    let triangle = "#include <metal_stdlib>\nusing T = metal::mesh<V, P, 3, 1, metal::topology::triangle>; [[mesh]] void main(T out) {}";
    assert!(proves_triangle_mesh(triangle));
    assert!(!proves_triangle_mesh(&format!(
        "{triangle} object_data int* payload;"
    )));
    assert!(!proves_triangle_mesh(&format!(
        "{triangle} threadgroup int* shared;"
    )));
    assert!(!proves_triangle_mesh(
        &triangle.replace("using T", "%:define triangle line\nusing T")
    ));
    assert!(!proves_triangle_mesh(
        "#include <metal_stdlib>\n// metal::mesh<V,P,3,1,metal::topology::triangle>\n[[mesh]] void main(int x) {}"
    ));
    assert!(!proves_triangle_mesh(
        &triangle.replace("metal::topology::triangle", "metal::topology::point")
    ));
    assert!(!proves_triangle_mesh(
        &triangle.replace("topology::triangle", "topology::line")
    ));
    assert!(!proves_triangle_mesh(&triangle.replace(
        "#include <metal_stdlib>",
        "#define triangle line\n#include <metal_stdlib>"
    )));
    assert!(!proves_triangle_mesh(
        &triangle.replace("metal::mesh", "mesh")
    ));
    assert!(!proves_triangle_mesh(&format!(
        "{triangle} using L = metal::mesh<V, P, 3, 1, metal::topology::line>;"
    )));
    assert!(!proves_triangle_mesh(
        "/*\n#include <metal_stdlib>\n*/ metal::mesh<V,P,3,1,metal::topology::triangle> value;"
    ));
    assert!(!proves_triangle_mesh(
        &triangle.replace("metal::topology::triangle", "Other")
    ));
    assert!(!proves_triangle_mesh(
        &triangle.replace(" 3, 1,", " MAX, 1,")
    ));
    assert!(!proves_triangle_mesh(
        &triangle.replace("using T", "\\\nusing T")
    ));
    assert!(!proves_triangle_mesh(
        &triangle.replace("using T", "/* using T")
    ));
}
#[test]
fn indirect_arguments_require_usage_alignment_and_complete_record() {
    let usage = api::BufferUsage::INDIRECT;
    assert!(valid_indirect(usage, 16, 4));
    assert!(!valid_indirect(api::BufferUsage::COPY_SRC, 16, 4));
    assert!(!valid_indirect(usage, 16, 3));
    assert!(!valid_indirect(usage, 15, 4));
    assert!(!valid_indirect(usage, u64::MAX, u64::MAX - 3));
}

#[test]
fn indirect_mesh_requires_apple9_and_selector() {
    assert!(indirect_mesh_supported(true, true));
    assert!(!indirect_mesh_supported(false, true));
    assert!(!indirect_mesh_supported(true, false));
    assert!(!indirect_mesh_supported(false, false));
}
