
use super::*;
use crate::backend::api::{GlBindingLocation, GlShaderResourceKind};
fn binding(binding: u32, count: u32) -> GlLogicalBinding {
    GlLogicalBinding {
        names: vec![format!("b{binding}")],
        location: GlBindingLocation { group: 0, binding },
        element: None,
        kind: GlShaderResourceKind::UniformBuffer,
        array_count: count,
        fixed_array: count > 1,
        pair: None,
        image_unit: None,
        sampled_type: None,
        image_type: None,
    }
}
#[test]
fn numeric_array_holes_and_duplicate_native_indices_are_consumed() {
    let binding = binding(0, 3);
    let mut blocks = vec![(4, "@10".into()), (5, "@10".into()), (9, "@12".into())];
    assert_eq!(
        block_indices(&binding, Some(10), &mut blocks).unwrap(),
        vec![(0, vec![4, 5]), (2, vec![9])]
    );
    assert!(blocks.is_empty());
}
#[test]
fn array_span_does_not_capture_next_binding() {
    let layout = GlPipelineLayout {
        bindings: vec![binding(0, 3), binding(1, 1)],
    };
    assert_eq!(
        at(&layout, GlShaderResourceKind::UniformBuffer, 2)
            .unwrap()
            .unwrap()
            .1,
        2
    );
    let next = at(&layout, GlShaderResourceKind::UniformBuffer, 3)
        .unwrap()
        .unwrap();
    assert_eq!(next.0.location.binding, 1);
    assert_eq!(next.1, 0);
}
#[test]
fn resource_namespaces_do_not_consume_each_others_native_slots() {
    let mut image = binding(1, 1);
    image.kind =
        GlShaderResourceKind::StorageImage(crate::backend::api::GlStorageImageAccess::WriteOnly);
    let layout = GlPipelineLayout {
        bindings: vec![binding(0, 8), image],
    };
    assert_eq!(
        at(
            &layout,
            GlShaderResourceKind::StorageImage(
                crate::backend::api::GlStorageImageAccess::WriteOnly
            ),
            0
        )
        .unwrap()
        .unwrap()
        .0
        .location
        .binding,
        1
    );
}
