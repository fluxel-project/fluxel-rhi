//! Admission boundaries for the fixed compute storage-array capability.

use super::{api, layout::validate_nonuniform_buffer_array};

#[test]
fn nonuniform_storage_array_layout_rejects_unproven_combinations() {
    let base = api::BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![api::ShaderStage::Compute],
        ty: api::BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size: 32,
            access: api::StorageAccess::Read,
        },
        count: api::BindingCount::Fixed(2),
        partially_bound: false,
        nonuniform_indexing: true,
    };
    validate_nonuniform_buffer_array(&base).unwrap();
    for entry in [
        api::BindGroupLayoutEntry {
            count: api::BindingCount::RuntimeSized { max_count: 2 },
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            partially_bound: true,
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            visibility: vec![api::ShaderStage::Fragment],
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            visibility: vec![api::ShaderStage::Compute, api::ShaderStage::Vertex],
            ..base.clone()
        },
    ] {
        assert!(matches!(
            validate_nonuniform_buffer_array(&entry),
            Err(api::Error::Unsupported(_))
        ));
    }
    // Native uniform indexing already has its own partial/runtime contracts.
    // The new non-uniform route must not silently remove those contracts.
    let native_uniform = api::BindGroupLayoutEntry {
        nonuniform_indexing: false,
        partially_bound: true,
        count: api::BindingCount::RuntimeSized { max_count: 2 },
        ..base
    };
    validate_nonuniform_buffer_array(&native_uniform).unwrap();
}
