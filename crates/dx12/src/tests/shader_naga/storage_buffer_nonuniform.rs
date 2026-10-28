//! Keep descriptor selection separate from byte addressing in raw buffer arrays.

use super::*;
use std::sync::Arc;

const READ: &str = include_str!("../../../tests/assets/storage_buffer_nonuniform_read.wgsl");
const WRITE: &str = include_str!("../../../tests/assets/storage_buffer_nonuniform_write.wgsl");

fn desc(source: &str) -> api::ShaderDesc {
    api::ShaderDesc {
        label: Some("storage buffer array lowering".into()),
        format: api::ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![api::ShaderEntryPoint {
            name: "main".into(),
            stage: api::ShaderStage::Compute,
            workgroup_size: Some([2, 1, 1]),
        }],
        required_features: vec![],
    }
}

fn lower(source: &str, access: api::StorageAccess) -> CompiledEntry {
    lower_checked(source, access, layout_entry(access)).unwrap()
}

fn layout_entry(access: api::StorageAccess) -> api::BindGroupLayoutEntry {
    api::BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![api::ShaderStage::Compute],
        ty: api::BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size: 32,
            access,
        },
        count: api::BindingCount::Fixed(2),
        partially_bound: false,
        nonuniform_indexing: true,
    }
}

fn lower_checked(
    source: &str,
    access: api::StorageAccess,
    entry: api::BindGroupLayoutEntry,
) -> Result<CompiledEntry> {
    let desc = desc(source);
    let class = if access == api::StorageAccess::Read {
        1
    } else {
        2
    };
    compile_variant_with_typed_map(
        &desc,
        &desc.entries[0],
        &[],
        &[(0, 0, 0, class, 2), (0, 2, 2, 2, 1)],
        &[
            (
                0,
                0,
                api::BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 32,
                    access,
                },
            ),
            (
                0,
                2,
                api::BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 8,
                    access: api::StorageAccess::ReadWrite,
                },
            ),
        ],
        &[],
        &[(0, entry)],
        None,
    )
}

#[test]
fn storage_binding_array_ir_separates_descriptor_from_member_access() {
    let desc = desc(READ);
    let module = naga::front::wgsl::parse_str(READ).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(&desc),
    )
    .validate(&module)
    .unwrap();
    let function = &module.entry_points[0].function;
    let function_info = info.get_entry_point(0);
    let mut found = false;
    for (_, expression) in function.expressions.iter() {
        if let naga::Expression::Access { base, .. } = *expression {
            let ty = function_info[base].ty.inner_with(&module.types);
            if let naga::TypeInner::Pointer { base, .. } = *ty {
                if matches!(
                    module.types[base].inner,
                    naga::TypeInner::BindingArray { .. }
                ) {
                    eprintln!("descriptor access parent: {:?}", module.types[base].inner);
                    found = true;
                }
            }
        }
    }
    assert!(
        found,
        "the regression must exercise the missing BindingArray pointer branch"
    );
}

#[test]
fn storage_array_read_emits_nonuniform_selector_and_nonzero_member_offset() {
    let compiled = lower(READ, api::StorageAccess::Read);
    eprintln!("{}", compiled.hlsl);
    assert!(
        compiled
            .hlsl
            .contains("values[NonUniformResourceIndex(i)].Load(0+16)")
    );
    assert!(compiled.hlsl.contains("register(t0"));
    assert!(compiled.hlsl.contains("register(u2"));
    assert!(!compiled.bytes.is_empty());
}

#[test]
fn storage_array_whole_struct_load_preserves_descriptor_selection() {
    let source = READ.replace(
        "output[i] = values[i].value.x;",
        "let item = values[i]; output[i] = item.value.x;",
    );
    let compiled = lower(&source, api::StorageAccess::Read);
    assert!(compiled.hlsl.contains("NonUniformResourceIndex"));
    assert!(!compiled.bytes.is_empty());
}

#[test]
fn storage_array_write_and_atomic_use_selected_buffer() {
    let compiled = lower(WRITE, api::StorageAccess::ReadWrite);
    eprintln!("{}", compiled.hlsl);
    assert!(compiled.hlsl.contains("NonUniformResourceIndex"));
    assert!(
        compiled
            .hlsl
            .contains("values[NonUniformResourceIndex(i)].Store(20,")
    );
    assert!(
        compiled
            .hlsl
            .contains("values[NonUniformResourceIndex(i)].InterlockedAdd(24,")
    );
    assert!(compiled.hlsl.contains("register(u0"));
    assert!(compiled.hlsl.contains("register(u2"));
    assert!(!compiled.bytes.is_empty());
}

#[test]
fn storage_array_static_selector_does_not_become_nonuniform() {
    let source = READ.replace("values[i].value.x", "values[1].value.x");
    let compiled = lower(&source, api::StorageAccess::Read);
    assert!(!compiled.hlsl.contains("NonUniformResourceIndex"));
    assert!(!compiled.bytes.is_empty());
}

#[test]
fn storage_array_runtime_and_length_queries_reject_before_writer() {
    for source in [
        READ.replace("binding_array<Value, 2>", "binding_array<Value>"),
        "struct Value { data: array<u32> };
         @group(0) @binding(0) var<storage, read> values: binding_array<Value, 2>;
         @group(0) @binding(2) var<storage, read_write> output: array<u32, 2>;
         @compute @workgroup_size(2) fn main(@builtin(local_invocation_index) i: u32) {
             output[i] = arrayLength(&values[i].data);
         }"
        .into(),
    ] {
        assert!(matches!(
            validate_source(&desc(&source)),
            Err(api::Error::Unsupported(_))
        ));
    }
}

#[test]
fn fixed_storage_shader_cannot_bypass_layout_provenance() {
    let base = layout_entry(api::StorageAccess::Read);
    for entry in [
        api::BindGroupLayoutEntry {
            nonuniform_indexing: false,
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            count: api::BindingCount::RuntimeSized { max_count: 2 },
            nonuniform_indexing: false,
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            partially_bound: true,
            nonuniform_indexing: false,
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            count: api::BindingCount::Fixed(3),
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            visibility: vec![api::ShaderStage::Fragment],
            ..base.clone()
        },
        api::BindGroupLayoutEntry {
            ty: api::BindingType::StorageBuffer {
                dynamic_offset: true,
                min_size: 32,
                access: api::StorageAccess::Read,
            },
            ..base
        },
    ] {
        assert!(matches!(
            lower_checked(READ, api::StorageAccess::Read, entry),
            Err(api::Error::Unsupported(_))
        ));
    }
}
