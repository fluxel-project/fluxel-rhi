//! Unit tests for feature-specific Naga admission.

use super::texture_atomic_only;

fn module(source: &str) -> naga::Module {
    naga::front::wgsl::parse_str(source).unwrap()
}

#[test]
fn texture_atomic_add_domain_rejects_other_image_atomic_operations() {
    let add = module(
        "@group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>; @compute @workgroup_size(1) fn main() { textureAtomicAdd(image, vec2i(0), 1u); }",
    );
    assert!(texture_atomic_only(
        &add,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));

    let max = module(
        "@group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>; @compute @workgroup_size(1) fn main() { if (true) { textureAtomicMax(image, vec2i(0), 1u); } }",
    );
    assert!(!texture_atomic_only(
        &max,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));
    assert!(texture_atomic_only(
        &max,
        naga::AtomicFunction::Max,
        naga::StorageFormat::R32Uint
    ));

    let min = module(
        "@group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>; @compute @workgroup_size(1) fn main() { textureAtomicMin(image, vec2i(0), 1u); }",
    );
    assert!(texture_atomic_only(
        &min,
        naga::AtomicFunction::Min,
        naga::StorageFormat::R32Uint
    ));
    assert!(!texture_atomic_only(
        &min,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));

    let signed_add = module(
        "@group(0) @binding(0) var image: texture_storage_2d<r32sint, atomic>; @compute @workgroup_size(1) fn main() { textureAtomicAdd(image, vec2i(0), 1); }",
    );
    assert!(texture_atomic_only(
        &signed_add,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Sint
    ));
    assert!(!texture_atomic_only(
        &signed_add,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));

    for (name, operation) in [
        ("textureAtomicMin", naga::AtomicFunction::Min),
        ("textureAtomicMax", naga::AtomicFunction::Max),
    ] {
        let source = format!(
            "@group(0) @binding(0) var image: texture_storage_2d<r32sint, atomic>; @compute @workgroup_size(1) fn main() {{ {name}(image, vec2i(0), -1); }}"
        );
        let signed_minmax = module(&source);
        assert!(texture_atomic_only(
            &signed_minmax,
            operation,
            naga::StorageFormat::R32Sint
        ));
        assert!(!texture_atomic_only(
            &signed_minmax,
            operation,
            naga::StorageFormat::R32Uint
        ));
    }

    let xor = module(
        "@group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>; @compute @workgroup_size(1) fn main() { textureAtomicXor(image, vec2i(0), 1u); }",
    );
    assert!(texture_atomic_only(
        &xor,
        naga::AtomicFunction::ExclusiveOr,
        naga::StorageFormat::R32Uint
    ));
    assert!(!texture_atomic_only(
        &xor,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));

    for (source, operation) in [
        ("textureAtomicAnd", naga::AtomicFunction::And),
        ("textureAtomicOr", naga::AtomicFunction::InclusiveOr),
    ] {
        let source = format!(
            "@group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>; @compute @workgroup_size(1) fn main() {{ {source}(image, vec2i(0), 1u); }}"
        );
        let bitwise = module(&source);
        assert!(texture_atomic_only(
            &bitwise,
            operation,
            naga::StorageFormat::R32Uint
        ));
        assert!(!texture_atomic_only(
            &bitwise,
            naga::AtomicFunction::Add,
            naga::StorageFormat::R32Uint
        ));
    }
    assert!(!texture_atomic_only(
        &add,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Sint
    ));

    let mixed_buffer_atomic = module(
        "@group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>; @group(0) @binding(1) var<storage, read_write> values: array<atomic<u32>>; @compute @workgroup_size(1) fn main() { textureAtomicAdd(image, vec2i(0), 1u); atomicAdd(&values[0], 1u); }",
    );
    assert!(!texture_atomic_only(
        &mixed_buffer_atomic,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));

    let mixed_workgroup_atomic = module(
        "@group(0) @binding(0) var image: texture_storage_2d<r32uint, atomic>; var<workgroup> workgroup_value: atomic<u32>; @compute @workgroup_size(1) fn main() { textureAtomicAdd(image, vec2i(0), 1u); atomicAdd(&workgroup_value, 1u); }",
    );
    assert!(!texture_atomic_only(
        &mixed_workgroup_atomic,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));

    let no_image_atomic = module("@compute @workgroup_size(1) fn main() {}");
    assert!(texture_atomic_only(
        &no_image_atomic,
        naga::AtomicFunction::Add,
        naga::StorageFormat::R32Uint
    ));
}
