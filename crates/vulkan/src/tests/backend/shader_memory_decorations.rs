use super::*;

fn instruction(opcode: u32, operands: &[u32]) -> Vec<u32> {
    let count = u32::try_from(operands.len() + 1).unwrap();
    let mut words = vec![(count << 16) | opcode];
    words.extend_from_slice(operands);
    words
}

fn module(memory_model: u32, decorations: &[(u32, u32, Option<u32>, u32)]) -> Vec<u32> {
    module_with_storage_buffers(memory_model, decorations, true, 1)
}

fn module_with_storage_buffers(
    memory_model: u32,
    decorations: &[(u32, u32, Option<u32>, u32)],
    block: bool,
    storage_buffer_count: u32,
) -> Vec<u32> {
    let mut words = vec![0x0723_0203, 0x0001_0000, 0, 16, 0];
    words.extend(instruction(14, &[0, memory_model])); // Logical GLSL450/Vulkan.
    words.extend(instruction(15, &[5, 8, 0x6e69_616d, 0])); // GLCompute `main`.
    words.extend(instruction(30, &[5, 1])); // %5 = OpTypeStruct %uint.
    if block {
        words.extend(instruction(71, &[5, 2])); // %5 = Block StorageBuffer structure.
    }
    words.extend(instruction(32, &[6, 12, 5])); // StorageBuffer pointer.
    for index in 0..storage_buffer_count {
        words.extend(instruction(59, &[6, 7 + index, 12])); // StorageBuffer variable.
    }
    for (opcode, target, member, decoration) in decorations {
        match (*opcode, member) {
            (71, None) => words.extend(instruction(71, &[*target, *decoration])),
            (72, Some(member)) => words.extend(instruction(72, &[*target, *member, *decoration])),
            _ => unreachable!(),
        }
    }
    words
}

fn desc(required_features: &[&str]) -> api::ShaderDesc {
    api::ShaderDesc {
        label: None,
        format: api::ShaderFormat::SpirV,
        bytes: std::sync::Arc::<[u8]>::from([]),
        entries: vec![api::ShaderEntryPoint {
            name: "main".into(),
            stage: api::ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: required_features
            .iter()
            .map(|value| (*value).into())
            .collect(),
    }
}

#[test]
fn admits_declared_coherent_storage_buffer_on_glsl450_model() {
    let words = module(1, &[(71, 7, None, 23), (72, 5, Some(0), 23)]);
    assert!(validate(&desc(&[api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV]), &words).is_ok());
}

#[test]
fn admits_declared_volatile_storage_buffer_on_glsl450_model() {
    let words = module(1, &[(71, 7, None, 21)]);
    let required = api::SHADER_FEATURE_MEMORY_VOLATILE_SPIRV;
    assert!(validate(&desc(&[required]), &words).is_ok());
}

#[test]
fn rejects_memory_decorations_under_vulkan_memory_model() {
    let words = module(3, &[(71, 7, None, 23)]);
    let required = api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn requires_declarations_to_match_storage_buffer_decorations() {
    let words = module(1, &[(71, 7, None, 23)]);
    assert!(matches!(
        validate(&desc(&[]), &words),
        Err(Error::Shader(_))
    ));
    let required = api::SHADER_FEATURE_MEMORY_VOLATILE_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Shader(_))
    ));
}

#[test]
fn rejects_coherent_decorations_on_non_storage_buffer_objects() {
    let mut words = module(1, &[]);
    // Add a decoration to the pointer type instead of its StorageBuffer target.
    words.extend(instruction(71, &[6, 23]));
    let required = api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_coherent_decorations_on_structure_types() {
    let words = module(1, &[(71, 5, None, 23)]);
    let required = api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn requires_block_on_the_storage_buffer_structure() {
    let words = module_with_storage_buffers(1, &[(71, 7, None, 23)], false, 1);
    let required = api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_multiple_storage_buffer_targets() {
    let words = module_with_storage_buffers(1, &[(71, 7, None, 23)], true, 2);
    let required = api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_malformed_extra_storage_buffer_target() {
    let mut words = module(1, &[(71, 7, None, 23)]);
    // %8 is intentionally not a pointer type. It must still count as an
    // attempted StorageBuffer declaration rather than being ignored.
    words.extend(instruction(59, &[8, 9, 12]));
    let required = api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_member_decorations_outside_storage_buffer_struct_bounds() {
    let words = module(1, &[(72, 5, Some(1), 23)]);
    let required = api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV;
    assert!(matches!(
        validate(&desc(&[required]), &words),
        Err(Error::Unsupported(_))
    ));
}
