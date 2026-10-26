//! Unit tests for the corresponding implementation module.

use super::*;

fn entry(stage: api::ShaderStage) -> api::ShaderEntryPoint {
    api::ShaderEntryPoint {
        name: "main".into(),
        stage,
        workgroup_size: None,
    }
}

#[test]
fn parses_all_core_subgroup_capability_families() {
    let mut words = vec![0x0723_0203, 0, 0, 0, 0];
    for capability in 61..=68 {
        words.extend([2 << 16 | OP_CAPABILITY, capability]);
    }
    assert_eq!(
        required_operations(&words).unwrap(),
        vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::VOTE
            | vk::SubgroupFeatureFlags::ARITHMETIC
            | vk::SubgroupFeatureFlags::BALLOT
            | vk::SubgroupFeatureFlags::SHUFFLE
            | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE
            | vk::SubgroupFeatureFlags::CLUSTERED
            | vk::SubgroupFeatureFlags::QUAD
    );
}

#[test]
fn rejects_artifact_operation_not_reported_by_device() {
    let entries = [entry(api::ShaderStage::Compute)];
    let words = [0x0723_0203, 0, 0, 0, 0, 2 << 16 | OP_CAPABILITY, 63];
    let error = validate(
        &entries,
        vk::ShaderStageFlags::COMPUTE,
        vk::SubgroupFeatureFlags::BASIC,
        true,
        required_operations(&words).unwrap(),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Unsupported("Vulkan subgroups")));
}

#[test]
fn limited_quad_support_admits_compute_and_fragment_entries() {
    let entries = [
        entry(api::ShaderStage::Compute),
        entry(api::ShaderStage::Fragment),
    ];
    validate(
        &entries,
        vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT,
        vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::QUAD,
        false,
        vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::QUAD,
    )
    .unwrap();
}

#[test]
fn limited_quad_support_rejects_vertex() {
    let error = validate(
        &[entry(api::ShaderStage::Vertex)],
        vk::ShaderStageFlags::VERTEX,
        vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::QUAD,
        false,
        vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::QUAD,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Unsupported("Vulkan subgroup quad operations")
    ));
}
