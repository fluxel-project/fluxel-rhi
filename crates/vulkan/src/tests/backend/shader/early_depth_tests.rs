//! Unit tests for Direct SPIR-V early-depth admission.

use super::*;

fn instruction(opcode: u32, operands: &[u32]) -> Vec<u32> {
    let count = u32::try_from(operands.len() + 1).unwrap();
    let mut words = vec![(count << 16) | opcode];
    words.extend_from_slice(operands);
    words
}

fn module(with_early_fragment_tests: bool) -> Vec<u32> {
    let mut words = vec![0x0723_0203, 0x0001_0000, 0, 16, 0];
    // Fragment %7 "main".
    words.extend(instruction(15, &[4, 7, 0x6e69_616d, 0]));
    if with_early_fragment_tests {
        words.extend(instruction(16, &[7, 9]));
    }
    words
}

fn module_with_undeclared_early_fragment_entry() -> Vec<u32> {
    let mut words = module(false);
    // Fragment %8 "other", which is intentionally absent from ShaderDesc.
    words.extend(instruction(15, &[4, 8, 0x6568_746f, 0x0000_0072]));
    words.extend(instruction(16, &[8, 9]));
    words
}

fn desc(name: &str, declares_early_depth: bool) -> api::ShaderDesc {
    api::ShaderDesc {
        label: None,
        format: api::ShaderFormat::SpirV,
        bytes: std::sync::Arc::from([]),
        entries: vec![api::ShaderEntryPoint {
            name: name.into(),
            stage: api::ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: declares_early_depth
            .then(|| api::SHADER_FEATURE_EARLY_DEPTH_TEST.into())
            .into_iter()
            .collect(),
    }
}

#[test]
fn admits_fragment_entry_with_early_fragment_tests_execution_mode() {
    assert!(validate_early_depth_route(&desc("main", true), &module(true)).is_ok());
}

#[test]
fn rejects_fragment_entry_without_early_fragment_tests_execution_mode() {
    assert!(matches!(
        validate_early_depth_route(&desc("main", true), &module(false)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_declared_fragment_entry_missing_from_spirv() {
    assert!(matches!(
        validate_early_depth_route(&desc("missing", true), &module(true)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_malformed_spirv_before_native_module_creation() {
    let mut words = module(true);
    words.push(16);
    assert!(matches!(
        validate_early_depth_route(&desc("main", true), &words),
        Err(Error::Shader(_))
    ));
}

#[test]
fn rejects_early_fragment_tests_without_the_required_feature() {
    assert!(matches!(
        validate_early_depth_route(&desc("main", false), &module(true)),
        Err(Error::Shader(_))
    ));
}

#[test]
fn admits_an_ordinary_fragment_without_early_depth_feature() {
    assert!(validate_early_depth_route(&desc("main", false), &module(false)).is_ok());
}

#[test]
fn rejects_an_undeclared_fragment_entry_using_early_fragment_tests_without_feature() {
    assert!(matches!(
        validate_early_depth_route(
            &desc("main", false),
            &module_with_undeclared_early_fragment_entry()
        ),
        Err(Error::Shader(_))
    ));
}
