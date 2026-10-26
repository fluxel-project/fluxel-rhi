//! Unit tests for the corresponding implementation module.

use super::{route, validate_entry_points};
use fluxel_rhi_core::api::{self, ShaderDesc, ShaderEntryPoint, ShaderFormat};
use std::sync::Arc;

#[test]
fn metallib_has_a_direct_route_without_changing_source_routes() {
    let mut desc = ShaderDesc {
        label: None,
        format: ShaderFormat::MetalLibrary,
        bytes: Arc::from([1]),
        entries: vec![],
        required_features: vec![],
    };
    assert!(matches!(route(&desc), api::ShaderRoute::Direct(target)
        if target.format == ShaderFormat::MetalLibrary && target.abi_key == "fluxel-metal-stage-slots-v2"));
    desc.format = ShaderFormat::MetalSource;
    assert!(matches!(route(&desc), api::ShaderRoute::Direct(_)));
    desc.format = ShaderFormat::Wgsl;
    assert!(matches!(route(&desc), api::ShaderRoute::Naga(_)));
    desc.format = ShaderFormat::Dxil;
    assert!(matches!(route(&desc), api::ShaderRoute::Unsupported));
}

#[test]
fn wgsl_entry_point_stage_must_match_core_descriptor() {
    let source = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: "vs".into(),
            stage: api::ShaderStage::Vertex,
            workgroup_size: None,
        }],
        required_features: Vec::new(),
    };
    assert!(validate_entry_points(&module, &desc.entries).is_ok());
    let mut mismatch = desc;
    mismatch.entries[0].stage = api::ShaderStage::Fragment;
    assert!(matches!(
        validate_entry_points(&module, &mismatch.entries),
        Err(api::Error::Shader(_))
    ));
}
