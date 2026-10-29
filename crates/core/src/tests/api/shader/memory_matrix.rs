//! SPIR-V memory-decoration and cooperative-matrix admission checks.
use super::*;

#[test]
fn spirv_memory_decoration_requirements_are_format_stage_and_capability_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |coherent, volatile| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                shader_memory_coherent_spirv: coherent,
                shader_memory_volatile_spirv: volatile,
                ..Default::default()
            },
        )
    };
    let desc = |format, stage, feature: &str| ShaderDesc {
        label: None,
        format,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([37, 1, 1]),
        }],
        required_features: vec![feature.into()],
    };
    assert!(matches!(
        device(false, false).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    let supported = device(true, true);
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderFormat::Wgsl,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Vertex,
            SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        )),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        device(true, false).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        )),
        ShaderRoute::Direct(_)
    ));
    assert!(matches!(
        device(false, true).shader_route(&desc(
            ShaderFormat::SpirV,
            ShaderStage::Compute,
            SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        )),
        ShaderRoute::Direct(_)
    ));
    let empty = ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![],
        required_features: vec![SHADER_FEATURE_MEMORY_COHERENT_SPIRV.into()],
    };
    assert!(matches!(
        supported.shader_route(&empty),
        ShaderRoute::Unsupported
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 2));
}

#[test]
fn cooperative_matrix_shader_requirement_is_compute_only_and_configuration_gated() {
    ROUTES.with(|v| v.set(0));
    let device = |supported| unsafe {
        Device::<Mock>::from_native(
            DeviceId::allocate().unwrap(),
            (),
            DeviceCapabilities {
                cooperative_matrix_configurations: if supported {
                    vec![CooperativeMatrixConfiguration {
                        m: 16,
                        n: 16,
                        k: 16,
                        a_type: CooperativeMatrixComponentType::Float16,
                        b_type: CooperativeMatrixComponentType::Float16,
                        c_type: CooperativeMatrixComponentType::Float32,
                        result_type: CooperativeMatrixComponentType::Float32,
                        saturating_accumulation: false,
                        scope: CooperativeMatrixScope::Subgroup,
                    }]
                } else {
                    vec![]
                },
                ..Default::default()
            },
        )
    };
    let desc = |stage| ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from([0u8; 20].as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: Some([64, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX.into()],
    };
    assert!(matches!(
        device(false).shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Unsupported
    ));
    let supported = device(true);
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Fragment)),
        ShaderRoute::Unsupported
    ));
    assert!(matches!(
        supported.shader_route(&desc(ShaderStage::Compute)),
        ShaderRoute::Direct(_)
    ));
    ROUTES.with(|v| assert_eq!(v.get(), 1));
}
