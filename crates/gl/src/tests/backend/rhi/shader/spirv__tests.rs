
use super::*;
use crate::backend::api::GlSpirvSource;
#[test]
fn fragment_stub_and_nonzero_sets_after_entry_are_checked() {
    let mut words = empty_fragment().words;
    validate_entry(&words, "main", ShaderStage::Fragment).unwrap();
    assert!(validate_entry(&words, "main", ShaderStage::Vertex).is_err());
    words.splice(18..18, [0x0004_0047, 1, 34, 2]);
    assert!(
        matches!(validate_entry(&words, "main", ShaderStage::Fragment), Err(Error::Shader(message)) if message.contains("DescriptorSet"))
    );
}
#[test]
fn entry_and_numeric_specialization_are_validated() {
    let words = vec![
        0x0723_0203,
        0x10000,
        0,
        8,
        0,
        (5 << 16) | 15,
        4,
        3,
        0x6e69616d,
        0,
        (3 << 16) | 22,
        1,
        32,
        (4 << 16) | 50,
        1,
        4,
        0,
        (4 << 16) | 71,
        4,
        1,
        7,
    ];
    validate_entry(&words, "main", ShaderStage::Fragment).unwrap();
    let mut source = GlShaderSource {
        stage: crate::backend::api::GlShaderStage::Fragment,
        dialect: crate::backend::api::GlShaderDialect::Desktop { version: 400 },
        entry_point: "main".into(),
        source_hash: crate::backend::api::GlShaderKey([0; 32]),
        text: String::new(),
        spirv: Some(GlSpirvSource {
            words,
            specialization: vec![],
        }),
        debug_name: None,
    };
    specialize(&mut source, &[("7".into(), SpecializationValue::F32(1.0))]).unwrap();
    assert_eq!(
        source.spirv.unwrap().specialization,
        vec![(7, 1.0f32.to_bits())]
    );
}
