use super::*;
use crate::backend::api::{GlShaderDialect, GlShaderKey, GlShaderStage, GlSpirvSource};

fn instruction(op: u32, operands: &[u32]) -> Vec<u32> {
    let mut out = vec![((operands.len() as u32 + 1) << 16) | op];
    out.extend_from_slice(operands);
    out
}
fn source(mut body: Vec<u32>, specs: Vec<(u32, u32)>) -> GlShaderSource {
    let mut words = vec![MAGIC, 0x10000, 0, 32, 0];
    words.append(&mut body);
    GlShaderSource {
        stage: GlShaderStage::Fragment,
        dialect: GlShaderDialect::Desktop { version: 460 },
        entry_point: "main".into(),
        source_hash: GlShaderKey([0; 32]),
        text: String::new(),
        spirv: Some(GlSpirvSource {
            words,
            specialization: specs,
        }),
        debug_name: None,
    }
}
fn sampler(variable: u32, binding: u32, location: u32) -> Vec<u32> {
    let mut v = Vec::new();
    v.extend(instruction(22, &[1, 32]));
    v.extend(instruction(25, &[2, 1, 1, 0, 0, 0, 1, 0]));
    v.extend(instruction(27, &[3, 2]));
    v.extend(instruction(32, &[4, 0, 3]));
    v.extend(instruction(59, &[4, variable, 0]));
    v.extend(instruction(71, &[variable, 33, binding]));
    v.extend(instruction(71, &[variable, 30, location]));
    v
}
#[test]
fn opaque_uniforms_sort_binding_not_declaration_order() {
    let mut body = sampler(10, 4, 8);
    body.extend(sampler(11, 1, 2));
    let got = uniforms([&source(body, vec![])]).unwrap();
    assert_eq!(
        got.iter().map(|u| u.binding).collect::<Vec<_>>(),
        vec![1, 4]
    );
}
#[test]
fn ubo_before_sampler_is_not_opaque() {
    let mut body = instruction(32, &[7, 2, 1]);
    body.extend(instruction(59, &[7, 9, 2]));
    body.extend(sampler(10, 1, 0));
    assert_eq!(uniforms([&source(body, vec![])]).unwrap().len(), 1);
}
#[test]
fn storage_image_is_distinct_from_sampler() {
    let mut body = Vec::new();
    body.extend(instruction(22, &[1, 32]));
    body.extend(instruction(25, &[2, 1, 1, 0, 0, 0, 2, 0]));
    body.extend(instruction(32, &[3, 0, 2]));
    body.extend(instruction(59, &[3, 4, 0]));
    body.extend(instruction(71, &[4, 33, 3]));
    body.extend(instruction(71, &[4, 30, 7]));
    let got = uniforms([&source(body, vec![])]).unwrap();
    assert_eq!(got[0].uniform_type, 0x904d);
    assert!(got[0].image);
}
#[test]
fn specialization_controls_array_length() {
    // Replace the variable pointer with an array pointer in a fresh module.
    let mut a = Vec::new();
    a.extend(instruction(22, &[1, 32]));
    a.extend(instruction(25, &[2, 1, 1, 0, 0, 0, 1, 0]));
    a.extend(instruction(27, &[3, 2]));
    a.extend(instruction(21, &[4, 32, 0]));
    a.extend(instruction(50, &[4, 5, 2]));
    a.extend(instruction(71, &[5, 1, 9]));
    a.extend(instruction(28, &[6, 3, 5]));
    a.extend(instruction(32, &[7, 0, 6]));
    a.extend(instruction(59, &[7, 8, 0]));
    a.extend(instruction(71, &[8, 33, 1]));
    a.extend(instruction(71, &[8, 30, 0]));
    assert_eq!(
        uniforms([&source(a, vec![(9, 6)])]).unwrap()[0].array_count,
        6
    );
}
