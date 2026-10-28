use super::*;

#[test]
fn distinct_capacities_on_shared_dynamic_type() {
    let source = "@group(0) @binding(0) var a: binding_array<texture_2d<f32>>; @group(1) @binding(2) var b: binding_array<texture_2d<f32>>; @fragment fn main() -> @location(0) vec4f { return textureLoad(a[1], vec2i(0), 0) + textureLoad(b[2], vec2i(0), 0); }";
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    specialize_with(&mut module, |binding| {
        Ok(if binding.group == 0 { 2 } else { 3 })
    })
    .unwrap();
    let sizes: Vec<_> = module
        .global_variables
        .iter()
        .map(|(_, global)| match module.types[global.ty].inner {
            naga::TypeInner::BindingArray {
                size: naga::ArraySize::Constant(n),
                ..
            } => n.get(),
            _ => panic!("array was not specialized"),
        })
        .collect();
    assert_eq!(sizes, [2, 3]);
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
#[test]
fn runtime_sample_array_reaches_existing_gl_lowering() {
    let source = "@group(0) @binding(0) var images: binding_array<texture_2d<f32>>; @group(1) @binding(0) var sm: binding_array<sampler>; @fragment fn main(@location(0) index: u32) -> @location(0) vec4f { return textureSampleLevel(images[index], sm[index], vec2f(0.5), 0.0); }";
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let (module, _) = naga::back::pipeline_constants::process_overrides(
        &module,
        &info,
        Some((naga::ShaderStage::Fragment, "main")),
        &Default::default(),
    )
    .unwrap();
    let mut module = module.into_owned();
    specialize_with(&mut module, |_| Ok(2)).unwrap();
    let mut aliases = std::collections::HashMap::new();
    super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
    super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 4).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(aliases.len(), 4);
}

#[test]
fn rejects_missing_or_zero_capacity() {
    let source = "@group(0) @binding(0) var a: binding_array<texture_2d<f32>>;";
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    assert!(specialize_with(&mut module, |_| Err(Error::InvalidState)).is_err());
    assert!(specialize_with(&mut module, |_| Ok(0)).is_err());
}
#[test]
fn fixed_array_shared_sample_and_query_are_lowered() {
    use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
    let source = b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>,2>; @group(1) @binding(0) var sm: sampler; @fragment fn main(@location(0) index: u32) -> @location(0) vec4f { let image = images[index]; let dims = textureDimensions(image); return textureSampleLevel(image, sm, vec2f(0.5), 0.0) + vec4f(f32(dims.x)); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: std::sync::Arc::from(source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    super::super::wgsl::lower_wgsl(
        crate::backend::profile::GlProfile::Desktop { major: 4, minor: 6 },
        &desc,
    )
    .unwrap();
}
