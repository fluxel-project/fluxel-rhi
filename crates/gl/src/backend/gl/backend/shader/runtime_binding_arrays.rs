//! Bounded runtime binding arrays specialized to the pipeline layout capacity.
//!
//! Each global gets its own interned type: Naga may share the original dynamic
//! type between bindings with different capacities. Core validates each bound
//! group independently and forbids shaders from accessing absent elements.

use super::super::types::GlPipelineLayout;
use fluxel_rhi_core::api::{BindingCount, Error, Result};

pub(super) fn specialize(
    module: &mut naga::Module,
    layout: Option<&GlPipelineLayout>,
) -> Result<()> {
    specialize_with(module, |binding| {
        let entry = layout
            .and_then(|layout| layout.groups.get(binding.group as usize))
            .and_then(|group| {
                group
                    .entries
                    .iter()
                    .find(|entry| entry.binding == binding.binding)
            })
            .ok_or(Error::InvalidState)?;
        match entry.count {
            BindingCount::RuntimeSized { max_count } => Ok(max_count),
            _ => Err(Error::InvalidState),
        }
    })
}

fn specialize_with(
    module: &mut naga::Module,
    mut capacity: impl FnMut(&naga::ResourceBinding) -> Result<u32>,
) -> Result<()> {
    let arrays: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| match module.types[global.ty].inner {
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Dynamic,
            } => Some((handle, base, global.binding.clone())),
            _ => None,
        })
        .collect();
    for (handle, base, binding) in arrays {
        let binding = binding.ok_or(Error::InvalidState)?;
        let max_count = capacity(&binding)?;
        let count = std::num::NonZeroU32::new(max_count).ok_or(Error::InvalidRange)?;
        let ty = naga::Type {
            name: None,
            inner: naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(count),
            },
        };
        let new_ty = module.types.insert(ty, naga::Span::UNDEFINED);
        module.global_variables[handle].ty = new_ty;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
            crate::backend::gl::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
    }
}
