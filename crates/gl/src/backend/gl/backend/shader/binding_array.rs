//! Fixed resource-array specialization for the GL shader compiler.
//!
//! The GLSL writer cannot emit Naga binding arrays. Each statically selected
//! element can instead use an ordinary resource global. The returned metadata
//! retains its original logical binding and element; synthetic binding numbers
//! are compiler-private and must never become Core binding numbers.

use std::collections::HashMap;

use fluxel_rhi_core::api::{Error, Result};
use naga::{Expression, GlobalVariable, Handle};

type Global = Handle<GlobalVariable>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Element {
    pub original: Global,
    pub index: u32,
}

/// Splits statically indexed accesses. Dynamic accesses remain untouched for a
/// later control-flow lowering pass; callers must not publish this partial
/// transformation as support for binding arrays.
pub(super) fn split_static(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
) -> Result<()> {
    let arrays: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, value)| match module.types[value.ty].inner {
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(count),
            } => Some((handle, base, count.get())),
            _ => None,
        })
        .collect();
    if arrays.is_empty() {
        return Ok(());
    }
    let constant_indices: HashMap<_, _> = module
        .constants
        .iter()
        .filter_map(|(handle, constant)| {
            let value = match module.global_expressions[constant.init] {
                Expression::Literal(naga::Literal::U32(value)) => value,
                Expression::Literal(naga::Literal::I32(value)) if value >= 0 => value as u32,
                _ => return None,
            };
            Some((handle, value))
        })
        .collect();
    let mut selected = std::collections::BTreeSet::new();
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, expression) in function.expressions.iter() {
            let (base, index) = match *expression {
                Expression::AccessIndex { base, index } => (base, index),
                Expression::Access { base, index } => {
                    let index = match function.expressions[index] {
                        Expression::Literal(naga::Literal::U32(value)) => value,
                        Expression::Literal(naga::Literal::I32(value)) if value >= 0 => {
                            value as u32
                        }
                        Expression::Constant(handle) => match constant_indices.get(&handle) {
                            Some(value) => *value,
                            None => continue,
                        },
                        _ => continue,
                    };
                    (base, index)
                }
                _ => continue,
            };
            if let Expression::GlobalVariable(global) = function.expressions[base] {
                if arrays
                    .iter()
                    .any(|(original, _, count)| *original == global && index < *count)
                {
                    selected.insert((global, index));
                }
            }
        }
    }
    let elements = ensure_elements(module, aliases, &selected)?;
    for function in module.functions.iter_mut().map(|(_, f)| f).chain(
        module
            .entry_points
            .iter_mut()
            .map(|entry| &mut entry.function),
    ) {
        let rewrites: Vec<_> = function
            .expressions
            .iter()
            .filter_map(|(handle, expression)| {
                let (base, index) = match *expression {
                    Expression::AccessIndex { base, index } => (base, index),
                    Expression::Access { base, index } => {
                        let index = match function.expressions[index] {
                            Expression::Literal(naga::Literal::U32(value)) => value,
                            Expression::Literal(naga::Literal::I32(value)) if value >= 0 => {
                                value as u32
                            }
                            Expression::Constant(handle) => *constant_indices.get(&handle)?,
                            _ => return None,
                        };
                        (base, index)
                    }
                    _ => return None,
                };
                let Expression::GlobalVariable(original) = function.expressions[base] else {
                    return None;
                };
                elements
                    .get(&(original, index))
                    .map(|synthetic| (handle, *synthetic))
            })
            .collect();
        let rewritten: std::collections::HashSet<_> =
            rewrites.iter().map(|(handle, _)| *handle).collect();
        for (handle, synthetic) in rewrites {
            function.expressions[handle] = Expression::GlobalVariable(synthetic);
        }
        remove_global_emits(&mut function.body, &rewritten);
    }
    Ok(())
}

/// Returns one canonical synthetic global for each original element.
pub(super) fn ensure_elements(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
    selected: &std::collections::BTreeSet<(Global, u32)>,
) -> Result<HashMap<(Global, u32), Global>> {
    let mut elements: HashMap<_, _> = aliases
        .iter()
        .map(|(&alias, element)| ((element.original, element.index), alias))
        .collect();
    let missing: Vec<_> = selected
        .iter()
        .filter(|key| !elements.contains_key(key))
        .copied()
        .collect();
    if missing.is_empty() {
        return Ok(elements);
    }
    let group = module
        .global_variables
        .iter()
        .filter_map(|(_, g)| g.binding.as_ref().map(|b| b.group))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(Error::InvalidRange)?;
    for (original, index) in missing {
        let naga::TypeInner::BindingArray {
            base,
            size: naga::ArraySize::Constant(count),
        } = module.types[module.global_variables[original].ty].inner
        else {
            return Err(Error::InvalidRange);
        };
        if index >= count.get() {
            return Err(Error::InvalidRange);
        }
        let mut value = module.global_variables[original].clone();
        value.ty = base;
        value.name = Some(format!(
            "fluxel_binding_array_{}_{}",
            original.index(),
            index
        ));
        value.binding = Some(naga::ResourceBinding {
            group,
            binding: u32::try_from(aliases.len()).map_err(|_| Error::InvalidRange)?,
        });
        let alias = module
            .global_variables
            .append(value, module.global_variables.get_span(original));
        aliases.insert(alias, Element { original, index });
        elements.insert((original, index), alias);
    }
    Ok(elements)
}

fn remove_global_emits(
    block: &mut naga::Block,
    rewritten: &std::collections::HashSet<Handle<Expression>>,
) {
    let old = std::mem::take(block);
    for (mut statement, span) in old.span_into_iter() {
        match &mut statement {
            naga::Statement::Emit(range) => {
                let mut first = None;
                let mut last = None;
                for handle in range.clone() {
                    if rewritten.contains(&handle) {
                        if let (Some(start), Some(end)) = (first.take(), last.take()) {
                            block.push(
                                naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                                span,
                            );
                        }
                    } else {
                        first.get_or_insert(handle);
                        last = Some(handle);
                    }
                }
                if let (Some(start), Some(end)) = (first, last) {
                    block.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(start, end)),
                        span,
                    );
                }
                continue;
            }
            naga::Statement::Block(inner) => remove_global_emits(inner, rewritten),
            naga::Statement::If { accept, reject, .. } => {
                remove_global_emits(accept, rewritten);
                remove_global_emits(reject, rewritten);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    remove_global_emits(&mut case.body, rewritten);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                remove_global_emits(body, rewritten);
                remove_global_emits(continuing, rewritten);
            }
            _ => {}
        }
        block.push(statement, span);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_texture_element_remains_valid_naga_ir() {
        let source = r#"
@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@fragment fn main() -> @location(0) vec4f {
    return textureLoad(images[1], vec2i(0), 0);
}
"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        split_static(&mut module, &mut aliases).unwrap();
        assert_eq!(aliases.len(), 1);
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Desktop(450),
            writer_flags: naga::back::glsl::WriterFlags::empty(),
            binding_map: Default::default(),
            zero_initialize_workgroup_memory: true,
        };
        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut text = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut text,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap();
        writer.write().unwrap();
        assert!(text.contains("texelFetch"));
    }

    #[test]
    fn reflection_preserves_original_binding_and_selected_element() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"const SELECTED: u32 = 1u; @group(1) @binding(3) var images: binding_array<texture_2d<f32>, 2>; @fragment fn main() -> @location(0) vec4f { return textureLoad(images[SELECTED], vec2i(0), 0); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        let (_, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::gl::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(bindings.iter().any(|binding| binding.group == 1
            && binding.binding == 3
            && binding.element == Some(1)));
    }

    #[test]
    fn separate_array_elements_keep_their_sampler_pairs() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>; @group(1) @binding(0) var samplers: binding_array<sampler, 2>; @fragment fn main() -> @location(0) vec4f { return textureSample(images[0], samplers[1], vec2f(0.5)) + textureSample(images[1], samplers[0], vec2f(0.5)); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        let (_, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::gl::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        for (texture, sampler) in [(0, 1), (1, 0)] {
            let image = bindings
                .iter()
                .find(|v| v.group == 0 && v.element == Some(texture))
                .unwrap();
            assert!(
                bindings
                    .iter()
                    .any(|v| v.group == 1 && v.element == Some(sampler) && v.name == image.name)
            );
        }
    }

    #[test]
    fn constant_array_element_can_be_passed_to_helper() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>; fn fetch(image: texture_2d<f32>) -> vec4f { return textureLoad(image, vec2i(0), 0); } @fragment fn main() -> @location(0) vec4f { return fetch(images[1]); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        let (_, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::gl::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(
            bindings
                .iter()
                .any(|v| v.binding == 0 && v.element == Some(1))
        );
    }

    #[test]
    fn storage_buffer_array_element_keeps_store_side_effect() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"struct Data { value: u32 }; @group(0) @binding(0) var<storage, read_write> buffers: binding_array<Data, 2>; @compute @workgroup_size(1) fn main() { buffers[1].value = 42u; }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: Vec::new(),
        };
        let (glsl, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::gl::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(
            bindings
                .iter()
                .any(|binding| binding.binding == 0 && binding.element == Some(1))
        );
        assert!(glsl.text.contains("42u"));
    }

    #[test]
    fn storage_image_array_element_keeps_store_and_unit() {
        use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
        let source = b"@group(0) @binding(2) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>; @compute @workgroup_size(1) fn main() { textureStore(images[1], vec2i(0), vec4f(1.)); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: std::sync::Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: Vec::new(),
        };
        let (glsl, bindings, _) = super::super::wgsl::lower_wgsl(
            crate::backend::gl::profile::GlProfile::Desktop { major: 4, minor: 6 },
            &desc,
        )
        .unwrap();
        assert!(bindings.iter().any(|binding| binding.binding == 2
            && binding.element == Some(1)
            && binding.image_unit.is_some()));
        assert!(glsl.text.contains("imageStore"));
    }
}
