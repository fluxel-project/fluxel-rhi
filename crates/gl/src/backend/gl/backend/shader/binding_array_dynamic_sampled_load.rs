//! Dynamic sampled-image loads specialized into per-element control flow.
//!
//! The result is captured at the original expression's Emit point. A private
//! local is reset on every evaluation, so an out-of-range selector reads zero
//! even inside loops. Unhandled dynamic accesses remain for the caller to reject.
use super::binding_array::Element;
use fluxel_rhi_core::api::{Error, Result};
use naga::{Expression, GlobalVariable, Handle, Statement};
use std::collections::{BTreeSet, HashMap, HashSet};
#[path = "binding_array_buffer_remap.rs"]
mod remap;
type Global = Handle<GlobalVariable>;
type Expr = Handle<Expression>;

/// Specializes direct dynamic sampled-image loads. The caller must merge alias
/// metadata, validate the result and reject all unlowered dynamic accesses.
/// On error the caller must discard the partially transformed module.
pub(super) fn lower(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
    sampled_limit: u32,
) -> Result<()> {
    let arrays: HashMap<_, _> = module
        .global_variables
        .iter()
        .filter_map(|(h, g)| {
            let naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(n),
            } = module.types[g.ty].inner
            else {
                return None;
            };
            matches!(
                module.types[base].inner,
                naga::TypeInner::Image {
                    class: naga::ImageClass::Sampled { .. } | naga::ImageClass::Depth { .. },
                    ..
                }
            )
            .then_some((h, (base, n.get())))
        })
        .collect();
    if arrays.is_empty() {
        return Ok(());
    }
    let mut used = BTreeSet::new();
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, expr) in function.expressions.iter() {
            if let Expression::ImageLoad { image, .. } = *expr {
                if let Some((g, _)) = selected(image, &function.expressions, &arrays) {
                    used.insert(g);
                }
            }
        }
    }
    if used.is_empty() {
        return Ok(());
    }
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("binding array input validation: {e:?}")))?;
    let mut slots = HashMap::new();
    for (h, f) in module.functions.iter_mut() {
        let count = load_count(f, &arrays);
        slots.insert(h, remap::reserve(f, count));
    }
    let entry_slots: Vec<_> = module
        .entry_points
        .iter_mut()
        .map(|e| {
            let count = load_count(&e.function, &arrays);
            remap::reserve(&mut e.function, count)
        })
        .collect();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("binding array input validation: {e:?}")))?;
    let mut total = 0u32;
    for g in &used {
        let n = arrays[g].1;
        if n > sampled_limit {
            return Err(Error::Unsupported(
                "dynamic sampled image array exceeds specialization limit",
            ));
        }
        total = total.checked_add(n).ok_or(Error::InvalidRange)?;
    }
    if total > sampled_limit {
        return Err(Error::Unsupported(
            "dynamic sampled image arrays exceed total specialization limit",
        ));
    }
    let mut selected = BTreeSet::new();
    for &original in &used {
        for index in 0..arrays[&original].1 {
            selected.insert((original, index));
        }
    }
    let mut unique = selected.clone();
    unique.extend(
        aliases
            .values()
            .filter(|element| arrays.contains_key(&element.original))
            .map(|element| (element.original, element.index)),
    );
    if unique.len() > sampled_limit as usize {
        return Err(Error::Unsupported(
            "dynamic sampled images exceed device image limit",
        ));
    }
    let canonical = super::binding_array::ensure_elements(module, aliases, &selected)?;
    let mut elements = HashMap::new();
    for original in used {
        let count = arrays[&original].1;
        elements.insert(
            original,
            (0..count)
                .map(|index| canonical[&(original, index)])
                .collect(),
        );
    }
    for (h, f) in module.functions.iter_mut() {
        rewrite(
            f,
            &info[h],
            &mut module.types,
            &arrays,
            &elements,
            &slots[&h],
        )?;
    }
    for (i, e) in module.entry_points.iter_mut().enumerate() {
        rewrite(
            &mut e.function,
            info.get_entry_point(i),
            &mut module.types,
            &arrays,
            &elements,
            &entry_slots[i],
        )?;
    }
    Ok(())
}

fn load_count(f: &naga::Function, arrays: &HashMap<Global, (Handle<naga::Type>, u32)>) -> usize {
    f.expressions
        .iter()
        .filter(|(_, e)| match **e {
            Expression::ImageLoad { image, .. } => {
                selected(image, &f.expressions, arrays).is_some()
            }
            _ => false,
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify(source: &str) {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 2).unwrap();
        assert_eq!(aliases.len(), 2);
        super::super::pairs::split(&mut module).unwrap();
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
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut output = String::new();
        naga::back::glsl::Writer::new(
            &mut output,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        assert!(
            output.contains("switch") && output.contains("texelFetch"),
            "{output}"
        );
    }

    #[test]
    fn d2_float_in_helper_loop() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
fn fetch_image(i: u32) -> vec4f { var value = vec4f(0.); for (var j=0; j<2; j++) { value += textureLoad(images[i], vec2i(j), 0); } return value; }
@compute @workgroup_size(1) fn main() { output[0] = fetch_image(selected); }"#,
        );
    }
    #[test]
    fn d2_array_sint() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d_array<i32>, 2>;
@group(0) @binding(1) var<uniform> selected: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4i>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 1, 0); }"#,
        );
    }
    #[test]
    fn d3_uint() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_3d<u32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4u>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec3i(0), 0); }"#,
        );
    }
    #[test]
    fn multisample() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_multisampled_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 1); }"#,
        );
    }
    #[test]
    fn depth_2d() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_depth_2d, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 0); }"#,
        );
    }
    #[test]
    fn limit_is_checked_before_alias_allocation() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 3>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[selected], vec2i(0), 0); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let globals = module.global_variables.len();
        let mut aliases = HashMap::new();
        assert!(matches!(
            lower(&mut module, &mut aliases, 2),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(module.global_variables.len(), globals);
        assert!(aliases.is_empty());
    }
    #[test]
    fn static_and_dynamic_share_aliases() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureLoad(images[1], vec2i(0), 0) + textureLoad(images[selected], vec2i(0), 0); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module, &mut aliases, 2).unwrap();
        assert_eq!(aliases.len(), 2);
        super::super::pairs::split(&mut module).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn shared_texture_load_and_sample() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samp: sampler;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let t = images[selected]; output[0] = textureLoad(t, vec2i(0), 0) + textureSampleLevel(t, samp, vec2f(0.2, 0.8), 0.); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 2).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 2).unwrap();
        assert_eq!(aliases.len(), 2);
        super::super::pairs::split(&mut module).unwrap();
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
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut output = String::new();
        naga::back::glsl::Writer::new(
            &mut output,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        assert!(
            output.contains("texelFetch") && output.contains("textureLod"),
            "{output}"
        );
    }

    #[test]
    fn shared_texture_multiple_loads() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var<uniform> selected: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let t = images[selected]; output[0] = textureLoad(t, vec2i(0), 0) + textureLoad(t, vec2i(1), 0); }"#,
        );
    }
}
fn selected(
    image: Expr,
    expressions: &naga::Arena<Expression>,
    arrays: &HashMap<Global, (Handle<naga::Type>, u32)>,
) -> Option<(Global, Expr)> {
    let Expression::Access { base, index } = expressions[image] else {
        return None;
    };
    if matches!(
        expressions[index],
        Expression::Literal(_) | Expression::Constant(_)
    ) {
        return None;
    }
    let Expression::GlobalVariable(g) = expressions[base] else {
        return None;
    };
    arrays.contains_key(&g).then_some((g, index))
}
struct Choice {
    selector: Expr,
    signed: bool,
    pointer: Expr,
    zero: Expr,
    loads: Vec<Expr>,
}
fn rewrite(
    f: &mut naga::Function,
    info: &naga::valid::FunctionInfo,
    types: &mut naga::UniqueArena<naga::Type>,
    arrays: &HashMap<Global, (Handle<naga::Type>, u32)>,
    elements: &HashMap<Global, Vec<Global>>,
    slots: &[Expr],
) -> Result<()> {
    let candidates: Vec<_> = f
        .expressions
        .iter()
        .filter_map(|(h, e)| {
            if let Expression::ImageLoad { image, .. } = *e {
                selected(image, &f.expressions, arrays).map(|(g, i)| (h, image, g, i))
            } else {
                None
            }
        })
        .collect();
    let mut counts = HashMap::new();
    for &(_, image, _, _) in &candidates {
        *counts.entry(image).or_insert(0usize) += 1;
    }
    let mut replacements = HashMap::new();
    for &(_, image, global, _) in &candidates {
        if info[image].ref_count == counts[&image] {
            replacements.insert(image, elements[&global][0]);
        }
    }
    let mut slots = slots.iter().copied();
    let mut choices = HashMap::new();
    let mut removed = HashSet::new();
    for (load, image, global, selector) in candidates {
        let _ = image;
        let pointer = slots.next().ok_or(Error::InvalidRange)?;
        let signed = match info[selector].ty.inner_with(types) {
            naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Sint,
                ..
            }) => true,
            naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Uint,
                ..
            }) => false,
            _ => return Err(Error::InvalidRange),
        };
        let value_type = types.insert(
            naga::Type {
                name: None,
                inner: info[load].ty.inner_with(types).clone(),
            },
            naga::Span::UNDEFINED,
        );
        let local = f.local_variables.append(
            naga::LocalVariable {
                name: None,
                ty: value_type,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let span = f.expressions.get_span(load);
        let template = f.expressions[load].clone();
        f.expressions[pointer] = Expression::LocalVariable(local);
        f.expressions[load] = Expression::Load { pointer };
        let zero = f
            .expressions
            .append(Expression::ZeroValue(value_type), span);
        let mut loads = Vec::new();
        for &alias in &elements[&global] {
            let global_expr = f
                .expressions
                .append(Expression::GlobalVariable(alias), span);
            let mut value = template.clone();
            if let Expression::ImageLoad { image, .. } = &mut value {
                *image = global_expr;
            }
            loads.push(f.expressions.append(value, span));
        }
        choices.insert(
            load,
            Choice {
                selector,
                signed,
                pointer,
                zero,
                loads,
            },
        );
    }
    for (expression, global) in replacements {
        f.expressions[expression] = Expression::GlobalVariable(global);
        removed.insert(expression);
    }
    transform(&mut f.body, &choices, &removed);
    Ok(())
}
fn transform(block: &mut naga::Block, choices: &HashMap<Expr, Choice>, removed: &HashSet<Expr>) {
    let old = std::mem::take(block);
    for (mut statement, span) in old.span_into_iter() {
        if let Statement::Emit(range) = &statement {
            let mut start = None;
            let mut end = None;
            for h in range.clone() {
                if removed.contains(&h) || choices.contains_key(&h) {
                    if let (Some(a), Some(b)) = (start.take(), end.take()) {
                        block.push(Statement::Emit(naga::Range::new_from_bounds(a, b)), span);
                    }
                    if let Some(choice) = choices.get(&h) {
                        block.push(
                            Statement::Store {
                                pointer: choice.pointer,
                                value: choice.zero,
                            },
                            span,
                        );
                        let mut cases = Vec::new();
                        for (i, load) in choice.loads.iter().enumerate() {
                            let mut body = naga::Block::new();
                            body.push(
                                Statement::Emit(naga::Range::new_from_bounds(*load, *load)),
                                span,
                            );
                            body.push(
                                Statement::Store {
                                    pointer: choice.pointer,
                                    value: *load,
                                },
                                span,
                            );
                            cases.push(naga::SwitchCase {
                                value: if choice.signed {
                                    naga::SwitchValue::I32(i as i32)
                                } else {
                                    naga::SwitchValue::U32(i as u32)
                                },
                                body,
                                fall_through: false,
                            });
                        }
                        cases.push(naga::SwitchCase {
                            value: naga::SwitchValue::Default,
                            body: naga::Block::new(),
                            fall_through: false,
                        });
                        block.push(
                            Statement::Switch {
                                selector: choice.selector,
                                cases,
                            },
                            span,
                        );
                        block.push(Statement::Emit(naga::Range::new_from_bounds(h, h)), span);
                    }
                } else {
                    start.get_or_insert(h);
                    end = Some(h);
                }
            }
            if let (Some(a), Some(b)) = (start, end) {
                block.push(Statement::Emit(naga::Range::new_from_bounds(a, b)), span);
            }
            continue;
        }
        match &mut statement {
            Statement::Block(b) => transform(b, choices, removed),
            Statement::If { accept, reject, .. } => {
                transform(accept, choices, removed);
                transform(reject, choices, removed);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    transform(&mut c.body, choices, removed);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                transform(body, choices, removed);
                transform(continuing, choices, removed);
            }
            _ => {}
        }
        block.push(statement, span);
    }
}
