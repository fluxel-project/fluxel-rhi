//! Specialize sampled image arrays into fixed image/sampler pairs.
//!
//! Each result is evaluated at its original Emit point. The default branch
//! returns zero, and the original coordinate, offset, layer and LOD operands
//! are reused without reevaluation.

use super::binding_array::Element;
use fluxel_rhi_core::api::{Error, Result};
use naga::{Expression, GlobalVariable, Handle, Statement};
use std::collections::{BTreeSet, HashMap, HashSet};
#[path = "binding_array_buffer_remap.rs"]
mod remap;

type Global = Handle<GlobalVariable>;
type Expr = Handle<Expression>;

#[derive(Clone, Copy)]
enum Resource {
    Fixed(Global),
    Dynamic {
        global: Global,
        selector: Expr,
        expression: Expr,
    },
}

fn shared_selector(a: Resource, b: Resource) -> bool {
    matches!((a, b), (Resource::Dynamic { selector: x, .. }, Resource::Dynamic { selector: y, .. }) if x == y)
}

fn resource(
    e: Expr,
    expressions: &naga::Arena<Expression>,
    arrays: &HashMap<Global, u32>,
) -> Option<Resource> {
    match expressions[e] {
        Expression::GlobalVariable(global) => Some(Resource::Fixed(global)),
        Expression::Access { base, index }
            if !matches!(
                expressions[index],
                Expression::Literal(_) | Expression::Constant(_)
            ) =>
        {
            if let Expression::GlobalVariable(global) = expressions[base] {
                if arrays.contains_key(&global) {
                    return Some(Resource::Dynamic {
                        global,
                        selector: index,
                        expression: e,
                    });
                }
            }
            None
        }
        _ => None,
    }
}

/// Caller must discard the module on error and reject remaining dynamic uses.
/// `pair_limit` bounds generated combined texture/sampler uniforms; limits for
/// other active resources must also be checked by the caller after linking.
pub(super) fn lower(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
    pair_limit: u32,
) -> Result<()> {
    let arrays: HashMap<_, _> = module
        .global_variables
        .iter()
        .filter_map(|(g, v)| match module.types[v.ty].inner {
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(n),
            } if matches!(
                module.types[base].inner,
                naga::TypeInner::Image { .. } | naga::TypeInner::Sampler { .. }
            ) =>
            {
                Some((g, n.get()))
            }
            _ => None,
        })
        .collect();
    if arrays.is_empty() {
        return Ok(());
    }
    let mut uses = Vec::new();
    for f in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, e) in f.expressions.iter() {
            if let Expression::ImageSample {
                image,
                sampler,
                level:
                    naga::SampleLevel::Zero
                    | naga::SampleLevel::Exact(_)
                    | naga::SampleLevel::Auto
                    | naga::SampleLevel::Bias(_)
                    | naga::SampleLevel::Gradient { .. },
                ..
            } = *e
            {
                let a = resource(image, &f.expressions, &arrays);
                let b = resource(sampler, &f.expressions, &arrays);
                if matches!(a, Some(Resource::Dynamic { .. }))
                    || matches!(b, Some(Resource::Dynamic { .. }))
                {
                    uses.push((
                        a.ok_or(Error::Unsupported(
                            "dynamic sampled texture has non-global image",
                        ))?,
                        b.ok_or(Error::Unsupported(
                            "dynamic sampled texture has non-global sampler",
                        ))?,
                    ));
                }
            }
        }
    }
    if uses.is_empty() {
        return Ok(());
    }
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("dynamic sampled input validation: {e:?}")))?;
    let mut function_slots = HashMap::new();
    for (h, f) in module.functions.iter_mut() {
        let count = sample_count(f, &arrays);
        function_slots.insert(h, remap::reserve(f, count));
    }
    let entry_slots: Vec<_> = module
        .entry_points
        .iter_mut()
        .map(|e| {
            let count = sample_count(&e.function, &arrays);
            remap::reserve(&mut e.function, count)
        })
        .collect();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("dynamic sampled input validation: {e:?}")))?;
    let mut selected = BTreeSet::new();
    let mut pairs = HashSet::new();
    for (image, sampler) in uses {
        let diagonal = shared_selector(image, sampler);
        let diagonal_len = if diagonal {
            match (image, sampler) {
                (Resource::Dynamic { global: a, .. }, Resource::Dynamic { global: b, .. }) => {
                    Some(arrays[&a].min(arrays[&b]))
                }
                _ => None,
            }
        } else {
            None
        };
        let keys = |r: Resource| -> Result<Vec<(Global, Option<u32>)>> {
            match r {
                Resource::Fixed(g) => Ok(vec![
                    aliases
                        .get(&g)
                        .map(|e| (e.original, Some(e.index)))
                        .unwrap_or((g, None)),
                ]),
                Resource::Dynamic { global, .. } => {
                    let n = diagonal_len.unwrap_or(arrays[&global]);
                    if n > pair_limit {
                        return Err(Error::Unsupported(
                            "dynamic sample array exceeds texture unit limit",
                        ));
                    }
                    Ok((0..n).map(|i| (global, Some(i))).collect())
                }
            }
        };
        let images = keys(image)?;
        let samplers = keys(sampler)?;
        let pair_count = if diagonal {
            images.len().min(samplers.len())
        } else {
            images
                .len()
                .checked_mul(samplers.len())
                .ok_or(Error::InvalidRange)?
        };
        if pair_count > pair_limit as usize {
            return Err(Error::Unsupported(
                "dynamic sample pairs exceed texture unit limit",
            ));
        }
        for (i, &a) in images.iter().enumerate() {
            for (j, &b) in samplers.iter().enumerate() {
                if diagonal && i != j {
                    continue;
                }
                pairs.insert((a, b));
                if pairs.len() > pair_limit as usize {
                    return Err(Error::Unsupported(
                        "dynamic sample pairs exceed texture unit limit",
                    ));
                }
            }
        }
        for r in [image, sampler] {
            if let Resource::Dynamic { global, .. } = r {
                for i in 0..diagonal_len.unwrap_or(arrays[&global]) {
                    selected.insert((global, i));
                }
            }
        }
    }
    let canonical = super::binding_array::ensure_elements(module, aliases, &selected)?;
    let mut elements = HashMap::new();
    for (global, index) in selected {
        elements
            .entry(global)
            .or_insert_with(Vec::new)
            .push((index, canonical[&(global, index)]));
    }
    for values in elements.values_mut() {
        values.sort_by_key(|v| v.0);
    }
    for (h, f) in module.functions.iter_mut() {
        rewrite(
            f,
            &info[h],
            &mut module.types,
            &arrays,
            &elements,
            &function_slots[&h],
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

fn sample_count(f: &naga::Function, arrays: &HashMap<Global, u32>) -> usize {
    f.expressions
        .iter()
        .filter(|(_, e)| match **e {
            Expression::ImageSample {
                image,
                sampler,
                level:
                    naga::SampleLevel::Zero
                    | naga::SampleLevel::Exact(_)
                    | naga::SampleLevel::Auto
                    | naga::SampleLevel::Bias(_)
                    | naga::SampleLevel::Gradient { .. },
                ..
            } => {
                matches!(
                    resource(image, &f.expressions, arrays),
                    Some(Resource::Dynamic { .. })
                ) || matches!(
                    resource(sampler, &f.expressions, arrays),
                    Some(Resource::Dynamic { .. })
                )
            }
            _ => false,
        })
        .count()
}

struct Dimension {
    selector: Expr,
    signed: bool,
}
struct Choice {
    dimensions: Vec<Dimension>,
    pointer: Expr,
    zero: Expr,
    samples: Vec<Expr>,
    sizes: Vec<usize>,
    gradients: Option<(Expr, Expr)>,
    gradient_range: Option<(Expr, Expr)>,
    eager: bool,
}
fn rewrite(
    f: &mut naga::Function,
    info: &naga::valid::FunctionInfo,
    types: &mut naga::UniqueArena<naga::Type>,
    arrays: &HashMap<Global, u32>,
    elements: &HashMap<Global, Vec<(u32, Global)>>,
    slots: &[Expr],
) -> Result<()> {
    let candidates: Vec<_> = f
        .expressions
        .iter()
        .filter_map(|(h, e)| match *e {
            Expression::ImageSample {
                image,
                sampler,
                level:
                    naga::SampleLevel::Zero
                    | naga::SampleLevel::Exact(_)
                    | naga::SampleLevel::Auto
                    | naga::SampleLevel::Bias(_)
                    | naga::SampleLevel::Gradient { .. },
                ..
            } => {
                let a = resource(image, &f.expressions, arrays)?;
                let b = resource(sampler, &f.expressions, arrays)?;
                (matches!(a, Resource::Dynamic { .. }) || matches!(b, Resource::Dynamic { .. }))
                    .then_some((h, a, b))
            }
            _ => None,
        })
        .collect();
    let mut references = HashMap::new();
    for (_, a, b) in &candidates {
        for r in [a, b] {
            if let Resource::Dynamic { expression, .. } = r {
                *references.entry(*expression).or_insert(0) += 1;
            }
        }
    }
    for (&expression, &count) in &references {
        if info[expression].ref_count != count {
            return Err(Error::Unsupported(
                "dynamic sample resource also used by an unlowered operation",
            ));
        }
    }
    let mut slots = slots.iter().copied();
    let mut choices = HashMap::new();
    let mut removed = HashSet::new();
    let mut replacements = HashMap::new();
    for (sample, image, sampler) in candidates {
        let diagonal = shared_selector(image, sampler);
        let mut dims = Vec::new();
        let mut resource_globals = Vec::new();
        for r in [image, sampler] {
            match r {
                Resource::Fixed(g) => resource_globals.push(vec![g]),
                Resource::Dynamic {
                    global,
                    selector,
                    expression,
                } => {
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
                    if !diagonal || dims.is_empty() {
                        dims.push(Dimension { selector, signed });
                    }
                    resource_globals.push(elements[&global].iter().map(|&(_, g)| g).collect());
                    replacements.insert(expression, elements[&global][0].1);
                    removed.insert(expression);
                }
            }
        }
        let pointer = slots.next().ok_or(Error::InvalidRange)?;
        let ty = info[sample].ty.handle().unwrap_or_else(|| {
            types.insert(
                naga::Type {
                    name: None,
                    inner: info[sample].ty.inner_with(types).clone(),
                },
                naga::Span::UNDEFINED,
            )
        });
        let local = f.local_variables.append(
            naga::LocalVariable {
                name: None,
                ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let span = f.expressions.get_span(sample);
        let template = f.expressions[sample].clone();
        // GLSL has no textureGrad overload for cube-array shadow samplers.
        // Evaluate all pure samples at the original uniform-control-flow point,
        // then select their values without moving implicit derivatives into a
        // selector-dependent branch. Partially bound units have valid dummies.
        let eager = matches!(&template, Expression::ImageSample {
            image, level: naga::SampleLevel::Auto, gather: None, depth_ref: Some(_), ..
        } if matches!(info[*image].ty.inner_with(types), naga::TypeInner::Image {
            dim: naga::ImageDimension::Cube, arrayed: true,
            class: naga::ImageClass::Depth { .. },
        }));
        f.expressions[pointer] = Expression::LocalVariable(local);
        f.expressions[sample] = Expression::Load { pointer };
        let zero = f.expressions.append(Expression::ZeroValue(ty), span);
        // Implicit derivatives must execute before divergent selector switches.
        // Keeping them at the original sample's Emit point preserves the
        // original control-flow uniformity requirements.
        let (gradients, gradient_range) = match &template {
            Expression::ImageSample {
                coordinate,
                level,
                gather: None,
                ..
            } if !eager
                && matches!(level, naga::SampleLevel::Auto | naga::SampleLevel::Bias(_)) =>
            {
                let bias = match level {
                    naga::SampleLevel::Bias(value) => Some(*value),
                    _ => None,
                };
                let mut x = f.expressions.append(
                    Expression::Derivative {
                        axis: naga::DerivativeAxis::X,
                        ctrl: naga::DerivativeControl::None,
                        expr: *coordinate,
                    },
                    span,
                );
                let mut y = f.expressions.append(
                    Expression::Derivative {
                        axis: naga::DerivativeAxis::Y,
                        ctrl: naga::DerivativeControl::None,
                        expr: *coordinate,
                    },
                    span,
                );
                let range_start = x;
                // Naga's GLSL backend cannot emit a bias together with an
                // explicit gradient. Scaling both gradients by 2^bias gives
                // the same mip-level offset while keeping derivatives outside
                // the divergent resource-selector switch.
                if let Some(bias) = bias {
                    let scale = f.expressions.append(
                        Expression::Math {
                            fun: naga::MathFunction::Exp2,
                            arg: bias,
                            arg1: None,
                            arg2: None,
                            arg3: None,
                        },
                        span,
                    );
                    x = f.expressions.append(
                        Expression::Binary {
                            op: naga::BinaryOperator::Multiply,
                            left: x,
                            right: scale,
                        },
                        span,
                    );
                    y = f.expressions.append(
                        Expression::Binary {
                            op: naga::BinaryOperator::Multiply,
                            left: y,
                            right: scale,
                        },
                        span,
                    );
                }
                (Some((x, y)), Some((range_start, y)))
            }
            _ => (None, None),
        };
        let mut samples = Vec::new();
        for (image_index, &i) in resource_globals[0].iter().enumerate() {
            for (sampler_index, &s) in resource_globals[1].iter().enumerate() {
                if diagonal && image_index != sampler_index {
                    continue;
                }
                let ie = f.expressions.append(Expression::GlobalVariable(i), span);
                let se = f.expressions.append(Expression::GlobalVariable(s), span);
                let mut e = template.clone();
                if let Expression::ImageSample {
                    image,
                    sampler,
                    level,
                    ..
                } = &mut e
                {
                    *image = ie;
                    *sampler = se;
                    if let Some((x, y)) = gradients {
                        *level = naga::SampleLevel::Gradient { x, y };
                    }
                }
                samples.push(f.expressions.append(e, span));
            }
        }
        let sizes = if diagonal {
            vec![resource_globals[0].len().min(resource_globals[1].len())]
        } else {
            [image, sampler]
                .iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    matches!(r, Resource::Dynamic { .. }).then_some(resource_globals[i].len())
                })
                .collect()
        };
        choices.insert(
            sample,
            Choice {
                dimensions: dims,
                pointer,
                zero,
                samples,
                sizes,
                gradients,
                gradient_range,
                eager,
            },
        );
    }
    for (expression, global) in replacements {
        f.expressions[expression] = Expression::GlobalVariable(global);
    }
    transform(&mut f.body, &choices, &removed);
    Ok(())
}

fn branches(choice: &Choice, depth: usize, base: usize, span: naga::Span) -> naga::Block {
    let mut block = naga::Block::new();
    if depth == choice.dimensions.len() {
        let value = choice.samples[base];
        if !choice.eager {
            block.push(
                Statement::Emit(naga::Range::new_from_bounds(value, value)),
                span,
            );
        }
        block.push(
            Statement::Store {
                pointer: choice.pointer,
                value,
            },
            span,
        );
        return block;
    }
    let stride: usize = choice.sizes[depth + 1..].iter().product();
    let mut cases = Vec::new();
    for i in 0..choice.sizes[depth] {
        cases.push(naga::SwitchCase {
            value: if choice.dimensions[depth].signed {
                naga::SwitchValue::I32(i as i32)
            } else {
                naga::SwitchValue::U32(i as u32)
            },
            body: branches(choice, depth + 1, base + i * stride, span),
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
            selector: choice.dimensions[depth].selector,
            cases,
        },
        span,
    );
    block
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
                    if let Some(c) = choices.get(&h) {
                        if let Some((start, end)) = c.gradient_range {
                            block.push(
                                Statement::Emit(naga::Range::new_from_bounds(start, end)),
                                span,
                            );
                        }
                        if c.eager {
                            for &sample in &c.samples {
                                block.push(
                                    Statement::Emit(naga::Range::new_from_bounds(sample, sample)),
                                    span,
                                );
                            }
                        }
                        block.push(
                            Statement::Store {
                                pointer: c.pointer,
                                value: c.zero,
                            },
                            span,
                        );
                        block.extend_block(branches(c, 0, 0, span));
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_selector_uses_only_diagonal_pairs() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 4>;
@group(1) @binding(0) var samplers: binding_array<sampler, 3>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureSample(images[i], samplers[i], uv);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 3).unwrap();
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
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut glsl = String::new();
        let reflection = naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        assert_eq!(reflection.texture_mapping.len(), 3, "{glsl}");
    }
    #[test]
    fn cube_array_shadow_samples_before_divergent_selection() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_depth_cube_array, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler_comparison, 2>;
@fragment fn main(@location(0) dir: vec3f, @location(1) @interpolate(flat) i: u32) -> @location(0) f32 {
    return textureSampleCompare(images[i], samplers[i], dir, 0, 0.5);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
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
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        let switch = glsl.find("switch").unwrap();
        assert!(
            glsl.find("texture(").unwrap() < switch && !glsl.contains("textureGrad"),
            "{glsl}"
        );
    }
    #[test]
    fn implicit_derivatives_are_computed_before_selector_switch() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureSample(images[i], samplers[i], uv);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
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
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        let switch = glsl.find("switch").unwrap();
        assert!(
            glsl.find("dFdx").unwrap() < switch
                && glsl.find("dFdy").unwrap() < switch
                && glsl.contains("textureGrad"),
            "{glsl}"
        );
    }

    #[test]
    fn dynamic_bias_is_lowered_into_scaled_gradients() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureSampleBias(images[i], samplers[i], uv, 1.0);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
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
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        let switch = glsl.find("switch").unwrap();
        assert!(
            glsl.find("dFdx").unwrap() < switch
                && glsl.find("dFdy").unwrap() < switch
                && glsl.contains("exp2")
                && glsl.contains("textureGrad"),
            "{glsl}"
        );
    }

    #[test]
    fn dynamic_gather_is_specialized_per_resource_pair() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@fragment fn main(@location(0) uv: vec2f, @location(1) @interpolate(flat) i: u32) -> @location(0) vec4f {
    return textureGather(0, images[i], samplers[i], uv);
}"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 4).unwrap();
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
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
            multiview: None,
        };
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
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
            glsl.contains("textureGather") && glsl.contains("switch"),
            "{glsl}"
        );
    }

    fn verify(source: &str, expected_aliases: usize, limit: u32) -> String {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module, &mut aliases, limit).unwrap();
        assert_eq!(aliases.len(), expected_aliases);
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
            output.contains("switch") && output.contains("textureLod"),
            "{output}"
        );
        output
    }
    #[test]
    fn dynamic_image_sampler_different_groups_and_offset() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d_array<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 3>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[ti], samplers[si], vec2f(0.25, 0.75), 1, 1.5, vec2i(1, -1)); }"#;
        let glsl = verify(source, 5, 6);
        assert!(glsl.contains("textureLodOffset"), "{glsl}");
    }
    #[test]
    fn static_texture_dynamic_sampler() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_3d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(1) @binding(1) var<uniform> si: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[1], samplers[si], vec3f(0.1, 0.2, 0.3), 2.); }"#,
            3,
            2,
        );
    }
    #[test]
    fn dynamic_texture_static_sampler() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[ti], samplers[1], vec2f(0.2), 0.); }"#,
            3,
            2,
        );
    }
    #[test]
    fn shared_selected_resources() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { let image = images[ti]; let sampler0 = samplers[si]; output[0] = textureSampleLevel(image, sampler0, vec2f(0.2), 0.) + textureSampleLevel(image, sampler0, vec2f(0.4), 1.); }"#,
            4,
            4,
        );
    }
    #[test]
    fn too_many_combined_pairs_are_rejected() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 3>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(images[ti], samplers[si], vec2f(0.2), 0.); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        assert!(matches!(
            lower(&mut module, &mut aliases, 5),
            Err(Error::Unsupported(_))
        ));
        assert!(aliases.is_empty());
    }
}
