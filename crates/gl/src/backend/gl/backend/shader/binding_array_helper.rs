//! Specialize resource helper parameters to dynamic binding-array selectors.
//!
//! Run after static array splitting and before dynamic array lowering. On error
//! the caller must discard the partially rewritten module.
use fluxel_rhi_core::api::{Error, Result};
use naga::{Expression as E, Handle, Statement as S};
use std::collections::HashMap;
#[path = "binding_array_buffer_remap.rs"]
mod remap;
type Function = Handle<naga::Function>;
type Global = Handle<naga::GlobalVariable>;
type Expr = Handle<E>;
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Resource {
    Static(Global),
    Dynamic(Global, naga::ScalarKind),
}
type Key = (Function, Vec<Option<Resource>>);
type Cache = HashMap<Key, Function>;

pub(super) fn lower(module: &mut naga::Module) -> Result<()> {
    if !module.global_variables.iter().any(|(_, global)| {
        matches!(
            module.types[global.ty].inner,
            naga::TypeInner::BindingArray { .. }
        )
    }) {
        return Ok(());
    }
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("helper input: {e:?}")))?;
    let mut entries = std::mem::take(&mut module.entry_points);
    let mut cache = HashMap::new();
    for (i, entry) in entries.iter_mut().enumerate() {
        let refs = entry
            .function
            .expressions
            .iter()
            .map(|(h, _)| (h, info.get_entry_point(i)[h].ref_count))
            .collect();
        visit_function(module, &mut entry.function, &mut cache, &info, refs)?;
    }
    module.entry_points = entries;
    Ok(())
}
fn visit_function(
    module: &mut naga::Module,
    f: &mut naga::Function,
    cache: &mut Cache,
    info: &naga::valid::ModuleInfo,
    refs: HashMap<Expr, usize>,
) -> Result<()> {
    let mut typifier = naga::front::Typifier::new();
    let context = naga::proc::ResolveContext::with_locals(module, &f.local_variables, &f.arguments);
    let mut kinds = HashMap::new();
    for (h, _) in f.expressions.iter() {
        typifier
            .grow(h, &f.expressions, &context)
            .map_err(|e| Error::Shader(format!("helper selector type: {e:?}")))?;
        if let naga::TypeInner::Scalar(s) = typifier.get(h, &module.types) {
            kinds.insert(h, s.kind);
        }
    }
    let mut consumed = HashMap::new();
    visit_block(
        module,
        &mut f.body,
        &mut f.expressions,
        cache,
        &kinds,
        info,
        &mut consumed,
    )?;
    let removed: std::collections::HashSet<_> = consumed
        .into_iter()
        .filter_map(|(h, n)| (refs.get(&h) == Some(&n)).then_some(h))
        .collect();
    remove_emits(&mut f.body, &removed);
    for h in removed {
        f.expressions[h] = E::Literal(naga::Literal::U32(0));
    }
    Ok(())
}
fn visit_block(
    module: &mut naga::Module,
    block: &mut naga::Block,
    expressions: &mut naga::Arena<E>,
    cache: &mut Cache,
    kinds: &HashMap<Expr, naga::ScalarKind>,
    info: &naga::valid::ModuleInfo,
    consumed: &mut HashMap<Expr, usize>,
) -> Result<()> {
    for stmt in block.iter_mut() {
        match stmt {
            S::Block(b) => visit_block(module, b, expressions, cache, kinds, info, consumed)?,
            S::If { accept, reject, .. } => {
                visit_block(module, accept, expressions, cache, kinds, info, consumed)?;
                visit_block(module, reject, expressions, cache, kinds, info, consumed)?;
            }
            S::Switch { cases, .. } => {
                for case in cases {
                    visit_block(
                        module,
                        &mut case.body,
                        expressions,
                        cache,
                        kinds,
                        info,
                        consumed,
                    )?;
                }
            }
            S::Loop {
                body, continuing, ..
            } => {
                visit_block(module, body, expressions, cache, kinds, info, consumed)?;
                visit_block(
                    module,
                    continuing,
                    expressions,
                    cache,
                    kinds,
                    info,
                    consumed,
                )?;
            }
            S::Call {
                function,
                arguments,
                result,
            } => {
                let original = *function;
                let called = &module.functions[original];
                let mut keys = vec![None; arguments.len()];
                let mut selectors = vec![None; arguments.len()];
                for (index, (argument, parameter)) in
                    arguments.iter().zip(&called.arguments).enumerate()
                {
                    if !matches!(
                        module.types[parameter.ty].inner,
                        naga::TypeInner::Image { .. } | naga::TypeInner::Sampler { .. }
                    ) {
                        continue;
                    }
                    if let E::GlobalVariable(global) = expressions[*argument] {
                        keys[index] = Some(Resource::Static(global));
                    }
                    if let E::Access {
                        base,
                        index: selector,
                    } = expressions[*argument]
                    {
                        if let E::GlobalVariable(global) = expressions[base] {
                            if matches!(
                                module.types[module.global_variables[global].ty].inner,
                                naga::TypeInner::BindingArray { .. }
                            ) {
                                // Naga validates array indexes as signed or unsigned 32-bit.
                                // Derive the type from validated function information separately below.
                                let kind = *kinds.get(&selector).ok_or(Error::InvalidRange)?;
                                keys[index] = Some(Resource::Dynamic(global, kind));
                                selectors[index] = Some(selector);
                            }
                        }
                    }
                }
                let key = (original, keys.clone());
                let replacement = if let Some(h) = cache.get(&key) {
                    *h
                } else {
                    let mut clone = called.clone();
                    clone.name = Some(format!(
                        "fluxel_array_helper_{}_{}_value",
                        original.index(),
                        cache.len()
                    ));
                    let count = clone.expressions.iter().filter(|(_, e)| matches!(e, E::FunctionArgument(i) if keys.get(*i as usize).is_some_and(Option::is_some))).count();
                    let old_refs: Vec<_> = clone
                        .expressions
                        .iter()
                        .map(|(h, _)| info[original][h].ref_count)
                        .collect();
                    let slots = remap::reserve(&mut clone, count * 2);
                    let refs = clone
                        .expressions
                        .iter()
                        .skip(count * 2)
                        .zip(old_refs)
                        .map(|((h, _), n)| (h, n))
                        .collect();
                    let handles: Vec<_> = clone
                        .expressions
                        .iter()
                        .filter_map(|(h, e)| match *e {
                            E::FunctionArgument(i)
                                if keys.get(i as usize).is_some_and(Option::is_some) =>
                            {
                                Some((h, i))
                            }
                            _ => None,
                        })
                        .collect();
                    let mut access = Vec::new();
                    for ((h, i), pair) in handles.into_iter().zip(slots.chunks_exact(2)) {
                        let (global, kind) = match keys[i as usize].unwrap() {
                            Resource::Static(global) => {
                                clone.expressions[h] = E::GlobalVariable(global);
                                continue;
                            }
                            Resource::Dynamic(global, kind) => (global, kind),
                        };
                        let ty = module.types.insert(
                            naga::Type {
                                name: None,
                                inner: naga::TypeInner::Scalar(naga::Scalar { kind, width: 4 }),
                            },
                            naga::Span::UNDEFINED,
                        );
                        clone.arguments[i as usize].ty = ty;
                        clone.expressions[pair[0]] = E::GlobalVariable(global);
                        clone.expressions[pair[1]] = E::FunctionArgument(i);
                        clone.expressions[h] = E::Access {
                            base: pair[0],
                            index: pair[1],
                        };
                        access.push(h);
                    }
                    let old = std::mem::take(&mut clone.body);
                    for h in access {
                        clone.body.push(
                            S::Emit(naga::Range::new_from_bounds(h, h)),
                            naga::Span::UNDEFINED,
                        );
                    }
                    clone.body.extend_block(old);
                    visit_function(module, &mut clone, cache, info, refs)?;
                    let h = module
                        .functions
                        .append(clone, module.functions.get_span(original));
                    cache.insert(key, h);
                    h
                };
                for (argument, selector) in arguments.iter_mut().zip(selectors) {
                    if let Some(s) = selector {
                        *consumed.entry(*argument).or_default() += 1;
                        *argument = s;
                    }
                }
                *function = replacement;
                if let Some(result) = result {
                    expressions[*result] = E::CallResult(replacement);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn remove_emits(block: &mut naga::Block, removed: &std::collections::HashSet<Expr>) {
    let old = std::mem::take(block);
    for (mut stmt, span) in old.span_into_iter() {
        if let S::Emit(range) = &stmt {
            let mut start = None;
            let mut end = None;
            for h in range.clone() {
                if removed.contains(&h) {
                    if let (Some(a), Some(b)) = (start.take(), end.take()) {
                        block.push(S::Emit(naga::Range::new_from_bounds(a, b)), span);
                    }
                } else {
                    start.get_or_insert(h);
                    end = Some(h);
                }
            }
            if let (Some(a), Some(b)) = (start, end) {
                block.push(S::Emit(naga::Range::new_from_bounds(a, b)), span);
            }
            continue;
        }
        match &mut stmt {
            S::Block(b) => remove_emits(b, removed),
            S::If { accept, reject, .. } => {
                remove_emits(accept, removed);
                remove_emits(reject, removed);
            }
            S::Switch { cases, .. } => {
                for c in cases {
                    remove_emits(&mut c.body, removed);
                }
            }
            S::Loop {
                body, continuing, ..
            } => {
                remove_emits(body, removed);
                remove_emits(continuing, removed);
            }
            _ => {}
        }
        block.push(stmt, span);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_two_resources() {
        let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(0) @binding(1) var<uniform> ti: u32;
@group(1) @binding(1) var<uniform> si: i32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
fn inner(t: texture_2d<f32>, s: sampler) -> vec4f { return textureSampleLevel(t, s, vec2f(0.2), 0.); }
fn outer(t: texture_2d<f32>, s: sampler) -> vec4f { return inner(t, s); }
@compute @workgroup_size(1) fn main() { output[0] = outer(textures[ti], samplers[si]); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 8).unwrap();
        super::super::pairs::split(&mut module).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let opts = naga::back::glsl::Options {
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
        let mut glsl = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl,
            &module,
            &info,
            &opts,
            &pipeline,
            Default::default(),
        )
        .unwrap()
        .write()
        .unwrap();
        assert!(
            glsl.contains("switch") && glsl.contains("textureLod"),
            "{glsl}"
        );
    }
    #[test]
    fn caller_without_resource_parameters_and_shared_access() {
        let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> index: u32;
@group(0) @binding(3) var<storage, read_write> output: array<vec4f>;
fn fetch_value(t: texture_2d<f32>) -> vec4f { return textureLoad(t, vec2i(0), 0); }
fn wrapper() -> vec4f { let t = textures[index]; return fetch_value(t) + textureLoad(t, vec2i(0), 0); }
@compute @workgroup_size(1) fn main() { output[0] = wrapper(); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        super::super::binding_array_dynamic_sampled_load::lower(&mut module, &mut aliases, 2)
            .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
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
            glsl.contains("switch") && glsl.contains("texelFetch"),
            "{glsl}"
        );
    }

    #[test]
    fn fragment_implicit_derivatives() {
        let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(0) @binding(1) var samplers: binding_array<sampler, 2>;
@group(0) @binding(2) var<uniform> index: u32;
fn sample_it(t: texture_2d<f32>, s: sampler, uv: vec2f) -> vec4f { return textureSample(t, s, uv); }
@fragment fn main(@location(0) uv: vec2f) -> @location(0) vec4f { return sample_it(textures[index], samplers[index], uv); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module).unwrap();
        super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 8).unwrap();
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
            glsl.contains("dFdx") && glsl.contains("textureGrad"),
            "{glsl}"
        );
    }
    #[test]
    fn static_texture_dynamic_sampler() {
        let source = r#"@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(0) var samplers: binding_array<sampler, 2>;
@group(1) @binding(1) var<uniform> index: u32;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
fn level(t: texture_2d<f32>, s: sampler) -> vec4f { return textureSampleLevel(t, s, vec2f(0.2), 0.); }
@compute @workgroup_size(1) fn main() { output[0] = level(textures[1], samplers[index]); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        super::super::binding_array::split_static(&mut module, &mut aliases).unwrap();
        lower(&mut module).unwrap();
        super::super::binding_array_dynamic_sample::lower(&mut module, &mut aliases, 8).unwrap();
        super::super::specialize::specialize(&mut module).unwrap();
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
            glsl.contains("switch") && glsl.contains("textureLod"),
            "{glsl}"
        );
    }
}
