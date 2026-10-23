//! Dynamic storage-image side effects specialized into fixed-resource switches.
//!
//! Only direct dynamically indexed image stores and image atomics are changed.
//! Other array uses remain for the caller to reject or lower separately. The
//! switch default performs no access, preserving robust out-of-range behavior.
use super::binding_array::Element;
use fluxel_rhi_core::api::{Error, Result};
use naga::{Expression, GlobalVariable, Handle, Statement};
use std::collections::{BTreeSet, HashMap, HashSet};

type Global = Handle<GlobalVariable>;
type Expr = Handle<Expression>;

/// Returns synthetic globals mapped to their original binding and array index.
/// The caller must merge this map with the static specialization's metadata,
/// validate the resulting module, and keep dynamic array capability closed until
/// all remaining uses are handled.
pub(super) fn lower(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
    image_limit: u32,
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
                    class: naga::ImageClass::Storage { .. },
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
    for (_, f) in module.functions.iter() {
        collect(&f.body, &f.expressions, &arrays, &mut used);
    }
    for e in &module.entry_points {
        collect(
            &e.function.body,
            &e.function.expressions,
            &arrays,
            &mut used,
        );
    }
    if used.is_empty() {
        return Ok(());
    }
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("binding array input validation: {e:?}")))?;
    let mut total = 0u32;
    for global in &used {
        let count = arrays[global].1;
        if count > image_limit {
            return Err(Error::Unsupported(
                "dynamic storage image array exceeds specialization limit",
            ));
        }
        total = total.checked_add(count).ok_or(Error::InvalidRange)?;
    }
    if total > image_limit {
        return Err(Error::Unsupported(
            "dynamic storage image arrays exceed total specialization limit",
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
    if unique.len() > image_limit as usize {
        return Err(Error::Unsupported(
            "dynamic storage images exceed device image limit",
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
    for (handle, f) in module.functions.iter_mut() {
        rewrite(f, &info[handle], &module.types, &elements)?;
    }
    for (index, entry) in module.entry_points.iter_mut().enumerate() {
        rewrite(
            &mut entry.function,
            info.get_entry_point(index),
            &module.types,
            &elements,
        )?;
    }
    Ok(())
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
    let Expression::GlobalVariable(global) = expressions[base] else {
        return None;
    };
    arrays.contains_key(&global).then_some((global, index))
}
fn collect(
    block: &naga::Block,
    expressions: &naga::Arena<Expression>,
    arrays: &HashMap<Global, (Handle<naga::Type>, u32)>,
    used: &mut BTreeSet<Global>,
) {
    for statement in block.iter() {
        match statement {
            Statement::ImageStore { image, .. } | Statement::ImageAtomic { image, .. } => {
                if let Some((g, _)) = selected(*image, expressions, arrays) {
                    used.insert(g);
                }
            }
            Statement::Block(b) => collect(b, expressions, arrays, used),
            Statement::If { accept, reject, .. } => {
                collect(accept, expressions, arrays, used);
                collect(reject, expressions, arrays, used);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    collect(&case.body, expressions, arrays, used);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect(body, expressions, arrays, used);
                collect(continuing, expressions, arrays, used);
            }
            _ => {}
        }
    }
}
fn rewrite(
    function: &mut naga::Function,
    info: &naga::valid::FunctionInfo,
    types: &naga::UniqueArena<naga::Type>,
    elements: &HashMap<Global, Vec<Global>>,
) -> Result<()> {
    let mut choices = HashMap::new();
    let mut refs = HashMap::new();
    let mut replacements = HashMap::new();
    scan(&function.body, &function.expressions, elements, &mut refs);
    for (image, count) in refs {
        let Expression::Access { base, index } = function.expressions[image] else {
            unreachable!()
        };
        let Expression::GlobalVariable(global) = function.expressions[base] else {
            unreachable!()
        };
        if info[image].ref_count == count {
            replacements.insert(image, elements[&global][0]);
        }
        let signed = match info[index].ty.inner_with(types) {
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
        let mut expressions = Vec::new();
        for &alias in &elements[&global] {
            expressions.push(function.expressions.append(
                Expression::GlobalVariable(alias),
                function.expressions.get_span(image),
            ));
        }
        choices.insert(image, (index, signed, expressions));
    }
    let removed: HashSet<_> = replacements.keys().copied().collect();
    for (image, global) in replacements {
        function.expressions[image] = Expression::GlobalVariable(global);
    }
    transform(&mut function.body, &choices, &removed);
    Ok(())
}
fn scan(
    block: &naga::Block,
    expressions: &naga::Arena<Expression>,
    elements: &HashMap<Global, Vec<Global>>,
    refs: &mut HashMap<Expr, usize>,
) {
    for statement in block.iter() {
        match statement {
            Statement::ImageStore { image, .. } | Statement::ImageAtomic { image, .. } => {
                if let Expression::Access { base, index } = expressions[*image] {
                    if let Expression::GlobalVariable(g) = expressions[base] {
                        if elements.contains_key(&g)
                            && !matches!(
                                expressions[index],
                                Expression::Literal(_) | Expression::Constant(_)
                            )
                        {
                            *refs.entry(*image).or_default() += 1;
                        }
                    }
                }
            }
            Statement::Block(b) => scan(b, expressions, elements, refs),
            Statement::If { accept, reject, .. } => {
                scan(accept, expressions, elements, refs);
                scan(reject, expressions, elements, refs);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    scan(&c.body, expressions, elements, refs);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                scan(body, expressions, elements, refs);
                scan(continuing, expressions, elements, refs);
            }
            _ => {}
        }
    }
}
type Choice = (Expr, bool, Vec<Expr>);
fn transform(block: &mut naga::Block, choices: &HashMap<Expr, Choice>, removed: &HashSet<Expr>) {
    let old = std::mem::take(block);
    for (mut statement, span) in old.span_into_iter() {
        if let Statement::Emit(range) = &statement {
            let mut start = None;
            let mut end = None;
            for h in range.clone() {
                if removed.contains(&h) {
                    if let (Some(a), Some(b)) = (start.take(), end.take()) {
                        block.push(Statement::Emit(naga::Range::new_from_bounds(a, b)), span);
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
        let image = match &statement {
            Statement::ImageStore { image, .. } | Statement::ImageAtomic { image, .. } => {
                Some(*image)
            }
            _ => None,
        };
        if let Some((selector, signed, aliases)) = image.and_then(|i| choices.get(&i)) {
            let mut cases = Vec::with_capacity(aliases.len() + 1);
            for (i, alias) in aliases.iter().enumerate() {
                let mut branch = statement.clone();
                match &mut branch {
                    Statement::ImageStore { image, .. } | Statement::ImageAtomic { image, .. } => {
                        *image = *alias
                    }
                    _ => unreachable!(),
                }
                let mut body = naga::Block::new();
                body.push(branch, span);
                cases.push(naga::SwitchCase {
                    value: if *signed {
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
                    selector: *selector,
                    cases,
                },
                span,
            );
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
    fn verify(source: &str, needle: &str) {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let mut aliases = HashMap::new();
        lower(&mut module, &mut aliases, 256).unwrap();
        assert_eq!(aliases.len(), 2);
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
        assert!(glsl.contains("switch"), "{glsl}");
        assert!(glsl.contains(needle), "{glsl}");
    }
    #[test]
    fn dynamic_store_in_helper_and_loop() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
@group(0) @binding(1) var<uniform> select: u32;
fn put(i:u32) { if (i < 4u) { for(var n=0u;n<1u;n++) { textureStore(images[i],vec2i(0),vec4f(1.)); } } }
@compute @workgroup_size(1) fn main() { put(select); }"#,
            "imageStore",
        );
    }
    #[test]
    fn signed_dynamic_store() {
        verify(
            r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 2>;
@group(0) @binding(1) var<uniform> select: i32;
@compute @workgroup_size(1) fn main() { textureStore(images[select],vec2i(0),vec4f(1.)); }"#,
            "imageStore",
        );
    }
    #[test]
    fn dynamic_atomic() {
        verify(
            r#"
@group(0) @binding(0) var images: binding_array<texture_storage_2d<r32uint, atomic>, 2>;
@group(0) @binding(1) var<uniform> select: u32;
@compute @workgroup_size(1) fn main() { textureAtomicAdd(images[select],vec2i(0),1u); }"#,
            "imageAtomicAdd",
        );
    }
    #[test]
    fn oversized_array_rejected_before_mutation() {
        let source = r#"@group(0) @binding(0) var images: binding_array<texture_storage_2d<rgba8unorm, write>, 65>;
@group(0) @binding(1) var<uniform> select:u32;
@compute @workgroup_size(1) fn main() { textureStore(images[select],vec2i(0),vec4f(1.)); }"#;
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        let before = module.global_variables.len();
        assert!(matches!(
            lower(&mut module, &mut HashMap::new(), 64),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(module.global_variables.len(), before);
    }
}
