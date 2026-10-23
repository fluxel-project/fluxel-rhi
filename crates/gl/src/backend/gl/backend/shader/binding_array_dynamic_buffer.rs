//! Specialization of dynamic storage-buffer array accesses.
//!
//! Loads, stores, atomics, and runtime array length are specialized at their
//! original operation sites. Unhandled pointer uses are rejected; callers must
//! discard the partially transformed module on failure.

use std::collections::{BTreeSet, HashMap, HashSet};

use fluxel_rhi_core::api::{Error, Result};
use naga::{Expression, GlobalVariable, Handle, Statement};

use super::binding_array::Element;
#[path = "binding_array_buffer_remap.rs"]
mod remap;

type Global = Handle<GlobalVariable>;
type Expr = Handle<Expression>;

#[derive(Clone)]
struct Path {
    root: Expr,
    global: Global,
    selector: Expr,
    tail: Vec<Expr>,
}

fn path(
    expressions: &naga::Arena<Expression>,
    mut pointer: Expr,
    arrays: &HashMap<Global, u32>,
) -> Option<Path> {
    let mut tail = Vec::new();
    loop {
        match expressions[pointer] {
            Expression::Access { base, index } => {
                if let Expression::GlobalVariable(global) = expressions[base] {
                    if arrays.contains_key(&global)
                        && !matches!(
                            expressions[index],
                            Expression::Literal(_) | Expression::Constant(_)
                        )
                    {
                        tail.reverse();
                        return Some(Path {
                            root: pointer,
                            global,
                            selector: index,
                            tail,
                        });
                    }
                }
                tail.push(pointer);
                pointer = base;
            }
            Expression::AccessIndex { base, .. } => {
                tail.push(pointer);
                pointer = base;
            }
            _ => return None,
        }
    }
}

/// On error, discard the module. Callers must reject all remaining dynamic
/// accesses and must not advertise array support on this partial lowering.
pub(super) fn lower(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
    limit: u32,
) -> Result<()> {
    lower_space(module, aliases, limit, false)
}

/// Uniform blocks are read-only, but use the same pointer-path specialization
/// as storage buffers. Their device binding limit is independent of SSBOs.
pub(super) fn lower_uniform(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
    limit: u32,
) -> Result<()> {
    lower_space(module, aliases, limit, true)
}

fn lower_space(
    module: &mut naga::Module,
    aliases: &mut HashMap<Global, Element>,
    limit: u32,
    uniform: bool,
) -> Result<()> {
    let arrays: HashMap<_, _> = module
        .global_variables
        .iter()
        .filter_map(|(h, g)| {
            if !(if uniform {
                matches!(g.space, naga::AddressSpace::Uniform)
            } else {
                matches!(g.space, naga::AddressSpace::Storage { .. })
            }) {
                return None;
            }
            match module.types[g.ty].inner {
                naga::TypeInner::BindingArray {
                    size: naga::ArraySize::Constant(count),
                    ..
                } => Some((h, count.get())),
                _ => None,
            }
        })
        .collect();
    if arrays.is_empty() {
        return Ok(());
    }
    let mut used = BTreeSet::new();
    for f in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, expr) in f.expressions.iter() {
            if let Expression::Access { base, index } = *expr {
                if let Expression::GlobalVariable(g) = f.expressions[base] {
                    if arrays.contains_key(&g)
                        && !matches!(
                            f.expressions[index],
                            Expression::Literal(_) | Expression::Constant(_)
                        )
                    {
                        used.insert(g);
                    }
                }
            }
        }
    }
    if used.is_empty() {
        return Ok(());
    }
    for f in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        reject_non_store_uses(&f.body, &f.expressions, &arrays)?;
    }
    let count = used
        .iter()
        .try_fold(0u32, |sum, g| sum.checked_add(arrays[g]))
        .ok_or(Error::InvalidRange)?;
    if count > limit {
        return Err(Error::Unsupported(
            "dynamic buffer array exceeds device binding limit",
        ));
    }
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("dynamic buffer input validation: {e:?}")))?;
    let mut function_slots = HashMap::new();
    for (h, f) in module.functions.iter_mut() {
        let count = slot_count(f, &arrays);
        function_slots.insert(h, remap::reserve(f, count));
    }
    let entry_slots: Vec<_> = module
        .entry_points
        .iter_mut()
        .map(|e| {
            let count = slot_count(&e.function, &arrays);
            remap::reserve(&mut e.function, count)
        })
        .collect();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| Error::Shader(format!("dynamic buffer input validation: {e:?}")))?;
    let mut selected = BTreeSet::new();
    for &g in &used {
        for index in 0..arrays[&g] {
            selected.insert((g, index));
        }
    }
    let canonical = super::binding_array::ensure_elements(module, aliases, &selected)?;
    let elements: HashMap<_, Vec<_>> = used
        .into_iter()
        .map(|g| (g, (0..arrays[&g]).map(|i| canonical[&(g, i)]).collect()))
        .collect();
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

fn slot_count(f: &naga::Function, arrays: &HashMap<Global, u32>) -> usize {
    if !f
        .expressions
        .iter()
        .any(|(h, _)| path(&f.expressions, h, arrays).is_some())
    {
        return 0;
    }
    f.expressions
        .iter()
        .filter(|(_, e)| {
            matches!(
                e,
                Expression::Load { .. }
                    | Expression::ArrayLength(_)
                    | Expression::AtomicResult { .. }
            )
        })
        .count()
}

#[derive(Clone)]
struct Choice {
    selector: Expr,
    signed: bool,
    branches: Vec<(Vec<Expr>, Expr)>,
}
struct LoadChoice {
    choice: Choice,
    pointer: Expr,
    zero: Expr,
    values: Vec<Expr>,
}
struct AtomicChoice {
    pointer: Expr,
    zero: Expr,
    results: Vec<Expr>,
    original_result: Expr,
}

fn rewrite(
    f: &mut naga::Function,
    info: &naga::valid::FunctionInfo,
    types: &mut naga::UniqueArena<naga::Type>,
    arrays: &HashMap<Global, u32>,
    elements: &HashMap<Global, Vec<Global>>,
    slots: &[Expr],
) -> Result<()> {
    let mut slots = slots.iter().copied();
    let original = f.expressions.clone();
    let mut paths = HashMap::new();
    find_stores(&f.body, &f.expressions, arrays, &mut paths);
    let mut atomics = Vec::new();
    find_atomics(&f.body, &f.expressions, arrays, &mut paths, &mut atomics);
    let mut choices = HashMap::new();
    let mut roots = HashMap::new();
    for (pointer, p) in paths {
        let signed = match info[p.selector].ty.inner_with(types) {
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
        let span = f.expressions.get_span(pointer);
        let mut branches = Vec::new();
        for &alias in &elements[&p.global] {
            let mut current = f
                .expressions
                .append(Expression::GlobalVariable(alias), span);
            let mut emitted = Vec::new();
            for &part in &p.tail {
                let expr = match f.expressions[part] {
                    Expression::Access { index, .. } => Expression::Access {
                        base: current,
                        index,
                    },
                    Expression::AccessIndex { index, .. } => Expression::AccessIndex {
                        base: current,
                        index,
                    },
                    _ => unreachable!(),
                };
                current = f.expressions.append(expr, span);
                emitted.push(current);
            }
            branches.push((emitted, current));
        }
        roots.insert(p.root, elements[&p.global][0]);
        choices.insert(
            pointer,
            Choice {
                selector: p.selector,
                signed,
                branches,
            },
        );
    }
    let loads: Vec<_> = f
        .expressions
        .iter()
        .filter_map(|(h, expr)| match *expr {
            Expression::Load { pointer } | Expression::ArrayLength(pointer) => {
                path(&f.expressions, pointer, arrays)
                    .map(|p| (h, pointer, p, matches!(*expr, Expression::ArrayLength(_))))
            }
            _ => None,
        })
        .collect();
    let mut load_choices = HashMap::new();
    let mut removed: HashSet<_> = roots.keys().copied().collect();
    for (load, pointer, p, length) in loads {
        let _ = pointer;
        let slot = slots.next().ok_or(Error::InvalidRange)?;
        let signed = match info[p.selector].ty.inner_with(types) {
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
        let ty = info[load].ty.handle().unwrap_or_else(|| {
            types.insert(
                naga::Type {
                    name: None,
                    inner: info[load].ty.inner_with(types).clone(),
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
        let span = f.expressions.get_span(load);
        let zero = f.expressions.append(Expression::ZeroValue(ty), span);
        let mut branches = Vec::new();
        let mut values = Vec::new();
        for &alias in &elements[&p.global] {
            let mut current = f
                .expressions
                .append(Expression::GlobalVariable(alias), span);
            let mut emitted = Vec::new();
            for &part in &p.tail {
                let expr = match original[part] {
                    Expression::Access { index, .. } => Expression::Access {
                        base: current,
                        index,
                    },
                    Expression::AccessIndex { index, .. } => Expression::AccessIndex {
                        base: current,
                        index,
                    },
                    _ => unreachable!(),
                };
                current = f.expressions.append(expr, span);
                emitted.push(current);
            }
            values.push(f.expressions.append(
                if length {
                    Expression::ArrayLength(current)
                } else {
                    Expression::Load { pointer: current }
                },
                span,
            ));
            branches.push((emitted, current));
        }
        roots.insert(p.root, elements[&p.global][0]);
        removed.insert(p.root);
        f.expressions[slot] = Expression::LocalVariable(local);
        f.expressions[load] = Expression::Load { pointer: slot };
        load_choices.insert(
            load,
            LoadChoice {
                choice: Choice {
                    selector: p.selector,
                    signed,
                    branches,
                },
                pointer: slot,
                zero,
                values,
            },
        );
    }
    let mut atomic_choices = HashMap::new();
    for (pointer, result) in atomics {
        if let Some(result) = result {
            let ty = info[result].ty.handle().unwrap_or_else(|| {
                types.insert(
                    naga::Type {
                        name: None,
                        inner: info[result].ty.inner_with(types).clone(),
                    },
                    naga::Span::UNDEFINED,
                )
            });
            let slot = slots.next().ok_or(Error::InvalidRange)?;
            let local = f.local_variables.append(
                naga::LocalVariable {
                    name: None,
                    ty,
                    init: None,
                },
                naga::Span::UNDEFINED,
            );
            let span = f.expressions.get_span(result);
            let zero = f.expressions.append(Expression::ZeroValue(ty), span);
            let template = f.expressions[result].clone();
            let results = (0..choices[&pointer].branches.len())
                .map(|_| f.expressions.append(template.clone(), span))
                .collect();
            f.expressions[slot] = Expression::LocalVariable(local);
            f.expressions[result] = Expression::Load { pointer: slot };
            atomic_choices.insert(
                result,
                AtomicChoice {
                    pointer: slot,
                    zero,
                    results,
                    original_result: result,
                },
            );
        }
    }
    reject_non_store_uses(&f.body, &original, arrays)?;
    for (root, alias) in roots {
        f.expressions[root] = Expression::GlobalVariable(alias);
    }
    transform(
        &mut f.body,
        &choices,
        &load_choices,
        &atomic_choices,
        &removed,
    );
    Ok(())
}

fn find_atomics(
    block: &naga::Block,
    expressions: &naga::Arena<Expression>,
    arrays: &HashMap<Global, u32>,
    paths: &mut HashMap<Expr, Path>,
    atomics: &mut Vec<(Expr, Option<Expr>)>,
) {
    for s in block.iter() {
        match s {
            Statement::Atomic {
                pointer, result, ..
            } => {
                if let Some(p) = path(expressions, *pointer, arrays) {
                    paths.insert(*pointer, p);
                    atomics.push((*pointer, *result));
                }
            }
            Statement::Block(b) => find_atomics(b, expressions, arrays, paths, atomics),
            Statement::If { accept, reject, .. } => {
                find_atomics(accept, expressions, arrays, paths, atomics);
                find_atomics(reject, expressions, arrays, paths, atomics);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    find_atomics(&c.body, expressions, arrays, paths, atomics);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                find_atomics(body, expressions, arrays, paths, atomics);
                find_atomics(continuing, expressions, arrays, paths, atomics);
            }
            _ => {}
        }
    }
}

fn find_stores(
    block: &naga::Block,
    expressions: &naga::Arena<Expression>,
    arrays: &HashMap<Global, u32>,
    out: &mut HashMap<Expr, Path>,
) {
    for s in block.iter() {
        match s {
            Statement::Store { pointer, .. } => {
                if let Some(p) = path(expressions, *pointer, arrays) {
                    out.insert(*pointer, p);
                }
            }
            Statement::Block(b) => find_stores(b, expressions, arrays, out),
            Statement::If { accept, reject, .. } => {
                find_stores(accept, expressions, arrays, out);
                find_stores(reject, expressions, arrays, out);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    find_stores(&c.body, expressions, arrays, out);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                find_stores(body, expressions, arrays, out);
                find_stores(continuing, expressions, arrays, out);
            }
            _ => {}
        }
    }
}

fn reject_non_store_uses(
    block: &naga::Block,
    expressions: &naga::Arena<Expression>,
    arrays: &HashMap<Global, u32>,
) -> Result<()> {
    for s in block.iter() {
        match s {
            Statement::Call { arguments, .. }
                if arguments
                    .iter()
                    .any(|&p| path(expressions, p, arrays).is_some()) =>
            {
                return Err(Error::Unsupported(
                    "dynamic buffer pointer escapes to a function",
                ));
            }
            Statement::Block(b) => reject_non_store_uses(b, expressions, arrays)?,
            Statement::If { accept, reject, .. } => {
                reject_non_store_uses(accept, expressions, arrays)?;
                reject_non_store_uses(reject, expressions, arrays)?;
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    reject_non_store_uses(&c.body, expressions, arrays)?;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                reject_non_store_uses(body, expressions, arrays)?;
                reject_non_store_uses(continuing, expressions, arrays)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn transform(
    block: &mut naga::Block,
    choices: &HashMap<Expr, Choice>,
    loads: &HashMap<Expr, LoadChoice>,
    atomics: &HashMap<Expr, AtomicChoice>,
    removed: &HashSet<Expr>,
) {
    let old = std::mem::take(block);
    for (mut statement, span) in old.span_into_iter() {
        if let Statement::Emit(range) = &statement {
            let mut start = None;
            let mut end = None;
            for h in range.clone() {
                if removed.contains(&h) || loads.contains_key(&h) {
                    if let (Some(a), Some(b)) = (start.take(), end.take()) {
                        block.push(Statement::Emit(naga::Range::new_from_bounds(a, b)), span);
                    }
                    if let Some(load) = loads.get(&h) {
                        block.push(
                            Statement::Store {
                                pointer: load.pointer,
                                value: load.zero,
                            },
                            span,
                        );
                        let mut cases = Vec::new();
                        for (i, ((emitted, _), value)) in
                            load.choice.branches.iter().zip(&load.values).enumerate()
                        {
                            let mut body = naga::Block::new();
                            if let Some(a) = emitted.first() {
                                body.push(
                                    Statement::Emit(naga::Range::new_from_bounds(*a, *value)),
                                    span,
                                );
                            } else {
                                body.push(
                                    Statement::Emit(naga::Range::new_from_bounds(*value, *value)),
                                    span,
                                );
                            }
                            body.push(
                                Statement::Store {
                                    pointer: load.pointer,
                                    value: *value,
                                },
                                span,
                            );
                            cases.push(naga::SwitchCase {
                                value: if load.choice.signed {
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
                                selector: load.choice.selector,
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
        if let Statement::Store { pointer, value } = statement {
            if let Some(choice) = choices.get(&pointer) {
                let mut cases = Vec::new();
                for (i, (emitted, ptr)) in choice.branches.iter().enumerate() {
                    let mut body = naga::Block::new();
                    if let (Some(a), Some(b)) = (emitted.first(), emitted.last()) {
                        body.push(Statement::Emit(naga::Range::new_from_bounds(*a, *b)), span);
                    }
                    body.push(
                        Statement::Store {
                            pointer: *ptr,
                            value,
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
                continue;
            }
            statement = Statement::Store { pointer, value };
        }
        if let Statement::Atomic {
            pointer,
            fun,
            value,
            result,
        } = statement
        {
            if let Some(choice) = choices.get(&pointer) {
                let atomic = result.and_then(|r| atomics.get(&r));
                if let Some(a) = atomic {
                    block.push(
                        Statement::Store {
                            pointer: a.pointer,
                            value: a.zero,
                        },
                        span,
                    );
                }
                let mut cases = Vec::new();
                for (i, (emitted, ptr)) in choice.branches.iter().enumerate() {
                    let mut body = naga::Block::new();
                    if let (Some(a), Some(b)) = (emitted.first(), emitted.last()) {
                        body.push(Statement::Emit(naga::Range::new_from_bounds(*a, *b)), span);
                    }
                    let branch_result = atomic.map(|a| a.results[i]);
                    body.push(
                        Statement::Atomic {
                            pointer: *ptr,
                            fun,
                            value,
                            result: branch_result,
                        },
                        span,
                    );
                    if let Some(a) = atomic {
                        body.push(
                            Statement::Store {
                                pointer: a.pointer,
                                value: a.results[i],
                            },
                            span,
                        );
                    }
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
                if let Some(a) = atomic {
                    block.push(
                        Statement::Emit(naga::Range::new_from_bounds(
                            a.original_result,
                            a.original_result,
                        )),
                        span,
                    );
                }
                continue;
            }
            statement = Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            };
        }
        match &mut statement {
            Statement::Block(b) => transform(b, choices, loads, atomics, removed),
            Statement::If { accept, reject, .. } => {
                transform(accept, choices, loads, atomics, removed);
                transform(reject, choices, loads, atomics, removed);
            }
            Statement::Switch { cases, .. } => {
                for c in cases {
                    transform(&mut c.body, choices, loads, atomics, removed);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                transform(body, choices, loads, atomics, removed);
                transform(continuing, choices, loads, atomics, removed);
            }
            _ => {}
        }
        block.push(statement, span);
    }
}

#[cfg(test)]
#[path = "binding_array_dynamic_buffer_tests.rs"]
mod tests;
