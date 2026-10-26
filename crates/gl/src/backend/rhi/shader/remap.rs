//! Rebuilds function expression handles after inserting pre-scoped resource
//! expressions. Every IR operand and statement is remapped as one transaction.
use naga::{Expression as E, Handle, Statement as S};
use std::collections::HashMap;
type Expr = Handle<E>;
type Global = Handle<naga::GlobalVariable>;
type Map = HashMap<Expr, Expr>;
fn h(value: &mut Expr, map: &Map) {
    *value = map[value];
}
fn opt(value: &mut Option<Expr>, map: &Map) {
    if let Some(value) = value {
        h(value, map);
    }
}
fn expression(e: &mut E, map: &Map) {
    match e {
        E::Compose { components, .. } => {
            for v in components {
                h(v, map);
            }
        }
        E::Access { base, index } => {
            h(base, map);
            h(index, map);
        }
        E::AccessIndex { base, .. } => h(base, map),
        E::Splat { value, .. } => h(value, map),
        E::Swizzle { vector, .. } => h(vector, map),
        E::Load { pointer } => h(pointer, map),
        E::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            ..
        } => {
            h(image, map);
            h(sampler, map);
            h(coordinate, map);
            opt(array_index, map);
            opt(offset, map);
            opt(depth_ref, map);
            match level {
                naga::SampleLevel::Exact(x) | naga::SampleLevel::Bias(x) => h(x, map),
                naga::SampleLevel::Gradient { x, y } => {
                    h(x, map);
                    h(y, map)
                }
                _ => {}
            }
        }
        E::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            h(image, map);
            h(coordinate, map);
            opt(array_index, map);
            opt(sample, map);
            opt(level, map)
        }
        E::ImageQuery { image, query } => {
            h(image, map);
            if let naga::ImageQuery::Size { level } = query {
                opt(level, map)
            }
        }
        E::Unary { expr, .. }
        | E::Derivative { expr, .. }
        | E::As { expr, .. }
        | E::ArrayLength(expr) => h(expr, map),
        E::Binary { left, right, .. } => {
            h(left, map);
            h(right, map)
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            h(condition, map);
            h(accept, map);
            h(reject, map)
        }
        E::Relational { argument, .. } => h(argument, map),
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            h(arg, map);
            opt(arg1, map);
            opt(arg2, map);
            opt(arg3, map)
        }
        E::RayQueryGetIntersection { query, .. } | E::RayQueryVertexPositions { query, .. } => {
            h(query, map)
        }
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_)
        | E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. }
        | E::RayQueryProceedResult => {}
    }
}
fn block(block: &mut naga::Block, map: &Map, first_map: &Map) {
    for s in block.iter_mut() {
        match s {
            S::Emit(range) => {
                let mut iter = range.clone();
                if let (Some(first), Some(last)) = (iter.next(), range.clone().last()) {
                    *range = naga::Range::new_from_bounds(first_map[&first], map[&last]);
                }
            }
            S::Block(b) => self::block(b, map, first_map),
            S::If {
                condition,
                accept,
                reject,
            } => {
                h(condition, map);
                self::block(accept, map, first_map);
                self::block(reject, map, first_map)
            }
            S::Switch { selector, cases } => {
                h(selector, map);
                for c in cases {
                    self::block(&mut c.body, map, first_map)
                }
            }
            S::Loop {
                body,
                continuing,
                break_if,
            } => {
                self::block(body, map, first_map);
                self::block(continuing, map, first_map);
                opt(break_if, map)
            }
            S::Return { value } => opt(value, map),
            S::Store { pointer, value } => {
                h(pointer, map);
                h(value, map)
            }
            S::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                h(image, map);
                h(coordinate, map);
                opt(array_index, map);
                h(value, map)
            }
            S::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                h(pointer, map);
                if let naga::AtomicFunction::Exchange { compare } = fun {
                    opt(compare, map)
                }
                h(value, map);
                opt(result, map)
            }
            S::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                h(image, map);
                h(coordinate, map);
                opt(array_index, map);
                h(value, map)
            }
            S::WorkGroupUniformLoad { pointer, result } => {
                h(pointer, map);
                h(result, map)
            }
            S::Call {
                arguments, result, ..
            } => {
                for a in arguments {
                    h(a, map)
                }
                opt(result, map)
            }
            S::RayQuery { query, fun } => {
                h(query, map);
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        h(acceleration_structure, map);
                        h(descriptor, map)
                    }
                    naga::RayQueryFunction::Proceed { result } => h(result, map),
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => h(hit_t, map),
                    _ => {}
                }
            }
            S::SubgroupBallot { result, predicate } => {
                h(result, map);
                opt(predicate, map)
            }
            S::SubgroupCollectiveOperation {
                argument, result, ..
            } => {
                h(argument, map);
                h(result, map)
            }
            S::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                match mode {
                    naga::GatherMode::Broadcast(x)
                    | naga::GatherMode::Shuffle(x)
                    | naga::GatherMode::ShuffleDown(x)
                    | naga::GatherMode::ShuffleUp(x)
                    | naga::GatherMode::ShuffleXor(x)
                    | naga::GatherMode::QuadBroadcast(x) => h(x, map),
                    _ => {}
                }
                h(argument, map);
                h(result, map)
            }
            S::Break | S::Continue | S::Kill | S::ControlBarrier(_) | S::MemoryBarrier(_) => {}
        }
    }
}
pub(super) fn rebuild(
    function: &mut naga::Function,
    pairs: &HashMap<(Global, Global), Global>,
    loads: &HashMap<Global, Global>,
) {
    let old = std::mem::take(&mut function.expressions);
    let mut arena = naga::Arena::new();
    let mut units = HashMap::new();
    for (&pair, &global) in pairs {
        units.insert(
            pair,
            arena.append(E::GlobalVariable(global), naga::Span::UNDEFINED),
        );
    }
    let mut load_units = HashMap::new();
    for (&original, &global) in loads {
        load_units.insert(
            original,
            arena.append(E::GlobalVariable(global), naga::Span::UNDEFINED),
        );
    }
    let mut map = Map::new();
    let mut first_map = Map::new();
    for (handle, original) in old.iter() {
        let target = if let E::ImageSample { image, sampler, .. } = original {
            match (&old[*image], &old[*sampler]) {
                (E::GlobalVariable(i), E::GlobalVariable(s)) => units.get(&(*i, *s)).copied(),
                _ => None,
            }
        } else {
            None
        };
        let load_target = if let E::ImageLoad { image, .. } = original {
            match old[*image] {
                E::GlobalVariable(global) => load_units.get(&global).copied(),
                _ => None,
            }
        } else {
            None
        };
        let mut next = original.clone();
        expression(&mut next, &map);
        if let (Some(target), E::ImageSample { image, .. }) = (target, &mut next) {
            *image = target;
        }
        if let (Some(target), E::ImageLoad { image, .. }) = (load_target, &mut next) {
            *image = target;
        }
        let first = arena.append(next, old.get_span(handle));
        let new = if load_target.is_some() {
            arena.append(
                E::AccessIndex {
                    base: first,
                    index: 0,
                },
                old.get_span(handle),
            )
        } else {
            first
        };
        first_map.insert(handle, first);
        map.insert(handle, new);
    }
    for (_, local) in function.local_variables.iter_mut() {
        opt(&mut local.init, &map)
    }
    function.named_expressions = std::mem::take(&mut function.named_expressions)
        .into_iter()
        .map(|(k, v)| (map[&k], v))
        .collect();
    block(&mut function.body, &map, &first_map);
    function.expressions = arena;
}
