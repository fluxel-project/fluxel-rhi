//! Insert expression slots before a function's existing expression arena.
//!
//! Naga requires every expression operand to precede its consumer. Rebuilding
//! the arena keeps the original order, then adjusts expression and statement
//! handles, including emit ranges and named expressions.

use naga::{Expression as E, Handle, Statement as S};

type Expr = Handle<E>;

pub(super) fn reserve(f: &mut naga::Function, count: usize) -> Vec<Expr> {
    if count == 0 {
        return Vec::new();
    }
    let old = std::mem::take(&mut f.expressions);
    let mut slots = Vec::with_capacity(count);
    for _ in 0..count {
        slots.push(
            f.expressions
                .append(E::Literal(naga::Literal::U32(0)), naga::Span::UNDEFINED),
        );
    }
    let mut map = Vec::with_capacity(old.len());
    for (h, e) in old.iter() {
        map.push(f.expressions.append(e.clone(), old.get_span(h)));
    }
    for (h, e) in f.expressions.iter_mut() {
        if h.index() >= count {
            expression(e, &map);
        }
    }
    for (_, local) in f.local_variables.iter_mut() {
        optional(&mut local.init, &map);
    }
    block(&mut f.body, &map);
    let names = std::mem::take(&mut f.named_expressions);
    for (old, name) in names {
        f.named_expressions.insert(map[old.index()], name);
    }
    slots
}

fn fix(h: &mut Expr, map: &[Expr]) {
    *h = map[h.index()];
}
fn optional(h: &mut Option<Expr>, map: &[Expr]) {
    if let Some(h) = h {
        fix(h, map);
    }
}

fn expression(e: &mut E, map: &[Expr]) {
    match e {
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
        E::Compose { components, .. } => {
            for h in components {
                fix(h, map);
            }
        }
        E::Access { base, index } => {
            fix(base, map);
            fix(index, map);
        }
        E::AccessIndex { base, .. } => fix(base, map),
        E::Splat { value, .. } => fix(value, map),
        E::Swizzle { vector, .. } => fix(vector, map),
        E::Load { pointer } => fix(pointer, map),
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
            fix(image, map);
            fix(sampler, map);
            fix(coordinate, map);
            optional(array_index, map);
            optional(offset, map);
            optional(depth_ref, map);
            match level {
                naga::SampleLevel::Auto | naga::SampleLevel::Zero => {}
                naga::SampleLevel::Exact(h) | naga::SampleLevel::Bias(h) => fix(h, map),
                naga::SampleLevel::Gradient { x, y } => {
                    fix(x, map);
                    fix(y, map);
                }
            }
        }
        E::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            fix(image, map);
            fix(coordinate, map);
            optional(array_index, map);
            optional(sample, map);
            optional(level, map);
        }
        E::ImageQuery { image, query } => {
            fix(image, map);
            if let naga::ImageQuery::Size { level } = query {
                optional(level, map);
            }
        }
        E::Unary { expr, .. }
        | E::Derivative { expr, .. }
        | E::As { expr, .. }
        | E::ArrayLength(expr) => fix(expr, map),
        E::Binary { left, right, .. } => {
            fix(left, map);
            fix(right, map);
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            fix(condition, map);
            fix(accept, map);
            fix(reject, map);
        }
        E::Relational { argument, .. } => fix(argument, map),
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            fix(arg, map);
            optional(arg1, map);
            optional(arg2, map);
            optional(arg3, map);
        }
        E::RayQueryGetIntersection { query, .. } | E::RayQueryVertexPositions { query, .. } => {
            fix(query, map)
        }
    }
}

fn block(b: &mut naga::Block, map: &[Expr]) {
    for s in b.iter_mut() {
        match s {
            S::Emit(range) => {
                let mut iter = range.clone();
                if let Some(first) = iter.next() {
                    let last = iter.last().unwrap_or(first);
                    *range = naga::Range::new_from_bounds(map[first.index()], map[last.index()]);
                }
            }
            S::Block(b) => block(b, map),
            S::If {
                condition,
                accept,
                reject,
            } => {
                fix(condition, map);
                block(accept, map);
                block(reject, map);
            }
            S::Switch { selector, cases } => {
                fix(selector, map);
                for c in cases {
                    block(&mut c.body, map);
                }
            }
            S::Loop {
                body,
                continuing,
                break_if,
            } => {
                block(body, map);
                block(continuing, map);
                optional(break_if, map);
            }
            S::Return { value } => optional(value, map),
            S::Store { pointer, value } => {
                fix(pointer, map);
                fix(value, map);
            }
            S::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                fix(image, map);
                fix(coordinate, map);
                optional(array_index, map);
                fix(value, map);
            }
            S::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                fix(pointer, map);
                if let naga::AtomicFunction::Exchange { compare } = fun {
                    optional(compare, map);
                }
                fix(value, map);
                optional(result, map);
            }
            S::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                fix(image, map);
                fix(coordinate, map);
                optional(array_index, map);
                if let naga::AtomicFunction::Exchange { compare } = fun {
                    optional(compare, map);
                }
                fix(value, map);
            }
            S::WorkGroupUniformLoad { pointer, result } => {
                fix(pointer, map);
                fix(result, map);
            }
            S::Call {
                arguments, result, ..
            } => {
                for h in arguments {
                    fix(h, map);
                }
                optional(result, map);
            }
            S::RayQuery { query, fun } => {
                fix(query, map);
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        fix(acceleration_structure, map);
                        fix(descriptor, map);
                    }
                    naga::RayQueryFunction::Proceed { result } => fix(result, map),
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => fix(hit_t, map),
                    naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            S::SubgroupBallot { result, predicate } => {
                fix(result, map);
                optional(predicate, map);
            }
            S::SubgroupCollectiveOperation {
                argument, result, ..
            } => {
                fix(argument, map);
                fix(result, map);
            }
            S::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                match mode {
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => fix(h, map),
                }
                fix(argument, map);
                fix(result, map);
            }
            S::Break | S::Continue | S::Kill | S::ControlBarrier(_) | S::MemoryBarrier(_) => {}
        }
    }
}
