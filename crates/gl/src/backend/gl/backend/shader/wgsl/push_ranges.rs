//! Find push-constant members referenced by reachable shader functions.

/// Collect the top-level members actually referenced by the selected entry's
/// reachable functions. Some GL drivers report every member of an active
/// struct uniform, including WGSL members never read by the shader.
pub(super) fn active_push_ranges(
    module: &naga::Module,
    entry_index: usize,
) -> Vec<std::ops::Range<u32>> {
    use naga::{Expression as E, Statement as S, TypeInner as T};
    fn root(
        function: &naga::Function,
        mut expr: naga::Handle<E>,
        module: &naga::Module,
    ) -> Option<(naga::Handle<naga::GlobalVariable>, Option<u32>)> {
        let mut member = None;
        loop {
            match function.expressions[expr] {
                E::GlobalVariable(handle)
                    if module.global_variables[handle].space
                        == naga::AddressSpace::PushConstant =>
                {
                    return Some((handle, member));
                }
                E::AccessIndex { base, index } => {
                    member = Some(index);
                    expr = base;
                }
                E::Access { base, .. } => {
                    member = None;
                    expr = base;
                }
                _ => return None,
            }
        }
    }
    fn add(
        module: &naga::Module,
        root: (naga::Handle<naga::GlobalVariable>, Option<u32>),
        out: &mut Vec<std::ops::Range<u32>>,
    ) {
        let ty = &module.types[module.global_variables[root.0].ty].inner;
        if let T::Struct { members, span } = ty {
            match root.1.and_then(|index| {
                members
                    .get(index as usize)
                    .map(|member| (index as usize, member.offset))
            }) {
                Some((index, start)) => {
                    out.push(start..members.get(index + 1).map_or(*span, |member| member.offset))
                }
                None => out.push(0..*span),
            }
        } else {
            // This range is only used to find overlapping reflected leaves;
            // the device limit is checked separately when the layout is made.
            out.push(0..u32::MAX);
        }
    }
    fn block(
        module: &naga::Module,
        function: &naga::Function,
        body: &naga::Block,
        visited: &mut std::collections::HashSet<naga::Handle<naga::Function>>,
        out: &mut Vec<std::ops::Range<u32>>,
    ) {
        for statement in body {
            match statement {
                S::Call {
                    function: called,
                    arguments,
                    ..
                } => {
                    for arg in arguments {
                        if let Some(value) = root(function, *arg, module) {
                            add(module, value, out);
                        }
                    }
                    if visited.insert(*called) {
                        walk(module, &module.functions[*called], visited, out);
                    }
                }
                S::Block(inner) => block(module, function, inner, visited, out),
                S::If { accept, reject, .. } => {
                    block(module, function, accept, visited, out);
                    block(module, function, reject, visited, out);
                }
                S::Switch { cases, .. } => {
                    for case in cases {
                        block(module, function, &case.body, visited, out);
                    }
                }
                S::Loop {
                    body, continuing, ..
                } => {
                    block(module, function, body, visited, out);
                    block(module, function, continuing, visited, out);
                }
                _ => {}
            }
        }
    }
    fn walk(
        module: &naga::Module,
        function: &naga::Function,
        visited: &mut std::collections::HashSet<naga::Handle<naga::Function>>,
        out: &mut Vec<std::ops::Range<u32>>,
    ) {
        for (handle, expression) in function.expressions.iter() {
            match expression {
                E::AccessIndex { .. } | E::Access { .. } => {
                    if let Some(value) = root(function, handle, module) {
                        add(module, value, out);
                    }
                }
                E::Load { pointer } => {
                    if let Some(value) = root(function, *pointer, module) {
                        add(module, value, out);
                    }
                }
                _ => {}
            }
        }
        block(module, function, &function.body, visited, out);
    }
    let mut out = Vec::new();
    walk(
        module,
        &module.entry_points[entry_index].function,
        &mut std::collections::HashSet::new(),
        &mut out,
    );
    out
}
