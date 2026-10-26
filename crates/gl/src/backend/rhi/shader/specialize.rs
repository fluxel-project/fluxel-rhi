//! Specializes WGSL helper functions for the concrete resource arguments at
//! each call site, allowing combined GLSL sampler uniforms to be reflected.
use fluxel_rhi_core::api::{Error, Result};
use std::collections::HashMap;
type Global = naga::Handle<naga::GlobalVariable>;
type Function = naga::Handle<naga::Function>;

pub(super) fn specialize(module: &mut naga::Module) -> Result<()> {
    let mut entries = std::mem::take(&mut module.entry_points);
    let mut cache = HashMap::new();
    for entry in &mut entries {
        visit_function(module, &mut entry.function, &mut cache)?;
    }
    module.entry_points = entries;
    Ok(())
}

type Cache = HashMap<(Function, Vec<Option<Global>>), Function>;
fn visit_function(
    module: &mut naga::Module,
    function: &mut naga::Function,
    cache: &mut Cache,
) -> Result<()> {
    let expressions = &mut function.expressions;
    visit_block(module, &mut function.body, expressions, cache)
}
fn visit_block(
    module: &mut naga::Module,
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    cache: &mut Cache,
) -> Result<()> {
    for statement in block.iter_mut() {
        match statement {
            naga::Statement::Block(inner) => visit_block(module, inner, expressions, cache)?,
            naga::Statement::If { accept, reject, .. } => {
                visit_block(module, accept, expressions, cache)?;
                visit_block(module, reject, expressions, cache)?;
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    visit_block(module, &mut case.body, expressions, cache)?;
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                visit_block(module, body, expressions, cache)?;
                visit_block(module, continuing, expressions, cache)?;
            }
            naga::Statement::Call {
                function,
                arguments,
                result,
            } => {
                let original = *function;
                let called = &module.functions[original];
                let mut globals = Vec::with_capacity(arguments.len());
                for (argument, parameter) in arguments.iter().zip(&called.arguments) {
                    let is_resource = matches!(
                        module.types[parameter.ty].inner,
                        naga::TypeInner::Image { .. } | naga::TypeInner::Sampler { .. }
                    );
                    let global = if is_resource {
                        match expressions[*argument] {
                            naga::Expression::GlobalVariable(global) => Some(global),
                            _ => {
                                return Err(Error::Unsupported(
                                    "GL resource function argument cannot be resolved",
                                ));
                            }
                        }
                    } else {
                        None
                    };
                    globals.push(global);
                }
                let key = (original, globals.clone());
                let replacement = if let Some(found) = cache.get(&key) {
                    *found
                } else {
                    let mut clone = called.clone();
                    clone.name = Some(format!(
                        "fluxel_specialized_{}_{}",
                        original.index(),
                        cache.len()
                    ));
                    for (_, expression) in clone.expressions.iter_mut() {
                        if let naga::Expression::FunctionArgument(index) = *expression {
                            if let Some(Some(global)) = globals.get(index as usize) {
                                *expression = naga::Expression::GlobalVariable(*global);
                            }
                        }
                    }
                    visit_function(module, &mut clone, cache)?;
                    let new = module
                        .functions
                        .append(clone, module.functions.get_span(original));
                    cache.insert(key, new);
                    new
                };
                *function = replacement;
                if let Some(result) = result {
                    expressions[*result] = naga::Expression::CallResult(replacement);
                }
            }
            _ => {}
        }
    }
    Ok(())
}
