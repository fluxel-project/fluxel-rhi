//! Replace texture level queries with backend-populated private scalar globals.
//!
//! Run after sampled array query specialization. The caller converts the
//! generated private declarations to uniforms and uploads each texture view's
//! mip count. This pass is intended for GLSL profiles without textureQueryLevels.
use fluxel_rhi_core::api::{Error, Result};
use naga::{Expression, GlobalVariable, Handle};
use std::collections::{HashMap, HashSet};
#[path = "binding_array_buffer_remap.rs"]
mod remap;

type Global = Handle<GlobalVariable>;

/// Returns (generated private global name, logical texture global).
/// The module must be discarded on error. Stage must be a distinct GLSL-safe
/// identifier per shader stage; the caller must validate the GLSL declaration.
pub(super) fn lower(module: &mut naga::Module, stage: &str) -> Result<Vec<(String, Global)>> {
    if stage.is_empty()
        || !stage
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
    {
        return Err(Error::InvalidRange);
    }
    let mut images = HashSet::new();
    for f in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, expr) in f.expressions.iter() {
            if let Expression::ImageQuery {
                image,
                query: naga::ImageQuery::NumLevels,
            } = *expr
            {
                match f.expressions[image] {
                    Expression::GlobalVariable(g) => {
                        images.insert(g);
                    }
                    _ => return Err(Error::Unsupported("unresolved texture levels query")),
                }
            }
        }
    }
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let mut images: Vec<_> = images.into_iter().collect();
    images.sort_by_key(|g| g.index());
    let mut used: HashSet<String> = module
        .global_variables
        .iter()
        .filter_map(|(_, g)| g.name.clone())
        .collect();
    let ty = module.types.insert(
        naga::Type {
            name: None,
            inner: naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Uint,
                width: 4,
            }),
        },
        naga::Span::UNDEFINED,
    );
    let mut mapping = HashMap::new();
    let mut result = Vec::new();
    for image in images {
        let base = format!("fluxel_internal_levels_{stage}_{}_value", image.index());
        let mut name = base.clone();
        let mut suffix = 0u32;
        while used.contains(&name) {
            suffix = suffix.checked_add(1).ok_or(Error::InvalidRange)?;
            name = format!("{base}_{suffix}_value");
        }
        used.insert(name.clone());
        let uniform = module.global_variables.append(
            GlobalVariable {
                name: Some(name.clone()),
                space: naga::AddressSpace::Private,
                binding: None,
                ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        mapping.insert(image, uniform);
        result.push((name, image));
    }
    for (_, f) in module.functions.iter_mut() {
        rewrite(f, &mapping)?;
    }
    for entry in &mut module.entry_points {
        rewrite(&mut entry.function, &mapping)?;
    }
    Ok(result)
}
fn rewrite(f: &mut naga::Function, mapping: &HashMap<Global, Global>) -> Result<()> {
    let count = f
        .expressions
        .iter()
        .filter(|(_, e)| {
            matches!(
                e,
                Expression::ImageQuery {
                    query: naga::ImageQuery::NumLevels,
                    ..
                }
            )
        })
        .count();
    let slots = remap::reserve(f, count);
    let queries: Vec<_> = f
        .expressions
        .iter()
        .filter_map(|(h, e)| match *e {
            Expression::ImageQuery {
                image,
                query: naga::ImageQuery::NumLevels,
            } => Some((h, image)),
            _ => None,
        })
        .collect();
    for ((query, image), slot) in queries.into_iter().zip(slots) {
        let Expression::GlobalVariable(texture) = f.expressions[image] else {
            return Err(Error::Unsupported("unresolved texture levels query"));
        };
        let uniform = *mapping.get(&texture).ok_or(Error::InvalidRange)?;
        f.expressions[slot] = Expression::GlobalVariable(uniform);
        f.expressions[query] = Expression::Load { pointer: slot };
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/backend/rhi/shader/texture_num_levels__tests.rs"]
mod tests;
