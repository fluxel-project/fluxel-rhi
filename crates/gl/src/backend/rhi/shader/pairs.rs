//! Splits direct WGSL image references into one Naga global per sampled pair.
//!
//! GLSL combines images and samplers into a single uniform. Synthetic globals
//! preserve the original logical texture binding in reflection while allowing
//! the writer to emit a distinct uniform for each image/sampler pair.
use fluxel_rhi_core::api::{Error, Result};
use std::collections::HashMap;

type Global = naga::Handle<naga::GlobalVariable>;

pub(super) fn split(module: &mut naga::Module) -> Result<HashMap<Global, Global>> {
    let mut pairs = Vec::new();
    let mut depth_loads = Vec::new();
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|ep| &ep.function))
    {
        for (_, expression) in function.expressions.iter() {
            if let naga::Expression::ImageLoad { image, .. } = *expression {
                if let naga::Expression::GlobalVariable(global) = function.expressions[image] {
                    if matches!(
                        module.types[module.global_variables[global].ty].inner,
                        naga::TypeInner::Image {
                            class: naga::ImageClass::Depth { .. },
                            ..
                        }
                    ) && !depth_loads.contains(&global)
                    {
                        depth_loads.push(global);
                    }
                }
            }
            if let naga::Expression::ImageSample { image, sampler, .. } = *expression {
                if let (
                    naga::Expression::GlobalVariable(image),
                    naga::Expression::GlobalVariable(sampler),
                ) = (&function.expressions[image], &function.expressions[sampler])
                {
                    let pair = (*image, *sampler);
                    if !pairs.contains(&pair) {
                        pairs.push(pair);
                    }
                }
            }
        }
    }
    let mut by_pair = HashMap::new();
    let mut originals = HashMap::new();
    if pairs.is_empty() && depth_loads.is_empty() {
        return Ok(originals);
    }
    let group = module
        .global_variables
        .iter()
        .filter_map(|(_, g)| g.binding.as_ref().map(|b| b.group))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(Error::InvalidRange)?;
    let pair_count = pairs.len();
    for (index, (image, sampler)) in pairs.into_iter().enumerate() {
        let mut cloned = module.global_variables[image].clone();
        cloned.name = Some(format!("fluxel_sampled_pair_{index}"));
        cloned.binding = Some(naga::ResourceBinding {
            group,
            binding: u32::try_from(index).map_err(|_| Error::InvalidRange)?,
        });
        let span = module.global_variables.get_span(image);
        let synthetic = module.global_variables.append(cloned, span);
        by_pair.insert((image, sampler), synthetic);
        originals.insert(synthetic, image);
    }
    let mut loads = HashMap::new();
    for (index, image) in depth_loads.into_iter().enumerate() {
        let mut cloned = module.global_variables[image].clone();
        let ty = &module.types[cloned.ty];
        let naga::TypeInner::Image {
            dim,
            arrayed,
            class: naga::ImageClass::Depth { multi },
        } = ty.inner
        else {
            unreachable!()
        };
        let ty = naga::Type {
            name: None,
            inner: naga::TypeInner::Image {
                dim,
                arrayed,
                class: naga::ImageClass::Sampled {
                    kind: naga::ScalarKind::Float,
                    multi,
                },
            },
        };
        cloned.ty = module.types.insert(ty, naga::Span::UNDEFINED);
        cloned.name = Some(format!("fluxel_depth_load_{index}"));
        cloned.binding = Some(naga::ResourceBinding {
            group,
            binding: u32::try_from(pair_count + index).map_err(|_| Error::InvalidRange)?,
        });
        let synthetic = module
            .global_variables
            .append(cloned, module.global_variables.get_span(image));
        loads.insert(image, synthetic);
        originals.insert(synthetic, image);
    }
    for (_, function) in module.functions.iter_mut() {
        super::remap::rebuild(function, &by_pair, &loads);
    }
    for entry in &mut module.entry_points {
        super::remap::rebuild(&mut entry.function, &by_pair, &loads);
    }
    Ok(originals)
}
