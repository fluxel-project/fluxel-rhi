//! Translate Naga resource reflection into the GL program binding ABI.

use super::*;

pub(super) fn is_storage_image(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
) -> bool {
    matches!(
        module.types[module.global_variables[global].ty].inner,
        naga::TypeInner::Image {
            class: naga::ImageClass::Storage { .. },
            ..
        }
    )
}

pub(super) fn collect_naga_bindings(
    module: &naga::Module,
    reflection: naga::back::glsl::ReflectionInfo,
    image_units: &naga::back::glsl::BindingMap,
    image_aliases: &std::collections::HashMap<
        naga::Handle<naga::GlobalVariable>,
        naga::Handle<naga::GlobalVariable>,
    >,
    array_elements: &std::collections::HashMap<
        naga::Handle<naga::GlobalVariable>,
        super::super::binding_array::Element,
    >,
) -> Vec<GlNagaBindingName> {
    let mut bindings = Vec::new();
    for (global, name) in reflection.uniforms {
        push_binding(
            &mut bindings,
            module,
            global,
            name,
            image_units,
            array_elements,
        );
    }
    for (name, mapping) in reflection.texture_mapping {
        // GLSL has a combined texture/sampler uniform. Both WebGPU bindings
        // must resolve to that one reflected name for layout validation.
        push_binding(
            &mut bindings,
            module,
            image_aliases
                .get(&mapping.texture)
                .copied()
                .unwrap_or(mapping.texture),
            name.clone(),
            image_units,
            array_elements,
        );
        if let Some(sampler) = mapping.sampler {
            push_binding(
                &mut bindings,
                module,
                sampler,
                name,
                image_units,
                array_elements,
            );
        }
    }
    bindings.sort_by(|left, right| {
        (left.group, left.binding, left.element, &left.name).cmp(&(
            right.group,
            right.binding,
            right.element,
            &right.name,
        ))
    });
    bindings.dedup_by(|left, right| {
        left.group == right.group
            && left.binding == right.binding
            && left.element == right.element
            && left.name == right.name
    });
    bindings
}

fn push_binding(
    output: &mut Vec<GlNagaBindingName>,
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
    name: String,
    image_units: &naga::back::glsl::BindingMap,
    array_elements: &std::collections::HashMap<
        naga::Handle<naga::GlobalVariable>,
        super::super::binding_array::Element,
    >,
) {
    let native_binding = module.global_variables[global].binding;
    let element = array_elements.get(&global);
    let logical_global = element.map_or(global, |element| element.original);
    if let Some(binding) = module.global_variables[logical_global].binding {
        output.push(GlNagaBindingName {
            group: binding.group,
            binding: binding.binding,
            element: element.map(|value| value.index),
            name,
            image_unit: native_binding
                .and_then(|binding| image_units.get(&binding).copied())
                .map(u32::from),
            resource_type: naga_resource_type(module, global),
        });
    }
}

fn naga_resource_type(
    module: &naga::Module,
    global: naga::Handle<naga::GlobalVariable>,
) -> Option<super::super::super::types::GlNagaResourceType> {
    use super::super::super::types::GlNagaResourceType as R;
    use api::{TextureSampleType as S, TextureViewDimension as D};
    match module.types[module.global_variables[global].ty].inner {
        naga::TypeInner::Sampler { comparison } => Some(R::Sampler { comparison }),
        naga::TypeInner::Image {
            dim,
            arrayed,
            class,
        } => {
            let dimension = match (dim, arrayed) {
                (naga::ImageDimension::D1, false) => D::D1,
                (naga::ImageDimension::D2, false) => D::D2,
                (naga::ImageDimension::D2, true) => D::D2Array,
                (naga::ImageDimension::D3, false) => D::D3,
                (naga::ImageDimension::Cube, false) => D::Cube,
                (naga::ImageDimension::Cube, true) => D::CubeArray,
                _ => return None,
            };
            let (sample, multisampled) = match class {
                naga::ImageClass::Sampled { kind, multi } => (
                    match kind {
                        naga::ScalarKind::Float => S::Float,
                        naga::ScalarKind::Sint => S::Sint,
                        naga::ScalarKind::Uint => S::Uint,
                        _ => return None,
                    },
                    multi,
                ),
                naga::ImageClass::Depth { multi } => (S::Depth, multi),
                naga::ImageClass::Storage { format, access } => {
                    let format = naga_storage_format(format)?;
                    let read = access.contains(naga::StorageAccess::LOAD);
                    let write = access.contains(naga::StorageAccess::STORE);
                    let access = match (read, write) {
                        (true, true) => api::StorageAccess::ReadWrite,
                        (true, false) => api::StorageAccess::Read,
                        (false, true) => api::StorageAccess::Write,
                        _ => return None,
                    };
                    return Some(R::StorageTexture {
                        dimension,
                        format,
                        access,
                    });
                }
                naga::ImageClass::External => return None,
            };
            Some(R::SampledTexture {
                dimension,
                sample,
                multisampled,
            })
        }
        _ => None,
    }
}

fn naga_storage_format(format: naga::StorageFormat) -> Option<api::TextureFormat> {
    use api::TextureFormat as T;
    use naga::StorageFormat as N;
    Some(match format {
        N::R8Unorm => T::R8Unorm,
        N::R8Snorm => T::R8Snorm,
        N::R8Uint => T::R8Uint,
        N::R8Sint => T::R8Sint,
        N::R16Uint => T::R16Uint,
        N::R16Sint => T::R16Sint,
        N::R16Float => T::R16Float,
        N::Rg8Unorm => T::Rg8Unorm,
        N::Rg8Snorm => T::Rg8Snorm,
        N::Rg8Uint => T::Rg8Uint,
        N::Rg8Sint => T::Rg8Sint,
        N::R32Uint => T::R32Uint,
        N::R32Sint => T::R32Sint,
        N::R32Float => T::R32Float,
        N::Rg16Uint => T::Rg16Uint,
        N::Rg16Sint => T::Rg16Sint,
        N::Rg16Float => T::Rg16Float,
        N::Rgba8Unorm => T::Rgba8Unorm,
        N::Rgba8Snorm => T::Rgba8Snorm,
        N::Rgba8Uint => T::Rgba8Uint,
        N::Rgba8Sint => T::Rgba8Sint,
        N::Bgra8Unorm => T::Bgra8Unorm,
        N::Rgb10a2Uint => T::Rgb10a2Uint,
        N::Rgb10a2Unorm => T::Rgb10a2Unorm,
        N::Rg11b10Ufloat => T::Rg11b10Ufloat,
        N::R64Uint => T::R64Uint,
        N::Rg32Uint => T::Rg32Uint,
        N::Rg32Sint => T::Rg32Sint,
        N::Rg32Float => T::Rg32Float,
        N::Rgba16Uint => T::Rgba16Uint,
        N::Rgba16Sint => T::Rgba16Sint,
        N::Rgba16Float => T::Rgba16Float,
        N::Rgba32Uint => T::Rgba32Uint,
        N::Rgba32Sint => T::Rgba32Sint,
        N::Rgba32Float => T::Rgba32Float,
        N::R16Unorm => T::R16Unorm,
        N::R16Snorm => T::R16Snorm,
        N::Rg16Unorm => T::Rg16Unorm,
        N::Rg16Snorm => T::Rg16Snorm,
        N::Rgba16Unorm => T::Rgba16Unorm,
        N::Rgba16Snorm => T::Rgba16Snorm,
    })
}

/// Assigns active WGSL storage images dense GLSL image units in logical
/// group/binding order. GLES image uniforms cannot be updated with
/// `glUniform1i`; the same unit is emitted as a GLSL binding qualifier and
/// carried into bind-group execution.
pub(super) fn naga_storage_image_units(
    module: &naga::Module,
    entry: &naga::valid::FunctionInfo,
) -> Result<naga::back::glsl::BindingMap> {
    let mut resources = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| {
            let binding = global.binding.as_ref()?;
            (!entry[handle].is_empty() && is_storage_image(module, handle))
                .then_some(binding.clone())
        })
        .collect::<Vec<_>>();
    resources.sort_by_key(|binding| (binding.group, binding.binding));
    resources.dedup();
    resources
        .into_iter()
        .enumerate()
        .map(|(unit, binding)| {
            let unit =
                u8::try_from(unit).map_err(|_| Error::Unsupported("too many storage images"))?;
            Ok((binding, unit))
        })
        .collect()
}
