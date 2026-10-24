//! Checks WGSL resource types against the public pipeline binding contract.
use fluxel_rhi_core::api::{self, Error, Result};

pub(super) fn validate_binding_type(
    module: &naga::Module,
    global: &naga::GlobalVariable,
    ty: &api::BindingType,
) -> Result<()> {
    let inner = match module.types[global.ty].inner {
        naga::TypeInner::BindingArray { base, .. } => &module.types[base].inner,
        ref inner => inner,
    };
    let matches = match (global.space, inner, ty) {
        (naga::AddressSpace::Uniform, _, api::BindingType::UniformBuffer { .. }) => true,
        (
            naga::AddressSpace::Storage { access },
            _,
            api::BindingType::StorageBuffer {
                access: declared, ..
            },
        ) => storage_access(access, *declared),
        (
            naga::AddressSpace::Handle,
            naga::TypeInner::Sampler { comparison },
            api::BindingType::Sampler {
                comparison: declared,
                ..
            },
        ) => comparison == declared,
        (
            naga::AddressSpace::Handle,
            naga::TypeInner::Image {
                dim,
                arrayed,
                class,
            },
            api::BindingType::SampledTexture {
                sample,
                dimension,
                multisampled,
            },
        ) => {
            dimension_matches(*dim, *arrayed, *dimension)
                && match class {
                    naga::ImageClass::Sampled { kind, multi } => {
                        *multi == *multisampled
                            && matches!(
                                (kind, sample),
                                (
                                    naga::ScalarKind::Float,
                                    api::TextureSampleType::Float
                                        | api::TextureSampleType::UnfilterableFloat
                                ) | (naga::ScalarKind::Sint, api::TextureSampleType::Sint)
                                    | (naga::ScalarKind::Uint, api::TextureSampleType::Uint)
                            )
                    }
                    naga::ImageClass::Depth { multi } => {
                        *multi == *multisampled && *sample == api::TextureSampleType::Depth
                    }
                    _ => false,
                }
        }
        (
            naga::AddressSpace::Handle,
            naga::TypeInner::Image {
                dim,
                arrayed,
                class: naga::ImageClass::Storage { format, access },
            },
            api::BindingType::StorageTexture {
                format: declared,
                dimension,
                access: expected,
            },
        ) => {
            dimension_matches(*dim, *arrayed, *dimension)
                && storage_format(*format) == *declared
                && storage_access(*access, *expected)
        }
        (
            naga::AddressSpace::Handle,
            naga::TypeInner::AccelerationStructure { .. },
            api::BindingType::AccelerationStructure,
        ) => true,
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(Error::Shader(
            "WGSL resource type does not match pipeline layout".into(),
        ))
    }
}
fn storage_access(actual: naga::StorageAccess, expected: api::StorageAccess) -> bool {
    let read = actual.contains(naga::StorageAccess::LOAD);
    let write = actual.contains(naga::StorageAccess::STORE);
    match expected {
        api::StorageAccess::Read => read && !write,
        api::StorageAccess::Write => !read && write,
        api::StorageAccess::ReadWrite => read && write,
    }
}
fn dimension_matches(
    dim: naga::ImageDimension,
    array: bool,
    expected: api::TextureViewDimension,
) -> bool {
    matches!(
        (dim, array, expected),
        (
            naga::ImageDimension::D1,
            false,
            api::TextureViewDimension::D1
        ) | (
            naga::ImageDimension::D2,
            false,
            api::TextureViewDimension::D2
        ) | (
            naga::ImageDimension::D2,
            true,
            api::TextureViewDimension::D2Array
        ) | (
            naga::ImageDimension::D3,
            false,
            api::TextureViewDimension::D3
        ) | (
            naga::ImageDimension::Cube,
            false,
            api::TextureViewDimension::Cube
        ) | (
            naga::ImageDimension::Cube,
            true,
            api::TextureViewDimension::CubeArray
        )
    )
}
fn storage_format(format: naga::StorageFormat) -> api::TextureFormat {
    match format {
        naga::StorageFormat::R8Unorm => api::TextureFormat::R8Unorm,
        naga::StorageFormat::R8Snorm => api::TextureFormat::R8Snorm,
        naga::StorageFormat::R8Uint => api::TextureFormat::R8Uint,
        naga::StorageFormat::R8Sint => api::TextureFormat::R8Sint,
        naga::StorageFormat::R16Uint => api::TextureFormat::R16Uint,
        naga::StorageFormat::R16Sint => api::TextureFormat::R16Sint,
        naga::StorageFormat::R16Float => api::TextureFormat::R16Float,
        naga::StorageFormat::Rg8Unorm => api::TextureFormat::Rg8Unorm,
        naga::StorageFormat::Rg8Snorm => api::TextureFormat::Rg8Snorm,
        naga::StorageFormat::Rg8Uint => api::TextureFormat::Rg8Uint,
        naga::StorageFormat::Rg8Sint => api::TextureFormat::Rg8Sint,
        naga::StorageFormat::R32Uint => api::TextureFormat::R32Uint,
        naga::StorageFormat::R32Sint => api::TextureFormat::R32Sint,
        naga::StorageFormat::R32Float => api::TextureFormat::R32Float,
        naga::StorageFormat::Rg16Uint => api::TextureFormat::Rg16Uint,
        naga::StorageFormat::Rg16Sint => api::TextureFormat::Rg16Sint,
        naga::StorageFormat::Rg16Float => api::TextureFormat::Rg16Float,
        naga::StorageFormat::Rgba8Unorm => api::TextureFormat::Rgba8Unorm,
        naga::StorageFormat::Rgba8Snorm => api::TextureFormat::Rgba8Snorm,
        naga::StorageFormat::Rgba8Uint => api::TextureFormat::Rgba8Uint,
        naga::StorageFormat::Rgba8Sint => api::TextureFormat::Rgba8Sint,
        naga::StorageFormat::Bgra8Unorm => api::TextureFormat::Bgra8Unorm,
        naga::StorageFormat::Rgb10a2Uint => api::TextureFormat::Rgb10a2Uint,
        naga::StorageFormat::Rgb10a2Unorm => api::TextureFormat::Rgb10a2Unorm,
        naga::StorageFormat::Rg11b10Ufloat => api::TextureFormat::Rg11b10Ufloat,
        naga::StorageFormat::R64Uint => api::TextureFormat::R64Uint,
        naga::StorageFormat::Rg32Uint => api::TextureFormat::Rg32Uint,
        naga::StorageFormat::Rg32Sint => api::TextureFormat::Rg32Sint,
        naga::StorageFormat::Rg32Float => api::TextureFormat::Rg32Float,
        naga::StorageFormat::Rgba16Uint => api::TextureFormat::Rgba16Uint,
        naga::StorageFormat::Rgba16Sint => api::TextureFormat::Rgba16Sint,
        naga::StorageFormat::Rgba16Float => api::TextureFormat::Rgba16Float,
        naga::StorageFormat::Rgba32Uint => api::TextureFormat::Rgba32Uint,
        naga::StorageFormat::Rgba32Sint => api::TextureFormat::Rgba32Sint,
        naga::StorageFormat::Rgba32Float => api::TextureFormat::Rgba32Float,
        naga::StorageFormat::R16Unorm => api::TextureFormat::R16Unorm,
        naga::StorageFormat::R16Snorm => api::TextureFormat::R16Snorm,
        naga::StorageFormat::Rg16Unorm => api::TextureFormat::Rg16Unorm,
        naga::StorageFormat::Rg16Snorm => api::TextureFormat::Rg16Snorm,
        naga::StorageFormat::Rgba16Unorm => api::TextureFormat::Rgba16Unorm,
        naga::StorageFormat::Rgba16Snorm => api::TextureFormat::Rgba16Snorm,
    }
}

pub(super) fn wgsl_binding_shape(
    module: &naga::Module,
    global: &naga::GlobalVariable,
) -> Result<(u8, u32)> {
    let (inner, count) = match module.types[global.ty].inner {
        naga::TypeInner::BindingArray {
            base,
            size: naga::ArraySize::Constant(count),
        } => (&module.types[base].inner, count.get()),
        naga::TypeInner::BindingArray { .. } => {
            return Err(Error::Unsupported("DX12 WGSL runtime binding array"));
        }
        ref inner => (inner, 1),
    };
    let class = match global.space {
        naga::AddressSpace::Uniform => 0,
        naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE) => 2,
        naga::AddressSpace::Storage { .. } => 1,
        naga::AddressSpace::Handle => match inner {
            naga::TypeInner::Sampler { .. } => 3,
            naga::TypeInner::Image {
                class: naga::ImageClass::Storage { .. },
                ..
            } => 2,
            naga::TypeInner::Image { .. } | naga::TypeInner::AccelerationStructure { .. } => 1,
            _ => return Err(Error::Unsupported("DX12 WGSL binding type")),
        },
        _ => return Err(Error::Unsupported("DX12 WGSL binding address space")),
    };
    Ok((class, count))
}

/// Turns WGSL's unbounded binding-array syntax into the exact nonzero capacity
/// declared by the selected Core pipeline layout.  D3D12 descriptor tables are
/// fixed-size root-signature ranges; Core supplies the per-layout bound rather
/// than letting a shader reserve an arbitrary device-wide maximum.
///
/// A fixed public array may have the same descriptor count, but it is a
/// distinct contract.  The parallel runtime-capacity map therefore proves the
/// matching binding was declared `RuntimeSized`, instead of inferring it from
/// capacity alone.
pub(super) fn specialize_runtime_arrays(
    module: &mut naga::Module,
    runtime_binding_capacities: Option<&[(u32, u32, u32)]>,
    require_layout: bool,
) -> Result<()> {
    let arrays: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, global)| match module.types[global.ty].inner {
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Dynamic,
            } => Some((handle, base, global.binding.clone())),
            _ => None,
        })
        .collect();
    for (handle, base, binding) in arrays {
        let binding = binding.ok_or(Error::Shader(
            "DX12 WGSL runtime binding array requires a binding".into(),
        ))?;
        let Some(capacities) = runtime_binding_capacities else {
            return Err(Error::Unsupported(
                "DX12 WGSL runtime binding array requires pipeline layout",
            ));
        };
        let capacity = capacities
            .iter()
            .find(|&&(group, slot, _)| group == binding.group && slot == binding.binding)
            .map(|&(_, _, capacity)| capacity)
            .ok_or_else(|| {
                if require_layout {
                    Error::Shader(
                        "WGSL runtime binding array requires RuntimeSized pipeline binding".into(),
                    )
                } else {
                    Error::Unsupported("DX12 WGSL runtime binding array requires pipeline layout")
                }
            })?;
        let size = std::num::NonZeroU32::new(capacity).ok_or(Error::InvalidRange)?;
        let ty = naga::Type {
            name: None,
            inner: naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(size),
            },
        };
        let ty = module.types.insert(ty, naga::Span::UNDEFINED);
        module.global_variables[handle].ty = ty;
    }
    Ok(())
}
