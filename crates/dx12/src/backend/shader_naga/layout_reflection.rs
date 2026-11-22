//! Selects the reflection route and extracts bindings from DX12 WGSL entries.

use fluxel_rhi_core::api::{self, Error, Result};

use super::{binding::storage_format, diagnostics, validation_capabilities};
use crate::backend::{Dx12, Dx12ShaderSource};

pub(in crate::backend) fn reflect(
    stage: api::ShaderStage,
    entry: &api::PipelineStage<Dx12>,
    runtime_arrays: &[api::RuntimeBindingCapacity],
) -> Result<Vec<(u32, api::BindGroupLayoutEntry)>> {
    // SAFETY: Core checks the shader module belongs to the reflecting device.
    let shader = unsafe { entry.module.native_ref() };
    if !shader
        .entries
        .iter()
        .any(|item| item.stage == stage && item.name == entry.entry)
    {
        return Err(Error::Shader("missing selected shader entry".into()));
    }
    let desc = match &shader.source {
        Some(Dx12ShaderSource::Wgsl(desc)) => desc,
        Some(Dx12ShaderSource::Hlsl(desc)) => {
            let api::ShaderFormat::Hlsl(model) = desc.format else {
                return Err(Error::InvalidState);
            };
            let text = std::str::from_utf8(&desc.bytes)
                .map_err(|_| Error::Shader("HLSL is not UTF-8".into()))?;
            let compiled = super::compile_hlsl_with_report(
                text,
                &entry.entry,
                stage,
                model,
                &entry.constants,
                api::ShaderDiagnosticSource::Hlsl,
                super::requires_native_16bit(desc),
            )?;
            return super::native_layout_reflection::reflect(
                &compiled.bytes,
                stage,
                runtime_arrays,
            );
        }
        None => {
            if shader.entries.len() != 1 {
                return Err(Error::Unsupported(
                    "DX12 layout inference requires single-entry DXIL",
                ));
            }
            if !entry.constants.is_empty() {
                return Err(Error::Unsupported("DX12 DXIL specialization"));
            }
            return super::native_layout_reflection::reflect(&shader.bytes, stage, runtime_arrays);
        }
    };
    let source =
        std::str::from_utf8(&desc.bytes).map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
    let module = diagnostics::parse_wgsl(source, super::immediates::required(desc))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        validation_capabilities(desc),
    )
    .validate(&module)
    .map_err(|error| diagnostics::wgsl_validation(source, error))?;
    let native_stage = match stage {
        api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
        api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
        api::ShaderStage::Compute => naga::ShaderStage::Compute,
        _ => return Err(Error::Unsupported("DX12 WGSL layout inference stage")),
    };
    let index = module
        .entry_points
        .iter()
        .position(|candidate| candidate.stage == native_stage && candidate.name == entry.entry)
        .ok_or_else(|| Error::Shader("missing selected WGSL entry".into()))?;
    let usage = info.get_entry_point(index);
    let mut layouts = naga::proc::Layouter::default();
    layouts
        .update(module.to_ctx())
        .map_err(|error| Error::Shader(error.to_string()))?;
    let mut bindings = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        if usage[handle].is_empty() {
            continue;
        }
        let Some(binding) = &global.binding else {
            continue;
        };
        let (count, ty) = match module.types[global.ty].inner {
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Constant(count),
            } => (api::BindingCount::Fixed(count.get()), base),
            naga::TypeInner::BindingArray {
                base,
                size: naga::ArraySize::Dynamic,
            } => {
                let max_count = runtime_arrays
                    .iter()
                    .find(|capacity| {
                        capacity.group == binding.group && capacity.binding == binding.binding
                    })
                    .ok_or(Error::Unsupported(
                        "runtime array layout inference requires a capacity",
                    ))?
                    .max_count;
                (api::BindingCount::RuntimeSized { max_count }, base)
            }
            naga::TypeInner::BindingArray { .. } => {
                return Err(Error::Unsupported("unresolved WGSL binding array size"));
            }
            _ => (api::BindingCount::One, global.ty),
        };
        // Sampling a float image uses a filterable default. Callers can choose
        // a nonfiltering sampler and unfilterable view in an explicit layout.
        let ty = match global.space {
            naga::AddressSpace::Uniform => api::BindingType::UniformBuffer {
                dynamic_offset: false,
                min_size: u64::from(layouts[ty].size),
            },
            naga::AddressSpace::Storage { access } => api::BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: u64::from(layouts[ty].size),
                access: storage_access(access)?,
            },
            naga::AddressSpace::Handle => match module.types[ty].inner {
                naga::TypeInner::Sampler { comparison } => api::BindingType::Sampler {
                    filtering: usage.sampling_set.iter().any(|pair| pair.sampler == handle),
                    comparison,
                },
                naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => image_binding(
                    dim,
                    arrayed,
                    class,
                    usage.sampling_set.iter().any(|pair| pair.image == handle),
                )?,
                naga::TypeInner::AccelerationStructure { .. } => {
                    api::BindingType::AccelerationStructure
                }
                _ => return Err(Error::Unsupported("DX12 WGSL layout inference resource")),
            },
            _ => {
                return Err(Error::Unsupported(
                    "DX12 WGSL layout inference address space",
                ));
            }
        };
        bindings.push((
            binding.group,
            api::BindGroupLayoutEntry {
                binding: binding.binding,
                visibility: vec![stage],
                ty,
                count,
                partially_bound: false,
                // Keep array layouts valid for both uniform and divergent indices.
                nonuniform_indexing: count != api::BindingCount::One,
            },
        ));
    }
    Ok(bindings)
}

fn storage_access(access: naga::StorageAccess) -> Result<api::StorageAccess> {
    if access.contains(naga::StorageAccess::ATOMIC) {
        return Ok(api::StorageAccess::Atomic);
    }
    match (
        access.contains(naga::StorageAccess::LOAD),
        access.contains(naga::StorageAccess::STORE),
    ) {
        (true, true) => Ok(api::StorageAccess::ReadWrite),
        (true, false) => Ok(api::StorageAccess::Read),
        (false, true) => Ok(api::StorageAccess::Write),
        (false, false) => Err(Error::Unsupported("WGSL storage access")),
    }
}

fn image_binding(
    dim: naga::ImageDimension,
    arrayed: bool,
    class: naga::ImageClass,
    sampled: bool,
) -> Result<api::BindingType> {
    use api::{TextureSampleType as Sample, TextureViewDimension as View};
    use naga::ImageDimension as Dim;
    let dimension = match (dim, arrayed) {
        (Dim::D1, false) => View::D1,
        (Dim::D2, false) => View::D2,
        (Dim::D2, true) => View::D2Array,
        (Dim::D3, false) => View::D3,
        (Dim::Cube, false) => View::Cube,
        (Dim::Cube, true) => View::CubeArray,
        _ => return Err(Error::Unsupported("DX12 inferred WGSL image dimension")),
    };
    let (sample, multisampled) = match class {
        naga::ImageClass::Storage { format, access } => {
            return Ok(api::BindingType::StorageTexture {
                format: storage_format(format),
                dimension,
                access: storage_access(access)?,
            });
        }
        naga::ImageClass::Depth { multi } => (Sample::Depth, multi),
        naga::ImageClass::Sampled { kind, multi } => (
            match kind {
                naga::ScalarKind::Float if sampled => Sample::Float,
                naga::ScalarKind::Float => Sample::UnfilterableFloat,
                naga::ScalarKind::Sint => Sample::Sint,
                naga::ScalarKind::Uint => Sample::Uint,
                _ => return Err(Error::Unsupported("DX12 inferred WGSL texture sample type")),
            },
            multi,
        ),
        _ => return Err(Error::Unsupported("DX12 inferred WGSL image class")),
    };
    Ok(api::BindingType::SampledTexture {
        sample,
        dimension,
        multisampled,
    })
}
