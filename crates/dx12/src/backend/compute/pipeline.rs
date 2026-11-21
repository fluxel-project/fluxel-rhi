//! DX12 compute descriptor admission, shader lowering and PSO creation.

use std::{mem::ManuallyDrop, sync::Arc};

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::*;

use super::super::pipeline::PipelineLayoutState;
use super::super::{Dx12, Dx12ComputePipeline, Dx12Device, Dx12Shader, Dx12ShaderSource};
use super::{ComputePipelineState, native_error, validate_workgroup_size};

/// HLSL `[WaveSize(N)]` requires SM 6.6 and a native wave-operation range.
/// D3D12 has no PSO field for this: the compiler bakes the requirement into
/// generated WGSL-derived DXIL after this admission check.
fn validate_required_wave_size(device: &Dx12Device, size: u32) -> Result<()> {
    if !size.is_power_of_two() {
        return Err(Error::InvalidRange);
    }
    if device.0.shader_model < 66 {
        return Err(Error::Unsupported(
            "DX12 required subgroup size needs shader model 6.6",
        ));
    }
    if !device
        .0
        .subgroup_widths
        .is_some_and(|widths| (widths.min..=widths.max).contains(&size))
    {
        return Err(Error::Unsupported("DX12 required subgroup size"));
    }
    Ok(())
}

struct AdmittedPipeline<'a> {
    shader: &'a Dx12Shader,
    entry: &'a api::ShaderEntryPoint,
    layout: &'a Arc<PipelineLayoutState>,
}

pub(super) fn support(device: &Dx12Device, desc: &api::ComputePipelineDesc<Dx12>) -> Result<()> {
    admit(device, desc).map(drop)
}

fn admit<'a>(
    device: &Dx12Device,
    desc: &'a api::ComputePipelineDesc<Dx12>,
) -> Result<AdmittedPipeline<'a>> {
    device.require_active()?;
    // SAFETY: Core checks the shader and layout belong to this device before either SPI call.
    let shader = unsafe { desc.compute.module.native_ref() };
    // SAFETY: Core checked layout provenance and `desc` keeps the layout alive.
    let layout = &unsafe { desc.layout.native_ref() }.0;
    let entry = shader
        .entries
        .iter()
        .find(|entry| entry.name == desc.compute.entry && entry.stage == api::ShaderStage::Compute)
        .ok_or_else(|| Error::Shader("DX12 shader has no requested compute entry".into()))?;
    if let Some(size) = entry.workgroup_size {
        validate_workgroup_size(size)?;
    }
    match &shader.source {
        Some(Dx12ShaderSource::Wgsl(_)) => {
            if let Some(size) = desc.required_subgroup_size {
                validate_required_wave_size(device, size)?;
            }
        }
        Some(Dx12ShaderSource::Hlsl(source)) => {
            if !layout.native_abi_compatible {
                return Err(Error::Unsupported("DX12 HLSL descriptor register overlap"));
            }
            let api::ShaderFormat::Hlsl(model) = source.format else {
                return Err(Error::InvalidState);
            };
            if let Some(size) = desc.required_subgroup_size {
                validate_required_wave_size(device, size)?;
                if model < 66 {
                    return Err(Error::Unsupported(
                        "DX12 HLSL required subgroup size needs shader model 6.6",
                    ));
                }
            }
        }
        None => {
            if !layout.native_abi_compatible {
                return Err(Error::Unsupported("DX12 DXIL descriptor register overlap"));
            }
            if !desc.compute.constants.is_empty() {
                return Err(Error::Unsupported("DX12 DXIL specialization"));
            }
            if desc.required_subgroup_size.is_some() {
                return Err(Error::Unsupported(
                    "DX12 DXIL required subgroup size is immutable in native artifact",
                ));
            }
            super::super::shader_naga::validate_dxil_visibility(
                &shader.bytes,
                entry.stage,
                layout,
            )?;
        }
    }
    Ok(AdmittedPipeline {
        shader,
        entry,
        layout,
    })
}

pub(super) fn create(
    device: &Dx12Device,
    desc: &api::ComputePipelineDesc<Dx12>,
) -> Result<Dx12ComputePipeline> {
    let AdmittedPipeline {
        shader,
        entry,
        layout,
    } = admit(device, desc)?;
    let (compiled, raw_storage_bindings, num_workgroups_constants, report) = match &shader.source {
        Some(Dx12ShaderSource::Wgsl(source)) => {
            let compiled = super::super::shader_naga::compile_pipeline_variant(
                source,
                entry,
                &desc.compute.constants,
                &layout.register_map,
                &layout.runtime_binding_capacities,
                &layout.binding_entries,
                super::super::shader_naga::WgslTarget {
                    shader_model: device.0.shader_model,
                    required_wave_size: desc.required_subgroup_size,
                    push_constants: layout.push_constants.as_ref(),
                },
            )?;
            (
                Some(compiled.bytes),
                compiled.raw_storage_bindings,
                compiled.num_workgroups_constants,
                compiled.report,
            )
        }
        Some(Dx12ShaderSource::Hlsl(source)) => {
            let api::ShaderFormat::Hlsl(model) = source.format else {
                return Err(Error::InvalidState);
            };
            let mut text = std::str::from_utf8(&source.bytes)
                .map_err(|_| Error::Shader("HLSL is not UTF-8".into()))?
                .to_owned();
            if let Some(size) = desc.required_subgroup_size {
                text = super::super::shader_naga::preprocess_direct_hlsl(
                    &text,
                    &desc.compute.constants,
                )?;
                text = super::super::shader_naga::inject_direct_hlsl(
                    &text,
                    &desc.compute.entry,
                    size,
                )?;
            }
            let origin = if desc.required_subgroup_size.is_some() {
                api::ShaderDiagnosticSource::GeneratedHlsl
            } else {
                api::ShaderDiagnosticSource::Hlsl
            };
            let compiled = super::super::shader_naga::compile_hlsl_with_report(
                &text,
                &desc.compute.entry,
                api::ShaderStage::Compute,
                model,
                &desc.compute.constants,
                origin,
                super::super::shader_naga::requires_native_16bit(source),
            )?;
            super::super::shader_naga::validate_native_visibility(
                &compiled.bytes,
                entry.stage,
                layout,
            )?;
            (Some(compiled.bytes), Vec::new(), false, compiled.report)
        }
        None => (None, Vec::new(), false, None),
    };
    let bytes: &[u8] = compiled.as_deref().unwrap_or(&shader.bytes);
    let (root, root_bytes, num_workgroups_root) = if num_workgroups_constants {
        let (root, bytes, index) =
            layout
                .num_workgroups_root
                .as_ref()
                .ok_or(Error::Unsupported(
                    "DX12 WGSL workgroup constants root budget",
                ))?;
        (root.clone(), bytes.as_slice(), Some(*index))
    } else {
        (
            layout.compute_root.clone(),
            layout.compute_root_bytes.as_slice(),
            None,
        )
    };
    let mut state_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        },
        ..Default::default()
    };
    // SAFETY: Core checked that the cache belongs to the pipeline device.
    let cache = desc
        .cache
        .as_ref()
        .map(|cache| unsafe { cache.native_ref() });
    let key = cache
        .map(|_| super::super::pipeline::cache::key(&[b"compute-v1", root_bytes, bytes]))
        .transpose()?;
    let cached = match (cache, key.as_ref()) {
        (Some(cache), Some(key)) => cache.0.lookup(key)?,
        _ => None,
    };
    if let Some(blob) = &cached {
        state_desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE {
            pCachedBlob: blob.as_ptr().cast(),
            CachedBlobSizeInBytes: blob.len(),
        };
    }
    // SAFETY: the root signature, shader and optional cached blob outlive both synchronous calls.
    let mut result = unsafe {
        device
            .0
            .native
            .CreateComputePipelineState::<ID3D12PipelineState>(&state_desc)
    };
    if cached.is_some()
        && result
            .as_ref()
            .is_err_and(super::super::pipeline::cache::is_compatibility_error)
    {
        state_desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
        // SAFETY: same descriptor, with the incompatible cache blob removed.
        result = unsafe {
            device
                .0
                .native
                .CreateComputePipelineState::<ID3D12PipelineState>(&state_desc)
        };
    }
    // SAFETY: release the manually-owned root signature even when creation failed.
    unsafe {
        ManuallyDrop::drop(&mut state_desc.pRootSignature);
    }
    let state = result.map_err(native_error)?;
    super::super::set_debug_name(&state, desc.label.as_deref());
    if let (Some(cache), Some(key)) = (cache, key.as_ref()) {
        // Cache extraction is an optimization; successful PSO creation remains usable if extraction fails.
        let _ = cache.0.store(key, &state);
    }
    Ok(Dx12ComputePipeline(Arc::new(ComputePipelineState {
        compilation_diagnostics: report
            .into_iter()
            .map(|report| api::PipelineShaderDiagnostics {
                stage: api::ShaderStage::Compute,
                entry: desc.compute.entry.clone(),
                report,
            })
            .collect(),
        root,
        layout: layout.clone(),
        state,
        num_workgroups_root,
        raw_storage_bindings,
    })))
}
