//! DX12 lowering for the compute domain.
//!
//! This module turns Core's already-validated compute pipeline and dispatch
//! operations directly into commands on the encoder's native command list. It
//! owns no scheduling or state tracking: cross-operation transitions and
//! submission order remain the caller's responsibility. Pipeline and bound
//! group clones are retained by the encoder solely until the submission fence
//! proves D3D12 has stopped reading their native objects.

use std::{mem::ManuallyDrop, sync::Arc};

use fluxel_rhi_core::api::{self, Error, Result};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

use super::{Dx12, Dx12ComputePipeline, Dx12Device, Dx12ShaderSource};

pub(crate) struct ComputePipelineState {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) layout: Arc<super::pipeline::PipelineLayoutState>,
    pub(crate) state: ID3D12PipelineState,
    pub(crate) raw_storage_bindings: Vec<(u32, u32)>,
    pub(crate) num_workgroups_root: Option<u32>,
}

/// Current compute root arguments. Descriptor and pipeline owners stay in the encoder.
#[derive(Default)]
pub(crate) struct ComputeSnapshot {
    roots: std::collections::BTreeMap<u32, super::encoder::raster_snapshot::RootValue>,
    constants: std::collections::BTreeMap<(u32, u32), u32>,
}

/// Restore only current caller state after an internal compute dispatch.
pub(crate) fn restore_compute_state(encoder: &mut super::encoder::Dx12Encoder) -> Result<()> {
    let ray_list = if encoder.ray_pipeline.is_some() {
        Some(
            encoder
                .list
                .cast::<ID3D12GraphicsCommandList4>()
                .map_err(native_error)?,
        )
    } else {
        None
    };
    if encoder.compute_pipeline.is_none() && encoder.ray_pipeline.is_none() {
        return Ok(());
    }
    // SAFETY: the encoder retains the pipeline, groups, and dynamic descriptor tables
    // through execution. Saved root indices and offsets were validated on binding.
    unsafe {
        if let Some(pipeline) = &encoder.ray_pipeline {
            encoder
                .list
                .SetComputeRootSignature(&pipeline.0.layout.compute_root);
            ray_list
                .as_ref()
                .expect("ray list")
                .SetPipelineState1(&pipeline.0.state);
        } else if let Some(pipeline) = &encoder.compute_pipeline {
            encoder.list.SetComputeRootSignature(&pipeline.0.root);
            encoder.list.SetPipelineState(&pipeline.0.state);
        }
        for (&root, value) in &encoder.compute_snapshot.roots {
            match value {
                super::encoder::raster_snapshot::RootValue::Table(handle) => {
                    encoder.list.SetComputeRootDescriptorTable(root, *handle)
                }
                super::encoder::raster_snapshot::RootValue::Cbv(address) => encoder
                    .list
                    .SetComputeRootConstantBufferView(root, *address),
            }
        }
        for (&(root, offset), &value) in &encoder.compute_snapshot.constants {
            encoder
                .list
                .SetComputeRoot32BitConstant(root, value, offset);
        }
    }
    Ok(())
}

/// Concrete static implementation selected by `Dx12::Compute`.
pub struct Compute;

impl fluxel_rhi_core::backend::ComputeOps<Dx12> for Compute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = super::pipeline::worker::ColdFuture<Dx12ComputePipeline>;

    fn create_pipeline<'a>(
        device: &'a Dx12Device,
        desc: &'a api::ComputePipelineDesc<Dx12>,
    ) -> Self::PipelineFuture<'a> {
        let device = (*device).clone();
        let desc = (*desc).clone();
        super::pipeline::worker::submit(move || create_pipeline(&device, &desc))
    }
    fn begin(_: &mut super::encoder::Dx12Encoder) -> Result<()> {
        // Ending a compute pass does not reset D3D12 root state. Keep the
        // validation snapshot aligned with the still-bound native arguments.
        Ok(())
    }
    fn set_pipeline(
        encoder: &mut super::encoder::Dx12Encoder,
        pipeline: &Dx12ComputePipeline,
    ) -> Result<()> {
        // SAFETY: the pipeline retains root signature and PSO COM interfaces;
        // the encoder owns this open list and keeps a pipeline clone below.
        unsafe {
            encoder.list.SetComputeRootSignature(&pipeline.0.root);
            encoder.list.SetPipelineState(&pipeline.0.state);
        }
        if !encoder
            .compute_pipeline
            .as_ref()
            .is_some_and(|old| old.0.root == pipeline.0.root)
        {
            encoder.compute_snapshot = Default::default();
        }
        encoder.compute_pipeline = Some(pipeline.clone());
        encoder.compute_bound_groups.clear();
        encoder.compute_pipelines.push(pipeline.clone());
        encoder.ray_pipeline = None;
        encoder.mesh_pipeline = None;
        Ok(())
    }
    fn set_bind_group(
        encoder: &mut super::encoder::Dx12Encoder,
        slot: u32,
        group: &super::Dx12BindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        let pipeline = encoder
            .compute_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?;
        let expected = pipeline
            .0
            .layout
            .groups
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        if !Arc::ptr_eq(expected, &group.0.layout) {
            return Err(Error::InvalidState);
        }
        if pipeline
            .0
            .raw_storage_bindings
            .iter()
            .any(|&(group_index, binding)| {
                group_index == slot && !group.0.storage_views_are_raw(binding)
            })
        {
            return Err(Error::InvalidState);
        }
        let layout = pipeline.0.layout.clone();
        bind_compute_group(encoder, &layout, slot, group, offsets)?;
        let slot = slot as usize;
        if encoder.compute_bound_groups.len() <= slot {
            encoder.compute_bound_groups.resize(slot + 1, None);
        }
        encoder.compute_bound_groups[slot] = Some(group.clone());
        Ok(())
    }
    fn dispatch(encoder: &mut super::encoder::Dx12Encoder, groups: [u32; 3]) -> Result<()> {
        validate_raw_bindings(encoder)?;
        // D3D12's guaranteed direct-dispatch count limits.  Keep this in
        // lockstep with `DeviceCapabilities::limits` in the public API.
        if groups.contains(&0)
            || groups
                .iter()
                .zip([65_535; 3])
                .any(|(&value, limit)| value > limit)
        {
            return Err(Error::InvalidRange);
        }
        // SAFETY: validated group counts are copied into the encoder-owned
        // open list; this call does not borrow caller memory.
        unsafe {
            if let Some(root) = encoder
                .compute_pipeline
                .as_ref()
                .and_then(|p| p.0.num_workgroups_root)
            {
                encoder
                    .list
                    .SetComputeRoot32BitConstants(root, 3, groups.as_ptr().cast(), 0);
            }
            encoder.list.Dispatch(groups[0], groups[1], groups[2]);
        }
        Ok(())
    }
    fn dispatch_indirect(
        encoder: &mut super::encoder::Dx12Encoder,
        buffer: &super::Dx12Buffer,
        offset: u64,
    ) -> Result<()> {
        validate_raw_bindings(encoder)?;
        if encoder.compute_pipeline.is_none() {
            return Err(Error::InvalidState);
        }
        validate_indirect(
            buffer.usage.contains(api::BufferUsage::INDIRECT),
            buffer.size,
            offset,
        )?;
        if encoder
            .compute_pipeline
            .as_ref()
            .is_some_and(|p| p.0.num_workgroups_root.is_some())
        {
            return dispatch_indirect_with_workgroups(encoder, buffer, offset);
        }
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: the descriptor and argument live through creation; no root signature is required.
        unsafe {
            encoder.device.0.native.CreateCommandSignature(
                &desc,
                None::<&ID3D12RootSignature>,
                &mut signature,
            )
        }
        .map_err(native_error)?;
        let signature =
            signature.ok_or_else(|| Error::Backend("DX12 dispatch signature missing".into()))?;
        // SAFETY: Core requires the caller to transition the argument buffer to INDIRECT_READ.
        // The buffer and signature are retained by the encoder until the completion fence.
        unsafe {
            encoder.list.ExecuteIndirect(
                &signature,
                1,
                &buffer.resource,
                offset,
                None::<&ID3D12Resource>,
                0,
            );
        }
        encoder.buffers.push(buffer.clone());
        encoder.indirect_signatures.push(signature);
        Ok(())
    }
    fn end(_: &mut super::encoder::Dx12Encoder) -> Result<()> {
        Ok(())
    }
    fn set_push_constants(
        encoder: &mut super::encoder::Dx12Encoder,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()> {
        let pipeline = encoder
            .compute_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?;
        let range = pipeline
            .0
            .layout
            .push_constants
            .as_ref()
            .ok_or(Error::Unsupported("DX12 pipeline has no push constants"))?;
        let end = offset
            .checked_add(u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?)
            .ok_or(Error::InvalidRange)?;
        if bytes.is_empty() || offset % 4 != 0 || bytes.len() % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        let root = range.compute_write(offset, end)?;
        let values: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("exact DWORD")))
            .collect();
        // SAFETY: the validated range belongs to the bound root signature and values live for this call.
        unsafe {
            encoder.list.SetComputeRoot32BitConstants(
                root,
                values.len() as u32,
                values.as_ptr().cast(),
                offset / 4,
            );
        }
        for (index, value) in values.into_iter().enumerate() {
            encoder
                .compute_snapshot
                .constants
                .insert((root, offset / 4 + index as u32), value);
        }
        Ok(())
    }
    fn abort(_: &mut super::encoder::Dx12Encoder) {}
}

fn validate_raw_bindings(encoder: &super::encoder::Dx12Encoder) -> Result<()> {
    let pipeline = encoder
        .compute_pipeline
        .as_ref()
        .ok_or(Error::InvalidState)?;
    for &(slot, binding) in &pipeline.0.raw_storage_bindings {
        let group = encoder
            .compute_bound_groups
            .get(slot as usize)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidState)?;
        if !group.0.storage_views_are_raw(binding) {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}

/// HLSL `[WaveSize(N)]` requires SM 6.6 and a native wave-operation range.
/// D3D12 has no PSO field for this: the compiler bakes the requirement into
/// generated WGSL-derived DXIL after this admission check.
fn validate_required_wave_size(device: &Dx12Device, size: u32) -> Result<()> {
    if !size.is_power_of_two() {
        return Err(Error::InvalidRange);
    }
    let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_6,
    };
    // SAFETY: `model` is exactly the structure D3D12 fills synchronously.
    let sm66 = unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&mut model as *mut D3D12_FEATURE_DATA_SHADER_MODEL).cast(),
            std::mem::size_of_val(&model) as u32,
        )
    }
    .is_ok()
        && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
    if !sm66 {
        return Err(Error::Unsupported(
            "DX12 required subgroup size needs shader model 6.6",
        ));
    }
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: `options` is exactly the structure D3D12 fills synchronously.
    let wave_ops = unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS1).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.WaveOps.as_bool();
    if !wave_ops || size < options.WaveLaneCountMin || size > options.WaveLaneCountMax {
        return Err(Error::Unsupported("DX12 required subgroup size"));
    }
    Ok(())
}

fn create_pipeline(
    device: &Dx12Device,
    desc: &api::ComputePipelineDesc<Dx12>,
) -> Result<Dx12ComputePipeline> {
    // SAFETY: pipeline descriptors are device-checked by Core; this borrows
    // immutable shader native state during synchronous PSO creation.
    let shader = unsafe { desc.compute.module.native_ref() };
    if !shader
        .entries
        .iter()
        .any(|entry| entry.name == desc.compute.entry && entry.stage == api::ShaderStage::Compute)
    {
        return Err(Error::Shader("DXIL has no requested compute entry".into()));
    }
    // SAFETY: same-device layout native state is cloned to retain its root
    // signature independently of the caller's descriptor borrow.
    let layout = unsafe { desc.layout.native_ref() }.0.clone();
    let (compiled, raw_storage_bindings, num_workgroups_constants) = match &shader.source {
        Some(Dx12ShaderSource::Wgsl(source)) => {
            if let Some(size) = desc.required_subgroup_size {
                validate_required_wave_size(device, size)?;
            }
            let entry = shader
                .entries
                .iter()
                .find(|entry| {
                    entry.name == desc.compute.entry && entry.stage == api::ShaderStage::Compute
                })
                .ok_or(Error::InvalidState)?;
            let compiled = super::shader_naga::compile_variant_with_typed_map(
                source,
                entry,
                &desc.compute.constants,
                &layout.register_map,
                &layout.binding_types,
                &layout.runtime_binding_capacities,
                &layout.storage_entries,
                desc.required_subgroup_size,
            )?;
            (
                Some(compiled.bytes),
                compiled.raw_storage_bindings,
                compiled.num_workgroups_constants,
            )
        }
        Some(Dx12ShaderSource::Hlsl(source)) => {
            if !layout.native_abi_compatible {
                return Err(Error::Unsupported("DX12 HLSL descriptor register overlap"));
            }
            let api::ShaderFormat::Hlsl(model) = source.format else {
                return Err(Error::InvalidState);
            };
            let mut text = std::str::from_utf8(&source.bytes)
                .map_err(|_| Error::Shader("HLSL is not UTF-8".into()))?
                .to_owned();
            if let Some(size) = desc.required_subgroup_size {
                validate_required_wave_size(device, size)?;
                if model < 66 {
                    return Err(Error::Unsupported(
                        "DX12 HLSL required subgroup size needs shader model 6.6",
                    ));
                }
                text = super::shader_naga::preprocess_direct_hlsl(&text, &desc.compute.constants)?;
                text = super::shader_naga::inject_direct_hlsl(&text, &desc.compute.entry, size)?;
            }
            (
                Some(super::shader_naga::compile_direct_hlsl(
                    &text,
                    &desc.compute.entry,
                    api::ShaderStage::Compute,
                    model,
                    &desc.compute.constants,
                )?),
                Vec::new(),
                false,
            )
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
            (None, Vec::new(), false)
        }
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
        .map(|_| super::pipeline::cache::key(&[b"compute-v1", root_bytes, bytes]))
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
            .is_err_and(super::pipeline::cache::is_compatibility_error)
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
    if let (Some(cache), Some(key)) = (cache, key.as_ref()) {
        // Cache extraction is an optimization; successful PSO creation remains usable if extraction fails.
        let _ = cache.0.store(key, &state);
    }
    Ok(Dx12ComputePipeline(Arc::new(ComputePipelineState {
        root,
        layout,
        state,
        num_workgroups_root,
        raw_storage_bindings,
    })))
}

fn dispatch_indirect_with_workgroups(
    encoder: &mut super::encoder::Dx12Encoder,
    buffer: &super::Dx12Buffer,
    offset: u64,
) -> Result<()> {
    let pipeline = encoder
        .compute_pipeline
        .as_ref()
        .ok_or(Error::InvalidState)?;
    let index = pipeline.0.num_workgroups_root.ok_or(Error::InvalidState)?;
    let root = pipeline.0.root.clone();
    let scratch = <Dx12 as fluxel_rhi_core::backend::ResourceBackend>::create_buffer(
        &encoder.device,
        &api::BufferDesc {
            label: None,
            size: 24,
            usage: api::BufferUsage::COPY_DST | api::BufferUsage::INDIRECT,
            memory: api::MemoryClass::Device,
        },
    )?;
    let arguments = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: index,
                    DestOffsetIn32BitValues: 0,
                    Num32BitValuesToSet: 3,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0::default(),
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: 24,
        NumArgumentDescs: 2,
        pArgumentDescs: arguments.as_ptr(),
        NodeMask: 0,
    };
    let mut signature = None;
    // SAFETY: descriptor and root are live during signature creation.
    unsafe {
        encoder
            .device
            .0
            .native
            .CreateCommandSignature(&desc, Some(&root), &mut signature)
    }
    .map_err(native_error)?;
    let signature: ID3D12CommandSignature = signature.ok_or(Error::InvalidState)?;
    super::encoder::barrier(
        &encoder.list,
        &buffer.resource,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    super::encoder::barrier(
        &encoder.list,
        &scratch.resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: the validated argument range and 24-byte scratch are retained through completion.
    unsafe {
        encoder
            .list
            .CopyBufferRegion(&scratch.resource, 0, &buffer.resource, offset, 12);
        encoder
            .list
            .CopyBufferRegion(&scratch.resource, 12, &buffer.resource, offset, 12);
    }
    super::encoder::barrier(
        &encoder.list,
        &buffer.resource,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    super::encoder::barrier(
        &encoder.list,
        &scratch.resource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    // SAFETY: scratch and signature remain live through GPU completion.
    unsafe {
        encoder.list.ExecuteIndirect(
            &signature,
            1,
            &scratch.resource,
            0,
            None::<&ID3D12Resource>,
            0,
        )
    };
    encoder.buffers.extend([buffer.clone(), scratch]);
    encoder.indirect_signatures.push(signature);
    Ok(())
}

fn native_error(error: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {error}"))
}

/// Install a compute-root bind group for both compute and ray pipelines.
pub(crate) fn bind_compute_group(
    encoder: &mut super::encoder::Dx12Encoder,
    layout: &super::pipeline::PipelineLayoutState,
    slot: u32,
    group: &super::Dx12BindGroup,
    offsets: &[u32],
) -> Result<()> {
    let expected = layout
        .groups
        .get(slot as usize)
        .ok_or(Error::InvalidRange)?;
    if !Arc::ptr_eq(expected, &group.0.layout) {
        return Err(Error::InvalidState);
    }
    let table = layout
        .tables
        .get(slot as usize)
        .ok_or(Error::InvalidRange)?;
    if group.0.has_views() != table.views.is_some()
        || group.0.has_samplers() != table.samplers.is_some()
    {
        return Err(Error::InvalidState);
    }
    if offsets.len() != table.dynamic_bindings.len() {
        return Err(Error::InvalidRange);
    }
    let mut addresses = Vec::with_capacity(offsets.len());
    let mut dynamic_views = Vec::new();
    for (&(binding, class), &offset) in table.dynamic_bindings.iter().zip(offsets) {
        if class != super::descriptor::DescriptorClass::Cbv {
            dynamic_views.push((binding, offset));
            continue;
        }
        let root = table
            .dynamic_cbvs
            .iter()
            .find(|root| root.binding == binding)
            .ok_or(Error::InvalidState)?;
        let uniform = group
            .0
            .uniform(root.binding)
            .filter(|value| value.dynamic_offset)
            .ok_or(Error::InvalidState)?;
        // SAFETY: the bind group retains the same-device native buffer.
        let base = unsafe { uniform.buffer.resource.GetGPUVirtualAddress() };
        addresses.push((
            root.root,
            dynamic_address(
                base,
                uniform.range.start,
                uniform.range.end,
                uniform.buffer.size,
                u64::from(offset),
                root.min_size,
            )?,
        ));
    }
    let dynamic_table = group.0.prepare_dynamic_views(&dynamic_views)?;
    // SAFETY: group heaps and table handles are same-device validated and
    // retained in `bound_groups` until this command list fence retires.
    unsafe {
        if let Some(root) = table.views {
            let handle = dynamic_table
                .as_ref()
                .map_or_else(|| group.0.view_gpu(), |table| table.gpu());
            encoder.list.SetComputeRootDescriptorTable(root, handle);
            if encoder.compute_pipeline.is_some() || encoder.ray_pipeline.is_some() {
                encoder.compute_snapshot.roots.insert(
                    root,
                    super::encoder::raster_snapshot::RootValue::Table(handle),
                );
            }
        }
        if let Some(root) = table.samplers {
            encoder
                .list
                .SetComputeRootDescriptorTable(root, group.0.sampler_gpu());
            if encoder.compute_pipeline.is_some() || encoder.ray_pipeline.is_some() {
                encoder.compute_snapshot.roots.insert(
                    root,
                    super::encoder::raster_snapshot::RootValue::Table(group.0.sampler_gpu()),
                );
            }
        }
        for (root, address) in addresses {
            encoder.list.SetComputeRootConstantBufferView(root, address);
            if encoder.compute_pipeline.is_some() || encoder.ray_pipeline.is_some() {
                encoder.compute_snapshot.roots.insert(
                    root,
                    super::encoder::raster_snapshot::RootValue::Cbv(address),
                );
            }
        }
    }
    if let Some(table) = dynamic_table {
        encoder.dynamic_view_tables.push(table);
    }
    encoder.bound_groups.push(group.clone());
    Ok(())
}

fn dynamic_address(
    base: u64,
    start: u64,
    end: u64,
    size: u64,
    offset: u64,
    minimum: u64,
) -> Result<u64> {
    if offset % 256 != 0 {
        return Err(Error::InvalidRange);
    }
    if end.checked_sub(start).is_none_or(|len| len < minimum)
        || end.checked_add(offset).is_none_or(|end| end > size)
    {
        return Err(Error::InvalidRange);
    }
    let start = start.checked_add(offset).ok_or(Error::InvalidRange)?;
    let address = base.checked_add(start).ok_or(Error::InvalidRange)?;
    if address % 256 != 0 {
        return Err(Error::InvalidRange);
    }
    Ok(address)
}

fn validate_indirect(usage: bool, size: u64, offset: u64) -> Result<()> {
    if !usage || offset % 4 != 0 || offset.checked_add(12).is_none_or(|end| end > size) {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/backend/compute.rs"]
mod tests;
