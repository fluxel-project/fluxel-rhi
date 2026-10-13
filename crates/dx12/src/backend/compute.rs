//! DX12 lowering for the compute domain.
//!
//! This module turns Core's already-validated compute pipeline and dispatch
//! operations directly into commands on the encoder's native command list. It
//! owns no scheduling or state tracking: cross-operation transitions and
//! submission order remain the caller's responsibility. Pipeline and bound
//! group clones are retained by the encoder solely until the submission fence
//! proves D3D12 has stopped reading their native objects.

use std::{
    future::{Ready, ready},
    mem::ManuallyDrop,
    sync::Arc,
};

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::*;

use super::{Dx12, Dx12ComputePipeline, Dx12Device};

pub(crate) struct ComputePipelineState {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) layout: Arc<super::pipeline::PipelineLayoutState>,
    pub(crate) state: ID3D12PipelineState,
}

/// Concrete static implementation selected by `Dx12::Compute`.
pub struct Compute;

impl api::ComputeOps<Dx12> for Compute {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<Result<Dx12ComputePipeline>>;

    fn create_pipeline<'a>(
        device: &'a Dx12Device,
        desc: &'a api::ComputePipelineDesc<Dx12>,
    ) -> Self::PipelineFuture<'a> {
        ready(create_pipeline(device, desc))
    }
    fn begin(_: &mut super::encoder::Dx12Encoder) -> Result<()> {
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
        encoder.compute_pipeline = Some(pipeline.clone());
        Ok(())
    }
    fn set_bind_group(
        encoder: &mut super::encoder::Dx12Encoder,
        slot: u32,
        group: &super::Dx12BindGroup,
        offsets: &[u32],
    ) -> Result<()> {
        if !offsets.is_empty() {
            return Err(Error::Unsupported("DX12 compute dynamic offsets"));
        }
        let pipeline = encoder
            .compute_pipeline
            .as_ref()
            .ok_or(Error::InvalidState)?;
        let table = pipeline
            .0
            .layout
            .tables
            .get(slot as usize)
            .ok_or(Error::InvalidRange)?;
        if group.0.has_views() != table.views.is_some()
            || group.0.has_samplers() != table.samplers.is_some()
        {
            return Err(Error::InvalidState);
        }
        // SAFETY: group heaps and table handles are same-device validated and
        // retained in `bound_groups` until this command list fence retires.
        unsafe {
            let heaps = [
                Some(group.0.heaps().view_heap().clone()),
                Some(group.0.heaps().sampler_heap().clone()),
            ];
            encoder.list.SetDescriptorHeaps(&heaps);
            if let Some(root) = table.views {
                encoder
                    .list
                    .SetComputeRootDescriptorTable(root, group.0.view_gpu());
            }
            if let Some(root) = table.samplers {
                encoder
                    .list
                    .SetComputeRootDescriptorTable(root, group.0.sampler_gpu());
            }
        }
        encoder.bound_groups.push(group.clone());
        Ok(())
    }
    fn dispatch(encoder: &mut super::encoder::Dx12Encoder, groups: [u32; 3]) -> Result<()> {
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
            encoder.list.Dispatch(groups[0], groups[1], groups[2]);
        }
        Ok(())
    }
    fn dispatch_indirect(
        _: &mut super::encoder::Dx12Encoder,
        _: &super::Dx12Buffer,
        _: u64,
    ) -> Result<()> {
        Err(Error::Unsupported("DX12 compute indirect dispatch"))
    }
    fn end(_: &mut super::encoder::Dx12Encoder) -> Result<()> {
        Ok(())
    }
    fn abort(_: &mut super::encoder::Dx12Encoder) {}
}

fn create_pipeline(
    device: &Dx12Device,
    desc: &api::ComputePipelineDesc<Dx12>,
) -> Result<Dx12ComputePipeline> {
    if !desc.compute.constants.is_empty()
        || desc.required_subgroup_size.is_some()
        || desc.cache.is_some()
    {
        return Err(Error::Unsupported("DX12 compute pipeline option"));
    }
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
    let bytes = &shader.bytes;
    let mut state_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(layout.root.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        },
        ..Default::default()
    };
    // SAFETY: root-signature COM reference and DXIL byte slice remain live for
    // the synchronous creation call; D3D12 copies the pipeline description.
    let state = unsafe {
        device
            .0
            .native
            .CreateComputePipelineState::<ID3D12PipelineState>(&state_desc)
    }
    .map_err(native_error)?;
    // SAFETY: this drops exactly the manually-owned root-signature clone put
    // into the FFI struct after D3D12 consumed it.
    unsafe {
        ManuallyDrop::drop(&mut state_desc.pRootSignature);
    }
    Ok(Dx12ComputePipeline(Arc::new(ComputePipelineState {
        root: layout.root.clone(),
        layout,
        state,
    })))
}

fn native_error(error: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {error}"))
}
