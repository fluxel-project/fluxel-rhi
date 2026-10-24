//! Native direct-queue command path for the DX12 backend.
//!
//! This module deliberately owns actual allocator/list/fence objects.  A
//! `Dx12RecordedWork` retains its allocator and upload resources until the
//! queue has accepted it, and `Dx12Completion` names the exact fence value
//! signalled after that submission.

use std::mem::ManuallyDrop;

use fluxel_rhi_core::api::{
    self, CompletionCoverage, CompletionPoint, CompletionState, DeferredBackend, Error, LaneId,
    PresentReceipt, StepPayload, SubmissionReceipt, SubmitOutcome, SubmitPacket,
};
use windows::{
    Win32::{
        Foundation::RECT,
        Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        Graphics::Direct3D12::*,
        Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC},
    },
    core::Interface,
};

use super::{
    Dx12, Dx12BindGroup, Dx12Buffer, Dx12ComputePipeline, Dx12Device, Dx12QuerySet,
    Dx12RasterPipeline, Dx12Texture, native_error,
};

mod aliasing;
mod blit_compute;
mod buffer_clear;
mod clear;
mod coordinates;
mod copy;
mod core;
mod depth_readback;
pub(crate) mod external_segment;
pub(super) mod fence_wait;
mod global_barrier;
mod heaps;
mod indirect;
pub(crate) mod integer_clear;
mod mipmap;
mod ownership;
pub(crate) mod query;
mod raster;
pub(crate) mod raster_snapshot;
mod readback;
mod record;
mod resolve;
pub(crate) mod secondary;
mod submit;
mod transition;
pub(crate) mod upload;
pub(crate) use blit_compute::BlitComputeCache;
pub(super) use coordinates::{scissor as validate_scissor, viewport as validate_viewport};
pub(crate) use core::DirectCommandPool;
pub use core::{Dx12Completion, Dx12Encoder, Dx12RecordedWork, RasterExtras};
pub(crate) use mipmap::MipBlitCache;
pub(super) use mipmap::{linear_mip_blit, validate_mip_blit};
pub(super) use raster::barrier;
pub(crate) use secondary::{Dx12SecondaryEncoder, Dx12SecondaryWork};
pub(super) use transition::{
    barrier_sub, buffer_state, copy_location, texture_state, texture_upload_buffer,
};

fn restore_after_internal_draw(encoder: &mut Dx12Encoder) -> Result<(), Error> {
    if encoder.pipeline.is_some() || encoder.mesh_pipeline.is_some() {
        raster_snapshot::restore(encoder)?;
    } else if let Some(ray) = &encoder.ray_pipeline {
        let list = encoder
            .list
            .cast::<ID3D12GraphicsCommandList4>()
            .map_err(native_error)?;
        // SAFETY: the state object is retained by the encoder until completion.
        unsafe { list.SetPipelineState1(&ray.0.state) };
    } else if let Some(compute) = &encoder.compute_pipeline {
        // SAFETY: the pipeline is retained by the encoder until completion.
        unsafe { encoder.list.SetPipelineState(&compute.0.state) };
    }
    Ok(())
}
