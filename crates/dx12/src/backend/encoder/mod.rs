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

mod coordinates;
mod copy;
mod core;
pub(super) mod fence_wait;
mod indirect;
mod mipmap;
pub(crate) mod query;
mod raster;
mod readback;
mod record;
pub(crate) mod secondary;
mod submit;
mod transition;
pub(super) use coordinates::{scissor as validate_scissor, viewport as validate_viewport};
pub(crate) use core::DirectCommandPool;
pub use core::{Dx12Completion, Dx12Encoder, Dx12RecordedWork, RasterExtras};
pub(crate) use mipmap::MipBlitCache;
pub(super) use mipmap::{linear_mip_blit, validate_mip_blit};
pub(super) use raster::{barrier, upload_buffer};
pub(crate) use secondary::{Dx12SecondaryEncoder, Dx12SecondaryWork};
pub(super) use transition::{
    barrier_sub, buffer_state, copy_location, texture_state, texture_upload_buffer,
};
