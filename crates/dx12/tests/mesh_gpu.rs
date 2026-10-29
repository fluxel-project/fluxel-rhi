//! Focused DX12 mesh pipeline GPU regressions.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[path = "mesh_gpu/depth_controls.rs"]
mod depth_controls;
#[path = "mesh_gpu/mrt_blend_stencil.rs"]
mod mrt_blend_stencil;
#[path = "mesh_gpu/multi_indirect.rs"]
mod multi_indirect;
#[path = "mesh_gpu/storage_view_reservations.rs"]
mod storage_view_reservations;
