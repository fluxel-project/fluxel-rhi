//! Focused DX12 sparse-texture GPU regressions.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[path = "sparse_gpu/mapping.rs"]
mod mapping;
#[path = "sparse_gpu/packed_tail.rs"]
mod packed_tail;
