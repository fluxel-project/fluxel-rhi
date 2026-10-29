//! Focused DX12 texture-clear GPU regressions.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[path = "clear_gpu/color_subresource.rs"]
mod color_subresource;
#[path = "clear_gpu/depth.rs"]
mod depth;
#[path = "clear_gpu/validation.rs"]
mod validation;
