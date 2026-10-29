//! Real GPU readback for multiple raster color targets.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "../support/mod.rs"]
mod support;
use support::block as block_on;

#[path = "color.rs"]
mod color;
#[path = "depth.rs"]
mod depth;
#[path = "resolve.rs"]
mod resolve;
#[path = "secondary.rs"]
mod secondary;
#[path = "validation.rs"]
mod validation;
