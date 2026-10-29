//! Vulkan clear regression test target.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "../support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block_on;
#[path = "depth_bias.rs"]
mod depth_bias;
#[path = "rect_attachment.rs"]
mod rect_attachment;
#[path = "subresources.rs"]
mod subresources;
#[path = "validation.rs"]
mod validation;
