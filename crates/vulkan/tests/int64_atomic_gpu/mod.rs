//! Vulkan Direct-SPIR-V Int64 atomic GPU target.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "buffer.rs"]
mod buffer;
#[path = "common.rs"]
mod common;
#[path = "workgroup.rs"]
mod workgroup;
