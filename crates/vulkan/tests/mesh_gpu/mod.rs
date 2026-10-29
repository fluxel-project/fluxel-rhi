//! Vulkan GPU regressions for mesh dispatch modes and raster output.
#![cfg(all(windows, feature = "hardware-tests"))]

#[path = "direct.rs"]
mod direct;
#[path = "generated.rs"]
mod generated;
#[path = "scenario.rs"]
mod scenario;
#[path = "../support/mod.rs"]
mod support;
