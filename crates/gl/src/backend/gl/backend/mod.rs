//! Static Core SPI boundary for the OpenGL family.
//!
//! This module owns only the concrete types shared by the Core backend traits.
//! Native context adoption, `glow` calls, and browser ownership stay in their
//! profile-specific private modules.  GL encoders issue calls immediately; no
//! recorded command list or submission replay is represented here.

pub(crate) mod cache;
mod command;
mod completion;
mod compute;
mod encoder;
mod extras;
mod immediate;
mod instance;
mod interop;
mod mapping;
mod pipeline;
mod push;
mod query;
pub(crate) mod raster;
mod readback;
mod resource;
pub(crate) mod shader;
pub(crate) mod spi;
mod surface;
mod types;

pub(crate) use surface::{GlSurface, GlSurfaceImage};
pub use types::{Gl, GlInstance};
pub(crate) use types::{GlDevice, GlDeviceState};

impl fluxel_rhi_core::api::ExecutionBackend for Gl {
    type Mode = fluxel_rhi_core::api::ImmediateExecution;
}
