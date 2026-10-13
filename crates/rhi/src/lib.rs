//! Fluxel's portable GPU execution facade.
//!
//! The public vocabulary is [`api`]. It has statically typed backend objects:
//! GPU command hot paths do not use `dyn Any`, runtime downcasts, or a backend
//! enum. RenderGraph and renderer code own scheduling, hazard analysis, frame
//! policy, and resource lifetime planning; the RHI executes the resulting work.
//!
//! Direct3D 12 and Vulkan are native Windows backends; WebGPU runs in browsers.

#![deny(missing_docs)]

/// The statically dispatched RHI vocabulary and backend contracts.
pub use fluxel_rhi_core::api;

/// Direct3D 12 is the active native backend on Windows.
#[cfg(all(feature = "dx12", windows))]
pub mod dx12 {
    /// The statically typed Direct3D 12 backend implementation.
    pub use fluxel_rhi_dx12::*;
}

/// Vulkan backend using statically typed native resources and commands.
#[cfg(feature = "vulkan")]
pub mod vulkan {
    /// Vulkan backend implementation and construction entry points.
    pub use fluxel_rhi_vulkan::*;
}

/// Browser WebGPU backend with statically typed resources and commands.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub mod webgpu {
    /// WebGPU backend implementation and construction entry points.
    pub use fluxel_rhi_webgpu::*;
}
