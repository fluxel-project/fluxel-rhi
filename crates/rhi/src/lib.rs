//! Fluxel RHI — the portable GPU execution contract.
//!
//! This crate is the boundary between Fluxel's engine layers and the GPU APIs
//! beneath them. It exposes one portable vocabulary for device ownership,
//! resources, shaders, recording, submission, completion, and presentation, and
//! it owns the rules that decide whether a portable operation is legal without
//! asking a driver.
//!
//! # What this crate is
//!
//! ```text
//! portable GPU execution vocabulary
//! + instance capability facts
//! + strict validation
//! + opaque logical handles
//! + explicit hazard/dependency validation
//! + logical submission/completion/presentation
//! + portable logical statistics / inventory
//! ```
//!
//! # What this crate is not
//!
//! ```text
//! a wrapper over one native API
//! the greatest common denominator of all platforms
//! a native-handle escape hatch
//! ```
//!
//! It is capability-layered rather than levelled down: a backend contributes the
//! facts about what it can do, and the portable vocabulary stays whole. A caller
//! therefore never learns whether it is on DX12, Vulkan, Metal, WebGPU, or the
//! GL family in order to write correct code — but a caller that needs a
//! capability asks the device for the fact rather than probing for a type.
//!
//! # Layering
//!
//! ```text
//! Renderer / material graph / custom pipeline policy
//!     -> RenderGraph declaration and object recipes
//!     -> GraphExecutionPlan
//!     -> RHI RecordedWork + SubmissionPlan
//!     -> backend-private lowering
//!     -> DX12 | Vulkan | Metal | WebGPU | GL family
//! ```
//!
//! RenderGraph owns declarations, versions, dependencies, culling, scheduling,
//! logical lifetime, and presentation intent. This crate owns portable
//! execution, device validation, submission, completion, presentation,
//! retirement, logical observation, and backend lowering. Keeping those apart is
//! what stops the graph from becoming a second source of truth about hazards.
//!
//! # Where the rules live
//!
//! Architecture decisions live under `documents/adr/`; the compact map is
//! `documents/design-rhi.md`. The public API is specified by this
//! crate's rustdoc and contract tests, not by a duplicate prose specification.
//!
//! # Status
//!
//! The workspace is built contract-first. [`api`] holds the public surface, written
//! from the specification with validation and refusal paths fixed before native
//! lowering is admitted. An unavailable lowering must fail structurally before
//! native work is accepted; reachable `todo!()` / `unimplemented!()` paths and
//! dummy success are not valid capability implementations. Portable contracts
//! live in `fluxel-rhi-core`; native implementations live in separate backend
//! crates, and this crate reexports their provider constructors.

#![deny(missing_docs)]

/// The portable RHI vocabulary and validation layer.
pub use fluxel_rhi_core::api;

/// Fixture-only evidence and fault-injection helpers.
#[cfg(all(
    feature = "test-support",
    not(any(
        feature = "native-gl-wgl",
        feature = "native-gles-egl",
        feature = "webgl2"
    ))
))]
#[doc(hidden)]
pub use fluxel_rhi_core::test_support;

/// Fixture-only native GL evidence helpers.
#[cfg(all(
    feature = "test-support",
    any(
        feature = "native-gl-wgl",
        feature = "native-gles-egl",
        feature = "webgl2"
    )
))]
#[doc(hidden)]
pub use fluxel_rhi_gl::test_support;

/// Opens the native Direct3D 12 provider.
#[cfg(all(feature = "dx12", windows))]
pub use fluxel_rhi_dx12::create_provider as create_dx12_provider;

/// Opens the native Vulkan provider.
#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
pub use fluxel_rhi_vulkan::create_provider as create_vulkan_provider;

/// Opens the native desktop OpenGL WGL provider for a live Windows host window.
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub use fluxel_rhi_gl::{WglProvider, create_wgl_provider};

/// Opens the native Metal provider.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub use fluxel_rhi_metal::create_provider as create_metal_provider;

/// Opens the browser WebGPU provider.
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub use fluxel_rhi_webgpu::create_provider as create_webgpu_provider;

/// Android Vulkan presentation evidence bridge.
#[cfg(all(feature = "android-wsi-evidence", target_os = "android"))]
#[doc(hidden)]
pub use fluxel_rhi_vulkan::android_vulkan_wsi;
