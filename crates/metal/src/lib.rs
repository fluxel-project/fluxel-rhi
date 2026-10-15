//! Apple Metal execution for Fluxel RHI.
//!
//! Metal objects remain private to this crate and command calls lower directly
//! to native encoders. RenderGraph owns ordering, hazards, and retirement; this
//! backend reports native execution and completion facts.

#![deny(missing_docs)]

/// The portable RHI API shared with other backends.
pub use fluxel_rhi_core::api;

mod backend;

/// The statically dispatched Metal backend.
pub use backend::Metal;
/// A Metal instance used for adapter discovery and device creation.
pub use backend::MetalInstance;
/// A native Metal Surface associated with a host-owned `CAMetalLayer`.
pub use backend::MetalSurface;
