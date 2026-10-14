//! Browser-owned WebGL2 discovery and execution.
//!
//! The Host/JS bridge owns canvas creation, DOM events, RAF, and context-loss
//! listeners. RHI creates and owns the WebGL2 context associated with that
//! Host-provided canvas, then gathers immutable evidence for its `ContextStamp`.

#![cfg(target_arch = "wasm32")]

mod completion;
mod discovery;
mod discovery_facts;
mod driver_identity;
mod encoder;
mod encoder_bindings;
mod exec_copy;
mod exec_copy_layout;
mod exec_framebuffer;
mod exec_multidraw;
mod exec_present;
mod exec_raster;
mod exec_shader;
mod exec_sync;
mod exec_timer;
mod exec_vertex;
mod format_map;
mod objects;
mod owner;
mod provider;
mod renderbuffer_facts;
mod resources;
mod shader_abi;
mod surface;
mod surface_core;

pub(crate) use completion::{BrowserCompletion, BrowserCompletionFuture};
pub(crate) use discovery::WebGl2BrowserDiscovery;
pub(crate) use owner::{
    BrowserBuffer, BrowserEncoder, BrowserOwner, BrowserRasterProgram, BrowserSampler,
    BrowserTexture, BrowserVertexArray,
};
pub(crate) use surface::{BrowserSurface, BrowserSurfaceImage};

/// The browser allocation path's exact immutable-storage admission predicate.
/// Backend capability publication consults this instead of duplicating the
/// format table, so a newly advertised format cannot later fail allocation
/// merely because the two tables drifted apart.
pub(crate) fn supports_texture_storage(format: crate::backend::gl::api::GlFormat) -> bool {
    format_map::internal_format(format).is_some()
}
