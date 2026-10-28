//! Browser-owned WebGL2 discovery and execution.
//!
//! The Host/JS bridge owns canvas creation, DOM events, RAF, and context-loss
//! listeners. RHI creates and owns the WebGL2 context associated with that
//! Host-provided canvas, then gathers immutable evidence for its `ContextStamp`.

#![cfg(target_arch = "wasm32")]

mod completion;
mod constant_vertex;
mod discovery;
mod discovery_facts;
mod driver_identity;
mod dummy_binding;
mod encoder;
mod encoder_bindings;
mod exec_shader;
mod exec_sync;
mod exec_timer;
mod format_map;
mod indexed_blend;
mod mapping;
mod multiview_probe;
mod objects;
mod ordinary_format;
mod owner;
mod polygon_offset;
mod provider;
mod push_constants;
mod query_core;
mod rasterization;
mod renderbuffer_facts;
mod resources;
mod surface;
mod surface_core;
mod texture_mip_uniform;

pub(crate) use completion::{BrowserCompletion, BrowserCompletionFuture};
pub(crate) use discovery::WebGl2BrowserDiscovery;
pub(crate) use encoder::{
    BrowserColorAttachment, BrowserDepthAttachment, BrowserResolveAttachment,
};
pub(crate) use owner::{
    BrowserBuffer, BrowserEncoder, BrowserOwner, BrowserRasterProgram, BrowserSampler,
    BrowserTexture, BrowserVertexArray,
};
pub(crate) use push_constants::BrowserPushUniformKind;
pub(crate) use surface::{BrowserSurface, BrowserSurfaceImage, BrowserSurfaceToken};

/// The browser allocation path's exact immutable-storage admission predicate.
/// Backend capability publication consults this instead of duplicating the
/// format table, so a newly advertised format cannot later fail allocation
/// merely because the two tables drifted apart.
pub(crate) fn supports_texture_storage(format: crate::backend::api::GlFormat) -> bool {
    format_map::internal_format(format).is_some()
}

#[cfg(test)]
#[path = "../../tests/backend/browser/array_clear.rs"]
mod webgl2_array_clear_raw;

#[cfg(test)]
#[path = "../../tests/backend/browser/multiview_attachment.rs"]
mod webgl2_multiview_attachment;

mod cube_mirror;
mod stencil_mirror;
