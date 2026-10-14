//! Transient state of one immediate GL command step.
//!
//! This keeps only state that GL itself makes mutable: the active pass extent,
//! VAO bindings, and selected pipeline. It owns neither a CPU command stream
//! nor submission policy. `issued` and `indeterminate` are close-reporting
//! facts required by Core after a direct native call has succeeded or may have
//! partially succeeded.

use std::rc::Rc;

use super::types::{GlComputePipeline, GlDeviceState, GlQuerySet, GlRasterPipeline, GlTextureView};

/// Immediate native encoder state for one Core command step.
pub struct GlEncoder {
    pub(crate) state: Rc<GlDeviceState>,
    pub(crate) device: fluxel_rhi_core::api::DeviceId,
    pub(crate) issued: bool,
    pub(crate) indeterminate: bool,
    /// Current pass extent is transient immediate state used only to construct
    /// a provider descriptor. It is not a recorded command or scheduler.
    pub(crate) raster_extent: Option<[u32; 2]>,
    /// Vertex-input setters describe the one mutable GL VAO. They are applied
    /// directly before draw because a VAO validation requires all slots at
    /// once; this is state, never retained draw work.
    pub(crate) vertex_bindings: Vec<crate::backend::gl::api::GlVertexBufferBinding>,
    pub(crate) index_binding: Option<crate::backend::gl::api::GlIndexBinding>,
    /// WebGL stores the element buffer in its VAO, but draw calls still need
    /// the current Core byte base and element width. This is immediate encoder
    /// state, not a deferred command record.
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser_index: Option<(u64, fluxel_rhi_core::api::IndexFormat)>,
    pub(crate) raster_pipeline: Option<GlRasterPipeline>,
    /// Query set declared by the current raster pass. This is only pass
    /// ownership state; individual query calls still issue immediately.
    pub(crate) occlusion_query_set: Option<GlQuerySet>,
    pub(crate) occlusion_query_active: bool,
    /// Selected compute pipeline for the current immediate compute pass. It
    /// retains only the linked native program and ABI needed by subsequent
    /// direct binds and dispatches; no dispatch packet is accumulated here.
    pub(crate) compute_pipeline: Option<GlComputePipeline>,
    /// Owned color/FBO objects for a surface pass that also uses a Core depth
    /// texture. Default framebuffer color cannot share an FBO with that depth
    /// texture, so the pass renders into this short-lived owned target and
    /// blits before it closes. `depth` retains the caller allocation because
    /// `RasterPassDesc` itself is borrowed only by `begin_raster`.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) surface_depth_pass: Option<GlSurfaceDepthPass>,
    /// User-owned texture attachments need a native FBO for exactly one
    /// immediate pass. The texture itself remains owned by Core resources.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) offscreen_framebuffer: Option<crate::backend::gl::api::FramebufferId>,
    /// A multisample offscreen pass may resolve directly into its acquired
    /// default framebuffer. The token only proves the lease still names the
    /// same configured target when the pass closes.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) offscreen_resolve: Option<GlOffscreenResolve>,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser: Option<crate::backend::gl::browser::BrowserEncoder>,
}

impl GlEncoder {
    pub(crate) fn new(state: Rc<GlDeviceState>, device: fluxel_rhi_core::api::DeviceId) -> Self {
        Self {
            state,
            device,
            issued: false,
            indeterminate: false,
            raster_extent: None,
            vertex_bindings: Vec::new(),
            index_binding: None,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            browser_index: None,
            raster_pipeline: None,
            occlusion_query_set: None,
            occlusion_query_active: false,
            compute_pipeline: None,
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            surface_depth_pass: None,
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            offscreen_framebuffer: None,
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            offscreen_resolve: None,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            browser: None,
        }
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) struct GlSurfaceDepthPass {
    pub(crate) target: crate::backend::gl::api::GlDefaultFramebufferTarget,
    /// A read-only generation/serial check before publishing the temporary
    /// color target. It never releases Core's exclusive surface image lease.
    pub(crate) surface_token: super::surface::GlSurfaceTargetToken,
    pub(crate) framebuffer: crate::backend::gl::api::FramebufferId,
    pub(crate) color: crate::backend::gl::api::TextureId,
    pub(crate) depth: GlTextureView,
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) struct GlOffscreenResolve {
    pub(crate) target: crate::backend::gl::api::GlDefaultFramebufferTarget,
    pub(crate) surface_token: super::surface::GlSurfaceTargetToken,
}
