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
    pub(crate) raster_multiview: Option<(u32, u32)>,
    /// Attachments outlive the descriptor used to begin an immediate pass.
    pub(crate) raster_views: Vec<GlTextureView>,
    /// Dynamic viewport/scissor survive pipeline selection just as they do in
    /// Core. `None` means no raster pass is active; a pass starts with the full
    /// target viewport and disabled scissor.
    pub(crate) raster_viewport: Option<crate::backend::api::GlViewport>,
    pub(crate) raster_scissor: Option<crate::backend::api::GlScissorRect>,
    pub(crate) raster_blend_constant: [f32; 4],
    pub(crate) raster_stencil_reference: u32,
    pub(crate) raster_depth_read_only: bool,
    pub(crate) raster_stencil_read_only: bool,
    pub(crate) raster_color_slots: Vec<bool>,
    /// Vertex-input setters describe the one mutable GL VAO. They are applied
    /// directly before draw because a VAO validation requires all slots at
    /// once; this is state, never retained draw work.
    pub(crate) vertex_bindings: Vec<crate::backend::api::GlVertexBufferBinding>,
    pub(crate) index_binding: Option<crate::backend::api::GlIndexBinding>,
    /// WebGL stores the element buffer in its VAO, but draw calls still need
    /// the current Core byte base and element width. This is immediate encoder
    /// state, not a deferred command record.
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser_index: Option<(u64, fluxel_rhi_core::api::IndexFormat)>,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser_vertex_bindings: Vec<(u32, super::types::GlBuffer, u64, u64)>,
    pub(crate) raster_pipeline: Option<GlRasterPipeline>,
    /// Current Core binding state, retained so a newly linked program can
    /// install the same resources at its own reflected binding points.
    pub(crate) bound_groups: Vec<(u32, super::types::GlBindGroup, Vec<u32>)>,
    pub(crate) push_bytes: [Vec<u8>; 3],
    /// Query set declared by the current raster pass. This is only pass
    /// ownership state; individual query calls still issue immediately.
    pub(crate) occlusion_query_set: Option<GlQuerySet>,
    pub(crate) occlusion_query_active: bool,
    pub(crate) pass_timestamp_end: Option<(GlQuerySet, u32)>,
    pub(crate) statistics_query_active: bool,
    pub(crate) statistics_query_set: Option<GlQuerySet>,
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
    pub(crate) offscreen_framebuffer: Option<crate::backend::api::FramebufferId>,
    /// A multisample offscreen pass may resolve directly into its acquired
    /// default framebuffer. The token only proves the lease still names the
    /// same configured target when the pass closes.
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) offscreen_resolve: Option<GlOffscreenResolve>,
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    pub(crate) native_texture_resolves: Vec<(
        crate::backend::api::GlTextureView,
        crate::backend::api::GlTextureView,
        Option<crate::backend::api::GlScissorRect>,
    )>,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser: Option<crate::backend::browser::BrowserEncoder>,
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub(crate) browser_surface_token: Option<crate::backend::browser::BrowserSurfaceToken>,
}

impl GlEncoder {
    pub(crate) fn new(state: Rc<GlDeviceState>, device: fluxel_rhi_core::api::DeviceId) -> Self {
        Self {
            state,
            device,
            issued: false,
            indeterminate: false,
            raster_extent: None,
            raster_multiview: None,
            raster_views: Vec::new(),
            raster_viewport: None,
            raster_scissor: None,
            raster_blend_constant: [0.0; 4],
            raster_stencil_reference: 0,
            raster_depth_read_only: false,
            raster_stencil_read_only: false,
            raster_color_slots: Vec::new(),
            vertex_bindings: Vec::new(),
            index_binding: None,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            browser_index: None,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            browser_vertex_bindings: Vec::new(),
            raster_pipeline: None,
            bound_groups: Vec::new(),
            push_bytes: std::array::from_fn(|_| Vec::new()),
            occlusion_query_set: None,
            occlusion_query_active: false,
            pass_timestamp_end: None,
            statistics_query_active: false,
            statistics_query_set: None,
            compute_pipeline: None,
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            surface_depth_pass: None,
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            offscreen_framebuffer: None,
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            offscreen_resolve: None,
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            native_texture_resolves: Vec::new(),
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            browser: None,
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            browser_surface_token: None,
        }
    }

    pub(crate) fn begin_raster_state(&mut self, extent: [u32; 2]) {
        self.bound_groups.clear();
        self.raster_extent = Some(extent);
        self.raster_viewport = Some(crate::backend::api::GlViewport {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
            min_depth: 0.0f32.to_bits(),
            max_depth: 1.0f32.to_bits(),
        });
        self.raster_scissor = None;
        self.raster_blend_constant = [0.0; 4];
        self.raster_stencil_reference = 0;
        self.raster_depth_read_only = false;
        self.raster_stencil_read_only = false;
    }

    pub(crate) fn set_raster_read_only(
        &mut self,
        desc: &fluxel_rhi_core::api::RasterPassDesc<'_, super::types::Gl>,
    ) {
        self.raster_multiview = desc
            .multiview_mask
            .map(|mask| (mask.count_ones(), mask.trailing_zeros()));
        self.raster_views.clear();
        for color in desc.colors.iter().flatten() {
            for attachment in std::iter::once(&color.view).chain(color.resolve_target.iter()) {
                if let fluxel_rhi_core::api::AttachmentView::Texture(view) = attachment {
                    // SAFETY: Core holds the descriptor's view borrow for this call.
                    self.raster_views.push(unsafe { view.native_ref() }.clone());
                }
            }
        }
        if let Some(depth) = &desc.depth_stencil {
            if let fluxel_rhi_core::api::AttachmentView::Texture(view) = &depth.view {
                // SAFETY: the descriptor keeps this borrowed view alive now.
                self.raster_views.push(unsafe { view.native_ref() }.clone());
            }
        }
        self.raster_color_slots = desc.colors.iter().map(Option::is_some).collect();
        self.raster_depth_read_only = desc
            .depth_stencil
            .as_ref()
            .is_some_and(|v| v.depth_read_only);
        self.raster_stencil_read_only = desc
            .depth_stencil
            .as_ref()
            .is_some_and(|v| v.stencil_read_only);
    }

    pub(crate) fn end_raster_state(&mut self) {
        self.raster_multiview = None;
        self.raster_views.clear();
        self.raster_extent = None;
        self.raster_viewport = None;
        self.raster_scissor = None;
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) struct GlSurfaceDepthPass {
    pub(crate) target: crate::backend::api::GlDefaultFramebufferTarget,
    /// A read-only generation/serial check before publishing the temporary
    /// color target. It never releases Core's exclusive surface image lease.
    pub(crate) surface_token: super::surface::GlSurfaceTargetToken,
    pub(crate) framebuffer: crate::backend::api::FramebufferId,
    pub(crate) color: crate::backend::api::TextureId,
    pub(crate) depth: Option<GlTextureView>,
    pub(crate) color_slot: u32,
    pub(crate) render_area: Option<crate::backend::api::GlScissorRect>,
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) struct GlOffscreenResolve {
    pub(crate) source: crate::backend::api::GlTextureView,
    pub(crate) target: crate::backend::api::GlDefaultFramebufferTarget,
    pub(crate) surface_token: super::surface::GlSurfaceTargetToken,
    pub(crate) render_area: Option<crate::backend::api::GlScissorRect>,
}
