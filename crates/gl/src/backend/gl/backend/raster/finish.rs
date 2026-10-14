//! Abort cleanup for one immediate GL raster pass.
//!
//! Aborting cannot replay or roll back GL calls already issued. A temporary
//! surface-depth FBO is destroyed without publishing its color to the default
//! framebuffer, and a provider failure keeps the encoder indeterminate.

use fluxel_rhi_core::api::RasterBackend;

use super::super::Gl;
use super::super::types::GlEncoder;

pub(super) fn abort(encoder: &mut GlEncoder) {
    if encoder.occlusion_query_active {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            use crate::backend::gl::api::GlOcclusionQueryApi as _;
            let result = encoder
                .state
                .with_native("abort active GL occlusion query", |provider| {
                    provider.end_occlusion_query()
                });
            if result.is_err() {
                encoder.indeterminate = true;
            }
        }
        encoder.occlusion_query_active = false;
    }
    encoder.occlusion_query_set = None;
    // Core calls this only while unwinding a poisoned pass. EndRenderPass is
    // the provider's legal exit from active-pass state; no retry is possible.
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    if let Some(browser) = encoder.browser.as_mut() {
        if browser.has_surface_depth_pass() {
            browser.abort_surface_depth_raster();
            encoder.raster_extent = None;
            encoder.raster_pipeline = None;
            return;
        }
        if browser.has_owned_framebuffer_pass() {
            browser.abort_offscreen_raster();
            encoder.raster_extent = None;
            encoder.raster_pipeline = None;
            return;
        }
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if let Some(pass) = encoder.surface_depth_pass.take() {
        use crate::backend::gl::api::{GlFramebufferApi as _, GlResourceApi as _};
        let result = encoder
            .state
            .with_native("abort GL surface depth pass", |provider| {
                let end = provider.end_render_pass();
                let framebuffer = provider.destroy_framebuffer(pass.framebuffer);
                let color = provider.destroy_texture_resource(pass.color);
                end.and(framebuffer).and(color)
            });
        if result.is_err() {
            encoder.indeterminate = true;
        }
        encoder.raster_extent = None;
        encoder.raster_pipeline = None;
        return;
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if let Some(framebuffer) = encoder.offscreen_framebuffer.take() {
        use crate::backend::gl::api::GlFramebufferApi as _;
        // An aborted pass never publishes a multisample resolve into the
        // acquired surface; dropping this lease proof also prevents later reuse.
        encoder.offscreen_resolve = None;
        let result = encoder
            .state
            .with_native("abort GL offscreen raster pass", |provider| {
                let end = provider.end_render_pass();
                let drop_framebuffer = provider.destroy_framebuffer(framebuffer);
                end.and(drop_framebuffer)
            });
        if result.is_err() {
            encoder.indeterminate = true;
        }
        encoder.raster_extent = None;
        encoder.raster_pipeline = None;
        return;
    }
    let _ = <Gl as RasterBackend>::end_raster(encoder).map_err(|_| encoder.indeterminate = true);
}
