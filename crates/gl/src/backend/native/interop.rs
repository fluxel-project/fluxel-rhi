//! Native interop's context-local dispatch pointer and state invalidation.
//!
//! Resource tables remain authoritative; foreign code must not delete or
//! mutate objects owned by the backend.

use super::provider::NativeGlProvider;
use crate::backend::{
    api::{GlError, GlFamilyApi as _, GlFamilyProfile},
    state::{ScopedRawAccess, StateEvent},
};

/// Small set of standard flags not otherwise installed by the RHI pipeline.
pub(crate) struct NativeInteropSnapshot {
    rasterizer_discard: bool,
    dither: bool,
    sample_coverage: bool,
    color_logic_op: Option<bool>,
}

impl NativeGlProvider {
    pub(crate) fn interop_dispatch(
        &mut self,
    ) -> Result<(std::rc::Rc<glow::Context>, NativeInteropSnapshot), GlError> {
        use glow::HasContext as _;
        self.assert_ready("acquire native interop")?;
        if self.pass.is_some() || self.active_query.is_some() || !self.active_statistics.is_empty()
        {
            return Err(Self::validation(
                "acquire native interop",
                "a GL pass or query is active",
            ));
        }
        // SAFETY: the owner context is current. All three enables are legal
        // in desktop GL 4+ and GLES 3+; COLOR_LOGIC_OP is desktop-only.
        let snapshot = unsafe {
            NativeInteropSnapshot {
                rasterizer_discard: self.gl.is_enabled(glow::RASTERIZER_DISCARD),
                dither: self.gl.is_enabled(glow::DITHER),
                sample_coverage: self.gl.is_enabled(glow::SAMPLE_COVERAGE),
                color_logic_op: matches!(
                    self.discovery.context().profile(),
                    GlFamilyProfile::Desktop { .. }
                )
                .then(|| self.gl.is_enabled(glow::COLOR_LOGIC_OP)),
            }
        };
        self.driver_error("snapshot GL native interop")?;
        Ok((self.gl.clone(), snapshot))
    }

    pub(crate) fn end_interop(&mut self, snapshot: &NativeInteropSnapshot) -> Result<(), GlError> {
        use glow::HasContext as _;
        let saved = self.pixel_store;
        // SAFETY: the original context was made current, and these operations
        // re-establish the provider's pixel-store and scratch-buffer baseline.
        unsafe {
            for (key, value) in [
                (glow::PACK_ALIGNMENT, i32::from(saved.pack_alignment)),
                (glow::PACK_ROW_LENGTH, saved.pack_row_length as i32),
                (glow::PACK_SKIP_PIXELS, saved.pack_skip_pixels as i32),
                (glow::PACK_SKIP_ROWS, saved.pack_skip_rows as i32),
                (glow::UNPACK_ALIGNMENT, i32::from(saved.unpack_alignment)),
                (glow::UNPACK_ROW_LENGTH, saved.unpack_row_length as i32),
                (glow::UNPACK_IMAGE_HEIGHT, saved.unpack_image_height as i32),
                (glow::UNPACK_SKIP_PIXELS, saved.unpack_skip_pixels as i32),
                (glow::UNPACK_SKIP_ROWS, saved.unpack_skip_rows as i32),
                (glow::UNPACK_SKIP_IMAGES, saved.unpack_skip_images as i32),
            ] {
                self.gl.pixel_store_i32(key, value);
            }
            self.gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            self.gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            for (capability, enabled) in [
                (glow::RASTERIZER_DISCARD, Some(snapshot.rasterizer_discard)),
                (glow::DITHER, Some(snapshot.dither)),
                (glow::SAMPLE_COVERAGE, Some(snapshot.sample_coverage)),
                (glow::COLOR_LOGIC_OP, snapshot.color_logic_op),
            ] {
                match enabled {
                    Some(true) => self.gl.enable(capability),
                    Some(false) => self.gl.disable(capability),
                    None => {}
                }
            }
        }
        self.current_program = None;
        self.bound_vertex_array = None;
        self.context_state
            .event(StateEvent::ScopedRawAccess(ScopedRawAccess::all()));
        self.driver_error("restore GL native interop")
    }
}
