//! KHR_debug command annotation lowering for native GL-family contexts.

use super::provider::NativeGlProvider;
use crate::backend::api::{GlError, GlFamilyApi as _};
use fluxel_rhi_core::backend::DebugMarkerOp;
use glow::HasContext as _;

impl NativeGlProvider {
    pub(crate) const fn debug_markers_supported(&self) -> bool {
        self.debug_markers
    }

    /// Emits one command annotation after bootstrap proved the complete
    /// KHR_debug/core entry-point group for this exact context generation.
    pub(crate) fn debug_marker(&mut self, op: DebugMarkerOp<'_>) -> Result<(), GlError> {
        const OP: &str = "GL debug marker";
        self.assert_ready(OP)?;
        if !self.debug_markers {
            return Err(GlError::Unsupported {
                operation: OP,
                reason: "KHR_debug marker entry points are unavailable",
            });
        }
        // SAFETY: `debug_markers` is set only after bootstrap proved these
        // functions against this current context; `with_native` keeps that
        // context current and the label borrow lasts for this call.
        unsafe {
            match op {
                DebugMarkerOp::Insert(label) => self.gl.debug_message_insert(
                    glow::DEBUG_SOURCE_APPLICATION,
                    glow::DEBUG_TYPE_MARKER,
                    0,
                    glow::DEBUG_SEVERITY_NOTIFICATION,
                    label,
                ),
                DebugMarkerOp::Push(label) => {
                    self.gl
                        .push_debug_group(glow::DEBUG_SOURCE_APPLICATION, 0, label)
                }
                DebugMarkerOp::Pop => self.gl.pop_debug_group(),
            }
        }
        self.defer_driver_error(OP)
    }
}
