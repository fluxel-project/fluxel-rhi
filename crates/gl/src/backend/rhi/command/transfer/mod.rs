//! Immediate buffer/texture copy, blit, and upload lowering for GL.
//!
//! These methods validate each transfer and issue it on the current context
//! before returning; only acceptance facts remain on the encoder.

use super::*;

mod copies;
mod uploads;

pub(super) use copies::{blit, copy_buffer, copy_texture};
pub(super) use uploads::{buffer_to_texture, texture_to_buffer, write_buffer, write_texture};

/// Lowers a Core texture region without changing the public coordinate system.
/// Native and browser executors perform their profile-specific Y conversion at
/// the call that touches GL, exactly once.
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
fn lower_region(value: TextureRegion<'_, Gl>) -> Result<crate::backend::api::GlTextureRegion> {
    // SAFETY: CommandBackend receives this region from Core while the Texture
    // borrow is live. The native reference is used synchronously only to read
    // immutable backend allocation identity and descriptor for lowering.
    let texture = unsafe { value.texture.native_ref() };
    crate::backend::translate::lower_texture_region(texture.id, texture.desc, &value)
}
