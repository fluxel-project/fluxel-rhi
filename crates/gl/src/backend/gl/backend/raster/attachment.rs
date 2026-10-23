//! Color attachment load/store/clear lowering shared by GL pass targets.

use fluxel_rhi_core::api::{ClearColor, LoadOp, Result, StoreOp};

pub(super) fn lower_load(value: LoadOp<ClearColor>) -> Result<crate::backend::gl::api::GlLoadOp> {
    Ok(match value {
        LoadOp::Clear(_) => crate::backend::gl::api::GlLoadOp::Clear,
        // Keeping old contents is legal for a discard load.
        LoadOp::Load | LoadOp::Discard => crate::backend::gl::api::GlLoadOp::Load,
    })
}
pub(super) fn lower_store(value: StoreOp) -> crate::backend::gl::api::GlStoreOp {
    match value {
        StoreOp::Store => crate::backend::gl::api::GlStoreOp::Store,
        StoreOp::Discard => crate::backend::gl::api::GlStoreOp::Discard,
    }
}
pub(super) fn lower_clear(
    value: LoadOp<ClearColor>,
) -> Result<crate::backend::gl::api::GlColorClearValue> {
    use crate::backend::gl::api::GlColorClearValue as Gl;
    Ok(match value {
        LoadOp::Clear(ClearColor::Float(value)) => Gl::Float(value.map(f32::to_bits)),
        LoadOp::Clear(ClearColor::Uint(value)) => Gl::Uint(value),
        LoadOp::Clear(ClearColor::Sint(value)) => Gl::Sint(value),
        LoadOp::Load | LoadOp::Discard => Gl::Float([0; 4]),
    })
}
