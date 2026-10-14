//! Color attachment load/store/clear lowering shared by GL pass targets.

use fluxel_rhi_core::api::{ClearColor, Error, LoadOp, Result, StoreOp};

pub(super) fn lower_load(value: LoadOp<ClearColor>) -> Result<crate::backend::gl::api::GlLoadOp> {
    Ok(match value {
        LoadOp::Load => crate::backend::gl::api::GlLoadOp::Load,
        LoadOp::Clear(_) => crate::backend::gl::api::GlLoadOp::Clear,
        LoadOp::Discard => return Err(Error::Unsupported("GL color load discard")),
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
    match value {
        LoadOp::Clear(ClearColor::Float([red, green, blue, alpha])) => {
            Ok(crate::backend::gl::api::GlColorClearValue {
                red: red.to_bits(),
                green: green.to_bits(),
                blue: blue.to_bits(),
                alpha: alpha.to_bits(),
            })
        }
        LoadOp::Load => Ok(crate::backend::gl::api::GlColorClearValue {
            red: 0,
            green: 0,
            blue: 0,
            alpha: 0,
        }),
        LoadOp::Clear(_) => Err(Error::Unsupported("GL integer color clear")),
        LoadOp::Discard => Err(Error::Unsupported("GL color load discard")),
    }
}
