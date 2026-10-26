//! Synchronisation routes. `ARB_texture_barrier` and `NV_texture_barrier`
//! remain distinct evidence because the latter uses a different entry token.
use super::{
    GL32, GL42, GL45, GLES30, GLES31, GlExtension as E, GlFeature as F, GlFunction as Fn, NEVER,
    Requirement,
};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::Sync => Requirement {
            desktop_core: GL32,
            gles_core: GLES30,
            extensions: &[E::ArbSync],
            functions: &[Fn::FenceSync, Fn::ClientWaitSync, Fn::DeleteSync],
        },
        F::MemoryBarrier => Requirement {
            desktop_core: GL42,
            gles_core: GLES31,
            extensions: &[E::ArbShaderImageLoadStore],
            functions: &[Fn::MemoryBarrier],
        },
        F::MemoryBarrierByRegion => Requirement {
            desktop_core: GL45,
            gles_core: GLES31,
            extensions: &[],
            functions: &[Fn::MemoryBarrierByRegion],
        },
        // ARB/core use `glTextureBarrier`; NV uses `glTextureBarrierNV`.
        // `GlFeatureProbe` chooses the route and checks its exact token.
        F::TextureBarrier => Requirement {
            desktop_core: GL45,
            gles_core: NEVER,
            extensions: &[E::ArbTextureBarrier, E::NvTextureBarrier],
            functions: &[],
        },
        _ => return None,
    })
}

#[cfg(test)]
#[path = "../../tests/backend/features/sync__tests.rs"]
mod tests;
