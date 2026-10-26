//! Feature-route requirements for compute dispatch.

use super::{GL42, GL43, GLES31, GlExtension as E, GlFeature as F, GlFunction as Fn, Requirement};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::Compute => Requirement {
            desktop_core: GL43,
            gles_core: GLES31,
            extensions: &[E::ArbComputeShader],
            functions: &[Fn::DispatchCompute],
        },
        F::IndirectDispatch => Requirement {
            desktop_core: GL43,
            gles_core: GLES31,
            extensions: &[E::ArbComputeShader],
            functions: &[Fn::DispatchComputeIndirect],
        },
        F::ShaderStorageBuffers => Requirement {
            desktop_core: GL43,
            gles_core: GLES31,
            extensions: &[E::ArbShaderStorageBufferObject],
            functions: &[Fn::ShaderStorageBlockBinding],
        },
        F::ImageLoadStore => Requirement {
            desktop_core: GL42,
            gles_core: GLES31,
            extensions: &[E::ArbShaderImageLoadStore],
            functions: &[Fn::BindImageTexture, Fn::MemoryBarrier],
        },
        _ => return None,
    })
}

#[cfg(test)]
#[path = "../../tests/backend/features/compute__tests.rs"]
mod tests;
