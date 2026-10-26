//! Feature-route requirements for texture operations.

use super::{
    GL30, GL42, GL43, GLES30, GlExtension as E, GlFeature as F, GlFunction as Fn, NEVER,
    Requirement,
};

pub(super) fn requirement(feature: F) -> Option<Requirement> {
    Some(match feature {
        F::SamplerAnisotropy => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::ExtTextureFilterAnisotropic],
            functions: &[Fn::TexParameterAnisotropy],
        },
        F::CompressionBcS3tc => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::ExtTextureCompressionS3tc],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionBcRgtc => Requirement {
            desktop_core: GL30,
            gles_core: NEVER,
            extensions: &[E::ExtTextureCompressionRgtc],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionBcBptc => Requirement {
            desktop_core: GL42,
            gles_core: NEVER,
            extensions: &[E::ArbTextureCompressionBptc],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionEtc2Eac => Requirement {
            desktop_core: GL43,
            gles_core: GLES30,
            extensions: &[E::ArbEs3Compatibility, E::OesCompressedEtc2Rgb8Texture],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionAstcLdr => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::KhrTextureCompressionAstcLdr],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::CompressionAstcHdr => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::KhrTextureCompressionAstcHdr],
            functions: &[Fn::CompressedTexImage2d, Fn::CompressedTexSubImage2d],
        },
        F::Multiview => Requirement {
            desktop_core: NEVER,
            gles_core: NEVER,
            extensions: &[E::OvrMultiview2],
            functions: &[Fn::FramebufferTextureMultiview],
        },
        _ => return None,
    })
}

#[cfg(test)]
#[path = "../../tests/backend/features/texture__tests.rs"]
mod tests;
