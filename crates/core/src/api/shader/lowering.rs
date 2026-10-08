//! Optional common source-to-SPIR-V shader lowering.
//!
//! Backends own their native shader-module object and device-loss handling.
//! This module owns the Naga frontend, validation, and SPIR-V generation shared
//! by those backends. The returned words are transient lowering output: the
//! public artifact and its content identity remain unchanged.

use std::sync::Arc;

use crate::api::error::{RhiError, RhiErrorKind, RhiResult};

use super::ShaderArtifact;
#[cfg(feature = "naga")]
use super::{GlslProfile, ShaderCode, ShaderStage};

/// Lowers a Naga-supported source artifact to SPIR-V.
///
/// Native SPIR-V deliberately does not pass through this helper. A caller that
/// owns a Vulkan module should give its words directly to Vulkan so the native
/// artifact path stays free of frontend work.
#[cfg(feature = "naga")]
#[doc(hidden)]
pub fn naga_spirv(artifact: &ShaderArtifact) -> RhiResult<Arc<[u32]>> {
    let shader_stage = naga_stage(artifact.stage)?;
    let module = match &artifact.code {
        ShaderCode::Wgsl(source) => naga::front::wgsl::parse_str(source).map_err(|error| {
            unsupported(format!("Naga could not parse the WGSL shader: {error}"))
        })?,
        ShaderCode::Glsl {
            version,
            profile,
            source,
        } => parse_glsl(source, *version, *profile, shader_stage)?,
        ShaderCode::GlslEs { .. } => {
            return Err(unsupported(
                "Naga's GLSL frontend does not support OpenGL ES source lowering",
            ));
        }
        _ => {
            return Err(unsupported(
                "Naga can lower only WGSL or desktop core GLSL to SPIR-V",
            ));
        }
    };
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| unsupported(format!("Naga rejected the shader: {error}")))?;
    let mut options = naga::back::spv::Options::default();
    // The source-language binding values already name the logical group and
    // slot. Preserve them exactly when lowering to Vulkan descriptor sets.
    options.fake_missing_bindings = false;
    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = global.binding {
            options.binding_map.insert(
                binding.clone(),
                naga::back::spv::BindingInfo {
                    descriptor_set: binding.group,
                    binding: binding.binding,
                    binding_array_size: None,
                },
            );
        }
    }
    naga::back::spv::write_vec(
        &module,
        &info,
        &options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage,
            entry_point: artifact.entry_point.clone(),
        }),
    )
    .map(Arc::from)
    .map_err(|error| {
        unsupported(format!(
            "Naga could not lower the shader to SPIR-V: {error}"
        ))
    })
}

#[cfg(not(feature = "naga"))]
pub(crate) fn naga_spirv(_: &ShaderArtifact) -> RhiResult<Arc<[u32]>> {
    Err(unsupported(
        "source-to-SPIR-V lowering requires the fluxel-rhi `naga` feature",
    ))
}

#[cfg(feature = "naga")]
fn parse_glsl(
    source: &str,
    expected_version: u16,
    profile: GlslProfile,
    stage: naga::ShaderStage,
) -> RhiResult<naga::Module> {
    let GlslProfile::Core = profile;
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(stage);
    let module = frontend
        .parse(&options, source)
        .map_err(|error| unsupported(format!("Naga could not parse the GLSL shader: {error}")))?;
    let metadata = frontend.metadata();
    if metadata.version != expected_version {
        return Err(unsupported(format!(
            "the GLSL artifact declares version {expected_version}, but the source declares version {}",
            metadata.version
        )));
    }
    if metadata.profile != naga::front::glsl::Profile::Core {
        return Err(unsupported("Naga can lower only core-profile desktop GLSL"));
    }
    Ok(module)
}

#[cfg(feature = "naga")]
fn naga_stage(stage: ShaderStage) -> RhiResult<naga::ShaderStage> {
    match stage {
        ShaderStage::Vertex => Ok(naga::ShaderStage::Vertex),
        ShaderStage::Fragment => Ok(naga::ShaderStage::Fragment),
        ShaderStage::Compute => Ok(naga::ShaderStage::Compute),
        _ => Err(unsupported(
            "Naga's SPIR-V lowering supports only vertex, fragment, and compute shader stages",
        )),
    }
}

fn unsupported(message: impl Into<String>) -> RhiError {
    RhiError::new(RhiErrorKind::Unsupported, message).at("NagaShaderLowering::naga_spirv")
}

#[cfg(all(test, feature = "naga"))]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::api::shader::{
        ArtifactHash, ArtifactProducerVersion, ShaderAbiVersion, ShaderInterface,
        ShaderRequirements,
    };

    #[test]
    fn desktop_glsl_vertex_source_lowers_to_spirv() {
        let artifact = ShaderArtifact::new(
            ShaderStage::Vertex,
            "main",
            ShaderCode::Glsl {
                version: 450,
                profile: GlslProfile::Core,
                source: Arc::from(
                    "#version 450 core\nlayout(location = 0) in vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }",
                ),
            },
            ShaderAbiVersion { major: 1, minor: 0 },
            ShaderInterface::new().with_writes_position(true),
            ShaderRequirements::new(),
            ArtifactHash([0x47; 32]),
            ArtifactProducerVersion {
                major: 0,
                minor: 16,
            },
        );
        let words = naga_spirv(&artifact).expect("Naga must lower GLSL 450 core vertex source");
        assert_eq!(words.first(), Some(&0x0723_0203));
    }

    #[test]
    fn wgsl_vertex_source_lowers_to_spirv_for_its_named_entry_point() {
        let artifact = ShaderArtifact::new(
            ShaderStage::Vertex,
            "vertex_main",
            ShaderCode::Wgsl(Arc::from(
                "@vertex fn vertex_main() -> @builtin(position) vec4<f32> {\n\
                     return vec4<f32>(0.0, 0.0, 0.0, 1.0);\n\
                 }",
            )),
            ShaderAbiVersion { major: 1, minor: 0 },
            ShaderInterface::new().with_writes_position(true),
            ShaderRequirements::new(),
            ArtifactHash([0x57; 32]),
            ArtifactProducerVersion {
                major: 0,
                minor: 16,
            },
        );
        let words = naga_spirv(&artifact).expect("Naga must lower the named WGSL entry point");
        assert_eq!(words.first(), Some(&0x0723_0203));
    }

    #[test]
    fn unsupported_stage_returns_a_structured_unsupported_error() {
        let error = naga_stage(ShaderStage::Task).expect_err("task stage is not lowered yet");
        assert_eq!(error.kind(), RhiErrorKind::Unsupported);
        assert_eq!(error.operation(), Some("NagaShaderLowering::naga_spirv"));
    }

    #[test]
    fn glsl_es_source_is_structurally_refused() {
        let artifact = ShaderArtifact::new(
            ShaderStage::Vertex,
            "main",
            ShaderCode::GlslEs {
                version: 310,
                source: Arc::from("#version 310 es\nvoid main() { gl_Position = vec4(0.0); }"),
            },
            ShaderAbiVersion { major: 1, minor: 0 },
            ShaderInterface::new().with_writes_position(true),
            ShaderRequirements::new(),
            ArtifactHash([0x45; 32]),
            ArtifactProducerVersion {
                major: 0,
                minor: 16,
            },
        );
        let error = naga_spirv(&artifact).expect_err("Naga 27 does not parse GLSL ES");
        assert_eq!(error.kind(), RhiErrorKind::Unsupported);
        assert!(error.message().contains("OpenGL ES"));
    }
}
