//! Select GLSL profiles and capabilities for the selected WGSL stage.

use super::*;

pub(super) fn uses_gather(module: &naga::Module, function: &naga::Function) -> bool {
    if function.expressions.iter().any(|(_, expr)| {
        matches!(
            expr,
            naga::Expression::ImageSample {
                gather: Some(_),
                ..
            }
        )
    }) {
        return true;
    }
    fn block(module: &naga::Module, statements: &naga::Block) -> bool {
        statements.iter().any(|statement| match statement {
            naga::Statement::Call { function, .. } => {
                uses_gather(module, &module.functions[*function])
            }
            naga::Statement::Block(inner) => block(module, inner),
            naga::Statement::If { accept, reject, .. } => {
                block(module, accept) || block(module, reject)
            }
            naga::Statement::Switch { cases, .. } => {
                cases.iter().any(|case| block(module, &case.body))
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => block(module, body) || block(module, continuing),
            _ => false,
        })
    }
    block(module, &function.body)
}

pub(super) fn naga_version(
    profile: crate::backend::gl::profile::GlProfile,
    stage: api::ShaderStage,
    storage: bool,
    gather: bool,
    multisampled_array: bool,
) -> Result<naga::back::glsl::Version> {
    let compute = stage == api::ShaderStage::Compute || storage;
    Ok(match profile {
        crate::backend::gl::profile::GlProfile::Embedded { major, minor }
            if multisampled_array && (major > 3 || major == 3 && minor >= 2) =>
        {
            naga::back::glsl::Version::Embedded {
                version: 320,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::Desktop { minor, .. } if compute && minor >= 3 => {
            naga::back::glsl::Version::Desktop(430)
        }
        crate::backend::gl::profile::GlProfile::Desktop { .. } if !compute => {
            naga::back::glsl::Version::Desktop(400)
        }
        crate::backend::gl::profile::GlProfile::Embedded { minor, .. } if compute && minor >= 1 => {
            naga::back::glsl::Version::Embedded {
                version: 310,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::Embedded { minor, .. }
            if (gather || multisampled_array) && !compute && minor >= 1 =>
        {
            naga::back::glsl::Version::Embedded {
                version: 310,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::Embedded { .. }
            if !compute && !gather && !multisampled_array =>
        {
            naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: false,
            }
        }
        crate::backend::gl::profile::GlProfile::WebGl2
            if !compute && !gather && !multisampled_array =>
        {
            naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            }
        }
        _ => return Err(Error::Unsupported("GL shader stage/profile")),
    })
}

#[cfg(test)]
#[test]
fn multisampled_array_selects_gles_310_only_when_available() {
    use crate::backend::gl::profile::GlProfile;
    use naga::back::glsl::Version;
    let stage = api::ShaderStage::Fragment;
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 1 },
            stage,
            false,
            false,
            true
        )
        .unwrap(),
        Version::Embedded {
            version: 310,
            is_webgl: false
        },
    );
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 2 },
            stage,
            false,
            false,
            true
        )
        .unwrap(),
        Version::Embedded {
            version: 320,
            is_webgl: false
        },
    );
    assert_eq!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 0 },
            stage,
            false,
            false,
            false
        )
        .unwrap(),
        Version::Embedded {
            version: 300,
            is_webgl: false
        },
    );
    assert!(
        naga_version(
            GlProfile::Embedded { major: 3, minor: 0 },
            stage,
            false,
            false,
            true
        )
        .is_err()
    );
    assert!(naga_version(GlProfile::WebGl2, stage, false, false, true).is_err());
}

pub(super) fn naga_stage(stage: api::ShaderStage) -> Result<naga::ShaderStage> {
    Ok(match stage {
        api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
        api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
        api::ShaderStage::Compute => naga::ShaderStage::Compute,
        _ => return Err(Error::Unsupported("GL WGSL shader stage")),
    })
}

pub(in crate::backend::gl::backend::shader) fn gl_stage(
    stage: api::ShaderStage,
) -> Result<crate::backend::gl::api::GlShaderStage> {
    Ok(match stage {
        api::ShaderStage::Vertex => crate::backend::gl::api::GlShaderStage::Vertex,
        api::ShaderStage::Fragment => crate::backend::gl::api::GlShaderStage::Fragment,
        api::ShaderStage::Compute => crate::backend::gl::api::GlShaderStage::Compute,
        _ => return Err(Error::Unsupported("GL WGSL shader stage")),
    })
}

/// Checks capability evidence rather than inferring compute from a version.
/// The native probe executed a dispatch; storage images are gated separately.
/// Browser WebGL2 has no compute route and remains false.
pub(in crate::backend::gl::backend::shader) fn compute_supported(device: &GlDevice) -> bool {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL compute capability", |provider| {
                let capabilities = provider.discovery().capabilities();
                Ok(capabilities.supports_compute())
            })
            .unwrap_or(false);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        false
    }
}

pub(super) fn gl_dialect(
    version: naga::back::glsl::Version,
) -> crate::backend::gl::api::GlShaderDialect {
    match version {
        naga::back::glsl::Version::Desktop(version) => {
            crate::backend::gl::api::GlShaderDialect::Desktop { version }
        }
        naga::back::glsl::Version::Embedded { version, .. } => {
            crate::backend::gl::api::GlShaderDialect::Embedded { version }
        }
    }
}
