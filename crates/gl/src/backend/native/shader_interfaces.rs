//! Native shader interface facts, with direct-SPIR-V identities kept in Rust.
//!
//! Driver names are used for GLSL only. SPIR-V vertex inputs and opaque uniforms
//! retain their explicit locations even when the driver discards `OpName`.

use super::provider::NativeGlProvider;
use super::spirv_opaque::OpaqueUniform;
use crate::backend::api::{GlError, GlShaderSource, GlVertexInputReflection};
use std::collections::BTreeMap;

const SAMPLER_TYPES: &[u32] = &[
    0x8B5E, 0x8B5F, 0x8B60, 0x8B62, 0x8DC1, 0x8B5D, 0x8B61, 0x8B63, 0x8B64, 0x8DC0, 0x8DC2, 0x8DC3,
    0x8DC9, 0x8DCD, 0x8DCE, 0x8DD0, 0x8DD1, 0x8DD5, 0x8DD6, 0x8DD8, 0x900C, 0x900D, 0x900E, 0x900F,
    0x9108, 0x9109, 0x910A, 0x910B, 0x910C, 0x910D, 0x8DC5, // SAMPLER_CUBE_SHADOW
    0x8DC4, // SAMPLER_2D_ARRAY_SHADOW
    0x8DCA, 0x8DCB, 0x8DCC, 0x8DCF, // INT_SAMPLER_2D, 3D, CUBE, 2D_ARRAY
    0x8DD2, 0x8DD3, 0x8DD4, 0x8DD7, // UNSIGNED_INT_SAMPLER_2D, 3D, CUBE, 2D_ARRAY
];

fn vertex_input_columns(type_: u32) -> Option<u8> {
    Some(match type_ {
        0x1404..=0x1406 | 0x140A => 1,
        0x8B50 | 0x8B53 | 0x8DC6 | 0x8FFC => 2,
        0x8B51 | 0x8B54 | 0x8DC7 | 0x8FFD => 3,
        0x8B52 | 0x8B55 | 0x8DC8 | 0x8FFE => 4,
        _ => return None,
    })
}

pub(super) fn is_sampler_type(type_: u32) -> bool {
    SAMPLER_TYPES.contains(&type_)
}

pub(super) fn vertex_inputs(
    provider: &NativeGlProvider,
    operation: &'static str,
    program: glow::NativeProgram,
    sources: &[&GlShaderSource],
) -> Result<Vec<GlVertexInputReflection>, GlError> {
    use glow::HasContext as _;
    if sources.iter().any(|source| source.spirv.is_some()) {
        return super::spirv_vertex::inputs(sources.iter().copied());
    }
    let mut inputs = Vec::new();
    // SAFETY: the caller retains this linked, current-context program through
    // the provider's exclusive reflection transaction. These calls only query it.
    unsafe {
        for index in 0..provider.gl.get_active_attributes(program) {
            let info = provider
                .gl
                .get_active_attribute(program, index)
                .ok_or_else(|| {
                    NativeGlProvider::validation(
                        operation,
                        "active vertex attribute was not reported",
                    )
                })?;
            if info.name.starts_with("gl_") {
                continue;
            }
            let columns = vertex_input_columns(info.atype).ok_or(GlError::Unsupported {
                operation,
                reason: "vertex input type is outside the common attribute vocabulary",
            })?;
            let location = provider
                .gl
                .get_attrib_location(program, &info.name)
                .ok_or_else(|| {
                    NativeGlProvider::validation(operation, "active vertex input has no location")
                })?;
            inputs.push(GlVertexInputReflection {
                name: info.name,
                location,
                columns,
            });
        }
    }
    Ok(inputs)
}

pub(super) fn opaque_uniforms(
    provider: &NativeGlProvider,
    operation: &'static str,
    program: glow::NativeProgram,
    sources: &[&GlShaderSource],
) -> Result<BTreeMap<String, OpaqueUniform>, GlError> {
    use glow::HasContext as _;
    let mut active = BTreeMap::new();
    for uniform in super::spirv_opaque::uniforms(sources.iter().copied())? {
        let mut value = [0_i32];
        // SAFETY: opaque uniforms are integer-valued GL handles. The verified
        // artifact supplies an explicit linked uniform location; one array
        // element returns one integer. The program remains current and alive.
        let error = unsafe {
            provider.gl.get_uniform_i32(
                program,
                &glow::NativeUniformLocation(uniform.location),
                &mut value,
            );
            provider.gl.get_error()
        };
        if error == glow::NO_ERROR {
            active.insert(
                format!("@spirv:{}:{}", uniform.binding, uniform.location),
                uniform,
            );
        } else if error == glow::CONTEXT_LOST {
            return Err(GlError::ContextLost { operation });
        } else if error == glow::OUT_OF_MEMORY {
            return Err(GlError::OutOfMemory { operation });
        } else if error != glow::INVALID_OPERATION {
            return Err(GlError::Driver {
                operation,
                message: format!("SPIR-V uniform activity query failed: 0x{error:04x}"),
            });
        }
        // An explicit location may be inactive after specialization. Native GL
        // reports INVALID_OPERATION for that query; it creates no assignment.
    }
    Ok(active)
}

/// Establishes the direct-artifact pairing convention without changing Core's
/// resource model: one sampler may serve all textures in a group, or equally
/// sized texture/sampler declaration lists pair in logical binding order.
pub(super) fn spirv_layout(
    layout: &crate::backend::api::GlPipelineLayout,
) -> Result<crate::backend::api::GlPipelineLayout, GlError> {
    use crate::backend::api::GlShaderResourceKind as K;
    let mut result = layout.clone();
    let groups: std::collections::BTreeSet<_> = result
        .bindings
        .iter()
        .map(|entry| entry.location.group)
        .collect();
    for group in groups {
        let mut textures: Vec<_> = result
            .bindings
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.location.group == group
                    && matches!(entry.kind, K::Texture | K::CombinedTextureSampler)
            })
            .map(|(index, entry)| (entry.location, entry.element, index))
            .collect();
        let mut samplers: Vec<_> = result
            .bindings
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.location.group == group && matches!(entry.kind, K::Sampler))
            .map(|(index, entry)| (entry.location, entry.element, index))
            .collect();
        textures.sort();
        samplers.sort();
        if samplers.is_empty() {
            continue;
        } // Texel-fetch-only declarations need no sampler object.
        if samplers.len() != 1 && samplers.len() != textures.len() && !textures.is_empty() {
            return Err(NativeGlProvider::validation(
                "assign GL SPIR-V pairs",
                "texture/sampler declaration lists do not satisfy the direct-artifact pairing ABI",
            ));
        }
        for (ordinal, (_, _, texture_index)) in textures.iter().enumerate() {
            let sampler_index = samplers[if samplers.len() == 1 { 0 } else { ordinal }].2;
            let texture_location = result.bindings[*texture_index].location;
            let sampler_location = result.bindings[sampler_index].location;
            result.bindings[*texture_index].pair = Some(sampler_location);
            let alias = format!(
                "@spirv_pair_{}_{}_{}",
                group, texture_location.binding, sampler_location.binding
            );
            result.bindings[*texture_index].names.push(alias.clone());
            result.bindings[sampler_index].names.push(alias);
        }
    }
    Ok(result)
}
