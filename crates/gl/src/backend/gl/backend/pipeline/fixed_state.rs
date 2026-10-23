//! Translate vertex layouts and depth/stencil state to the GL ABI.

use super::*;

pub(super) fn lower_vertex_layout(
    values: &[api::VertexBufferLayout],
) -> Result<crate::backend::gl::api::GlVertexLayout> {
    let buffers = values
        .iter()
        .enumerate()
        .map(|(slot, value)| {
            Ok(crate::backend::gl::api::GlVertexBufferLayout {
                slot: u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                stride: u32::try_from(value.stride).map_err(|_| Error::InvalidRange)?,
                step_mode: match value.step_mode {
                    VertexStepMode::Vertex => crate::backend::gl::api::GlVertexStepMode::Vertex,
                    VertexStepMode::Instance => crate::backend::gl::api::GlVertexStepMode::Instance,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let attributes = values
        .iter()
        .enumerate()
        .flat_map(|(slot, buffer)| {
            buffer
                .attributes
                .iter()
                .map(move |attribute| (slot, attribute))
        })
        .map(|(slot, attribute)| {
            Ok(crate::backend::gl::api::GlVertexAttribute {
                location: attribute.location,
                buffer_slot: u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                format: lower_vertex_format(attribute.format)?,
                offset: u32::try_from(attribute.offset).map_err(|_| Error::InvalidRange)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(crate::backend::gl::api::GlVertexLayout {
        buffers,
        attributes,
    })
}

/// Lowers depth state without inventing a format conversion. Attachment
/// validation later checks that this declared format matches the pass view.
pub(super) fn lower_depth_stencil(
    value: api::DepthStencilState,
) -> Result<crate::backend::gl::api::GlDepthStencilState> {
    if !matches!(
        value.format,
        api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth32FloatStencil8
            | api::TextureFormat::Stencil8
    ) || !value.depth_bias_slope_scale.is_finite()
        || !value.depth_bias_clamp.is_finite()
    {
        return Err(Error::Unsupported("GL depth-stencil state"));
    }
    let face = |face: api::StencilFaceState| crate::backend::gl::api::GlStencilFaceState {
        compare: crate::backend::gl::translate_raster::lower_compare(face.compare),
        fail_op: lower_stencil_op(face.fail),
        depth_fail_op: lower_stencil_op(face.depth_fail),
        pass_op: lower_stencil_op(face.pass),
        read_mask: value.stencil_read_mask,
        write_mask: value.stencil_write_mask,
    };
    Ok(crate::backend::gl::api::GlDepthStencilState {
        depth_write_enabled: value.depth_write_enabled,
        depth_compare: crate::backend::gl::translate_raster::lower_compare(value.depth_compare),
        // Shared native/browser ABI: polygon offset factor, then units, then
        // clamp. Units are a float in GL even though Core exposes an integer.
        depth_bias: [
            value.depth_bias_slope_scale.to_bits(),
            (value.depth_bias_constant as f32).to_bits(),
            value.depth_bias_clamp.to_bits(),
        ],
        stencil_front: face(value.stencil_front),
        stencil_back: face(value.stencil_back),
        stencil_reference: 0,
    })
}

fn lower_stencil_op(value: api::StencilOperation) -> crate::backend::gl::api::GlStencilOperation {
    use crate::backend::gl::api::GlStencilOperation as Gl;
    match value {
        api::StencilOperation::Keep => Gl::Keep,
        api::StencilOperation::Zero => Gl::Zero,
        api::StencilOperation::Replace => Gl::Replace,
        api::StencilOperation::IncrementClamp => Gl::IncrementClamp,
        api::StencilOperation::DecrementClamp => Gl::DecrementClamp,
        api::StencilOperation::Invert => Gl::Invert,
        api::StencilOperation::IncrementWrap => Gl::IncrementWrap,
        api::StencilOperation::DecrementWrap => Gl::DecrementWrap,
    }
}
