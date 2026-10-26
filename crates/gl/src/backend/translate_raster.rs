//! Pure lowering of current Core raster fixed-function descriptors.
//!
//! Dynamic viewport, scissor and draw commands are lowered by the immediate
//! encoder. This module creates no GL objects and does not cache or issue GL.

use fluxel_rhi_core::api::{
    self, BlendFactor, BlendOperation, CompareFunction, CullMode, FrontFace, MultisampleState,
    PrimitiveState, PrimitiveTopology,
};

use super::api::*;

/// Converts Core primitive state to the fixed GL topology and winding state.
pub(crate) fn lower_primitive(
    value: PrimitiveState,
) -> api::Result<(GlPrimitiveTopology, GlCullMode, GlFrontFace)> {
    let topology = match value.topology {
        PrimitiveTopology::PointList => GlPrimitiveTopology::Points,
        PrimitiveTopology::LineList => GlPrimitiveTopology::Lines,
        PrimitiveTopology::LineStrip => GlPrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => GlPrimitiveTopology::Triangles,
        PrimitiveTopology::TriangleStrip => GlPrimitiveTopology::TriangleStrip,
    };
    let cull = match value.cull_mode {
        CullMode::None => GlCullMode::None,
        CullMode::Front => GlCullMode::Front,
        CullMode::Back => GlCullMode::Back,
    };
    // Vertex position lowering reflects Y, which reverses triangle winding.
    let front = match value.front_face {
        FrontFace::Ccw => GlFrontFace::Clockwise,
        FrontFace::Cw => GlFrontFace::CounterClockwise,
    };
    Ok((topology, cull, front))
}

#[cfg(test)]
#[path = "../tests/backend/translate_raster__tests.rs"]
mod tests;

/// Converts Core blend factors without approximating dual-source factors.
pub(crate) fn lower_blend_component(
    value: api::BlendComponent,
    alpha: bool,
) -> api::Result<GlBlendComponent> {
    Ok(GlBlendComponent {
        src_factor: lower_blend_factor(value.src_factor, alpha)?,
        dst_factor: lower_blend_factor(value.dst_factor, alpha)?,
        operation: match value.operation {
            BlendOperation::Add => GlBlendOperation::Add,
            BlendOperation::Subtract => GlBlendOperation::Subtract,
            BlendOperation::ReverseSubtract => GlBlendOperation::ReverseSubtract,
            BlendOperation::Min => GlBlendOperation::Min,
            BlendOperation::Max => GlBlendOperation::Max,
        },
    })
}

pub(crate) fn lower_compare(value: CompareFunction) -> GlDepthCompareFunction {
    match value {
        CompareFunction::Never => GlDepthCompareFunction::Never,
        CompareFunction::Less => GlDepthCompareFunction::Less,
        CompareFunction::Equal => GlDepthCompareFunction::Equal,
        CompareFunction::LessEqual => GlDepthCompareFunction::LessEqual,
        CompareFunction::Greater => GlDepthCompareFunction::Greater,
        CompareFunction::NotEqual => GlDepthCompareFunction::NotEqual,
        CompareFunction::GreaterEqual => GlDepthCompareFunction::GreaterEqual,
        CompareFunction::Always => GlDepthCompareFunction::Always,
    }
}

/// Preserves all Core coverage bits and normalizes inactive samples.
pub(crate) fn lower_multisample(value: MultisampleState) -> api::Result<GlMultisampleState> {
    // Bits beyond the active sample count have no observable effect.
    let active = if value.count >= 64 {
        u64::MAX
    } else {
        (1u64 << value.count) - 1
    };
    let mask = value.mask | !active;
    Ok(GlMultisampleState {
        sample_count: value.count,
        alpha_to_coverage_enabled: value.alpha_to_coverage_enabled,
        sample_mask: mask,
    })
}

fn lower_blend_factor(value: BlendFactor, alpha: bool) -> api::Result<GlBlendFactor> {
    Ok(match value {
        BlendFactor::Zero => GlBlendFactor::Zero,
        BlendFactor::One => GlBlendFactor::One,
        BlendFactor::SrcColor => GlBlendFactor::Src,
        BlendFactor::OneMinusSrcColor => GlBlendFactor::OneMinusSrc,
        BlendFactor::SrcAlpha => GlBlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => GlBlendFactor::OneMinusSrcAlpha,
        BlendFactor::DstColor => GlBlendFactor::Dst,
        BlendFactor::OneMinusDstColor => GlBlendFactor::OneMinusDst,
        BlendFactor::DstAlpha => GlBlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => GlBlendFactor::OneMinusDstAlpha,
        BlendFactor::Constant => GlBlendFactor::Constant,
        BlendFactor::OneMinusConstant => GlBlendFactor::OneMinusConstant,
        BlendFactor::SrcAlphaSaturated => GlBlendFactor::SrcAlphaSaturated,
        BlendFactor::Src1 => {
            if alpha {
                GlBlendFactor::Src1Alpha
            } else {
                GlBlendFactor::Src1
            }
        }
        BlendFactor::OneMinusSrc1 => {
            if alpha {
                GlBlendFactor::OneMinusSrc1Alpha
            } else {
                GlBlendFactor::OneMinusSrc1
            }
        }
    })
}
