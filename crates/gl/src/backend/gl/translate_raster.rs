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
    if !matches!(value.polygon_mode, api::PolygonMode::Fill)
        || value.unclipped_depth
        || value.conservative
    {
        return Err(api::Error::Unsupported("GL primitive mode"));
    }
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
    // Front-facing tests and culling use GL's same winding selector. Viewport
    // origin conversion adjusts the rectangle, not this declaration.
    let front = match value.front_face {
        FrontFace::Ccw => GlFrontFace::CounterClockwise,
        FrontFace::Cw => GlFrontFace::Clockwise,
    };
    Ok((topology, cull, front))
}

/// Converts Core blend factors without approximating dual-source factors.
pub(crate) fn lower_blend_component(value: api::BlendComponent) -> api::Result<GlBlendComponent> {
    Ok(GlBlendComponent {
        src_factor: lower_blend_factor(value.src_factor)?,
        dst_factor: lower_blend_factor(value.dst_factor)?,
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

/// Lowers the portable sample mask after rejecting high bits GL cannot carry.
pub(crate) fn lower_multisample(value: MultisampleState) -> api::Result<GlMultisampleState> {
    if value.mask >> 32 != 0 {
        return Err(api::Error::Unsupported("GL sample mask wider than 32 bits"));
    }
    Ok(GlMultisampleState {
        sample_count: value.count,
        alpha_to_coverage_enabled: value.alpha_to_coverage_enabled,
        sample_mask: value.mask as u32,
    })
}

fn lower_blend_factor(value: BlendFactor) -> api::Result<GlBlendFactor> {
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
        BlendFactor::Src1 | BlendFactor::OneMinusSrc1 => {
            return Err(api::Error::Unsupported("GL dual-source blending"));
        }
    })
}
