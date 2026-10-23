//! Closed fixed-raster enum lowering for native GL calls.
//!
//! These helpers have no provider state and cannot issue GL. Keeping them
//! separate from pass validation makes the executable state transition in the
//! parent module easier to audit.

use crate::backend::gl::api::{
    GlBlendFactor, GlBlendOperation, GlDepthCompareFunction, GlPrimitiveTopology,
    GlStencilOperation,
};

pub(crate) const fn topology_mode(topology: GlPrimitiveTopology) -> u32 {
    match topology {
        GlPrimitiveTopology::Points => glow::POINTS,
        GlPrimitiveTopology::Lines => glow::LINES,
        GlPrimitiveTopology::LineStrip => glow::LINE_STRIP,
        GlPrimitiveTopology::Triangles => glow::TRIANGLES,
        GlPrimitiveTopology::TriangleStrip => glow::TRIANGLE_STRIP,
    }
}

pub(super) const fn compare_function(compare: GlDepthCompareFunction) -> u32 {
    match compare {
        GlDepthCompareFunction::Never => glow::NEVER,
        GlDepthCompareFunction::Less => glow::LESS,
        GlDepthCompareFunction::Equal => glow::EQUAL,
        GlDepthCompareFunction::LessEqual => glow::LEQUAL,
        GlDepthCompareFunction::Greater => glow::GREATER,
        GlDepthCompareFunction::NotEqual => glow::NOTEQUAL,
        GlDepthCompareFunction::GreaterEqual => glow::GEQUAL,
        GlDepthCompareFunction::Always => glow::ALWAYS,
    }
}

pub(super) const fn stencil_operation(operation: GlStencilOperation) -> u32 {
    match operation {
        GlStencilOperation::Keep => glow::KEEP,
        GlStencilOperation::Zero => glow::ZERO,
        GlStencilOperation::Replace => glow::REPLACE,
        GlStencilOperation::IncrementClamp => glow::INCR,
        GlStencilOperation::DecrementClamp => glow::DECR,
        GlStencilOperation::Invert => glow::INVERT,
        GlStencilOperation::IncrementWrap => glow::INCR_WRAP,
        GlStencilOperation::DecrementWrap => glow::DECR_WRAP,
    }
}

pub(super) const fn blend_factor(factor: GlBlendFactor) -> u32 {
    match factor {
        GlBlendFactor::Zero => glow::ZERO,
        GlBlendFactor::One => glow::ONE,
        GlBlendFactor::Src => glow::SRC_COLOR,
        GlBlendFactor::OneMinusSrc => glow::ONE_MINUS_SRC_COLOR,
        GlBlendFactor::SrcAlpha => glow::SRC_ALPHA,
        GlBlendFactor::OneMinusSrcAlpha => glow::ONE_MINUS_SRC_ALPHA,
        GlBlendFactor::Dst => glow::DST_COLOR,
        GlBlendFactor::OneMinusDst => glow::ONE_MINUS_DST_COLOR,
        GlBlendFactor::DstAlpha => glow::DST_ALPHA,
        GlBlendFactor::OneMinusDstAlpha => glow::ONE_MINUS_DST_ALPHA,
        GlBlendFactor::Constant => glow::CONSTANT_COLOR,
        GlBlendFactor::OneMinusConstant => glow::ONE_MINUS_CONSTANT_COLOR,
        GlBlendFactor::SrcAlphaSaturated => glow::SRC_ALPHA_SATURATE,
        GlBlendFactor::Src1 => 0x88F9,
        GlBlendFactor::OneMinusSrc1 => 0x88FA,
        GlBlendFactor::Src1Alpha => 0x8589,
        GlBlendFactor::OneMinusSrc1Alpha => 0x88FB,
    }
}

pub(super) const fn blend_operation(operation: GlBlendOperation) -> u32 {
    match operation {
        GlBlendOperation::Add => glow::FUNC_ADD,
        GlBlendOperation::Subtract => glow::FUNC_SUBTRACT,
        GlBlendOperation::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
        GlBlendOperation::Min => glow::MIN,
        GlBlendOperation::Max => glow::MAX,
    }
}
