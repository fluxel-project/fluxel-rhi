//! Pure fixed-raster enum lowering to the shared OpenGL token values.
//!
//! Native GL and WebGL issue calls through different drivers, but these closed
//! Core-to-GL mappings have the same numeric values on both paths.

use super::{
    GlAddressMode, GlBlendFactor, GlBlendOperation, GlCompareFunction, GlDepthCompareFunction,
    GlFilterMode, GlFormat, GlMipmapFilterMode, GlPrimitiveTopology, GlStencilOperation,
};

pub(crate) const fn raster_topology_mode(topology: GlPrimitiveTopology) -> u32 {
    match topology {
        GlPrimitiveTopology::Points => 0x0000,
        GlPrimitiveTopology::Lines => 0x0001,
        GlPrimitiveTopology::LineStrip => 0x0003,
        GlPrimitiveTopology::Triangles => 0x0004,
        GlPrimitiveTopology::TriangleStrip => 0x0005,
    }
}

pub(crate) const fn raster_compare_function(compare: GlDepthCompareFunction) -> u32 {
    match compare {
        GlDepthCompareFunction::Never => 0x0200,
        GlDepthCompareFunction::Less => 0x0201,
        GlDepthCompareFunction::Equal => 0x0202,
        GlDepthCompareFunction::LessEqual => 0x0203,
        GlDepthCompareFunction::Greater => 0x0204,
        GlDepthCompareFunction::NotEqual => 0x0205,
        GlDepthCompareFunction::GreaterEqual => 0x0206,
        GlDepthCompareFunction::Always => 0x0207,
    }
}

pub(crate) const fn raster_stencil_operation(operation: GlStencilOperation) -> u32 {
    match operation {
        GlStencilOperation::Keep => 0x1E00,
        GlStencilOperation::Zero => 0,
        GlStencilOperation::Replace => 0x1E01,
        GlStencilOperation::IncrementClamp => 0x1E02,
        GlStencilOperation::DecrementClamp => 0x1E03,
        GlStencilOperation::Invert => 0x150A,
        GlStencilOperation::IncrementWrap => 0x8507,
        GlStencilOperation::DecrementWrap => 0x8508,
    }
}

pub(crate) const fn raster_blend_factor(factor: GlBlendFactor) -> u32 {
    match factor {
        GlBlendFactor::Zero => 0,
        GlBlendFactor::One => 1,
        GlBlendFactor::Src => 0x0300,
        GlBlendFactor::OneMinusSrc => 0x0301,
        GlBlendFactor::SrcAlpha => 0x0302,
        GlBlendFactor::OneMinusSrcAlpha => 0x0303,
        GlBlendFactor::Dst => 0x0306,
        GlBlendFactor::OneMinusDst => 0x0307,
        GlBlendFactor::DstAlpha => 0x0304,
        GlBlendFactor::OneMinusDstAlpha => 0x0305,
        GlBlendFactor::Constant => 0x8001,
        GlBlendFactor::OneMinusConstant => 0x8002,
        GlBlendFactor::SrcAlphaSaturated => 0x0308,
        GlBlendFactor::Src1 => 0x88F9,
        GlBlendFactor::OneMinusSrc1 => 0x88FA,
        GlBlendFactor::Src1Alpha => 0x8589,
        GlBlendFactor::OneMinusSrc1Alpha => 0x88FB,
    }
}

pub(crate) const fn raster_blend_operation(operation: GlBlendOperation) -> u32 {
    match operation {
        GlBlendOperation::Add => 0x8006,
        GlBlendOperation::Subtract => 0x800A,
        GlBlendOperation::ReverseSubtract => 0x800B,
        GlBlendOperation::Min => 0x8007,
        GlBlendOperation::Max => 0x8008,
    }
}

pub(crate) const fn sampler_address_mode(mode: GlAddressMode) -> i32 {
    match mode {
        GlAddressMode::ClampToEdge => 0x812F,
        GlAddressMode::Repeat => 0x2901,
        GlAddressMode::MirroredRepeat => 0x8370,
    }
}

pub(crate) const fn sampler_mag_filter(mode: GlFilterMode) -> i32 {
    match mode {
        GlFilterMode::Nearest => 0x2600,
        GlFilterMode::Linear => 0x2601,
    }
}

pub(crate) const fn sampler_min_filter(min: GlFilterMode, mip: GlMipmapFilterMode) -> i32 {
    match (min, mip) {
        (GlFilterMode::Nearest, GlMipmapFilterMode::Nearest) => 0x2700,
        (GlFilterMode::Linear, GlMipmapFilterMode::Nearest) => 0x2701,
        (GlFilterMode::Nearest, GlMipmapFilterMode::Linear) => 0x2702,
        (GlFilterMode::Linear, GlMipmapFilterMode::Linear) => 0x2703,
    }
}

pub(crate) const fn sampler_compare_function(compare: GlCompareFunction) -> i32 {
    match compare {
        GlCompareFunction::Never => 0x0200,
        GlCompareFunction::Less => 0x0201,
        GlCompareFunction::Equal => 0x0202,
        GlCompareFunction::LessEqual => 0x0203,
        GlCompareFunction::Greater => 0x0204,
        GlCompareFunction::NotEqual => 0x0205,
        GlCompareFunction::GreaterEqual => 0x0206,
        GlCompareFunction::Always => 0x0207,
    }
}

pub(crate) const fn depth_attachment_point(format: GlFormat) -> Option<u32> {
    match format {
        GlFormat::Depth16Unorm | GlFormat::Depth24Unorm | GlFormat::Depth32Float => Some(0x8D00),
        GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8 => Some(0x821A),
        GlFormat::Stencil8 => Some(0x8D20),
        _ => None,
    }
}

pub(crate) const fn has_stencil_plane(format: GlFormat) -> bool {
    matches!(
        format,
        GlFormat::Depth24PlusStencil8 | GlFormat::Depth32FloatStencil8 | GlFormat::Stencil8
    )
}

#[cfg(test)]
#[path = "../../tests/backend/api/lowering__tests.rs"]
mod tests;
