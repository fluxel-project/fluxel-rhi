//! Shared native graphics fixed-state mapping and device feature checks.
use super::super::VulkanDevice;
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};

pub(crate) fn validate_attachment_format(
    device: &VulkanDevice,
    format: vk::Format,
    depth: bool,
    blend: bool,
) -> Result<()> {
    // SAFETY: the physical device and instance remain live through pipeline creation.
    let properties = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_format_properties(device.0.physical, format)
    };
    let required = if depth {
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
    } else {
        vk::FormatFeatureFlags::COLOR_ATTACHMENT
    };
    if !properties.optimal_tiling_features.contains(required)
        || (blend
            && !properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND))
    {
        return Err(Error::Unsupported("Vulkan device attachment format"));
    }
    Ok(())
}
pub(crate) fn polygon_mode(mode: api::PolygonMode) -> vk::PolygonMode {
    match mode {
        api::PolygonMode::Fill => vk::PolygonMode::FILL,
        api::PolygonMode::Line => vk::PolygonMode::LINE,
        api::PolygonMode::Point => vk::PolygonMode::POINT,
    }
}
pub(crate) fn validate_primitive(
    device: &VulkanDevice,
    primitive: &api::PrimitiveState,
    depth: Option<&api::DepthStencilState>,
) -> Result<()> {
    let f = &device.0.features;
    if !matches!(primitive.polygon_mode, api::PolygonMode::Fill)
        && f.fill_mode_non_solid != vk::TRUE
    {
        return Err(Error::Unsupported("Vulkan fillModeNonSolid"));
    }
    if primitive.unclipped_depth && f.depth_clamp != vk::TRUE {
        return Err(Error::Unsupported("Vulkan depthClamp"));
    }
    if depth.is_some_and(|d| d.depth_bias_clamp != 0.0) && f.depth_bias_clamp != vk::TRUE {
        return Err(Error::Unsupported("Vulkan depthBiasClamp"));
    }
    if primitive.conservative && !device.0.conservative_rasterization {
        return Err(Error::Unsupported("Vulkan conservative rasterization"));
    }
    Ok(())
}
pub(crate) fn validate_blends(
    device: &VulkanDevice,
    targets: &[Option<api::ColorTargetState>],
) -> Result<()> {
    // SAFETY: the physical device belongs to this live instance.
    let limit = unsafe {
        device
            .0
            .instance
            .instance
            .get_physical_device_properties(device.0.physical)
    }
    .limits
    .max_color_attachments;
    if targets.len() > limit as usize {
        return Err(Error::Unsupported("Vulkan maxColorAttachments"));
    }
    let features = &device.0.features;
    let state = |target: &Option<api::ColorTargetState>| {
        target.as_ref().map(|t| {
            let component = |c: api::BlendComponent| {
                (c.src_factor as u8, c.dst_factor as u8, c.operation as u8)
            };
            (
                t.write_mask,
                t.blend.map(|b| (component(b.color), component(b.alpha))),
            )
        })
    };
    if features.independent_blend != vk::TRUE
        && targets
            .windows(2)
            .any(|pair| state(&pair[0]) != state(&pair[1]))
    {
        return Err(Error::Unsupported("Vulkan independentBlend"));
    }
    if features.dual_src_blend != vk::TRUE
        && targets.iter().flatten().filter_map(|t| t.blend).any(|b| {
            [
                b.color.src_factor,
                b.color.dst_factor,
                b.alpha.src_factor,
                b.alpha.dst_factor,
            ]
            .iter()
            .any(|f| matches!(f, api::BlendFactor::Src1 | api::BlendFactor::OneMinusSrc1))
        })
    {
        return Err(Error::Unsupported("Vulkan dualSrcBlend"));
    }
    Ok(())
}
pub(crate) fn blend_factor(value: api::BlendFactor, alpha: bool) -> Result<vk::BlendFactor> {
    Ok(match value {
        api::BlendFactor::Zero => vk::BlendFactor::ZERO,
        api::BlendFactor::One => vk::BlendFactor::ONE,
        api::BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        api::BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        api::BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        api::BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        api::BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        api::BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        api::BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        api::BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        api::BlendFactor::Constant => {
            if alpha {
                vk::BlendFactor::CONSTANT_ALPHA
            } else {
                vk::BlendFactor::CONSTANT_COLOR
            }
        }
        api::BlendFactor::OneMinusConstant => {
            if alpha {
                vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR
            }
        }
        api::BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        api::BlendFactor::Src1 => {
            if alpha {
                vk::BlendFactor::SRC1_ALPHA
            } else {
                vk::BlendFactor::SRC1_COLOR
            }
        }
        api::BlendFactor::OneMinusSrc1 => {
            if alpha {
                vk::BlendFactor::ONE_MINUS_SRC1_ALPHA
            } else {
                vk::BlendFactor::ONE_MINUS_SRC1_COLOR
            }
        }
    })
}

pub(crate) fn blend_op(value: api::BlendOperation) -> Result<vk::BlendOp> {
    Ok(match value {
        api::BlendOperation::Add => vk::BlendOp::ADD,
        api::BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        api::BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        api::BlendOperation::Min => vk::BlendOp::MIN,
        api::BlendOperation::Max => vk::BlendOp::MAX,
    })
}
pub(crate) fn compare(v: api::CompareFunction) -> vk::CompareOp {
    match v {
        api::CompareFunction::Never => vk::CompareOp::NEVER,
        api::CompareFunction::Less => vk::CompareOp::LESS,
        api::CompareFunction::Equal => vk::CompareOp::EQUAL,
        api::CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        api::CompareFunction::Greater => vk::CompareOp::GREATER,
        api::CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        api::CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        api::CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}
pub(crate) fn stencil_face(
    face: api::StencilFaceState,
    read: u32,
    write: u32,
) -> vk::StencilOpState {
    vk::StencilOpState::default()
        .fail_op(stencil_op(face.fail))
        .pass_op(stencil_op(face.pass))
        .depth_fail_op(stencil_op(face.depth_fail))
        .compare_op(compare(face.compare))
        .compare_mask(read)
        .write_mask(write)
}
fn stencil_op(op: api::StencilOperation) -> vk::StencilOp {
    match op {
        api::StencilOperation::Keep => vk::StencilOp::KEEP,
        api::StencilOperation::Zero => vk::StencilOp::ZERO,
        api::StencilOperation::Replace => vk::StencilOp::REPLACE,
        api::StencilOperation::Invert => vk::StencilOp::INVERT,
        api::StencilOperation::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        api::StencilOperation::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        api::StencilOperation::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        api::StencilOperation::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}
pub(crate) fn stencil_disabled(state: &api::DepthStencilState) -> bool {
    matches!(state.stencil_front.compare, api::CompareFunction::Always)
        && matches!(state.stencil_back.compare, api::CompareFunction::Always)
        && [state.stencil_front, state.stencil_back]
            .iter()
            .all(|face| {
                matches!(face.fail, api::StencilOperation::Keep)
                    && matches!(face.depth_fail, api::StencilOperation::Keep)
                    && matches!(face.pass, api::StencilOperation::Keep)
            })
}

pub(crate) fn stencil_writes(d: &api::DepthStencilState) -> bool {
    d.stencil_write_mask != 0
        && [d.stencil_front, d.stencil_back].iter().any(|f| {
            [f.fail, f.depth_fail, f.pass]
                .iter()
                .any(|op| !matches!(op, api::StencilOperation::Keep))
        })
}
