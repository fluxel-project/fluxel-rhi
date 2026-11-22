//! Shared DX12 fixed-function raster state lowering.
use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::{
    Foundation::{FALSE, TRUE},
    Graphics::{Direct3D12::*, Dxgi::Common::*},
};

use super::super::{Dx12Device, texture};

const COLOR_SLOTS: usize = D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize;

fn validate_depth(state: Option<api::DepthStencilState>) -> Result<Option<api::DepthStencilState>> {
    let depth = match state {
        None => None,
        Some(state)
            if matches!(
                state.format,
                api::TextureFormat::Depth16Unorm
                    | api::TextureFormat::Depth32Float
                    | api::TextureFormat::Depth24Plus
            ) && stencil_is_disabled(&state) =>
        {
            Some(state)
        }
        Some(state)
            if matches!(
                state.format,
                api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
            ) =>
        {
            Some(state)
        }
        Some(state)
            if state.format == api::TextureFormat::Stencil8
                && !state.depth_write_enabled
                && state.depth_compare == api::CompareFunction::Always
                && state.depth_bias_constant == 0
                && state.depth_bias_slope_scale == 0.0
                && state.depth_bias_clamp == 0.0 =>
        {
            Some(state)
        }
        Some(_) => {
            return Err(Error::Unsupported("DX12 depth/stencil pipeline format"));
        }
    };
    Ok(depth)
}

pub(crate) struct Attachments {
    pub(crate) depth: Option<api::DepthStencilState>,
    pub(crate) dual_source: Option<[u8; 2]>,
    pub(crate) formats: [DXGI_FORMAT; COLOR_SLOTS],
    pub(crate) blends: [D3D12_RENDER_TARGET_BLEND_DESC; COLOR_SLOTS],
}

fn uses_dual_source(blend: api::BlendState) -> bool {
    [
        blend.color.src_factor,
        blend.color.dst_factor,
        blend.alpha.src_factor,
        blend.alpha.dst_factor,
    ]
    .into_iter()
    .any(api::BlendFactor::uses_secondary_source)
}

fn dual_source_outputs(target: &api::ColorTargetState, alpha_to_coverage: bool) -> Option<[u8; 2]> {
    use api::{BlendFactor as Factor, TextureFormat::*};
    let blend = target.blend.filter(|&blend| uses_dual_source(blend))?;
    let channels = match target.format {
        R8Unorm | R8Snorm | R8Uint | R8Sint | R16Unorm | R16Snorm | R16Uint | R16Sint
        | R16Float | R32Uint | R32Sint | R32Float => 0b0001,
        Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint | Rg16Unorm | Rg16Snorm | Rg16Uint | Rg16Sint
        | Rg16Float | Rg32Uint | Rg32Sint | Rg32Float => 0b0011,
        Rg11b10Ufloat | Rgb9e5Ufloat => 0b0111,
        _ => 0b1111,
    };
    const ALPHA: u8 = 0b1000;
    let writes = target.write_mask & channels;
    let rgb = writes & !ALPHA;
    let mut first = writes;
    let mut second = 0;
    if alpha_to_coverage {
        first |= ALPHA;
    }
    if rgb != 0 {
        for factor in [blend.color.src_factor, blend.color.dst_factor] {
            match factor {
                Factor::SrcAlpha | Factor::OneMinusSrcAlpha | Factor::SrcAlphaSaturated => {
                    first |= ALPHA
                }
                Factor::Src1 | Factor::OneMinusSrc1 => second |= rgb,
                Factor::Src1Alpha | Factor::OneMinusSrc1Alpha => second |= ALPHA,
                _ => {}
            }
        }
    }
    if writes & ALPHA != 0
        && [blend.alpha.src_factor, blend.alpha.dst_factor]
            .into_iter()
            .any(Factor::uses_secondary_source)
    {
        second |= ALPHA;
    }
    Some([first, second])
}

pub(crate) fn lower_attachments(
    device: &Dx12Device,
    targets: &[Option<api::ColorTargetState>],
    depth: Option<api::DepthStencilState>,
    multisample: api::MultisampleState,
) -> Result<Attachments> {
    // Keep known device removal distinct from an unsupported attachment format.
    device.require_active()?;
    if !matches!(multisample.count, 1 | 2 | 4 | 8 | 16 | 32) {
        return Err(Error::Unsupported("DX12 pipeline sample count"));
    }
    if multisample.mask > u64::from(u32::MAX) {
        return Err(Error::InvalidRange);
    }
    if multisample.alpha_to_coverage_enabled && multisample.count == 1 {
        return Err(Error::InvalidState);
    }
    let depth = validate_depth(depth)?;
    if depth.is_some_and(|state| {
        !texture::format_has_support(device, state.format, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)
            || !texture::sample_count_supported(device, state.format, multisample.count)
    }) {
        return Err(Error::Unsupported(
            "DX12 depth/stencil attachment sample count or format",
        ));
    }
    let (formats, blends) = lower_targets(device, targets, multisample.count)?;
    Ok(Attachments {
        depth,
        dual_source: targets
            .first()
            .and_then(Option::as_ref)
            .and_then(|target| dual_source_outputs(target, multisample.alpha_to_coverage_enabled)),
        formats,
        blends,
    })
}

fn lower_targets(
    device: &Dx12Device,
    targets: &[Option<api::ColorTargetState>],
    samples: u32,
) -> Result<(
    [DXGI_FORMAT; COLOR_SLOTS],
    [D3D12_RENDER_TARGET_BLEND_DESC; COLOR_SLOTS],
)> {
    if targets.len() > COLOR_SLOTS {
        return Err(Error::InvalidRange);
    }
    let mut formats = [DXGI_FORMAT_UNKNOWN; COLOR_SLOTS];
    let mut blends = [D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: FALSE,
        LogicOpEnable: FALSE,
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: 0,
    }; COLOR_SLOTS];
    for (index, target) in targets.iter().enumerate() {
        let Some(target) = target else { continue };
        formats[index] = color_format(target.format)?;
        let mut required = D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
        if samples > 1 {
            required |= D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET;
        }
        if target.blend.is_some() {
            required |= D3D12_FORMAT_SUPPORT1_BLENDABLE;
        }
        if !texture::format_has_support(device, target.format, required) {
            return Err(Error::Unsupported("DX12 color target format or blending"));
        }
        if !texture::sample_count_supported(device, target.format, samples) {
            return Err(Error::Unsupported("DX12 color attachment sample count"));
        }
        if let Some(blend) = target.blend {
            if uses_dual_source(blend) && (index != 0 || targets.len() != 1) {
                return Err(Error::Unsupported(
                    "DX12 dual source blend with multiple targets",
                ));
            }
        }
        if target.write_mask & !15 != 0 {
            return Err(Error::InvalidRange);
        }
        blends[index] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: if target.blend.is_some() { TRUE } else { FALSE },
            LogicOpEnable: FALSE,
            SrcBlend: target.blend.map_or(Ok(D3D12_BLEND_ONE), |x| {
                blend_factor(x.color.src_factor, false)
            })?,
            DestBlend: target.blend.map_or(Ok(D3D12_BLEND_ZERO), |x| {
                blend_factor(x.color.dst_factor, false)
            })?,
            BlendOp: target
                .blend
                .map_or(D3D12_BLEND_OP_ADD, |x| blend_op(x.color.operation)),
            SrcBlendAlpha: target.blend.map_or(Ok(D3D12_BLEND_ONE), |x| {
                blend_factor(x.alpha.src_factor, true)
            })?,
            DestBlendAlpha: target.blend.map_or(Ok(D3D12_BLEND_ZERO), |x| {
                blend_factor(x.alpha.dst_factor, true)
            })?,
            BlendOpAlpha: target
                .blend
                .map_or(D3D12_BLEND_OP_ADD, |x| blend_op(x.alpha.operation)),
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: target.write_mask,
        };
    }
    Ok((formats, blends))
}

pub(crate) fn lower_raster(
    primitive: api::PrimitiveState,
    multisample: api::MultisampleState,
    depth: Option<api::DepthStencilState>,
) -> Result<D3D12_RASTERIZER_DESC> {
    if primitive.conservative
        && (!matches!(primitive.polygon_mode, api::PolygonMode::Fill)
            || !matches!(
                primitive.topology,
                api::PrimitiveTopology::TriangleList | api::PrimitiveTopology::TriangleStrip
            ))
    {
        return Err(Error::Unsupported(
            "DX12 conservative rasterization requires filled triangles",
        ));
    }
    let raster = D3D12_RASTERIZER_DESC {
        FillMode: match primitive.polygon_mode {
            api::PolygonMode::Fill => D3D12_FILL_MODE_SOLID,
            api::PolygonMode::Line => D3D12_FILL_MODE_WIREFRAME,
            _ => return Err(Error::Unsupported("DX12 polygon mode")),
        },
        CullMode: match primitive.cull_mode {
            api::CullMode::None => D3D12_CULL_MODE_NONE,
            api::CullMode::Front => D3D12_CULL_MODE_FRONT,
            api::CullMode::Back => D3D12_CULL_MODE_BACK,
        },
        FrontCounterClockwise: if matches!(primitive.front_face, api::FrontFace::Ccw) {
            TRUE
        } else {
            FALSE
        },
        DepthBias: depth.map_or(0, |d| d.depth_bias_constant),
        DepthBiasClamp: depth.map_or(0., |d| d.depth_bias_clamp),
        SlopeScaledDepthBias: depth.map_or(0., |d| d.depth_bias_slope_scale),
        DepthClipEnable: if primitive.unclipped_depth {
            FALSE
        } else {
            TRUE
        },
        MultisampleEnable: if multisample.count > 1 { TRUE } else { FALSE },
        AntialiasedLineEnable: FALSE,
        ForcedSampleCount: 0,
        ConservativeRaster: if primitive.conservative {
            D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
        } else {
            D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
        },
    };
    Ok(raster)
}

pub(crate) fn lower_depth(depth: Option<api::DepthStencilState>) -> D3D12_DEPTH_STENCIL_DESC {
    let default_stencil = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let depth_state = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: if depth.is_some_and(|d| d.format != api::TextureFormat::Stencil8) {
            TRUE
        } else {
            FALSE
        },
        DepthWriteMask: if depth.is_some_and(|d| d.depth_write_enabled) {
            D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: depth.map_or(D3D12_COMPARISON_FUNC_ALWAYS, |d| {
            comparison(d.depth_compare)
        }),
        StencilEnable: if depth.is_some_and(|d| {
            matches!(
                d.format,
                api::TextureFormat::Depth24PlusStencil8
                    | api::TextureFormat::Depth32FloatStencil8
                    | api::TextureFormat::Stencil8
            )
        }) {
            TRUE
        } else {
            FALSE
        },
        StencilReadMask: depth.map_or(0, |d| d.stencil_read_mask as u8),
        StencilWriteMask: depth.map_or(0, |d| d.stencil_write_mask as u8),
        FrontFace: depth.map_or(default_stencil, |d| stencil_face(d.stencil_front)),
        BackFace: depth.map_or(default_stencil, |d| stencil_face(d.stencil_back)),
    };
    depth_state
}

pub(crate) fn depth_format(depth: Option<api::DepthStencilState>) -> DXGI_FORMAT {
    depth.map_or(DXGI_FORMAT_UNKNOWN, |d| texture::dxgi_format(d.format))
}

pub(crate) fn per_face_stencil(
    depth: D3D12_DEPTH_STENCIL_DESC,
    masks: api::StencilMasks,
) -> D3D12_DEPTH_STENCIL_DESC2 {
    let face = |op: D3D12_DEPTH_STENCILOP_DESC, read, write| D3D12_DEPTH_STENCILOP_DESC1 {
        StencilFailOp: op.StencilFailOp,
        StencilDepthFailOp: op.StencilDepthFailOp,
        StencilPassOp: op.StencilPassOp,
        StencilFunc: op.StencilFunc,
        StencilReadMask: read,
        StencilWriteMask: write,
    };
    D3D12_DEPTH_STENCIL_DESC2 {
        DepthEnable: depth.DepthEnable,
        DepthWriteMask: depth.DepthWriteMask,
        DepthFunc: depth.DepthFunc,
        StencilEnable: depth.StencilEnable,
        FrontFace: face(depth.FrontFace, masks.front_read, masks.front_write),
        BackFace: face(depth.BackFace, masks.back_read, masks.back_write),
        DepthBoundsTestEnable: FALSE,
    }
}

pub(crate) fn stencil_writes(
    depth: Option<api::DepthStencilState>,
    masks: Option<api::StencilMasks>,
) -> bool {
    depth.is_some_and(|d| {
        let write_masks = masks.map_or([d.stencil_write_mask as u8; 2], |masks| {
            [masks.front_write, masks.back_write]
        });
        [d.stencil_front, d.stencil_back]
            .into_iter()
            .zip(write_masks)
            .any(|(f, mask)| {
                mask != 0
                    && (!matches!(f.fail, api::StencilOperation::Keep)
                        || !matches!(f.depth_fail, api::StencilOperation::Keep)
                        || !matches!(f.pass, api::StencilOperation::Keep))
            })
    })
}

fn blend_factor(value: api::BlendFactor, alpha: bool) -> Result<D3D12_BLEND> {
    Ok(match value {
        api::BlendFactor::Zero => D3D12_BLEND_ZERO,
        api::BlendFactor::One => D3D12_BLEND_ONE,
        api::BlendFactor::SrcColor => {
            if alpha {
                D3D12_BLEND_SRC_ALPHA
            } else {
                D3D12_BLEND_SRC_COLOR
            }
        }
        api::BlendFactor::OneMinusSrcColor => {
            if alpha {
                D3D12_BLEND_INV_SRC_ALPHA
            } else {
                D3D12_BLEND_INV_SRC_COLOR
            }
        }
        api::BlendFactor::DstColor => {
            if alpha {
                D3D12_BLEND_DEST_ALPHA
            } else {
                D3D12_BLEND_DEST_COLOR
            }
        }
        api::BlendFactor::OneMinusDstColor => {
            if alpha {
                D3D12_BLEND_INV_DEST_ALPHA
            } else {
                D3D12_BLEND_INV_DEST_COLOR
            }
        }
        api::BlendFactor::SrcAlpha => D3D12_BLEND_SRC_ALPHA,
        api::BlendFactor::OneMinusSrcAlpha => D3D12_BLEND_INV_SRC_ALPHA,
        api::BlendFactor::DstAlpha => D3D12_BLEND_DEST_ALPHA,
        api::BlendFactor::OneMinusDstAlpha => D3D12_BLEND_INV_DEST_ALPHA,
        api::BlendFactor::Constant => D3D12_BLEND_BLEND_FACTOR,
        api::BlendFactor::OneMinusConstant => D3D12_BLEND_INV_BLEND_FACTOR,
        api::BlendFactor::SrcAlphaSaturated => {
            if alpha {
                D3D12_BLEND_ONE
            } else {
                D3D12_BLEND_SRC_ALPHA_SAT
            }
        }
        api::BlendFactor::Src1 => {
            if alpha {
                D3D12_BLEND_SRC1_ALPHA
            } else {
                D3D12_BLEND_SRC1_COLOR
            }
        }
        api::BlendFactor::OneMinusSrc1 => {
            if alpha {
                D3D12_BLEND_INV_SRC1_ALPHA
            } else {
                D3D12_BLEND_INV_SRC1_COLOR
            }
        }
        api::BlendFactor::Src1Alpha => D3D12_BLEND_SRC1_ALPHA,
        api::BlendFactor::OneMinusSrc1Alpha => D3D12_BLEND_INV_SRC1_ALPHA,
    })
}

fn blend_op(value: api::BlendOperation) -> D3D12_BLEND_OP {
    match value {
        api::BlendOperation::Add => D3D12_BLEND_OP_ADD,
        api::BlendOperation::Subtract => D3D12_BLEND_OP_SUBTRACT,
        api::BlendOperation::ReverseSubtract => D3D12_BLEND_OP_REV_SUBTRACT,
        api::BlendOperation::Min => D3D12_BLEND_OP_MIN,
        api::BlendOperation::Max => D3D12_BLEND_OP_MAX,
    }
}
pub(crate) fn stencil_is_disabled(d: &api::DepthStencilState) -> bool {
    let inactive = |s: api::StencilFaceState| {
        matches!(s.compare, api::CompareFunction::Always)
            && matches!(s.fail, api::StencilOperation::Keep)
            && matches!(s.depth_fail, api::StencilOperation::Keep)
            && matches!(s.pass, api::StencilOperation::Keep)
    };
    inactive(d.stencil_front) && inactive(d.stencil_back)
}
fn stencil_face(face: api::StencilFaceState) -> D3D12_DEPTH_STENCILOP_DESC {
    let op = |value| match value {
        api::StencilOperation::Keep => D3D12_STENCIL_OP_KEEP,
        api::StencilOperation::Zero => D3D12_STENCIL_OP_ZERO,
        api::StencilOperation::Replace => D3D12_STENCIL_OP_REPLACE,
        api::StencilOperation::Invert => D3D12_STENCIL_OP_INVERT,
        api::StencilOperation::IncrementClamp => D3D12_STENCIL_OP_INCR_SAT,
        api::StencilOperation::DecrementClamp => D3D12_STENCIL_OP_DECR_SAT,
        api::StencilOperation::IncrementWrap => D3D12_STENCIL_OP_INCR,
        api::StencilOperation::DecrementWrap => D3D12_STENCIL_OP_DECR,
    };
    D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: op(face.fail),
        StencilDepthFailOp: op(face.depth_fail),
        StencilPassOp: op(face.pass),
        StencilFunc: comparison(face.compare),
    }
}
pub(crate) fn comparison(v: api::CompareFunction) -> D3D12_COMPARISON_FUNC {
    match v {
        api::CompareFunction::Never => D3D12_COMPARISON_FUNC_NEVER,
        api::CompareFunction::Less => D3D12_COMPARISON_FUNC_LESS,
        api::CompareFunction::Equal => D3D12_COMPARISON_FUNC_EQUAL,
        api::CompareFunction::LessEqual => D3D12_COMPARISON_FUNC_LESS_EQUAL,
        api::CompareFunction::Greater => D3D12_COMPARISON_FUNC_GREATER,
        api::CompareFunction::NotEqual => D3D12_COMPARISON_FUNC_NOT_EQUAL,
        api::CompareFunction::GreaterEqual => D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        api::CompareFunction::Always => D3D12_COMPARISON_FUNC_ALWAYS,
    }
}
fn color_format(v: api::TextureFormat) -> Result<DXGI_FORMAT> {
    if super::super::texture::color_texel_bytes(v).is_some() {
        Ok(super::super::texture::dxgi_format(v))
    } else {
        Err(Error::Unsupported("DX12 render-target format"))
    }
}
