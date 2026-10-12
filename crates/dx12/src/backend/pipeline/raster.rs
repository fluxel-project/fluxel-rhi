//! Fixed-function raster PSO lowering and public pipeline validation.
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    future::{Ready, ready},
    mem::ManuallyDrop,
    sync::Arc,
};
use windows::{
    Win32::{
        Foundation::{FALSE, TRUE},
        Graphics::{Direct3D12::*, Dxgi::Common::*},
    },
    core::PCSTR,
};

use super::super::{Dx12, Dx12Device, Dx12RasterPipeline};
use super::{PipelineLayoutState, create_view_instanced_pso};

pub(crate) struct RasterPipelineState {
    pub(in crate::backend) root: ID3D12RootSignature,
    pub(in crate::backend) layout: Arc<PipelineLayoutState>,
    pub(in crate::backend) state: ID3D12PipelineState,
    pub(in crate::backend) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(in crate::backend) vertex_strides: Vec<u32>,
    pub(in crate::backend) color_format: api::TextureFormat,
    pub(in crate::backend) samples: u32,
    pub(in crate::backend) multiview: Option<u32>,
}
impl api::PipelineBackend for Dx12 {
    type RasterPipelineFuture<'a> = Ready<Result<Dx12RasterPipeline>>;
    fn create_raster_pipeline<'a>(
        device: &'a Dx12Device,
        desc: &'a api::RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        ready(create_raster(device, desc))
    }
}

fn create_raster(
    device: &Dx12Device,
    desc: &api::RasterPipelineDesc<Dx12>,
) -> Result<Dx12RasterPipeline> {
    if desc.fragment.is_none() {
        return Err(Error::Unsupported(
            "DX12 raster pipeline without fragment DXIL",
        ));
    }
    if !matches!(
        desc.primitive.topology,
        api::PrimitiveTopology::TriangleList
    ) || desc.primitive.strip_index_format.is_some()
    {
        return Err(Error::Unsupported("DX12 primitive topology"));
    }
    if !desc.vertex.constants.is_empty()
        || desc
            .fragment
            .as_ref()
            .is_some_and(|stage| !stage.constants.is_empty())
    {
        return Err(Error::Unsupported("DX12 shader specialization constants"));
    }
    // D32_FLOAT has no stencil plane.  Do not silently discard requested stencil
    // behavior: the fixed-function PSO lowering below enables only depth.
    let depth = match desc.depth_stencil {
        None => None,
        Some(state)
            if state.format == api::TextureFormat::Depth32Float && stencil_is_disabled(&state) =>
        {
            Some(state)
        }
        Some(_) => {
            return Err(Error::Unsupported(
                "DX12 depth state requires Depth32Float without stencil operations",
            ));
        }
    };
    if !matches!(desc.multisample.count, 1 | 2 | 4 | 8) {
        return Err(Error::Unsupported(
            "DX12 multisample count (supported: 1, 2, 4, 8)",
        ));
    }
    if desc.multisample.alpha_to_coverage_enabled && desc.multisample.count == 1 {
        return Err(Error::InvalidState);
    }
    if desc.multisample.mask > u64::from(u32::MAX) {
        return Err(Error::InvalidRange);
    }
    if desc.cache.is_some() {
        return Err(Error::Unsupported("DX12 pipeline cache"));
    }
    if desc.multiview.is_some_and(|views| views != 2) {
        return Err(Error::Unsupported("DX12 multiview view count"));
    }
    if desc.color_targets.len() != 1 || desc.color_targets[0].is_none() {
        return Err(Error::Unsupported(
            "DX12 pipeline requires one color target",
        ));
    }
    let vs = stage(&desc.vertex, api::ShaderStage::Vertex)?;
    let ps = stage(desc.fragment.as_ref().unwrap(), api::ShaderStage::Fragment)?;
    // The command lowering needs the public strides to construct D3D12 vertex
    // buffer views.  Keep this validation adjacent to input-layout lowering so
    // a pipeline can never be created with a stride the command path cannot
    // represent.
    let vertex_strides: Result<Vec<u32>> = desc
        .vertex_buffers
        .iter()
        .map(|buffer| u32::try_from(buffer.stride).map_err(|_| Error::InvalidRange))
        .collect();
    let vertex_strides = vertex_strides?;
    let names: Vec<Vec<u8>> = desc
        .vertex_buffers
        .iter()
        .flat_map(|x| x.attributes.iter())
        .map(|_| b"LOCATION\0".to_vec())
        .collect();
    let mut n = 0usize;
    let elements: Result<Vec<_>> = desc
        .vertex_buffers
        .iter()
        .enumerate()
        .flat_map(|(slot, b)| b.attributes.iter().map(move |a| (slot, b, a)))
        .map(|(slot, b, a)| {
            let name = &names[n];
            n += 1;
            Ok(D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr()),
                SemanticIndex: a.location,
                Format: vertex_format(a.format)?,
                InputSlot: slot as u32,
                AlignedByteOffset: u32::try_from(a.offset).map_err(|_| Error::InvalidRange)?,
                InputSlotClass: match b.step_mode {
                    api::VertexStepMode::Vertex => D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    api::VertexStepMode::Instance => D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                },
                InstanceDataStepRate: match b.step_mode {
                    api::VertexStepMode::Vertex => 0,
                    api::VertexStepMode::Instance => 1,
                },
            })
        })
        .collect();
    let elements = elements?;
    let target = desc.color_targets[0].as_ref().unwrap();
    let format = color_format(target.format)?;
    if target.blend.is_some() || target.write_mask != 15 {
        return Err(Error::Unsupported("DX12 color blend/write mask"));
    }
    let blend = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: FALSE,
        LogicOpEnable: FALSE,
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    let raster = D3D12_RASTERIZER_DESC {
        FillMode: match desc.primitive.polygon_mode {
            api::PolygonMode::Fill => D3D12_FILL_MODE_SOLID,
            _ => return Err(Error::Unsupported("DX12 polygon mode")),
        },
        CullMode: match desc.primitive.cull_mode {
            api::CullMode::None => D3D12_CULL_MODE_NONE,
            api::CullMode::Front => D3D12_CULL_MODE_FRONT,
            api::CullMode::Back => D3D12_CULL_MODE_BACK,
        },
        FrontCounterClockwise: if matches!(desc.primitive.front_face, api::FrontFace::Ccw) {
            TRUE
        } else {
            FALSE
        },
        DepthBias: depth.map_or(0, |d| d.depth_bias_constant),
        DepthBiasClamp: depth.map_or(0., |d| d.depth_bias_clamp),
        SlopeScaledDepthBias: depth.map_or(0., |d| d.depth_bias_slope_scale),
        DepthClipEnable: if desc.primitive.unclipped_depth {
            FALSE
        } else {
            TRUE
        },
        MultisampleEnable: if desc.multisample.count > 1 {
            TRUE
        } else {
            FALSE
        },
        AntialiasedLineEnable: FALSE,
        ForcedSampleCount: 0,
        ConservativeRaster: if desc.primitive.conservative {
            D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
        } else {
            D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
        },
    };
    let stencil = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let layout = unsafe { desc.layout.native_ref() }.0.clone();
    let root = layout.root.clone();
    let depth_state = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: if depth.is_some() { TRUE } else { FALSE },
        DepthWriteMask: if depth.is_some_and(|d| d.depth_write_enabled) {
            D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: depth.map_or(D3D12_COMPARISON_FUNC_ALWAYS, |d| {
            comparison(d.depth_compare)
        }),
        StencilEnable: FALSE,
        StencilReadMask: 0,
        StencilWriteMask: 0,
        FrontFace: stencil,
        BackFace: stencil,
    };
    let top = topology(desc.primitive.topology);
    let view_locations = [
        D3D12_VIEW_INSTANCE_LOCATION {
            ViewportArrayIndex: 0,
            RenderTargetArrayIndex: 0,
        },
        D3D12_VIEW_INSTANCE_LOCATION {
            ViewportArrayIndex: 1,
            RenderTargetArrayIndex: 1,
        },
    ];
    let mut pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: bytecode(vs),
        PS: bytecode(ps),
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: if desc.multisample.alpha_to_coverage_enabled {
                TRUE
            } else {
                FALSE
            },
            IndependentBlendEnable: FALSE,
            RenderTarget: [blend; 8],
        },
        SampleMask: desc.multisample.mask as u32,
        RasterizerState: raster,
        DepthStencilState: depth_state,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: elements.as_ptr(),
            NumElements: elements.len() as u32,
        },
        PrimitiveTopologyType: top,
        NumRenderTargets: 1,
        RTVFormats: [
            format,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
        ],
        DSVFormat: if depth.is_some() {
            DXGI_FORMAT_D32_FLOAT
        } else {
            DXGI_FORMAT_UNKNOWN
        },
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: desc.multisample.count,
            Quality: 0,
        },
        ..Default::default()
    };
    let state = if desc.multiview.is_some() {
        create_view_instanced_pso(&device.0.native, &pso, &root, &view_locations)?
    } else {
        unsafe {
            device
                .0
                .native
                .CreateGraphicsPipelineState::<ID3D12PipelineState>(&pso)
        }
        .map_err(native_error)?
    };
    unsafe { ManuallyDrop::drop(&mut pso.pRootSignature) };
    Ok(Dx12RasterPipeline(Arc::new(RasterPipelineState {
        root,
        layout,
        state,
        topology: top,
        vertex_strides,
        color_format: desc.color_targets[0]
            .as_ref()
            .expect("validated color target")
            .format,
        samples: desc.multisample.count,
        multiview: desc.multiview,
    })))
}
fn stencil_is_disabled(d: &api::DepthStencilState) -> bool {
    let inactive = |s: api::StencilFaceState| {
        matches!(s.compare, api::CompareFunction::Always)
            && matches!(s.fail, api::StencilOperation::Keep)
            && matches!(s.depth_fail, api::StencilOperation::Keep)
            && matches!(s.pass, api::StencilOperation::Keep)
    };
    inactive(d.stencil_front) && inactive(d.stencil_back)
}
fn comparison(v: api::CompareFunction) -> D3D12_COMPARISON_FUNC {
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
fn stage<'a>(s: &'a api::PipelineStage<Dx12>, need: api::ShaderStage) -> Result<&'a [u8]> {
    let n = unsafe { s.module.native_ref() };
    if !n
        .entries
        .iter()
        .any(|e| e.name == s.entry && e.stage == need)
    {
        return Err(Error::Shader(format!(
            "DXIL has no requested {:?} entry {}",
            need, s.entry
        )));
    };
    Ok(&n.bytes)
}
fn bytecode(v: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: v.as_ptr().cast(),
        BytecodeLength: v.len(),
    }
}
fn vertex_format(v: api::VertexFormat) -> Result<DXGI_FORMAT> {
    match v {
        api::VertexFormat::Float32(1) => Ok(DXGI_FORMAT_R32_FLOAT),
        api::VertexFormat::Float32(2) => Ok(DXGI_FORMAT_R32G32_FLOAT),
        api::VertexFormat::Float32(3) => Ok(DXGI_FORMAT_R32G32B32_FLOAT),
        api::VertexFormat::Float32(4) => Ok(DXGI_FORMAT_R32G32B32A32_FLOAT),
        api::VertexFormat::Sint32(1) => Ok(DXGI_FORMAT_R32_SINT),
        api::VertexFormat::Sint32(2) => Ok(DXGI_FORMAT_R32G32_SINT),
        api::VertexFormat::Sint32(3) => Ok(DXGI_FORMAT_R32G32B32_SINT),
        api::VertexFormat::Sint32(4) => Ok(DXGI_FORMAT_R32G32B32A32_SINT),
        _ => Err(Error::Unsupported("DX12 vertex format")),
    }
}
fn color_format(v: api::TextureFormat) -> Result<DXGI_FORMAT> {
    match v {
        api::TextureFormat::Rgba8Unorm => Ok(DXGI_FORMAT_R8G8B8A8_UNORM),
        api::TextureFormat::Bgra8Unorm => Ok(DXGI_FORMAT_B8G8R8A8_UNORM),
        _ => Err(Error::Unsupported("DX12 render-target format")),
    }
}
fn topology(v: api::PrimitiveTopology) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match v {
        api::PrimitiveTopology::PointList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
        api::PrimitiveTopology::LineList | api::PrimitiveTopology::LineStrip => {
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
        }
        _ => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
    }
}

fn native_error(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}
