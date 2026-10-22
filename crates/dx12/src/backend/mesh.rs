//! Direct3D 12 mesh-shader capability and native execution.
//!
//! Hardware support is conditional on mesh tier and shader model; the public
//! capability is enabled only after the full recording path is validated.
mod ops;
use super::{Dx12, Dx12Device, Dx12MeshPipeline, pipeline::PipelineLayoutState};
use fluxel_rhi_core::api::{self, Error, Result};
pub use ops::Mesh;
use std::sync::Arc;
use windows::Win32::{
    Foundation::{FALSE, TRUE},
    Graphics::Dxgi::Common::*,
};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

pub(crate) struct MeshPipelineState {
    pub(crate) layout: Arc<PipelineLayoutState>,
    pub(crate) state: ID3D12PipelineState,
    pub(crate) color_format: api::TextureFormat,
    pub(crate) samples: u32,
    pub(crate) depth_format: Option<api::TextureFormat>,
}

#[repr(C, align(8))]
struct Subobject<T> {
    tag: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    value: T,
}
#[repr(C)]
struct Stream {
    root: Subobject<*mut std::ffi::c_void>,
    task: Subobject<D3D12_SHADER_BYTECODE>,
    mesh: Subobject<D3D12_SHADER_BYTECODE>,
    fragment: Subobject<D3D12_SHADER_BYTECODE>,
    blend: Subobject<D3D12_BLEND_DESC>,
    sample_mask: Subobject<u32>,
    raster: Subobject<D3D12_RASTERIZER_DESC>,
    depth: Subobject<D3D12_DEPTH_STENCIL_DESC>,
    depth_format: Subobject<DXGI_FORMAT>,
    topology: Subobject<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    formats: Subobject<D3D12_RT_FORMAT_ARRAY>,
    samples: Subobject<DXGI_SAMPLE_DESC>,
}
fn sub<T>(tag: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, value: T) -> Subobject<T> {
    Subobject { tag, value }
}

fn shader<'a>(stage: &'a api::PipelineStage<Dx12>, expected: api::ShaderStage) -> Result<&'a [u8]> {
    if !stage.constants.is_empty() {
        return Err(Error::Unsupported("DX12 mesh shader specialization"));
    }
    // SAFETY: Core validates that every shader belongs to the pipeline device.
    let shader = unsafe { stage.module.native_ref() };
    if !shader
        .entries
        .iter()
        .any(|entry| entry.name == stage.entry && entry.stage == expected)
    {
        return Err(Error::InvalidState);
    }
    Ok(&shader.bytes)
}
fn bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}

pub(super) fn create_pipeline(
    device: &Dx12Device,
    desc: &api::MeshPipelineDesc<Dx12>,
) -> Result<Dx12MeshPipeline> {
    if !device.0.mesh_supported {
        return Err(Error::Unsupported("DX12 mesh shader tier 1"));
    }
    if desc.cache.is_some()
        || desc.targets.len() != 1
        || desc.primitive.strip_index_format.is_some()
        || desc.primitive.conservative
        || desc.primitive.unclipped_depth
        || !matches!(
            desc.primitive.topology,
            api::PrimitiveTopology::TriangleList
        )
        || !matches!(desc.primitive.polygon_mode, api::PolygonMode::Fill)
        || !matches!(desc.multisample.count, 1 | 2 | 4 | 8)
        || desc.multisample.mask > u64::from(u32::MAX)
    {
        return Err(Error::Unsupported("DX12 mesh pipeline state"));
    }
    let depth = match desc.depth_stencil {
        None => None,
        Some(state)
            if state.format == api::TextureFormat::Depth32Float
                && super::pipeline::stencil_is_disabled(&state) =>
        {
            Some(state)
        }
        Some(_) => return Err(Error::Unsupported("DX12 mesh depth state")),
    };
    let target = desc.targets[0].as_ref().ok_or(Error::InvalidState)?;
    if target.blend.is_some() || target.write_mask != 15 {
        return Err(Error::Unsupported("DX12 mesh blend state"));
    }
    let format = match target.format {
        api::TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        api::TextureFormat::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        _ => return Err(Error::Unsupported("DX12 mesh color format")),
    };
    let mesh = shader(&desc.mesh, api::ShaderStage::Mesh)?;
    let task = desc
        .task
        .as_ref()
        .map(|s| shader(s, api::ShaderStage::Task))
        .transpose()?;
    let fragment = desc
        .fragment
        .as_ref()
        .map(|s| shader(s, api::ShaderStage::Fragment))
        .transpose()?;
    // SAFETY: the layout was validated by Core and the cloned Arc retains it.
    let layout = unsafe { desc.layout.native_ref() }.0.clone();
    let target_blend = D3D12_RENDER_TARGET_BLEND_DESC {
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
        FillMode: D3D12_FILL_MODE_SOLID,
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
        DepthBiasClamp: depth.map_or(0.0, |d| d.depth_bias_clamp),
        SlopeScaledDepthBias: depth.map_or(0.0, |d| d.depth_bias_slope_scale),
        DepthClipEnable: TRUE,
        MultisampleEnable: if desc.multisample.count > 1 {
            TRUE
        } else {
            FALSE
        },
        AntialiasedLineEnable: FALSE,
        ForcedSampleCount: 0,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    };
    let stencil = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let stream = Stream {
        root: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            layout.root.as_raw(),
        ),
        task: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
            task.map_or(D3D12_SHADER_BYTECODE::default(), bytecode),
        ),
        mesh: sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, bytecode(mesh)),
        fragment: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
            fragment.map_or(D3D12_SHADER_BYTECODE::default(), bytecode),
        ),
        blend: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
            D3D12_BLEND_DESC {
                AlphaToCoverageEnable: if desc.multisample.alpha_to_coverage_enabled {
                    TRUE
                } else {
                    FALSE
                },
                IndependentBlendEnable: FALSE,
                RenderTarget: [target_blend; 8],
            },
        ),
        sample_mask: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            desc.multisample.mask as u32,
        ),
        raster: sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, raster),
        depth: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
            D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: if depth.is_some() { TRUE } else { FALSE },
                DepthWriteMask: if depth.is_some_and(|d| d.depth_write_enabled) {
                    D3D12_DEPTH_WRITE_MASK_ALL
                } else {
                    D3D12_DEPTH_WRITE_MASK_ZERO
                },
                DepthFunc: depth.map_or(D3D12_COMPARISON_FUNC_ALWAYS, |d| {
                    super::pipeline::comparison(d.depth_compare)
                }),
                StencilEnable: FALSE,
                StencilReadMask: 0,
                StencilWriteMask: 0,
                FrontFace: stencil,
                BackFace: stencil,
            },
        ),
        depth_format: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            if depth.is_some() {
                DXGI_FORMAT_D32_FLOAT
            } else {
                DXGI_FORMAT_UNKNOWN
            },
        ),
        topology: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ),
        formats: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            D3D12_RT_FORMAT_ARRAY {
                RTFormats: [
                    format,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                ],
                NumRenderTargets: 1,
            },
        ),
        samples: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
            DXGI_SAMPLE_DESC {
                Count: desc.multisample.count,
                Quality: 0,
            },
        ),
    };
    let native = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: std::mem::size_of_val(&stream),
        pPipelineStateSubobjectStream: (&stream as *const Stream).cast_mut().cast(),
    };
    let native_device = device
        .0
        .native
        .cast::<ID3D12Device2>()
        .map_err(super::native_error)?;
    // SAFETY: the aligned stream, root signature, and shader bytecode remain
    // live for the duration of this synchronous PSO creation call.
    let state = unsafe { native_device.CreatePipelineState::<ID3D12PipelineState>(&native) }
        .map_err(super::native_error)?;
    Ok(Dx12MeshPipeline(Arc::new(MeshPipelineState {
        layout,
        state,
        color_format: target.format,
        samples: desc.multisample.count,
        depth_format: depth.map(|d| d.format),
    })))
}

pub(super) fn probe(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    let mut shader = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_5,
    };
    // SAFETY: both feature structures match their query selectors and are
    // valid mutable storage for these synchronous feature queries.
    let tier = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS7,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;
    let shader_model = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&mut shader as *mut D3D12_FEATURE_DATA_SHADER_MODEL).cast(),
            std::mem::size_of_val(&shader) as u32,
        )
    }
    .is_ok()
        && shader.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_5.0;
    tier && shader_model && device.cast::<ID3D12Device2>().is_ok()
}
