//! Fixed-function raster PSO lowering and public pipeline validation.
//!
//! This module translates the supported Core raster subset into immutable D3D12
//! PSOs. It rejects state without a faithful native representation instead of
//! introducing emulation. Temporary ABI arrays and DXIL slices remain alive
//! through synchronous PSO creation; the completed pipeline then owns its COM
//! objects and layout mapping for later command encoding.
use fluxel_rhi_core::api::{self, Error, Result};
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};
use windows::{
    Win32::{
        Foundation::{FALSE, TRUE},
        Graphics::{Direct3D::*, Direct3D12::*, Dxgi::Common::*},
    },
    core::PCSTR,
};

use super::super::{Dx12, Dx12Device, Dx12RasterPipeline, Dx12ShaderSource};
use super::fixed::{
    depth_format, lower_depth, lower_raster, lower_targets, stencil_writes, validate_depth,
};
use super::{PipelineLayoutState, create_view_instanced_pso};

pub(crate) struct RasterPipelineState {
    pub(in crate::backend) root: ID3D12RootSignature,
    pub(in crate::backend) layout: Arc<PipelineLayoutState>,
    pub(in crate::backend) state: ID3D12PipelineState,
    pub(in crate::backend) topology: D3D_PRIMITIVE_TOPOLOGY,
    pub(in crate::backend) strip_index_format: Option<api::IndexFormat>,
    pub(in crate::backend) raw_storage_bindings: Vec<(u32, u32)>,
    pub(in crate::backend) vertex_index_root: Option<u32>,
    /// At most one native ABI signature for each indexed/non-indexed variant.
    /// Keeping it with this root prevents stale pointer keys and device-wide
    /// retention of every historical pipeline.
    pub(in crate::backend) vertex_indirect_signatures: Mutex<[Option<ID3D12CommandSignature>; 2]>,
    pub(in crate::backend) depth_writes: bool,
    pub(in crate::backend) stencil_writes: bool,
    pub(in crate::backend) vertex_strides: Vec<u32>,
    pub(in crate::backend) color_formats: Vec<Option<api::TextureFormat>>,
    pub(in crate::backend) depth_format: Option<api::TextureFormat>,
    pub(in crate::backend) samples: u32,
    pub(in crate::backend) multiview: Option<u32>,
    pub(in crate::backend) dynamic_depth_bias: bool,
}
impl fluxel_rhi_core::backend::PipelineBackend for Dx12 {
    type RasterPipelineFuture<'a> = super::worker::ColdFuture<Dx12RasterPipeline>;
    fn create_raster_pipeline<'a>(
        device: &'a Dx12Device,
        desc: &'a api::RasterPipelineDesc<Self>,
        dynamic_depth_bias: Option<api::DynamicDepthBias>,
    ) -> Self::RasterPipelineFuture<'a> {
        let device = (*device).clone();
        let desc = (*desc).clone();
        super::worker::submit(move || create_raster(&device, &desc, dynamic_depth_bias))
    }
}

fn create_raster(
    device: &Dx12Device,
    desc: &api::RasterPipelineDesc<Dx12>,
    dynamic_depth_bias: Option<api::DynamicDepthBias>,
) -> Result<Dx12RasterPipeline> {
    if dynamic_depth_bias.is_some() && !device.0.dynamic_depth_bias_supported {
        return Err(Error::Unsupported("DX12 dynamic depth bias"));
    }
    if desc.primitive.strip_index_format.is_some()
        && !matches!(
            desc.primitive.topology,
            api::PrimitiveTopology::LineStrip | api::PrimitiveTopology::TriangleStrip
        )
    {
        return Err(Error::InvalidState);
    }
    let depth = validate_depth(desc.depth_stencil)?;
    if !matches!(desc.multisample.count, 1 | 2 | 4 | 8 | 16 | 32) {
        return Err(Error::Unsupported(
            "DX12 multisample count (supported: 1, 2, 4, 8, 16, 32)",
        ));
    }
    if desc.multisample.alpha_to_coverage_enabled && desc.multisample.count == 1 {
        return Err(Error::InvalidState);
    }
    if desc.multisample.mask > u64::from(u32::MAX) {
        return Err(Error::InvalidRange);
    }
    if desc
        .multiview
        .is_some_and(|views| !(1..=4).contains(&views))
    {
        return Err(Error::Unsupported("DX12 multiview view count"));
    }
    if desc.color_targets.len() > 8 {
        return Err(Error::InvalidRange);
    }
    // SAFETY: Core validated the layout belongs to this device.
    let layout = unsafe { desc.layout.native_ref() }.0.clone();
    let (vs, mut raw_storage_bindings, vertex_index_constants) =
        stage(&desc.vertex, api::ShaderStage::Vertex, &layout)?;
    let ps = desc
        .fragment
        .as_ref()
        .map(|s| stage(s, api::ShaderStage::Fragment, &layout))
        .transpose()?;
    if let Some((_, bindings, _)) = &ps {
        raw_storage_bindings.extend(bindings);
    }
    raw_storage_bindings.sort_unstable();
    raw_storage_bindings.dedup();
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
    // Naga emits LOC<n>; caller-provided DXIL follows the LOCATION<n> ABI.
    // SAFETY: Core has validated this shader's device identity.
    let wgsl_vertex = matches!(
        unsafe { desc.vertex.module.native_ref() }.source,
        Some(Dx12ShaderSource::Wgsl(_))
    );
    let names: Vec<Vec<u8>> = desc
        .vertex_buffers
        .iter()
        .flat_map(|x| x.attributes.iter())
        .map(|_| {
            if wgsl_vertex {
                b"LOC\0".to_vec()
            } else {
                b"LOCATION\0".to_vec()
            }
        })
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
    let (formats, blends) = lower_targets(&desc.color_targets)?;
    let mut raster = lower_raster(desc.primitive, desc.multisample, depth)?;
    if dynamic_depth_bias.is_some() {
        // D3D12 requires all three static rasterizer depth-bias fields to be
        // zero when the PSO opts into dynamic depth bias.
        raster.DepthBias = 0;
        raster.DepthBiasClamp = 0.0;
        raster.SlopeScaledDepthBias = 0.0;
    }
    // SAFETY: Core validated the layout belongs to this device; cloning the
    // Arc retains its root signature after the public descriptor borrow ends.
    let (root, root_bytes, vertex_index_root) = if vertex_index_constants {
        let (root, bytes, index) = layout
            .vertex_index_root
            .as_ref()
            .ok_or(Error::Unsupported("DX12 WGSL vertex constants root budget"))?;
        (root.clone(), bytes, Some(*index))
    } else {
        (layout.root.clone(), &layout.root_bytes, None)
    };
    let depth_state = lower_depth(depth);
    let top = topology(desc.primitive.topology);
    let view_locations: Vec<_> = (0..desc.multiview.unwrap_or(0))
        .map(|index| D3D12_VIEW_INSTANCE_LOCATION {
            ViewportArrayIndex: 0,
            RenderTargetArrayIndex: index,
        })
        .collect();
    let mut pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Install the COM clone only after all fallible cache preparation.
        pRootSignature: ManuallyDrop::new(None),
        VS: bytecode(&vs),
        PS: ps
            .as_ref()
            .map_or_else(D3D12_SHADER_BYTECODE::default, |(bytes, _, _)| {
                bytecode(bytes)
            }),
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: if desc.multisample.alpha_to_coverage_enabled {
                TRUE
            } else {
                FALSE
            },
            IndependentBlendEnable: TRUE,
            RenderTarget: blends,
        },
        SampleMask: desc.multisample.mask as u32,
        RasterizerState: raster,
        DepthStencilState: depth_state,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: elements.as_ptr(),
            NumElements: elements.len() as u32,
        },
        PrimitiveTopologyType: top,
        IBStripCutValue: match desc.primitive.strip_index_format {
            Some(api::IndexFormat::Uint16) => D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF,
            Some(api::IndexFormat::Uint32) => D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF,
            None => D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
        },
        NumRenderTargets: desc.color_targets.len() as u32,
        RTVFormats: formats,
        DSVFormat: depth_format(depth),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: desc.multisample.count,
            Quality: 0,
        },
        ..Default::default()
    };
    if dynamic_depth_bias.is_some() {
        pso.Flags = D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS;
    }
    let settings = format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
        desc.vertex_buffers,
        desc.color_targets,
        desc.primitive,
        desc.depth_stencil,
        desc.multisample,
        desc.multiview,
        dynamic_depth_bias.is_some(),
    );
    let cache = desc.cache.as_ref().map(|c| unsafe { c.native_ref() });
    let key = cache
        .map(|_| {
            super::cache::key(&[
                b"raster-v1",
                root_bytes,
                &vs,
                ps.as_ref()
                    .map_or(&[][..], |(bytes, _, _)| bytes.as_slice()),
                settings.as_bytes(),
            ])
        })
        .transpose()?;
    let cached = match (cache, key.as_ref()) {
        (Some(c), Some(k)) => c.0.lookup(k)?,
        _ => None,
    };
    if let Some(bytes) = &cached {
        pso.CachedPSO = D3D12_CACHED_PIPELINE_STATE {
            pCachedBlob: bytes.as_ptr().cast(),
            CachedBlobSizeInBytes: bytes.len(),
        };
    }
    pso.pRootSignature = ManuallyDrop::new(Some(root.clone()));
    let state = if desc.multiview.is_some() {
        create_view_instanced_pso(
            &device.0.native,
            &pso,
            &root,
            &view_locations,
            cached.as_deref(),
        )
    } else {
        // SAFETY: all PSO pointer fields refer to locals retained for the
        // synchronous D3D12 creation call, and the root COM clone remains
        // manually owned until it is explicitly dropped below.
        let create = || unsafe {
            device
                .0
                .native
                .CreateGraphicsPipelineState::<ID3D12PipelineState>(&pso)
        };
        match create() {
            Err(error) if cached.is_some() && super::cache::is_compatibility_error(&error) => {
                pso.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
                unsafe {
                    device
                        .0
                        .native
                        .CreateGraphicsPipelineState::<ID3D12PipelineState>(&pso)
                }
                .map_err(native_error)
            }
            result => result.map_err(native_error),
        }
    };
    // SAFETY: CreateGraphicsPipelineState has consumed the descriptor; this
    // releases exactly the COM clone stored in the ManuallyDrop field.
    unsafe { ManuallyDrop::drop(&mut pso.pRootSignature) };
    let state = state?;
    if let (Some(cache), Some(key)) = (cache, key.as_ref()) {
        let _ = cache.0.store(key, &state);
    }
    // This finite private utility is required only by WGSL vertex-index
    // indirect draws. `create_raster` itself runs in the bounded cold worker,
    // so recording never compiles its DXC/PSO payload on the caller thread.
    if vertex_index_root.is_some() {
        let _ = device.0.indirect_pack_cache.get_or_create(device)?;
    }
    Ok(Dx12RasterPipeline(Arc::new(RasterPipelineState {
        root,
        layout,
        state,
        topology: native_topology(desc.primitive.topology),
        strip_index_format: desc.primitive.strip_index_format,
        raw_storage_bindings,
        vertex_index_root,
        vertex_indirect_signatures: Mutex::new([None, None]),
        depth_writes: depth.is_some_and(|d| d.depth_write_enabled),
        stencil_writes: stencil_writes(depth),
        vertex_strides,
        color_formats: desc
            .color_targets
            .iter()
            .map(|t| t.as_ref().map(|t| t.format))
            .collect(),
        depth_format: depth.map(|state| state.format),
        samples: desc.multisample.count,
        multiview: desc.multiview,
        dynamic_depth_bias: dynamic_depth_bias.is_some(),
    })))
}

fn stage(
    s: &api::PipelineStage<Dx12>,
    need: api::ShaderStage,
    layout: &super::layout::PipelineLayoutState,
) -> Result<(Vec<u8>, Vec<(u32, u32)>, bool)> {
    // SAFETY: Core's pipeline descriptor validates device identity. The return
    // slice borrows the stage, which outlives PSO construction in create_raster.
    let n = unsafe { s.module.native_ref() };
    let entry = n
        .entries
        .iter()
        .find(|e| e.name == s.entry && e.stage == need)
        .ok_or_else(|| {
            Error::Shader(format!(
                "DXIL has no requested {:?} entry {}",
                need, s.entry
            ))
        })?;
    match &n.source {
        Some(Dx12ShaderSource::Wgsl(source)) => {
            let compiled = super::super::shader_naga::compile_variant_with_typed_map(
                source,
                entry,
                &s.constants,
                &layout.register_map,
                &layout.binding_types,
                &layout.runtime_binding_capacities,
                None,
            )?;
            Ok((
                compiled.bytes,
                compiled.raw_storage_bindings,
                compiled.vertex_index_constants,
            ))
        }
        Some(Dx12ShaderSource::Hlsl(source)) => {
            if !layout.native_abi_compatible {
                return Err(Error::Unsupported("DX12 HLSL descriptor register overlap"));
            }
            let api::ShaderFormat::Hlsl(model) = source.format else {
                return Err(Error::InvalidState);
            };
            let text = std::str::from_utf8(&source.bytes)
                .map_err(|_| Error::Shader("HLSL is not UTF-8".into()))?;
            Ok((
                super::super::shader_naga::compile_direct_hlsl(
                    text,
                    &s.entry,
                    need,
                    model,
                    &s.constants,
                )?,
                Vec::new(),
                false,
            ))
        }
        None if !layout.native_abi_compatible => {
            Err(Error::Unsupported("DX12 DXIL descriptor register overlap"))
        }
        None if !s.constants.is_empty() => Err(Error::Unsupported("DX12 DXIL specialization")),
        None => Ok((n.bytes.to_vec(), Vec::new(), false)),
    }
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
        api::VertexFormat::Uint32(1) => Ok(DXGI_FORMAT_R32_UINT),
        api::VertexFormat::Uint32(2) => Ok(DXGI_FORMAT_R32G32_UINT),
        api::VertexFormat::Uint32(3) => Ok(DXGI_FORMAT_R32G32B32_UINT),
        api::VertexFormat::Uint32(4) => Ok(DXGI_FORMAT_R32G32B32A32_UINT),
        api::VertexFormat::Float16(1) => Ok(DXGI_FORMAT_R16_FLOAT),
        api::VertexFormat::Float16(2) => Ok(DXGI_FORMAT_R16G16_FLOAT),
        api::VertexFormat::Float16(4) => Ok(DXGI_FORMAT_R16G16B16A16_FLOAT),
        api::VertexFormat::Uint16(1) => Ok(DXGI_FORMAT_R16_UINT),
        api::VertexFormat::Uint16(2) => Ok(DXGI_FORMAT_R16G16_UINT),
        api::VertexFormat::Uint16(4) => Ok(DXGI_FORMAT_R16G16B16A16_UINT),
        api::VertexFormat::Sint16(1) => Ok(DXGI_FORMAT_R16_SINT),
        api::VertexFormat::Sint16(2) => Ok(DXGI_FORMAT_R16G16_SINT),
        api::VertexFormat::Sint16(4) => Ok(DXGI_FORMAT_R16G16B16A16_SINT),
        api::VertexFormat::Unorm16(1) => Ok(DXGI_FORMAT_R16_UNORM),
        api::VertexFormat::Unorm16(2) => Ok(DXGI_FORMAT_R16G16_UNORM),
        api::VertexFormat::Unorm16(4) => Ok(DXGI_FORMAT_R16G16B16A16_UNORM),
        api::VertexFormat::Snorm16(1) => Ok(DXGI_FORMAT_R16_SNORM),
        api::VertexFormat::Snorm16(2) => Ok(DXGI_FORMAT_R16G16_SNORM),
        api::VertexFormat::Snorm16(4) => Ok(DXGI_FORMAT_R16G16B16A16_SNORM),
        api::VertexFormat::Uint8(1) => Ok(DXGI_FORMAT_R8_UINT),
        api::VertexFormat::Uint8(2) => Ok(DXGI_FORMAT_R8G8_UINT),
        api::VertexFormat::Uint8(4) => Ok(DXGI_FORMAT_R8G8B8A8_UINT),
        api::VertexFormat::Sint8(1) => Ok(DXGI_FORMAT_R8_SINT),
        api::VertexFormat::Sint8(2) => Ok(DXGI_FORMAT_R8G8_SINT),
        api::VertexFormat::Sint8(4) => Ok(DXGI_FORMAT_R8G8B8A8_SINT),
        api::VertexFormat::Unorm8(1) => Ok(DXGI_FORMAT_R8_UNORM),
        api::VertexFormat::Unorm8(2) => Ok(DXGI_FORMAT_R8G8_UNORM),
        api::VertexFormat::Unorm8(4) => Ok(DXGI_FORMAT_R8G8B8A8_UNORM),
        api::VertexFormat::Snorm8(1) => Ok(DXGI_FORMAT_R8_SNORM),
        api::VertexFormat::Snorm8(2) => Ok(DXGI_FORMAT_R8G8_SNORM),
        api::VertexFormat::Snorm8(4) => Ok(DXGI_FORMAT_R8G8B8A8_SNORM),
        _ => Err(Error::Unsupported("DX12 vertex format")),
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
fn native_topology(v: api::PrimitiveTopology) -> D3D_PRIMITIVE_TOPOLOGY {
    match v {
        api::PrimitiveTopology::PointList => D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        api::PrimitiveTopology::LineList => D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        api::PrimitiveTopology::LineStrip => D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        api::PrimitiveTopology::TriangleList => D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        api::PrimitiveTopology::TriangleStrip => D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    }
}

fn native_error(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}
