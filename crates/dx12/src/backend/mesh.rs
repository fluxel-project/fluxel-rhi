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
    pub(crate) color_formats: Vec<Option<api::TextureFormat>>,
    pub(crate) samples: u32,
    pub(crate) depth_format: Option<api::TextureFormat>,
    pub(crate) depth_writes: bool,
    pub(crate) stencil_writes: bool,
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
    cached: Subobject<D3D12_CACHED_PIPELINE_STATE>,
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
    if desc.primitive.strip_index_format.is_some()
        || !matches!(
            desc.primitive.topology,
            api::PrimitiveTopology::TriangleList
        )
        || !matches!(desc.multisample.count, 1 | 2 | 4 | 8 | 16 | 32)
        || desc.multisample.mask > u64::from(u32::MAX)
    {
        return Err(Error::Unsupported("DX12 mesh pipeline state"));
    }
    if desc.multisample.alpha_to_coverage_enabled && desc.multisample.count == 1 {
        return Err(Error::InvalidState);
    }
    let depth = super::pipeline::fixed::validate_depth(desc.depth_stencil)?;
    let (formats, blends) = super::pipeline::fixed::lower_targets(&desc.targets)?;
    let raster = super::pipeline::fixed::lower_raster(desc.primitive, desc.multisample, depth)?;
    let depth_state = super::pipeline::fixed::lower_depth(depth);
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
    if !layout.native_abi_compatible {
        return Err(Error::Unsupported("DX12 DXIL descriptor register overlap"));
    }
    // SAFETY: Core validates that the cache belongs to the same device.
    let cache = desc
        .cache
        .as_ref()
        .map(|cache| unsafe { cache.native_ref() });
    let fixed = format!(
        "{:?}|{:?}|{:?}|{:?}",
        desc.primitive, desc.depth_stencil, desc.multisample, desc.targets
    );
    let key = cache
        .map(|_| {
            super::pipeline::cache::key(&[
                b"mesh-v1",
                &layout.root_bytes,
                task.unwrap_or(&[]),
                mesh,
                fragment.unwrap_or(&[]),
                fixed.as_bytes(),
            ])
        })
        .transpose()?;
    let cached = match (cache, key.as_ref()) {
        (Some(cache), Some(key)) => cache.0.lookup(key)?,
        _ => None,
    };
    let mut stream = Stream {
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
                IndependentBlendEnable: TRUE,
                RenderTarget: blends,
            },
        ),
        sample_mask: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            desc.multisample.mask as u32,
        ),
        raster: sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, raster),
        depth: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
            depth_state,
        ),
        depth_format: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            super::pipeline::fixed::depth_format(depth),
        ),
        topology: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ),
        formats: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            D3D12_RT_FORMAT_ARRAY {
                RTFormats: formats,
                NumRenderTargets: desc.targets.len() as u32,
            },
        ),
        cached: sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,
            cached
                .as_ref()
                .map_or(D3D12_CACHED_PIPELINE_STATE::default(), |blob| {
                    D3D12_CACHED_PIPELINE_STATE {
                        pCachedBlob: blob.as_ptr().cast(),
                        CachedBlobSizeInBytes: blob.len(),
                    }
                }),
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
    // SAFETY: stream, root signature, shader bytes and cached blob outlive the synchronous call.
    let mut result = unsafe { native_device.CreatePipelineState::<ID3D12PipelineState>(&native) };
    if cached.is_some()
        && result
            .as_ref()
            .is_err_and(super::pipeline::cache::is_compatibility_error)
    {
        stream.cached.value = D3D12_CACHED_PIPELINE_STATE::default();
        let retry = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: std::mem::size_of_val(&stream),
            pPipelineStateSubobjectStream: (&mut stream as *mut Stream).cast(),
        };
        // SAFETY: the rebuilt descriptor points to the updated live stream.
        result = unsafe { native_device.CreatePipelineState::<ID3D12PipelineState>(&retry) };
    }
    let state = result.map_err(super::native_error)?;
    if let (Some(cache), Some(key)) = (cache, key.as_ref()) {
        // The usable pipeline is independent of whether cache extraction succeeds.
        let _ = cache.0.store(key, &state);
    }
    Ok(Dx12MeshPipeline(Arc::new(MeshPipelineState {
        layout,
        state,
        color_formats: desc
            .targets
            .iter()
            .map(|t| t.as_ref().map(|t| t.format))
            .collect(),
        samples: desc.multisample.count,
        depth_format: depth.map(|d| d.format),
        depth_writes: depth.is_some_and(|d| d.depth_write_enabled),
        stencil_writes: super::pipeline::fixed::stencil_writes(depth),
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
