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

pub(super) const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION: u32 =
    D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
pub(super) const MAX_DISPATCH_WORKGROUPS: u32 = 1 << 22;
pub(super) const SHADER_LIMITS: api::MeshShaderLimits = api::MeshShaderLimits {
    max_mesh_workgroup_invocations: 128,
    max_task_workgroup_invocations: 128,
    max_mesh_output_vertices: 256,
    max_mesh_output_primitives: 256,
    max_mesh_output_layers: 8,
    max_task_payload_bytes: 16 * 1024,
    max_mesh_output_bytes: 32 * 1024,
    max_task_payload_and_mesh_output_bytes: Some(47 * 1024),
};

pub(super) fn shader_limits(device: &ID3D12Device) -> api::MeshShaderLimits {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
    // SAFETY: OPTIONS9 matches the feature selector and remains writable
    // through the synchronous query.
    let full_range = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS9,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS9).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options
            .MeshShaderSupportsFullRangeRenderTargetArrayIndex
            .as_bool();
    api::MeshShaderLimits {
        max_mesh_output_layers: if full_range {
            D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION
        } else {
            SHADER_LIMITS.max_mesh_output_layers
        },
        ..SHADER_LIMITS
    }
}

pub(crate) struct MeshPipelineState {
    pub(crate) layout: Arc<PipelineLayoutState>,
    pub(crate) state: ID3D12PipelineState,
    pub(crate) color_formats: Vec<Option<api::TextureFormat>>,
    pub(crate) samples: u32,
    pub(crate) depth_format: Option<api::TextureFormat>,
    pub(crate) depth_writes: bool,
    pub(crate) stencil_writes: bool,
    pub(crate) multiview: Option<u32>,
}

#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct Subobject<T> {
    tag: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    value: T,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Stream<D> {
    root: Subobject<*mut std::ffi::c_void>,
    task: Subobject<D3D12_SHADER_BYTECODE>,
    mesh: Subobject<D3D12_SHADER_BYTECODE>,
    fragment: Subobject<D3D12_SHADER_BYTECODE>,
    blend: Subobject<D3D12_BLEND_DESC>,
    sample_mask: Subobject<u32>,
    raster: Subobject<D3D12_RASTERIZER_DESC>,
    depth: Subobject<D>,
    depth_format: Subobject<DXGI_FORMAT>,
    topology: Subobject<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    formats: Subobject<D3D12_RT_FORMAT_ARRAY>,
    samples: Subobject<DXGI_SAMPLE_DESC>,
    cached: Subobject<D3D12_CACHED_PIPELINE_STATE>,
}
#[repr(C)]
struct MultiviewStream<D> {
    base: Stream<D>,
    view_instancing: Subobject<D3D12_VIEW_INSTANCING_DESC>,
}

fn create_native_stream<T>(
    device: &ID3D12Device2,
    stream: &T,
) -> windows::core::Result<ID3D12PipelineState> {
    let native = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: std::mem::size_of::<T>(),
        pPipelineStateSubobjectStream: std::ptr::from_ref(stream).cast_mut().cast(),
    };
    // SAFETY: callers use a typed mesh stream and retain every borrowed native
    // payload through the synchronous CreatePipelineState call.
    unsafe { device.CreatePipelineState::<ID3D12PipelineState>(&native) }
}

impl<D: Copy> Stream<D> {
    fn with_depth<D2>(self, depth: Subobject<D2>) -> Stream<D2> {
        Stream {
            root: self.root,
            task: self.task,
            mesh: self.mesh,
            fragment: self.fragment,
            blend: self.blend,
            sample_mask: self.sample_mask,
            raster: self.raster,
            depth,
            depth_format: self.depth_format,
            topology: self.topology,
            formats: self.formats,
            samples: self.samples,
            cached: self.cached,
        }
    }

    fn create(
        mut self,
        device: &ID3D12Device2,
        has_cache: bool,
        view_locations: Option<&[D3D12_VIEW_INSTANCE_LOCATION]>,
    ) -> Result<ID3D12PipelineState> {
        let create = |stream: &Self| {
            if let Some(locations) = view_locations {
                create_native_stream(
                    device,
                    &MultiviewStream {
                        base: *stream,
                        view_instancing: sub(
                            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING,
                            super::pipeline::view_instancing_desc(locations),
                        ),
                    },
                )
            } else {
                create_native_stream(device, stream)
            }
        };
        match create(&self) {
            Err(error) if has_cache && super::pipeline::cache::is_compatibility_error(&error) => {
                self.cached.value = D3D12_CACHED_PIPELINE_STATE::default();
                create(&self).map_err(super::native_error)
            }
            result => result.map_err(super::native_error),
        }
    }
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
    if shader.source.is_some() {
        return Err(Error::Unsupported(
            "DX12 mesh pipeline requires DXIL shader stages",
        ));
    }
    if expected == api::ShaderStage::Fragment && shader.requires_raster_primitive_index {
        return Err(Error::Unsupported(
            "DX12 mesh shader cannot supply the rasterizer primitive index",
        ));
    }
    let entry = shader
        .entries
        .iter()
        .find(|entry| entry.name == stage.entry && entry.stage == expected)
        .ok_or(Error::InvalidState)?;
    let max_invocations = match expected {
        api::ShaderStage::Mesh => SHADER_LIMITS.max_mesh_workgroup_invocations,
        api::ShaderStage::Task => SHADER_LIMITS.max_task_workgroup_invocations,
        _ => return Ok(&shader.bytes),
    };
    // This validates the caller's declaration. Native PSO creation validates
    // the DXIL, including artifacts with no declared workgroup size.
    if let Some(size) = entry.workgroup_size {
        let invocations = size.into_iter().try_fold(1u32, u32::checked_mul);
        if invocations.is_none_or(|count| count == 0 || count > max_invocations) {
            return Err(Error::InvalidRange);
        }
    }
    Ok(&shader.bytes)
}
fn bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}

fn topology(
    device: &Dx12Device,
    topology: api::PrimitiveTopology,
) -> std::result::Result<D3D12_PRIMITIVE_TOPOLOGY_TYPE, &'static str> {
    if !device.0.mesh_supported {
        return Err("DX12 mesh shader tier 1");
    }
    match topology {
        api::PrimitiveTopology::LineList => Ok(D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE),
        api::PrimitiveTopology::TriangleList => Ok(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE),
        _ => Err("DX12 mesh output topology"),
    }
}

pub(super) fn create_pipeline(
    device: &Dx12Device,
    desc: &api::MeshPipelineDesc<Dx12>,
    masks: Option<api::StencilMasks>,
    multiview: Option<u32>,
) -> Result<Dx12MeshPipeline> {
    if let Some(views) = multiview {
        if !device.0.mesh_multiview {
            return Err(Error::Unsupported("DX12 mesh multiview"));
        }
        if !(1..=D3D12_MAX_VIEW_INSTANCE_COUNT).contains(&views) {
            return Err(Error::InvalidRange);
        }
    }
    if masks.is_some() && !device.0.stencil_reference_per_face {
        return Err(Error::Unsupported("DX12 per-face stencil masks"));
    }
    let topology = topology(device, desc.primitive.topology).map_err(Error::Unsupported)?;
    if desc.primitive.strip_index_format.is_some()
        || !matches!(desc.multisample.count, 1 | 2 | 4 | 8 | 16 | 32)
        || desc.multisample.mask > u64::from(u32::MAX)
    {
        return Err(Error::Unsupported("DX12 mesh pipeline state"));
    }
    if desc.multisample.alpha_to_coverage_enabled && desc.multisample.count == 1 {
        return Err(Error::InvalidState);
    }
    let depth = super::pipeline::fixed::validate_depth(desc.depth_stencil)?;
    let (formats, blends) = super::pipeline::fixed::lower_targets(device, &desc.targets)?;
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
    let mut fixed = format!(
        "{:?}|{:?}|{:?}|{:?}",
        desc.primitive, desc.depth_stencil, desc.multisample, desc.targets
    );
    if let Some(masks) = masks {
        fixed.push_str(&format!("|{masks:?}"));
    }
    if let Some(views) = multiview {
        fixed.push_str(&format!("|views={views}"));
    }
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
            topology,
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
    let native_device = device
        .0
        .native
        .cast::<ID3D12Device2>()
        .map_err(super::native_error)?;
    let view_locations = multiview.map(super::pipeline::view_locations);
    let state = match masks {
        Some(masks) => stream
            .with_depth(sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2,
                super::pipeline::fixed::per_face_stencil(depth_state, masks),
            ))
            .create(&native_device, cached.is_some(), view_locations.as_deref())?,
        None => stream.create(&native_device, cached.is_some(), view_locations.as_deref())?,
    };
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
        stencil_writes: super::pipeline::fixed::stencil_writes(depth, masks),
        multiview,
    })))
}

pub(super) fn probe(device: &ID3D12Device, shader_model: u16) -> bool {
    if shader_model < 65 {
        return false;
    }
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    // SAFETY: the feature structure matches its query selector and
    // remains valid throughout this synchronous feature query.
    let tier = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS7,
            (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(),
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;
    tier && device.cast::<ID3D12Device2>().is_ok()
}
