//! Internal graphics lowering for 2D color rectangle blits.
//!
//! Each array layer is rendered independently, using device-wide descriptor
//! heaps and a cached pipeline for the destination and shader numeric type.
//! A temporary target supports COPY_DST resources without render-target flags.
//! Descriptors and temporary resources remain alive through GPU completion.

use super::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// Device-lifetime native state for one destination format.
/// Descriptor heaps remain per encoder because their descriptors are rewritten
/// for each source/target mip pair and must outlive command execution.
pub(crate) struct MipBlitPipeline {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) state: ID3D12PipelineState,
}

pub(crate) struct MipBlitCache(Mutex<HashMap<(i32, u8, u8), Arc<MipBlitPipeline>>>);

impl MipBlitCache {
    pub(crate) fn get_or_create(
        &self,
        device: &ID3D12Device,
        format: DXGI_FORMAT,
        source_view: u8,
        kind: u8,
    ) -> Result<Arc<MipBlitPipeline>, Error> {
        let mut cached = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pipeline) = cached.get(&(format.0, source_view, kind)) {
            return Ok(pipeline.clone());
        }
        let pipeline = Arc::new(create_pipeline(device, format, source_view, kind)?);
        cached.insert((format.0, source_view, kind), pipeline.clone());
        Ok(pipeline)
    }
}

impl Default for MipBlitCache {
    fn default() -> Self {
        Self(Mutex::new(HashMap::new()))
    }
}

pub(crate) fn validate_mip_blit(
    texture: &super::super::texture::TextureState,
    region: &api::TextureRegion<'_, Dx12>,
    source: bool,
) -> Result<(), Error> {
    if (super::super::texture::color_texel_bytes(texture.desc.format).is_none()
        && !(source && super::super::texture::bc_block_bytes(texture.desc.format).is_some()))
        || region.aspect != api::TextureAspects::COLOR
        || region.mip >= texture.desc.mip_levels
        || texture.desc.samples != 1
        || region.extent[2] == 0
        || region.extent[0] == 0
        || region.extent[1] == 0
    {
        return Err(Error::Unsupported("DX12 color texture blit"));
    }
    let width = (texture.desc.extent[0] >> region.mip).max(1);
    let height = (texture.desc.extent[1] >> region.mip).max(1);
    let depth = if texture.desc.dimension == api::TextureDimension::D3 {
        (texture.desc.extent[2] >> region.mip).max(1)
    } else {
        texture.desc.array_layers
    };
    if region.origin[2]
        .checked_add(region.extent[2])
        .is_none_or(|v| v > depth)
        || region.origin[0]
            .checked_add(region.extent[0])
            .is_none_or(|v| v > width)
        || region.origin[1]
            .checked_add(region.extent[1])
            .is_none_or(|v| v > height)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub(crate) fn blit_kind(format: api::TextureFormat) -> u8 {
    if super::super::texture::integer_clear_bits(format, api::ClearColor::Sint([0; 4])).is_some() {
        2
    } else if super::super::texture::integer_clear_bits(format, api::ClearColor::Uint([0; 4]))
        .is_some()
    {
        1
    } else {
        0
    }
}

pub(crate) fn temporary_blit_target(
    encoder: &mut Dx12Encoder,
    size: [u32; 2],
    format: DXGI_FORMAT,
    dimension: api::TextureDimension,
) -> Result<ID3D12Resource, Error> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: match dimension {
            api::TextureDimension::D1 => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            api::TextureDimension::D2 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            api::TextureDimension::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        },
        Width: u64::from(size[0]),
        Height: size[1],
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: the initialized descriptors and output live through this synchronous call.
    unsafe {
        encoder.device.0.native.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut resource,
        )
    }
    .map_err(native_error)?;
    resource.ok_or(Error::OutOfMemory)
}

pub(crate) fn copy_temporary_blit(
    encoder: &mut Dx12Encoder,
    temporary: ID3D12Resource,
    target: &ID3D12Resource,
    target_subresource: u32,
    origin: [u32; 3],
) {
    barrier_sub(
        &encoder.list,
        &temporary,
        0,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    let mut from = copy_location(
        &temporary,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
        false,
        0,
    );
    let mut to = copy_location(
        target,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
        false,
        target_subresource,
    );
    // SAFETY: both subresources are in copy states; their validated extents fit the destination.
    unsafe {
        encoder
            .list
            .CopyTextureRegion(&to, origin[0], origin[1], origin[2], &from, None);
        ManuallyDrop::drop(&mut from.pResource);
        ManuallyDrop::drop(&mut to.pResource);
    }
}

fn create_pipeline(
    device: &ID3D12Device,
    format: DXGI_FORMAT,
    source_view: u8,
    kind: u8,
) -> Result<MipBlitPipeline, Error> {
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        },
    ];
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ranges[0],
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 5,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ranges[1],
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let root_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 3,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob = None;
    // SAFETY: `root_desc` points only into the local `ranges`/`params` arrays,
    // which remain alive through serialization; D3D12 copies the result into
    // the returned blob.
    unsafe {
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
    }
    .map_err(native_error)?;
    let blob = blob.ok_or(Error::OutOfMemory)?;
    // SAFETY: the blob owns this pointer and its reported byte range until the
    // blob is dropped after `CreateRootSignature` has copied the serialized data.
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    // SAFETY: `bytes` is the validated serialized root signature above and
    // `device` owns the native factory call.
    let root = unsafe { device.CreateRootSignature::<ID3D12RootSignature>(0, bytes) }
        .map_err(native_error)?;
    let vs = include_bytes!("../../assets/blit.vs.dxil");
    let ps: &[u8] = match (kind, source_view) {
        (0, 0) => include_bytes!("../../assets/blit.ps.dxil"),
        (0, 1) => include_bytes!("../../assets/blit_array.ps.dxil"),
        (1, 0) => include_bytes!("../../assets/blit_uint.ps.dxil"),
        (1, 1) => include_bytes!("../../assets/blit_uint_array.ps.dxil"),
        (2, 0) => include_bytes!("../../assets/blit_sint.ps.dxil"),
        (2, 1) => include_bytes!("../../assets/blit_sint_array.ps.dxil"),
        (0, 2) => include_bytes!("../../assets/blit_d1.ps.dxil"),
        (1, 2) => include_bytes!("../../assets/blit_uint_d1.ps.dxil"),
        (2, 2) => include_bytes!("../../assets/blit_sint_d1.ps.dxil"),
        (0, 3) => include_bytes!("../../assets/blit_d3.ps.dxil"),
        (1, 3) => include_bytes!("../../assets/blit_uint_d3.ps.dxil"),
        (2, 3) => include_bytes!("../../assets/blit_sint_d3.ps.dxil"),
        _ => return Err(Error::InvalidState),
    };
    let blend = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: windows::Win32::Foundation::FALSE,
        LogicOpEnable: windows::Win32::Foundation::FALSE,
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    let mut pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr().cast(),
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr().cast(),
            BytecodeLength: ps.len(),
        },
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: windows::Win32::Foundation::FALSE,
            IndependentBlendEnable: windows::Win32::Foundation::FALSE,
            RenderTarget: [blend; 8],
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: windows::Win32::Foundation::FALSE,
            DepthBias: 0,
            DepthBiasClamp: 0.,
            SlopeScaledDepthBias: 0.,
            DepthClipEnable: windows::Win32::Foundation::TRUE,
            MultisampleEnable: windows::Win32::Foundation::FALSE,
            AntialiasedLineEnable: windows::Win32::Foundation::FALSE,
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        },
        DepthStencilState: Default::default(),
        InputLayout: Default::default(),
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: [
            format,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        ],
        DSVFormat: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    // SAFETY: every pointer in `pso` references local shader/format data that
    // remains live for this call; D3D12 copies pipeline state at creation.
    let state = unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&pso) };
    // SAFETY: release the temporary root-signature clone after PSO creation.
    unsafe { ManuallyDrop::drop(&mut pso.pRootSignature) };
    let state = state.map_err(native_error)?;
    Ok(MipBlitPipeline { root, state })
}

pub(crate) fn linear_mip_blit(
    encoder: &mut Dx12Encoder,
    source: &ID3D12Resource,
    source_mip: u32,
    source_layer: u32,
    source_mips: u32,
    source_view: u8,
    source_z: f32,
    kind: u8,
    source_rect: [u32; 4],
    source_size: [u32; 2],
    source_format: DXGI_FORMAT,
    target: &ID3D12Resource,
    target_mip: u32,
    target_layer: u32,
    target_mips: u32,
    target_dimension: api::TextureDimension,
    target_array: bool,
    target_rect: [u32; 4],
    target_format: DXGI_FORMAT,
    filter: api::FilterMode,
) -> Result<(), Error> {
    let list = &encoder.list;
    // D3D12 has no fixed-function scaled blit.  This tiny native fullscreen
    // pass samples the source with a linear sampler and writes the destination.
    let mut device = None;
    // SAFETY: `source` is retained by the encoder and the out parameter is a
    // valid Option slot for the COM interface returned by D3D12.
    unsafe { source.GetDevice::<ID3D12Device>(&mut device) }.map_err(native_error)?;
    let device = device.ok_or(Error::OutOfMemory)?;
    let pipeline =
        encoder
            .mip_blit_cache
            .get_or_create(&device, target_format, source_view, kind)?;
    let root = &pipeline.root;
    let state = &pipeline.state;
    let descriptors = encoder.device.0.descriptors.temporary_pair()?;
    // SAFETY: the RTV heap descriptor is initialized; its COM reference is
    // retained until the command list using its handle is fence-retired.
    let rtv_heap = unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .map_err(native_error)?;
    // SAFETY: each non-empty heap above remains alive through all descriptor
    // writes and command recording below, so its start handles remain valid.
    let srv = descriptors.view_cpu();
    let sh = descriptors.sampler_cpu();
    // SAFETY: identical non-empty retained-heap invariant as `srv` and `sh`.
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    let view_gpu = descriptors.view_gpu();
    let sampler_gpu = descriptors.sampler_gpu();
    // SAFETY: source/target resources and all three heaps are retained by the
    // encoder; the view ranges name validated mip indices and initialized descriptors.
    unsafe {
        device.CreateShaderResourceView(
            source,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: source_format,
                ViewDimension: if source_view == 2 {
                    D3D12_SRV_DIMENSION_TEXTURE1DARRAY
                } else if source_view == 3 {
                    D3D12_SRV_DIMENSION_TEXTURE3D
                } else if source_view == 1 {
                    D3D12_SRV_DIMENSION_TEXTURE2DARRAY
                } else {
                    D3D12_SRV_DIMENSION_TEXTURE2D
                },
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: if source_view == 2 {
                    D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture1DArray: D3D12_TEX1D_ARRAY_SRV {
                            MostDetailedMip: source_mip,
                            MipLevels: 1,
                            FirstArraySlice: source_layer,
                            ArraySize: 1,
                            ResourceMinLODClamp: 0.,
                        },
                    }
                } else if source_view == 3 {
                    D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture3D: D3D12_TEX3D_SRV {
                            MostDetailedMip: source_mip,
                            MipLevels: 1,
                            ResourceMinLODClamp: 0.,
                        },
                    }
                } else if source_view == 1 {
                    D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                            MostDetailedMip: source_mip,
                            MipLevels: 1,
                            FirstArraySlice: source_layer,
                            ArraySize: 1,
                            PlaneSlice: 0,
                            ResourceMinLODClamp: 0.,
                        },
                    }
                } else {
                    D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MostDetailedMip: source_mip,
                            MipLevels: 1,
                            PlaneSlice: 0,
                            ResourceMinLODClamp: 0.,
                        },
                    }
                },
            }),
            srv,
        );
        device.CreateSampler(
            &D3D12_SAMPLER_DESC {
                Filter: if filter == api::FilterMode::Nearest {
                    D3D12_FILTER_MIN_MAG_MIP_POINT
                } else {
                    D3D12_FILTER_MIN_MAG_MIP_LINEAR
                },
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MipLODBias: 0.,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_NONE,
                BorderColor: [0.; 4],
                MinLOD: 0.,
                MaxLOD: 0.,
            },
            sh,
        );
        device.CreateRenderTargetView(
            target,
            Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                Format: target_format,
                ViewDimension: if target_dimension == api::TextureDimension::D1 {
                    D3D12_RTV_DIMENSION_TEXTURE1DARRAY
                } else if target_dimension == api::TextureDimension::D3 {
                    D3D12_RTV_DIMENSION_TEXTURE3D
                } else if target_array {
                    D3D12_RTV_DIMENSION_TEXTURE2DARRAY
                } else {
                    D3D12_RTV_DIMENSION_TEXTURE2D
                },
                Anonymous: if target_dimension == api::TextureDimension::D1 {
                    D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture1DArray: D3D12_TEX1D_ARRAY_RTV {
                            MipSlice: target_mip,
                            FirstArraySlice: target_layer,
                            ArraySize: 1,
                        },
                    }
                } else if target_dimension == api::TextureDimension::D3 {
                    D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture3D: D3D12_TEX3D_RTV {
                            MipSlice: target_mip,
                            FirstWSlice: target_layer,
                            WSize: 1,
                        },
                    }
                } else if target_array {
                    D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                            MipSlice: target_mip,
                            FirstArraySlice: target_layer,
                            ArraySize: 1,
                            PlaneSlice: 0,
                        },
                    }
                } else {
                    D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_RTV {
                            MipSlice: target_mip,
                            PlaneSlice: 0,
                        },
                    }
                },
            }),
            rtv,
        );
    }
    // Record-owned heaps must survive even if restoring a previous pipeline fails.
    encoder.mip_blit_descriptors.push(descriptors);
    encoder.mip_blit_heaps.push(rtv_heap);
    barrier_sub(
        list,
        source,
        source_mip
            + if source_view == 3 {
                0
            } else {
                source_layer * source_mips
            },
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    );
    barrier_sub(
        list,
        target,
        target_mip
            + if target_dimension == api::TextureDimension::D3 {
                0
            } else {
                target_layer * target_mips
            },
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    // SAFETY: this encoder exclusively records `list`; root state, descriptor
    // heaps, and attachment handles above remain retained through submission.
    unsafe {
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.SetGraphicsRootSignature(root);
        list.SetPipelineState(state);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.RSSetViewports(&[D3D12_VIEWPORT {
            TopLeftX: target_rect[0] as f32,
            TopLeftY: target_rect[1] as f32,
            Width: target_rect[2] as f32,
            Height: target_rect[3] as f32,
            MinDepth: 0.,
            MaxDepth: 1.,
        }]);
        list.RSSetScissorRects(&[RECT {
            left: target_rect[0] as i32,
            top: target_rect[1] as i32,
            right: (target_rect[0] + target_rect[2]) as i32,
            bottom: (target_rect[1] + target_rect[3]) as i32,
        }]);
        list.SetGraphicsRootDescriptorTable(0, view_gpu);
        list.SetGraphicsRootDescriptorTable(2, sampler_gpu);
        let uv = [
            source_rect[0] as f32 / source_size[0] as f32,
            source_rect[1] as f32 / source_size[1] as f32,
            source_rect[2] as f32 / source_size[0] as f32,
            source_rect[3] as f32 / source_size[1] as f32,
            source_z,
        ];
        list.SetGraphicsRoot32BitConstants(1, 5, uv.as_ptr().cast(), 0);
        list.DrawInstanced(3, 1, 0, 0);
    }
    barrier_sub(
        list,
        source,
        source_mip
            + if source_view == 3 {
                0
            } else {
                source_layer * source_mips
            },
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    barrier_sub(
        list,
        target,
        target_mip
            + if target_dimension == api::TextureDimension::D3 {
                0
            } else {
                target_layer * target_mips
            },
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    if let Some(compute) = &encoder.compute_pipeline {
        // SAFETY: the previous PSO remains owned by the encoder.
        unsafe { list.SetPipelineState(&compute.0.state) };
    }
    if let Some(ray) = &encoder.ray_pipeline {
        let list4 = list
            .cast::<ID3D12GraphicsCommandList4>()
            .map_err(native_error)?;
        // SAFETY: the prior ray state object is retained by the encoder.
        unsafe { list4.SetPipelineState1(&ray.0.state) };
    }
    Ok(())
}
