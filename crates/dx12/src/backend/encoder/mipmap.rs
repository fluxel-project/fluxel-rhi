use super::*;
use std::sync::{Arc, Mutex};

/// Device-lifetime native state for the fixed RGBA8 linear mip blit pass.
/// Descriptor heaps remain per encoder because their descriptors are rewritten
/// for each source/target mip pair and must outlive command execution.
pub(crate) struct MipBlitPipeline {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) state: ID3D12PipelineState,
}

pub(crate) struct MipBlitCache(Mutex<Option<Arc<MipBlitPipeline>>>);

impl MipBlitCache {
    pub(crate) fn get_or_create(
        &self,
        device: &ID3D12Device,
    ) -> Result<Arc<MipBlitPipeline>, Error> {
        let mut cached = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pipeline) = cached.as_ref() {
            return Ok(pipeline.clone());
        }
        let pipeline = Arc::new(create_pipeline(device)?);
        *cached = Some(pipeline.clone());
        Ok(pipeline)
    }
}

impl Default for MipBlitCache {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}

pub(crate) fn validate_mip_blit(
    texture: &super::super::texture::TextureState,
    region: &api::TextureRegion<'_, Dx12>,
    source: bool,
) -> Result<(), Error> {
    if texture.desc.format != api::TextureFormat::Rgba8Unorm
        || region.aspect != api::TextureAspects::COLOR
        || region.mip >= texture.desc.mip_levels
        || region.origin != [0; 3]
        || region.extent[2] != 1
    {
        return Err(Error::Unsupported("DX12 RGBA8 full-mip blit"));
    }
    let width = (texture.desc.extent[0] >> region.mip).max(1);
    let height = (texture.desc.extent[1] >> region.mip).max(1);
    if region.extent != [width, height, 1] {
        return Err(Error::Unsupported("DX12 partial mip blit"));
    }
    if source && (width == 1 && height == 1) {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

fn create_pipeline(device: &ID3D12Device) -> Result<MipBlitPipeline, Error> {
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
        NumParameters: 2,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob = None;
    unsafe {
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
    }
    .map_err(native_error)?;
    let blob = blob.ok_or(Error::OutOfMemory)?;
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    let root = unsafe { device.CreateRootSignature::<ID3D12RootSignature>(0, bytes) }
        .map_err(native_error)?;
    let vs = include_bytes!("../../assets/mipmap.vs.dxil");
    let ps = include_bytes!("../../assets/mipmap.ps.dxil");
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
    let pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
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
            DXGI_FORMAT_R8G8B8A8_UNORM,
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
    let state = unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&pso) }
        .map_err(native_error)?;
    Ok(MipBlitPipeline { root, state })
}

pub(crate) fn linear_mip_blit(
    encoder: &mut Dx12Encoder,
    source: &ID3D12Resource,
    source_mip: u32,
    target: &ID3D12Resource,
    target_mip: u32,
    width: u32,
    height: u32,
) -> Result<(), Error> {
    let list = &encoder.list;
    // D3D12 has no fixed-function scaled blit.  This tiny native fullscreen
    // pass samples the source with a linear sampler and writes the destination.
    let mut device = None;
    unsafe { source.GetDevice::<ID3D12Device>(&mut device) }.map_err(native_error)?;
    let device = device.ok_or(Error::OutOfMemory)?;
    let pipeline = encoder.mip_blit_cache.get_or_create(&device)?;
    let root = &pipeline.root;
    let state = &pipeline.state;
    let views = unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    }
    .map_err(native_error)?;
    let samplers = unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    }
    .map_err(native_error)?;
    let rtv_heap = unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .map_err(native_error)?;
    let srv = unsafe { views.GetCPUDescriptorHandleForHeapStart() };
    let sh = unsafe { samplers.GetCPUDescriptorHandleForHeapStart() };
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    unsafe {
        device.CreateShaderResourceView(
            source,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: source_mip,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.,
                    },
                },
            }),
            srv,
        );
        device.CreateSampler(
            &D3D12_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
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
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_RTV {
                        MipSlice: target_mip,
                        PlaneSlice: 0,
                    },
                },
            }),
            rtv,
        );
    }
    barrier_sub(
        list,
        source,
        source_mip,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    );
    barrier_sub(
        list,
        target,
        target_mip,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    unsafe {
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.SetGraphicsRootSignature(root);
        list.SetPipelineState(state);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.RSSetViewports(&[D3D12_VIEWPORT {
            TopLeftX: 0.,
            TopLeftY: 0.,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.,
            MaxDepth: 1.,
        }]);
        list.RSSetScissorRects(&[RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        }]);
        let heaps = [Some(views.clone()), Some(samplers.clone())];
        list.SetDescriptorHeaps(&heaps);
        list.SetGraphicsRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
        list.SetGraphicsRootDescriptorTable(1, samplers.GetGPUDescriptorHandleForHeapStart());
        list.DrawInstanced(3, 1, 0, 0);
    }
    barrier_sub(
        list,
        source,
        source_mip,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    barrier_sub(
        list,
        target,
        target_mip,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COMMON,
    );
    encoder.mip_blit_heaps.push(views);
    encoder.mip_blit_heaps.push(samplers);
    encoder.mip_blit_heaps.push(rtv_heap);
    Ok(())
}
