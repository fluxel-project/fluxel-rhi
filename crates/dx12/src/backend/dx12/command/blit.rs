//! Shader implementation of Direct3D 12 filtered texture blits.
//!
//! D3D12 has no scaled image-copy instruction.  Rendering into the caller's
//! destination would require `ALLOW_RENDER_TARGET` at texture creation, but a
//! portable `COPY_DST` texture deliberately has no such hidden requirement.
//! This lowering therefore renders into a private temporary RT, then copies the
//! result into the destination.  The temporary is retained through the batch
//! fence, just like upload staging.

use std::{mem::ManuallyDrop, slice};

use windows::Win32::Foundation::{FALSE, TRUE};
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_ROOT_SIGNATURE_VERSION_1,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::api::command::copy::{BlitFilter, TextureBlit};
use crate::api::resource::texture::{TextureDimension, mip_extent};
use crate::backend::dx12::failure::{Dx12Failure, ref_native};
use crate::backend::dx12::platform::facts::dxgi_format;

use super::dx12_texture;
use super::transfer::CommittedBatch;
use super::transition::Transitions;

const BLIT_VS: &[u8] = include_bytes!("blit_vs.dxil");
const BLIT_PS: &[u8] = include_bytes!("blit_ps.dxil");

/// Records one D2, one-layer filtered blit.  The capability table only exposes
/// this baseline; array/cube/3D variants need a different shader interface
/// rather than silently sampling layer zero.
pub(super) fn lower_texture_blit(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    blit: &TextureBlit,
    committed: &mut CommittedBatch,
) -> Result<(), Dx12Failure> {
    let src_desc = blit.src.descriptor();
    let dst_desc = blit.dst.descriptor();
    if src_desc.dimension != TextureDimension::D2
        || dst_desc.dimension != TextureDimension::D2
        || src_desc.array_layers != 1
        || dst_desc.array_layers != 1
        || blit.src_subresource.layer_count != 1
        || blit.dst_subresource.layer_count != 1
    {
        return Err(unsupported(
            "a non-single-layer 2D shader blit",
            "the embedded fullscreen shader samples Texture2D and has no array/cube/3D ABI",
        ));
    }
    let format = dxgi_format(dst_desc.format).ok_or_else(|| {
        unsupported(
            "a shader-blit destination format without a DXGI mapping",
            "DX12 cannot create the private render target",
        )
    })?;
    let source = dx12_texture(&blit.src)?;
    let destination = dx12_texture(&blit.dst)?;
    let src_subresource =
        blit.src_subresource.mip_level + blit.src_subresource.base_layer * src_desc.mip_levels;
    let dst_subresource =
        blit.dst_subresource.mip_level + blit.dst_subresource.base_layer * dst_desc.mip_levels;
    let same_subresource = blit.src.id() == blit.dst.id() && src_subresource == dst_subresource;

    let (root, pso) = create_pipeline(device, format)?;
    let (srv_heap, srv_cpu, srv_gpu) =
        visible_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)?;
    let (sampler_heap, sampler_cpu, sampler_gpu) =
        visible_heap(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)?;
    let (rtv_heap, rtv) = rtv_heap(device)?;

    let source_extent = mip_extent(
        src_desc.extent,
        src_desc.dimension,
        blit.src_subresource.mip_level,
    );
    let temporary = temporary_target(
        device,
        format,
        blit.dst_extent.width,
        blit.dst_extent.height,
    )?;
    unsafe {
        device.CreateShaderResourceView(
            source.resource(),
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: dxgi_format(src_desc.format).ok_or_else(|| {
                    unsupported(
                        "a shader-blit source format without a DXGI mapping",
                        "DX12 cannot create its SRV",
                    )
                })?,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: blit.src_subresource.mip_level,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            }),
            srv_cpu,
        );
        device.CreateSampler(
            &D3D12_SAMPLER_DESC {
                Filter: match blit.filter {
                    BlitFilter::Nearest => D3D12_FILTER_MIN_MAG_MIP_POINT,
                    BlitFilter::Linear => D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                    _ => {
                        return Err(unsupported(
                            "an unknown blit filter",
                            "the DX12 shader path has no sampler mapping",
                        ));
                    }
                },
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: 0.0,
            },
            sampler_cpu,
        );
        device.CreateRenderTargetView(
            &temporary,
            Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_RTV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            }),
            rtv,
        );
    }

    let mut entering = Transitions::default();
    entering.push_subresource(
        source.resource(),
        src_subresource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    );
    entering.push(
        &temporary,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    entering.record(list);
    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: blit.dst_extent.width as f32,
        Height: blit.dst_extent.height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: i32::try_from(blit.dst_extent.width).map_err(|_| {
            unsupported(
                "a shader-blit width above i32",
                "D3D12 scissor rectangles use LONG",
            )
        })?,
        bottom: i32::try_from(blit.dst_extent.height).map_err(|_| {
            unsupported(
                "a shader-blit height above i32",
                "D3D12 scissor rectangles use LONG",
            )
        })?,
    };
    let constants = [
        blit.src_origin.x as f32 / source_extent.width as f32,
        blit.src_origin.y as f32 / source_extent.height as f32,
        (blit.src_origin.x + blit.src_extent.width) as f32 / source_extent.width as f32,
        (blit.src_origin.y + blit.src_extent.height) as f32 / source_extent.height as f32,
    ];
    unsafe {
        list.SetDescriptorHeaps(&[Some(srv_heap.clone()), Some(sampler_heap.clone())]);
        list.SetGraphicsRootSignature(&root);
        list.SetPipelineState(&pso);
        list.SetGraphicsRootDescriptorTable(0, srv_gpu);
        list.SetGraphicsRootDescriptorTable(1, sampler_gpu);
        list.SetGraphicsRoot32BitConstants(2, constants.len() as u32, constants.as_ptr().cast(), 0);
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.RSSetViewports(&[viewport]);
        list.RSSetScissorRects(&[scissor]);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.DrawInstanced(3, 1, 0, 0);
    }

    let mut copy_source = texture_location(&temporary, 0);
    let mut copy_destination = texture_location(destination.resource(), dst_subresource);
    let box_ = D3D12_BOX {
        left: 0,
        top: 0,
        front: 0,
        right: blit.dst_extent.width,
        bottom: blit.dst_extent.height,
        back: 1,
    };
    let mut between = Transitions::default();
    between.push(
        &temporary,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    between.push_subresource(
        destination.resource(),
        dst_subresource,
        if same_subresource {
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        } else {
            D3D12_RESOURCE_STATE_COMMON
        },
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    between.record(list);
    unsafe {
        list.CopyTextureRegion(
            &copy_destination,
            blit.dst_origin.x,
            blit.dst_origin.y,
            0,
            &copy_source,
            Some(&box_),
        );
        drop_copy_location(&mut copy_source);
        drop_copy_location(&mut copy_destination);
    }
    let mut leaving = Transitions::default();
    if !same_subresource {
        leaving.push_subresource(
            source.resource(),
            src_subresource,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
    }
    leaving.push(
        &temporary,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.push_subresource(
        destination.resource(),
        dst_subresource,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    leaving.record(list);

    committed
        .raster_descriptor_heaps
        .extend([srv_heap, sampler_heap, rtv_heap]);
    committed.blit_root_signatures.push(root);
    committed.blit_pipeline_states.push(pso);
    committed.blit_resources.push(temporary);
    Ok(())
}

fn create_pipeline(
    device: &ID3D12Device,
    format: DXGI_FORMAT,
) -> Result<(ID3D12RootSignature, ID3D12PipelineState), Dx12Failure> {
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
    ];
    let parameters = [
        descriptor_table(&ranges[0]),
        descriptor_table(&ranges[1]),
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let signature_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob = None;
    unsafe {
        D3D12SerializeRootSignature(
            &signature_desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            None,
        )
    }
    .map_err(|error| ref_native(&error))?;
    let blob = blob.ok_or_else(|| {
        unsupported(
            "a missing shader-blit root signature blob",
            "D3D12 serialization reported success without a blob",
        )
    })?;
    let bytes =
        unsafe { slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    let root = unsafe { device.CreateRootSignature::<ID3D12RootSignature>(0, bytes) }
        .map_err(|error| ref_native(&error))?;

    let disabled = D3D12_RENDER_TARGET_BLEND_DESC {
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
    let stencil = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: bytecode(BLIT_VS),
        PS: bytecode(BLIT_PS),
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: FALSE,
            IndependentBlendEnable: FALSE,
            RenderTarget: [disabled; 8],
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: FALSE,
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: TRUE,
            MultisampleEnable: FALSE,
            AntialiasedLineEnable: FALSE,
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: FALSE,
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            StencilEnable: FALSE,
            StencilReadMask: 0,
            StencilWriteMask: 0,
            FrontFace: stencil,
            BackFace: stencil,
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC::default(),
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
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
        DSVFormat: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let pso = unsafe { device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&pso_desc) }
        .map_err(|error| ref_native(&error))?;
    // `pRootSignature` owns a manually-managed reference in the generated ABI.
    let mut pso_desc = pso_desc;
    unsafe { ManuallyDrop::drop(&mut pso_desc.pRootSignature) };
    Ok((root, pso))
}

fn descriptor_table(range: &D3D12_DESCRIPTOR_RANGE) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    }
}

fn visible_heap(
    device: &ID3D12Device,
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
) -> Result<
    (
        ID3D12DescriptorHeap,
        D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_GPU_DESCRIPTOR_HANDLE,
    ),
    Dx12Failure,
> {
    let heap = unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    }
    .map_err(|error| ref_native(&error))?;
    Ok((
        heap.clone(),
        unsafe { heap.GetCPUDescriptorHandleForHeapStart() },
        unsafe { heap.GetGPUDescriptorHandleForHeapStart() },
    ))
}

fn rtv_heap(
    device: &ID3D12Device,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Dx12Failure> {
    let heap = unsafe {
        device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        })
    }
    .map_err(|error| ref_native(&error))?;
    Ok((heap.clone(), unsafe {
        heap.GetCPUDescriptorHandleForHeapStart()
    }))
}

fn temporary_target(
    device: &ID3D12Device,
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
) -> Result<ID3D12Resource, Dx12Failure> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    };
    let mut target = None;
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut target,
        )
    }
    .map_err(|error| ref_native(&error))?;
    target.ok_or_else(|| {
        unsupported(
            "a missing shader-blit temporary texture",
            "CreateCommittedResource reported success without a resource",
        )
    })
}

fn texture_location(resource: &ID3D12Resource, subresource: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(resource.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: subresource,
        },
    }
}

unsafe fn drop_copy_location(location: &mut D3D12_TEXTURE_COPY_LOCATION) {
    unsafe { ManuallyDrop::drop(&mut location.pResource) };
}

fn bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}

fn unsupported(what: &'static str, why: &'static str) -> Dx12Failure {
    Dx12Failure::Unsupported { what, why }
}
