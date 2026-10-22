//! Compute-queue rectangle blit using a retained typed UAV staging texture.
use super::*;

pub(crate) fn blit_compute(
    encoder: &mut Dx12Encoder,
    src: &super::super::texture::TextureState,
    dst: &super::super::texture::TextureState,
    source: &api::TextureRegion<'_, Dx12>,
    target: &api::TextureRegion<'_, Dx12>,
    filter: api::FilterMode,
) -> Result<(), Error> {
    use windows::Win32::Graphics::Dxgi::Common::*;
    let source_view = match src.desc.dimension {
        api::TextureDimension::D1 => 2,
        api::TextureDimension::D3 => 3,
        api::TextureDimension::D2 => u8::from(src.desc.array_layers > 1),
    };
    let target_view = match dst.desc.dimension {
        api::TextureDimension::D1 => 0,
        api::TextureDimension::D2 => 1,
        api::TextureDimension::D3 => 2,
    };
    let kind = super::mipmap::blit_kind(src.desc.format);
    let (uav_format, srgb) = match dst.desc.format {
        api::TextureFormat::Rgba8UnormSrgb => (DXGI_FORMAT_R8G8B8A8_UNORM, true),
        api::TextureFormat::Bgra8UnormSrgb => (DXGI_FORMAT_B8G8R8A8_UNORM, true),
        _ => (super::super::texture::dxgi_format(dst.desc.format), false),
    };
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: uav_format,
        ..Default::default()
    };
    // SAFETY: the feature query receives the matching initialized support structure.
    let typed = unsafe {
        encoder.device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &mut support as *mut _ as *mut _,
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support
            .Support1
            .contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
        && support
            .Support2
            .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
    // The hardware-test hook forces the fallback on devices whose typed UAV
    // support would otherwise make the packing path unreachable.
    #[cfg(feature = "hardware-tests")]
    let typed = typed && std::env::var_os("FLUXEL_DX12_TEST_BUFFER_BLIT").is_none();
    let pixel_bytes = super::super::texture::color_texel_bytes(dst.desc.format)
        .ok_or(Error::Unsupported("DX12 blit buffer format"))?;
    let format_code = if typed {
        0
    } else {
        buffer_format_code(dst.desc.format)?
    };
    let row_pitch = target.extent[0]
        .checked_mul(pixel_bytes)
        .and_then(|n| n.checked_add(255))
        .ok_or(Error::InvalidRange)?
        & !255;
    let buffer_size = u64::from(row_pitch) * u64::from(target.extent[1]);
    let shader: &[u8] = if !typed {
        match (source_view, kind) {
            (0, 0) => include_bytes!("../../assets/blit_buffer_d2_float.dxil"),
            (0, 1) => include_bytes!("../../assets/blit_buffer_d2_uint.dxil"),
            (0, 2) => include_bytes!("../../assets/blit_buffer_d2_sint.dxil"),
            (1, 0) => include_bytes!("../../assets/blit_buffer_array_float.dxil"),
            (1, 1) => include_bytes!("../../assets/blit_buffer_array_uint.dxil"),
            (1, 2) => include_bytes!("../../assets/blit_buffer_array_sint.dxil"),
            (2, 0) => include_bytes!("../../assets/blit_buffer_d1_float.dxil"),
            (2, 1) => include_bytes!("../../assets/blit_buffer_d1_uint.dxil"),
            (2, 2) => include_bytes!("../../assets/blit_buffer_d1_sint.dxil"),
            (3, 0) => include_bytes!("../../assets/blit_buffer_d3_float.dxil"),
            (3, 1) => include_bytes!("../../assets/blit_buffer_d3_uint.dxil"),
            (3, 2) => include_bytes!("../../assets/blit_buffer_d3_sint.dxil"),
            _ => return Err(Error::InvalidState),
        }
    } else {
        match (source_view, target_view, kind) {
            (0, 0, 0) => include_bytes!("../../assets/blit_cs_d2_d1_float.dxil"),
            (0, 0, 1) => include_bytes!("../../assets/blit_cs_d2_d1_uint.dxil"),
            (0, 0, 2) => include_bytes!("../../assets/blit_cs_d2_d1_sint.dxil"),
            (0, 1, 0) => include_bytes!("../../assets/blit_cs_d2_d2_float.dxil"),
            (0, 1, 1) => include_bytes!("../../assets/blit_cs_d2_d2_uint.dxil"),
            (0, 1, 2) => include_bytes!("../../assets/blit_cs_d2_d2_sint.dxil"),
            (0, 2, 0) => include_bytes!("../../assets/blit_cs_d2_d3_float.dxil"),
            (0, 2, 1) => include_bytes!("../../assets/blit_cs_d2_d3_uint.dxil"),
            (0, 2, 2) => include_bytes!("../../assets/blit_cs_d2_d3_sint.dxil"),
            (1, 0, 0) => include_bytes!("../../assets/blit_cs_array_d1_float.dxil"),
            (1, 0, 1) => include_bytes!("../../assets/blit_cs_array_d1_uint.dxil"),
            (1, 0, 2) => include_bytes!("../../assets/blit_cs_array_d1_sint.dxil"),
            (1, 1, 0) => include_bytes!("../../assets/blit_cs_array_d2_float.dxil"),
            (1, 1, 1) => include_bytes!("../../assets/blit_cs_array_d2_uint.dxil"),
            (1, 1, 2) => include_bytes!("../../assets/blit_cs_array_d2_sint.dxil"),
            (1, 2, 0) => include_bytes!("../../assets/blit_cs_array_d3_float.dxil"),
            (1, 2, 1) => include_bytes!("../../assets/blit_cs_array_d3_uint.dxil"),
            (1, 2, 2) => include_bytes!("../../assets/blit_cs_array_d3_sint.dxil"),
            (2, 0, 0) => include_bytes!("../../assets/blit_cs_d1_d1_float.dxil"),
            (2, 0, 1) => include_bytes!("../../assets/blit_cs_d1_d1_uint.dxil"),
            (2, 0, 2) => include_bytes!("../../assets/blit_cs_d1_d1_sint.dxil"),
            (2, 1, 0) => include_bytes!("../../assets/blit_cs_d1_d2_float.dxil"),
            (2, 1, 1) => include_bytes!("../../assets/blit_cs_d1_d2_uint.dxil"),
            (2, 1, 2) => include_bytes!("../../assets/blit_cs_d1_d2_sint.dxil"),
            (2, 2, 0) => include_bytes!("../../assets/blit_cs_d1_d3_float.dxil"),
            (2, 2, 1) => include_bytes!("../../assets/blit_cs_d1_d3_uint.dxil"),
            (2, 2, 2) => include_bytes!("../../assets/blit_cs_d1_d3_sint.dxil"),
            (3, 0, 0) => include_bytes!("../../assets/blit_cs_d3_d1_float.dxil"),
            (3, 0, 1) => include_bytes!("../../assets/blit_cs_d3_d1_uint.dxil"),
            (3, 0, 2) => include_bytes!("../../assets/blit_cs_d3_d1_sint.dxil"),
            (3, 1, 0) => include_bytes!("../../assets/blit_cs_d3_d2_float.dxil"),
            (3, 1, 1) => include_bytes!("../../assets/blit_cs_d3_d2_uint.dxil"),
            (3, 1, 2) => include_bytes!("../../assets/blit_cs_d3_d2_sint.dxil"),
            (3, 2, 0) => include_bytes!("../../assets/blit_cs_d3_d3_float.dxil"),
            (3, 2, 1) => include_bytes!("../../assets/blit_cs_d3_d3_uint.dxil"),
            (3, 2, 2) => include_bytes!("../../assets/blit_cs_d3_d3_sint.dxil"),
            _ => return Err(Error::InvalidState),
        }
    };
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            ..Default::default()
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            ..Default::default()
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
            NumDescriptors: 1,
            ..Default::default()
        },
    ];
    let table = |range: &D3D12_DESCRIPTOR_RANGE| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let constants = |register, count| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: register,
                RegisterSpace: 0,
                Num32BitValues: count,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let params = [
        table(&ranges[0]),
        table(&ranges[1]),
        table(&ranges[2]),
        constants(0, 5),
        constants(1, 6),
    ];
    let root_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        ..Default::default()
    };
    let mut blob = None;
    // SAFETY: serialized root pointers remain live during this call.
    unsafe {
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
    }
    .map_err(native_error)?;
    let blob = blob.ok_or(Error::OutOfMemory)?;
    // SAFETY: the blob owns the returned bytes until this creation returns.
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    let device = &encoder.device.0.native;
    // SAFETY: serialized bytes are from D3D12SerializeRootSignature.
    let root = unsafe { device.CreateRootSignature::<ID3D12RootSignature>(0, bytes) }
        .map_err(native_error)?;
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: shader.as_ptr().cast(),
            BytecodeLength: shader.len(),
        },
        ..Default::default()
    };
    // SAFETY: shader and root signature remain valid throughout pipeline creation.
    let state = unsafe { device.CreateComputePipelineState::<ID3D12PipelineState>(&desc) };
    // SAFETY: release the temporary cloned root reference.
    unsafe { ManuallyDrop::drop(&mut desc.pRootSignature) };
    let state = state.map_err(native_error)?;
    let output_desc = encoder.device.0.descriptors.temporary_pair()?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let resource_desc = if !typed {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: buffer_size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        }
    } else {
        D3D12_RESOURCE_DESC {
            Dimension: match dst.desc.dimension {
                api::TextureDimension::D1 => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
                api::TextureDimension::D2 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                api::TextureDimension::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            },
            Width: u64::from(target.extent[0]),
            Height: target.extent[1],
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: uav_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        }
    };
    let mut output = None;
    // SAFETY: creation descriptors and output slot remain valid for this call.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut output,
        )
    }
    .map_err(native_error)?;
    let output: ID3D12Resource = output.ok_or(Error::OutOfMemory)?;
    let uav = if !typed {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: u32::try_from(buffer_size / 4).map_err(|_| Error::InvalidRange)?,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        }
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: uav_format,
            ViewDimension: match target_view {
                0 => D3D12_UAV_DIMENSION_TEXTURE1D,
                1 => D3D12_UAV_DIMENSION_TEXTURE2D,
                _ => D3D12_UAV_DIMENSION_TEXTURE3D,
            },
            Anonymous: match target_view {
                0 => D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture1D: D3D12_TEX1D_UAV { MipSlice: 0 },
                },
                1 => D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
                _ => D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_UAV {
                        MipSlice: 0,
                        FirstWSlice: 0,
                        WSize: 1,
                    },
                },
            },
        }
    };
    // SAFETY: the output and descriptor slot are retained until GPU completion.
    unsafe {
        device.CreateUnorderedAccessView(
            &output,
            None::<&ID3D12Resource>,
            Some(&uav),
            output_desc.view_cpu(),
        );
    }
    // Reserve every per-layer slot before recording commands. A later allocation
    // failure must not leave earlier dispatches referencing released objects.
    let source_descriptors = (0..target.extent[2])
        .map(|layer| {
            // Exercise allocation failure before recording without exhausting the heap.
            #[cfg(feature = "hardware-tests")]
            if std::env::var("FLUXEL_DX12_TEST_BLIT_FAIL_LAYER")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                == Some(layer)
            {
                return Err(Error::OutOfMemory);
            }
            let _ = layer;
            encoder.device.0.descriptors.temporary_pair()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_gpu = output_desc.view_gpu();
    encoder.mip_blit_descriptors.push(output_desc);
    encoder.uploads.push(output.clone());
    encoder
        .blit_compute_objects
        .push((root.clone(), state.clone()));
    let source_size = [
        (src.desc.extent[0] >> source.mip).max(1),
        (src.desc.extent[1] >> source.mip).max(1),
    ];
    for (layer, source_desc) in (0..target.extent[2]).zip(source_descriptors) {
        let source_layer = if source_view == 3 {
            0
        } else {
            source.origin[2] + layer
        };
        let source_z = if source_view == 3 {
            (source.origin[2] as f32
                + (layer as f32 + 0.5) * source.extent[2] as f32 / target.extent[2] as f32)
                / (src.desc.extent[2] >> source.mip).max(1) as f32
        } else {
            0.0
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: super::super::texture::dxgi_format(src.desc.format),
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ViewDimension: match source_view {
                0 => D3D12_SRV_DIMENSION_TEXTURE2D,
                1 => D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                2 => D3D12_SRV_DIMENSION_TEXTURE1DARRAY,
                _ => D3D12_SRV_DIMENSION_TEXTURE3D,
            },
            Anonymous: match source_view {
                0 => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: source.mip,
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
                1 => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: source.mip,
                        MipLevels: 1,
                        FirstArraySlice: source_layer,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
                2 => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture1DArray: D3D12_TEX1D_ARRAY_SRV {
                        MostDetailedMip: source.mip,
                        MipLevels: 1,
                        FirstArraySlice: source_layer,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
                _ => D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture3D: D3D12_TEX3D_SRV {
                        MostDetailedMip: source.mip,
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            },
        };
        // SAFETY: each temporary sampler slot is retained through GPU completion.
        unsafe {
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
                    ComparisonFunc: D3D12_COMPARISON_FUNC_NONE,
                    MaxLOD: 0.0,
                    ..Default::default()
                },
                source_desc.sampler_cpu(),
            );
        }
        // Each iteration needs independent descriptors because GPU execution is deferred.
        // SAFETY: descriptors refer to retained resources with validated subresources.
        unsafe {
            device.CreateShaderResourceView(&src.resource, Some(&srv), source_desc.view_cpu());
        }
        let source_gpu = source_desc.view_gpu();
        let sampler_gpu = source_desc.sampler_gpu();
        encoder.mip_blit_descriptors.push(source_desc);
        let source_sub = source.mip
            + if source_view == 3 {
                0
            } else {
                source_layer * src.desc.mip_levels
            };
        barrier_sub(
            &encoder.list,
            &src.resource,
            source_sub,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let uv = [
            source.origin[0] as f32 / source_size[0] as f32,
            source.origin[1] as f32 / source_size[1] as f32,
            source.extent[0] as f32 / source_size[0] as f32,
            source.extent[1] as f32 / source_size[1] as f32,
            source_z,
        ];
        let sizes = [
            target.extent[0],
            target.extent[1],
            u32::from(srgb),
            row_pitch,
            format_code,
            pixel_bytes,
        ];
        // SAFETY: root arguments and descriptors remain live through submission.
        unsafe {
            encoder.list.SetComputeRootSignature(&root);
            encoder.list.SetPipelineState(&state);
            encoder.list.SetComputeRootDescriptorTable(0, source_gpu);
            encoder.list.SetComputeRootDescriptorTable(1, output_gpu);
            encoder.list.SetComputeRootDescriptorTable(2, sampler_gpu);
            encoder
                .list
                .SetComputeRoot32BitConstants(3, 5, uv.as_ptr().cast(), 0);
            encoder
                .list
                .SetComputeRoot32BitConstants(4, 6, sizes.as_ptr().cast(), 0);
            encoder.list.Dispatch(
                if typed {
                    target.extent[0].div_ceil(8)
                } else {
                    target.extent[0]
                        .div_ceil((4 / pixel_bytes).max(1))
                        .div_ceil(8)
                },
                target.extent[1].div_ceil(8),
                1,
            );
        }
        barrier_sub(
            &encoder.list,
            &src.resource,
            source_sub,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        barrier_sub(
            &encoder.list,
            &output,
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let mut from = copy_location(
            &output,
            if typed {
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default()
            } else {
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::super::texture::dxgi_format(dst.desc.format),
                        Width: target.extent[0],
                        Height: target.extent[1],
                        Depth: 1,
                        RowPitch: row_pitch,
                    },
                }
            },
            !typed,
            0,
        );
        let target_sub = target.mip
            + if target_view == 2 {
                0
            } else {
                (target.origin[2] + layer) * dst.desc.mip_levels
            };
        let mut to = copy_location(
            &dst.resource,
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(),
            false,
            target_sub,
        );
        // SAFETY: both resources are in copy states; the target region was validated.
        unsafe {
            encoder.list.CopyTextureRegion(
                &to,
                target.origin[0],
                target.origin[1],
                if target_view == 2 {
                    target.origin[2] + layer
                } else {
                    0
                },
                &from,
                None,
            );
            ManuallyDrop::drop(&mut from.pResource);
            ManuallyDrop::drop(&mut to.pResource);
        }
        if layer + 1 < target.extent[2] {
            barrier_sub(
                &encoder.list,
                &output,
                0,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
        }
    }
    super::super::compute::restore_compute_state(encoder)?;
    Ok(())
}

fn buffer_format_code(format: api::TextureFormat) -> Result<u32, Error> {
    use api::TextureFormat::*;
    let (bits, count, kind, bgra) = match format {
        R8Unorm => (8, 1, 0, false),
        R8Snorm => (8, 1, 1, false),
        R8Uint => (8, 1, 2, false),
        R8Sint => (8, 1, 3, false),
        Rg8Unorm => (8, 2, 0, false),
        Rg8Snorm => (8, 2, 1, false),
        Rg8Uint => (8, 2, 2, false),
        Rg8Sint => (8, 2, 3, false),
        Rgba8Unorm => (8, 4, 0, false),
        Rgba8Snorm => (8, 4, 1, false),
        Rgba8Uint => (8, 4, 2, false),
        Rgba8Sint => (8, 4, 3, false),
        R16Unorm => (16, 1, 0, false),
        R16Snorm => (16, 1, 1, false),
        R16Uint => (16, 1, 2, false),
        R16Sint => (16, 1, 3, false),
        R16Float => (16, 1, 2, false),
        Rg16Unorm => (16, 2, 0, false),
        Rg16Snorm => (16, 2, 1, false),
        Rg16Uint => (16, 2, 2, false),
        Rg16Sint => (16, 2, 3, false),
        Rg16Float => (16, 2, 2, false),
        Rgba16Unorm => (16, 4, 0, false),
        Rgba16Snorm => (16, 4, 1, false),
        Rgba16Uint => (16, 4, 2, false),
        Rgba16Sint => (16, 4, 3, false),
        Rgba16Float => (16, 4, 2, false),
        R32Uint => (32, 1, 2, false),
        R32Sint => (32, 1, 3, false),
        R32Float => (32, 1, 2, false),
        Rg32Uint => (32, 2, 2, false),
        Rg32Sint => (32, 2, 3, false),
        Rg32Float => (32, 2, 2, false),
        Rgba32Uint => (32, 4, 2, false),
        Rgba32Sint => (32, 4, 3, false),
        Rgba32Float => (32, 4, 2, false),
        Rgba8UnormSrgb => (8, 4, 0, false),
        Bgra8Unorm | Bgra8UnormSrgb => (8, 4, 0, true),
        Rgb10a2Unorm | Rgb10a2Uint => (10, 4, 4, false),
        Rg11b10Ufloat => (11, 3, 5, false),
        Rgb9e5Ufloat => (9, 3, 6, false),
        _ => {
            return Err(Error::Unsupported(
                "DX12 compute blit packed float buffer format",
            ));
        }
    };
    Ok(bits | (count << 8) | (kind << 16) | (u32::from(bgra) << 24))
}
