//! Focused real-GPU operations output assertions.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_partial_attachment_clear_preserves_pixels_outside_area() {
    let d = open().await;
    let t = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &t,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    for (area, color) in [
        (
            Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            [0.0, 0.0, 1.0, 1.0],
        ),
        (
            Rect {
                x: 2,
                y: 3,
                width: 3,
                height: 2,
            },
            [1.0, 0.0, 0.0, 1.0],
        ),
    ] {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float(color)),
            store: StoreOp::Store,
        })];
        e.begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: area,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap()
        .finish()
        .unwrap();
    }
    let output = buffer(
        &d,
        8 * 256,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    e.texture_to_buffer(
        TextureRegion {
            texture: &t,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..8 * 256).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the texture copy and readback ticket precede the awaited point.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for y in 0..8 {
        for x in 0..8 {
            let expected = if (2..5).contains(&x) && (3..5).contains(&y) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            assert_eq!(
                &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4],
                &expected,
                "pixel {x},{y}"
            );
        }
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_compressed_blocks_upload_and_copy_without_byte_changes() {
    let d = open().await;
    let formats = [
        (TextureFormat::Bc1RgbaUnorm, 8usize),
        (TextureFormat::Etc2Rgb8Unorm, 8),
        (
            TextureFormat::Astc {
                block: [4, 4],
                srgb: false,
                hdr: false,
            },
            16,
        ),
    ];
    for (format, block_bytes) in formats {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        };
        if !matches!(d.texture_support(&desc), Support::Supported(_)) {
            console_log!("SKIP compressed {:?}: adapter format capability", format);
            continue;
        }
        let t = d.create_texture(&desc).unwrap();
        let out = buffer(
            &d,
            256,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            MemoryClass::Device,
        );
        let bytes: Vec<u8> = (0..block_bytes)
            .map(|n| (n as u8).wrapping_mul(17))
            .collect();
        let mut e = d.encoder(d.lanes()[0].id).unwrap();
        e.write_texture(
            TextureRegion {
                texture: &t,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: block_bytes as u32,
                rows_per_image: 4,
            },
            &bytes,
        )
        .unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &t,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 4,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&out, 0..block_bytes as u64).unwrap();
        let point = submit(&d, e).await;
        // SAFETY: the point covers the compressed copy and readback.
        assert_eq!(
            unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
            bytes,
            "{:?}",
            format
        );
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_compute_constants_and_indirect_dispatch_change_output() {
    let d = open().await;
    let out = buffer(
        &d,
        256,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let indirect = buffer(
        &d,
        16,
        BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 4,
                    access: StorageAccess::ReadWrite,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &out,
                    range: 0..4,
                },
            }],
        })
        .unwrap();
    let pipeline_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let source = "override factor:u32=2u; @group(0) @binding(0) var<storage,read_write> output:atomic<u32>; @compute @workgroup_size(1) fn cs(){ atomicAdd(&output,factor); }";
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        })
        .await
        .unwrap();
    let compute = d.compute().unwrap();
    let pipeline = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout: pipeline_layout,
            compute: PipelineStage {
                module: shader,
                entry: "cs".into(),
                constants: vec![("factor".into(), SpecializationValue::U32(7))],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(&out, 0, &0u32.to_le_bytes()).unwrap();
    e.write_buffer(
        &indirect,
        0,
        &[3u32, 1, 1]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut pass = compute.begin(&mut e).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch_indirect(&indirect, 0).unwrap();
    pass.finish().unwrap();
    let ticket = e.readback_buffer(&out, 0..4).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the awaited point covers the indirect dispatch and readback.
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        21u32.to_le_bytes()
    );
}

#[wasm_bindgen_test(async)]
async fn gpu_blit_integer_depth_and_volume_paths_produce_expected_texels() {
    let d = open().await;
    for (format, dimension, source_extent, target_extent, aspect, filter) in [
        (
            TextureFormat::Rgba8Uint,
            TextureDimension::D2,
            [4, 4, 1],
            [2, 2, 1],
            TextureAspects::COLOR,
            FilterMode::Nearest,
        ),
        (
            TextureFormat::Depth32Float,
            TextureDimension::D2,
            [4, 4, 1],
            [2, 2, 1],
            TextureAspects::DEPTH,
            FilterMode::Nearest,
        ),
        (
            TextureFormat::Rgba8Unorm,
            TextureDimension::D3,
            [4, 4, 4],
            [2, 2, 2],
            TextureAspects::COLOR,
            FilterMode::Linear,
        ),
    ] {
        let usage = TextureUsage::COPY_DST | TextureUsage::COPY_SRC | TextureUsage::TEXTURE_BINDING;
        let src = d
            .create_texture(&TextureDesc {
                label: None,
                dimension,
                extent: source_extent,
                array_layers: 1,
                format,
                mip_levels: 1,
                samples: 1,
                usage: if aspect == TextureAspects::DEPTH {
                    usage | TextureUsage::RENDER_ATTACHMENT
                } else {
                    usage
                },
                view_formats: vec![],
            })
            .unwrap();
        let dst = d
            .create_texture(&TextureDesc {
                label: None,
                dimension,
                extent: target_extent,
                array_layers: 1,
                format,
                mip_levels: 1,
                samples: 1,
                usage,
                view_formats: vec![],
            })
            .unwrap();
        let expected = if aspect == TextureAspects::DEPTH {
            0.25f32.to_le_bytes()
        } else {
            [64, 128, 192, 255]
        };
        let mut e = d.encoder(d.lanes()[0].id).unwrap();
        if aspect == TextureAspects::DEPTH {
            e.clear_texture(&src, 0..1, 0..1, TextureClearValue::Depth(0.25))
                .unwrap();
        } else {
            let data =
                expected.repeat((source_extent[0] * source_extent[1] * source_extent[2]) as usize);
            e.write_texture(
                TextureRegion {
                    texture: &src,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: source_extent,
                    aspect,
                },
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: source_extent[0] * 4,
                    rows_per_image: source_extent[1],
                },
                &data,
            )
            .unwrap();
        }
        e.blit(
            TextureRegion {
                texture: &src,
                mip: 0,
                origin: [0, 0, 0],
                extent: source_extent,
                aspect,
            },
            TextureRegion {
                texture: &dst,
                mip: 0,
                origin: [0, 0, 0],
                extent: target_extent,
                aspect,
            },
            filter,
        )
        .unwrap();
        let len = u64::from(target_extent[1] * target_extent[2]) * 256;
        let output = buffer(
            &d,
            len,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            MemoryClass::Device,
        );
        e.texture_to_buffer(
            TextureRegion {
                texture: &dst,
                mip: 0,
                origin: [0, 0, 0],
                extent: target_extent,
                aspect,
            },
            &output,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: target_extent[1],
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&output, 0..len).unwrap();
        let point = submit(&d, e).await;
        // SAFETY: the submitted recording covers the blit and readback.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        for row in 0..(target_extent[1] * target_extent[2]) as usize {
            for x in 0..target_extent[0] as usize {
                assert_eq!(
                    &bytes[row * 256 + x * 4..row * 256 + x * 4 + 4],
                    &expected,
                    "{:?} row {row} x {x}",
                    format
                );
            }
        }
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_immediate_arrays_alignment_and_pointer_helpers_keep_byte_layout() {
    let d = open().await;
    if !d.capabilities().push_constants || d.capabilities().limits.max_push_constant_bytes < 64 {
        console_log!("SKIP immediate address space/native byte budget");
        return;
    }
    let out = buffer(
        &d,
        256,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 4,
                    access: StorageAccess::ReadWrite,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &out,
                    range: 0..4,
                },
            }],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Compute],
                range: 0..64,
            }],
        })
        .unwrap();
    let source = "struct Inner { @align(32) value:u32 } struct Push { values:array<u32,4>, inner:Inner } var<push_constant> push:Push; @group(0) @binding(0) var<storage,read_write> output:atomic<u32>; fn identity(push:u32)->u32{return push;} fn read(p:ptr<push_constant,u32>)->u32{return *p;} @compute @workgroup_size(4) fn cs(@builtin(local_invocation_index) i:u32){atomicAdd(&output, identity(push.values[i]) + read(&push.inner.value));}";
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([4, 1, 1]),
            }],
            required_features: vec![],
        })
        .await
        .unwrap();
    let compute = d.compute().unwrap();
    let pipeline = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let mut data = [0u8; 64];
    for (i, v) in [1u32, 2, 3, 4].into_iter().enumerate() {
        data[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    data[32..36].copy_from_slice(&17u32.to_le_bytes());
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(&out, 0, &0u32.to_le_bytes()).unwrap();
    let mut pass = compute.begin(&mut e).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.push_constants(0, &data).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    let ticket = e.readback_buffer(&out, 0..4).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the dispatch consuming these immediate bytes precedes the point.
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        78u32.to_le_bytes()
    );
}
