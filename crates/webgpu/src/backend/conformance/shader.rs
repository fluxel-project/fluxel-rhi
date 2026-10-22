//! Focused shader ABI and subgroup-size GPU checks.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_scalar_push_pointer_helper_reads_written_value() {
    let d = open().await;
    if !d.capabilities().push_constants || d.capabilities().limits.max_push_constant_bytes < 4 {
        console_log!("SKIP immediate address space unavailable");
        return;
    }
    let out = buffer(
        &d,
        256,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
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
                range: 0..4,
            }],
        })
        .unwrap();
    let source = "var<push_constant> p:u32; @group(0) @binding(0) var<storage,read_write> out:u32; fn read_value(q:ptr<push_constant,u32>)->u32{return *q;} @compute @workgroup_size(1) fn cs(){out=read_value(&p);}";
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
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    let mut pass = compute.begin(&mut e).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.push_constants(0, &12345u32.to_le_bytes()).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    let ticket = e.readback_buffer(&out, 0..4).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the completion point covers the compute dispatch and readback.
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        12345u32.to_le_bytes()
    );
}

#[wasm_bindgen_test(async)]
async fn gpu_required_subgroup_size_matches_shader_builtin() {
    let gpu = super::super::js::browser_gpu().unwrap();
    let adapter = wasm_bindgen_futures::JsFuture::from(
        super::super::js::request_adapter(&gpu, None).unwrap(),
    )
    .await
    .unwrap();
    if !super::super::js::has_feature(&adapter, "subgroup-size-control") {
        console_log!("SKIP subgroup-size-control unavailable");
        return;
    }
    let info = super::super::js::property(&adapter, "info").unwrap();
    let size = super::super::js::property(&info, "subgroupMinSize")
        .unwrap()
        .as_f64()
        .unwrap() as u32;
    let d = open().await;
    let out = buffer(
        &d,
        256,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
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
            push_constants: &[],
        })
        .unwrap();
    let source = format!(
        "enable subgroups; @group(0) @binding(0) var<storage,read_write> out:atomic<u32>; @compute @workgroup_size({size}) fn cs(@builtin(subgroup_size) s:u32){{ atomicStore(&out,s); }}"
    );
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([size, 1, 1]),
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
            required_subgroup_size: Some(size),
            cache: None,
        })
        .await
        .unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    let mut pass = compute.begin(&mut e).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    let ticket = e.readback_buffer(&out, 0..4).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the completion point covers the compute dispatch and readback.
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        size.to_le_bytes()
    );
}

#[wasm_bindgen_test(async)]
async fn gpu_multi_draw_count_keeps_attachment_queries_and_render_state() {
    let d = open().await;
    assert!(d.capabilities().multi_draw_count);
    let source = "struct O { @builtin(position) p:vec4f,@location(0) @interpolate(flat) id:u32 } @vertex fn vs(@builtin(vertex_index) i:u32)->O { let id=i/3u; let j=i%3u; let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); var o:O; o.p=vec4f((p[j].x+1.)/3.-1.+f32(id)*2./3.,p[j].y,0.5,1.); o.id=id; return o; } @fragment fn fs(v:O)->@location(0) vec4f { if v.id==0u { return vec4f(1.,0.,0.,1.); } if v.id==1u { return vec4f(0.,1.,0.,1.); } return vec4f(0.,0.,1.,1.); }";
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![
                ShaderEntryPoint {
                    name: "vs".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "fs".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: vec![],
        })
        .await
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 15,
            })],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u64::MAX,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [9, 6, 1],
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
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let args = buffer(
        &d,
        32,
        BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let counts = buffer(
        &d,
        8,
        BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let queries = d.queries(QueryKind::Occlusion).unwrap();
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1,
        })
        .unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.write_buffer(
        &args,
        0,
        &[3u32, 1, 0, 0, 3, 1, 3, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    e.write_buffer(
        &counts,
        0,
        &[1u32, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 9,
                height: 6,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: Some(&set),
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    queries.begin_occlusion(&mut pass, &set, 0).unwrap();
    let extras = d.raster_extras().unwrap();
    extras
        .multi_draw_count(&mut pass, &args, 0, &counts, 0, 2, 16, false)
        .unwrap();
    extras
        .multi_draw_count(&mut pass, &args, 0, &counts, 4, 2, 16, false)
        .unwrap();
    pass.draw(6..9, 0..1).unwrap();
    queries.end_occlusion(&mut pass).unwrap();
    pass.finish().unwrap();
    let image = buffer(
        &d,
        1536,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    e.texture_to_buffer(
        TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [9, 6, 1],
            aspect: TextureAspects::COLOR,
        },
        &image,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 6,
        },
    )
    .unwrap();
    let image_ticket = e.readback_buffer(&image, 0..1536).unwrap();
    let result = buffer(
        &d,
        256,
        BufferUsage::QUERY_RESOLVE | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    queries.resolve(&mut e, &set, 0..1, &result, 0).unwrap();
    let query_ticket = e.readback_buffer(&result, 0..8).unwrap();
    let indexed_args = buffer(
        &d,
        48,
        BufferUsage::INDIRECT | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let indices = buffer(
        &d,
        24,
        BufferUsage::INDEX | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    e.write_buffer(
        &indexed_args,
        0,
        &[3u32, 1, 0, 0, 0, 99, 3, 1, 3, 0, 0, 99]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    e.write_buffer(
        &indices,
        0,
        &[0u32, 1, 2, 3, 4, 5]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    e.write_buffer(&counts, 0, &3u32.to_le_bytes()).unwrap();
    let mut indexed_pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 9,
                height: 6,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    indexed_pass.set_pipeline(&pipeline).unwrap();
    indexed_pass
        .set_index_buffer(&indices, 0..24, IndexFormat::Uint32)
        .unwrap();
    extras
        .multi_draw_count(&mut indexed_pass, &indexed_args, 0, &counts, 0, 2, 24, true)
        .unwrap();
    indexed_pass.finish().unwrap();
    let indexed_image = buffer(
        &d,
        1536,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    e.texture_to_buffer(
        TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [9, 6, 1],
            aspect: TextureAspects::COLOR,
        },
        &indexed_image,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 6,
        },
    )
    .unwrap();
    let indexed_ticket = e.readback_buffer(&indexed_image, 0..1536).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the point covers render, query reduction, and copies.
    let pixels = unsafe { d.read_after(image_ticket, &point).await }
        .unwrap()
        .bytes;
    for (x, expected) in [
        (1, [255, 0, 0, 255]),
        (4, [0, 0, 0, 255]),
        (7, [0, 0, 255, 255]),
    ] {
        assert_eq!(
            &pixels[2 * 256 + x * 4..2 * 256 + x * 4 + 4],
            &expected,
            "pixel {x}"
        );
    }
    // SAFETY: the same point covers query reduction and readback.
    let raw = unsafe { d.read_after(query_ticket, &point).await }
        .unwrap()
        .bytes;
    assert!(u64::from_le_bytes(raw.try_into().unwrap()) > 0);
    // SAFETY: the indexed pass and copy precede the completion point.
    let indexed_pixels = unsafe { d.read_after(indexed_ticket, &point).await }
        .unwrap()
        .bytes;
    for (x, expected) in [
        (1, [255, 0, 0, 255]),
        (4, [0, 255, 0, 255]),
        (7, [0, 0, 0, 255]),
    ] {
        assert_eq!(
            &indexed_pixels[2 * 256 + x * 4..2 * 256 + x * 4 + 4],
            &expected,
            "indexed pixel {x}"
        );
    }
}
