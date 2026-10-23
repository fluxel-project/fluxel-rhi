//! Desktop depth CPU and PBO uploads with exact mip readback.
use super::*;

pub(super) async fn depth_pbo_upload(d: &Device<Gl>, format: TextureFormat, cpu: bool) {
    let depth32 = format == TextureFormat::Depth32Float;
    let mip = u32::from(cpu);
    let volume = TextureDesc {
        label: None,
        dimension: TextureDimension::D3,
        extent: [4, 4, 2],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    assert!(
        matches!(d.texture_support(&volume), Support::Unsupported(_)),
        "GL depth textures must reject a 3D target: {format:?}"
    );
    let (upload_offset, bytes_per_row) = if depth32 { (4, 16) } else { (6, 8) };
    let input = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let support_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    assert!(matches!(
        d.texture_support(&support_desc),
        Support::Supported(_)
    ));

    let source = br#"
@group(0) @binding(0) var image: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f {
 let first_row=textureLoad(image,vec2i(0,0),0);
 let second_row=textureLoad(image,vec2i(0,1),0);
 return vec4f(first_row,second_row,0.,1.);
}
"#;
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_slice()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Depth,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
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
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
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
    let input_view = d
        .create_view(
            &input,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: mip..mip + 1,
                layers: 0..1,
            },
        )
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&input_view),
            }],
        })
        .unwrap();
    let output = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let output_view = d
        .create_view(
            &output,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let upload = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let read = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let depth_read = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    // Nonzero buffer offset, row padding, and asymmetric texels distinguish
    // both uploaded rows from untouched storage.
    let mut bytes = [0xadu8; 32];
    if depth32 {
        for (row, values) in [[0.25f32, 1.0f32], [0.0, 1.0]].into_iter().enumerate() {
            let start = upload_offset as usize + row * bytes_per_row as usize;
            bytes[start..start + 4].copy_from_slice(&values[0].to_ne_bytes());
            bytes[start + 4..start + 8].copy_from_slice(&values[1].to_ne_bytes());
        }
    } else {
        for (row, values) in [[0x4000u16, 0xffffu16], [0x0000, 0xffff]]
            .into_iter()
            .enumerate()
        {
            let start = upload_offset as usize + row * bytes_per_row as usize;
            bytes[start..start + 2].copy_from_slice(&values[0].to_ne_bytes());
            bytes[start + 2..start + 4].copy_from_slice(&values[1].to_ne_bytes());
        }
    }
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.; 4])),
        store: StoreOp::Store,
    })];
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let rejected_plan = common::single_commands_plan(&fixture, 519).unwrap();
    // SAFETY: the resources remain alive while validating the rejected aspect.
    let mut rejected_session =
        unsafe { common::begin_commands(&fixture, rejected_plan, CompletionRequest::End) }.unwrap();
    let mut rejected_run = rejected_session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin invalid depth PBO upload"));
    assert!(matches!(
        rejected_run.commands().buffer_to_texture(
            &upload,
            TexelCopyBufferLayout {
                offset: upload_offset,
                bytes_per_row,
                rows_per_image: 2,
            },
            TextureRegion {
                texture: &input,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
        ),
        Err(Error::Unsupported(_))
    ));
    drop(rejected_run);
    drop(rejected_session);

    let plan = common::single_commands_plan(&fixture, 520).unwrap();
    // SAFETY: every resource remains live through the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin depth PBO upload"));
    let e = run.commands();
    e.write_buffer(&upload, 0, &bytes).unwrap();
    let upload_layout = TexelCopyBufferLayout {
        offset: upload_offset,
        bytes_per_row,
        rows_per_image: 2,
    };
    let region = TextureRegion {
        texture: &input,
        mip,
        origin: [0, 0, 0],
        extent: [2, 2, 1],
        aspect: TextureAspects::DEPTH,
    };
    if cpu {
        e.write_texture(region, upload_layout, &bytes).unwrap();
    } else {
        e.buffer_to_texture(&upload, upload_layout, region).unwrap();
    }
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &output,
            mip: 0,
            origin: [0, 0, 0],
            extent: [1, 1, 1],
            aspect: TextureAspects::COLOR,
        },
        &read,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 4,
            rows_per_image: 1,
        },
    )
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &input,
            mip,
            origin: [0, 0, 0],
            extent: [2, 2, 1],
            aspect: TextureAspects::DEPTH,
        },
        &depth_read,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row,
            rows_per_image: 2,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&read, 0..4).unwrap();
    let depth_ticket = e.readback_buffer(&depth_read, 0..32).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish depth PBO upload"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the point covers the PBO upload, sample, render, and readback.
    let actual = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(actual, [64, 0, 0, 255]);
    let depth = unsafe { d.read_after(depth_ticket, &point).await }
        .unwrap()
        .bytes;
    if depth32 {
        assert_eq!(
            &depth[0..8],
            &[0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x80, 0x3f]
        );
        assert_eq!(
            &depth[16..24],
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f]
        );
    } else {
        assert_eq!(&depth[0..4], &[0x00, 0x40, 0xff, 0xff]);
        assert_eq!(&depth[8..12], &[0x00, 0x00, 0xff, 0xff]);
    }
    if cpu {
        for aspect in [TextureAspects::COLOR, TextureAspects::STENCIL] {
            let plan = common::single_commands_plan(&fixture, 237).unwrap();
            // SAFETY: the context and texture remain alive for this rejected scope.
            let mut session =
                unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
            let mut run = session
                .begin_commands()
                .unwrap_or_else(|_| panic!("begin invalid depth upload"));
            let result = run.commands().write_texture(
                TextureRegion {
                    texture: &input,
                    mip,
                    origin: [0, 0, 0],
                    extent: [2, 2, 1],
                    aspect,
                },
                upload_layout,
                &bytes,
            );
            assert!(
                matches!(result, Err(Error::Unsupported(_))),
                "{format:?} {aspect:?}: {result:?}"
            );
        }
    }
    eprintln!(
        "EXECUTED {format:?}: {} upload at mip {mip}, padded rows, sample and exact depth readback",
        if cpu { "CPU" } else { "PBO" }
    );
}
