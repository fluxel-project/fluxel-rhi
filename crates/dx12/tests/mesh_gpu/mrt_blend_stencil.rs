//! Mesh MRT blend and stencil control GPU readback.

use super::*;

#[test]
fn mesh_mrt_blend_and_stencil_control_readback() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !d.capabilities().mesh {
        eprintln!("SKIP: mesh shader tier unavailable");
        return;
    }
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let shader = |bytes: &'static [u8], name: &str, stage| {
        block_on(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Dxil,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: name.into(),
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        }))
        .unwrap()
    };
    let vs = shader(
        include_bytes!("../assets/mesh/mesh.dxil"),
        "main",
        ShaderStage::Mesh,
    );
    let ps = shader(
        include_bytes!("../assets/mesh/pixel.dxil"),
        "main",
        ShaderStage::Fragment,
    );
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let face = StencilFaceState {
        compare: CompareFunction::Equal,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let mesh = d.mesh().unwrap();
    let cache = d.create_pipeline_cache(None).unwrap();
    let mesh_desc = |cache| MeshPipelineDesc {
        layout: layout.clone(),
        task: None,
        mesh: PipelineStage {
            module: vs.clone(),
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps.clone(),
            entry: "main".into(),
            constants: vec![],
        }),
        targets: vec![
            Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                }),
                write_mask: 15,
            }),
            Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 0,
            }),
        ],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 255,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        cache: Some(cache),
    };
    block_on(mesh.create_pipeline(mesh_desc(cache.clone()))).unwrap();
    let serialized = d.serialize_pipeline_cache(&cache).unwrap();
    assert!(
        serialized.len() > 16,
        "native mesh PSO blob must be recorded"
    );
    let restored = d.create_pipeline_cache(Some(&serialized)).unwrap();
    let pipeline = block_on(mesh.create_pipeline(mesh_desc(restored))).unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 4,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 2,
            format: TextureFormat::Depth24PlusStencil8,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let views = |texture: &Texture<_>, format, aspect| {
        (0..4)
            .map(|layer| {
                d.create_view(
                    texture,
                    &TextureViewDesc {
                        format,
                        dimension: TextureViewDimension::D2Array,
                        aspects: aspect,
                        mips: 0..1,
                        layers: layer..layer + 1,
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let colors = views(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let depths = (0..2)
        .map(|layer| {
            d.create_view(
                &depth,
                &TextureViewDesc {
                    format: TextureFormat::Depth24PlusStencil8,
                    dimension: TextureViewDimension::D2Array,
                    aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                    mips: 0..1,
                    layers: layer..layer + 1,
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16384,
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })
        .unwrap();
    // A pipeline requiring depth cannot bind into a color-only pass.
    {
        let mut invalid = d.encoder(lane).unwrap();
        let color_attachment = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[0]),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let mut pass = invalid
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &color_attachment,
                depth_stencil: None,
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        assert!(matches!(
            mesh.bind_pipeline(&mut pass, &pipeline),
            Err(Error::InvalidState)
        ));
    }
    let mut e = d.encoder(lane).unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to, layouts| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &color,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..4,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    for layer in 0..2 {
        let attachments = [
            Some(ColorAttachment {
                view: AttachmentView::Texture(&colors[layer]),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.25, 0.0, 0.0, 1.0])),
                store: StoreOp::Store,
            }),
            Some(ColorAttachment {
                view: AttachmentView::Texture(&colors[layer + 2]),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0, 1.0, 0.0, 1.0])),
                store: StoreOp::Store,
            }),
        ];
        let depth_attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(&depths[layer]),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
            stencil: Some((LoadOp::Clear(layer as u32), StoreOp::Store)),
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &attachments,
                depth_stencil: Some(depth_attachment),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: 16.0,
            height: 16.0,
            min_depth: 0.0,
            max_depth: 1.0,
        })
        .unwrap();
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .unwrap();
        pass.stencil_reference(1).unwrap();
        mesh.bind_pipeline(&mut pass, &pipeline).unwrap();
        mesh.dispatch(&mut pass, [1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    for layer in 0..4 {
        e.texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, layer],
                extent: [16, 16, 1],
                aspect: TextureAspects::COLOR,
            },
            &dst,
            TexelCopyBufferLayout {
                offset: u64::from(layer) * 4096,
                bytes_per_row: 256,
                rows_per_image: 16,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&dst, 0..16384).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755321,
    };
    // SAFETY: the one-step plan uses this device's graphics lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the work is consumed once on its recording lane.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the awaited completion point covers the readback.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    let center = 8 * 256 + 8 * 4;
    assert_eq!(&bytes[center..center + 4], &[64, 0, 0, 255]);
    assert_eq!(&bytes[4096 + center..4096 + center + 4], &[255, 0, 0, 255]);
    for layer in 2..4 {
        assert_eq!(
            &bytes[layer * 4096 + center..layer * 4096 + center + 4],
            &[0, 255, 0, 255]
        );
    }
}
