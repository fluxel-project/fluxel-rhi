//! Sparse render target and depth attachment GPU coverage.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(f: F) -> F::Output {
    let w = std::task::Waker::noop();
    let mut c = std::task::Context::from_waker(w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut c) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn sparse_render_and_depth_control_readback() {
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
        include_bytes!("assets/mesh/mesh.dxil"),
        "main",
        ShaderStage::Mesh,
    );
    let ps = shader(
        include_bytes!("assets/mesh/pixel.dxil"),
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
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let mesh = d.mesh().unwrap();
    let pipeline = block_on(mesh.create_pipeline(MeshPipelineDesc {
        layout,
        task: None,
        mesh: PipelineStage {
            module: vs,
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "main".into(),
            constants: vec![],
        }),
        targets: vec![Some(ColorTargetState {
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
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 0,
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
        cache: None,
    }))
    .unwrap();
    let sparse = d.sparse().expect("sparse capability");
    let color = sparse
        .texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [256, 256, 1],
            array_layers: 2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        })
        .unwrap();
    let depth = sparse
        .texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [256, 256, 1],
            array_layers: 2,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let views = |texture: &Texture<_>, format, aspect| {
        (0..2)
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
    let depths = views(&depth, TextureFormat::Depth32Float, TextureAspects::DEPTH);
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8192,
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
                layers: 0..2,
            },
            usage: StateTransition { from, to },
            layout: Some(layouts),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Undefined, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    // Newly allocated DX12 depth resources start in DEPTH_WRITE.
    e.clear_texture(&depth, 0..1, 0..2, TextureClearValue::Depth(0.0))
        .unwrap();
    e.clear_texture(&depth, 0..1, 1..2, TextureClearValue::Depth(1.0))
        .unwrap();
    for layer in 0..2 {
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&colors[layer]),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let depth_attachment = DepthStencilAttachment {
            view: AttachmentView::Texture(&depths[layer]),
            depth: Some((LoadOp::Load, StoreOp::Store)),
            stencil: None,
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
    for layer in 0..2 {
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
    let ticket = e.readback_buffer(&dst, 0..8192).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755321,
    };
    // A generic non-RT allocation cannot back an RT/DS texture.
    let invalid_plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 755320,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::SparseBind,
                waits: vec![],
            }]),
        )
    };
    let invalid_step = invalid_plan.step(0);
    let grid = sparse.tiling(&color).unwrap().tile_grid;
    let invalid = SparseBind {
        resource: color.clone(),
        tiles: vec![SparseTile {
            mip: 0,
            layer: 0,
            origin: [0, 0, 0],
            count: [1, 1, 1],
        }],
        allocation: Some(sparse.allocation(65536).unwrap()),
        allocation_offset: 0,
    };
    assert!(grid[0] > 0 && grid[1] > 0);
    // SAFETY: mismatched heap class must be rejected before queue handoff.
    let rejected = unsafe {
        d.queue().submit(SubmitPacket {
            plan: invalid_plan,
            steps: vec![BoundStep {
                step: invalid_step,
                payload: StepPayload::SparseBind(vec![invalid]),
            }],
            completion: CompletionRequest::End,
        })
    };
    assert!(matches!(
        rejected,
        SubmitOutcome::Rejected {
            error: Error::InvalidState,
            work: RetryableSubmit::Fresh(_)
        }
    ));
    let mut binds = Vec::new();
    for texture in [&color, &depth] {
        let tiling = sparse.tiling(texture).unwrap();
        for layer in 0..2 {
            let tile = if let Some(tail) = tiling.packed_tail {
                let mut tile = SparseTile::packed_tail(tail);
                tile.layer = layer;
                tile
            } else {
                SparseTile {
                    mip: 0,
                    layer,
                    origin: [0, 0, 0],
                    count: tiling.tile_grid,
                }
            };
            let count =
                u64::from(tile.count[0]) * u64::from(tile.count[1]) * u64::from(tile.count[2]);
            binds.push(SparseBind {
                resource: texture.clone(),
                tiles: vec![tile],
                allocation: Some(
                    sparse
                        .allocation_for(texture, count * tiling.tile_bytes)
                        .unwrap(),
                ),
                allocation_offset: 0,
            });
        }
    }
    // SAFETY: the mapping step is ordered before both attachment uses on the same queue.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::SparseBind,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let steps = vec![
        BoundStep {
            step: plan.step(0),
            payload: StepPayload::SparseBind(binds),
        },
        BoundStep {
            step: plan.step(1),
            payload: StepPayload::Commands(vec![e.finish().unwrap()]),
        },
    ];
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps,
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
    assert_eq!(&bytes[center..center + 4], &[0, 0, 0, 255]);
    assert_eq!(&bytes[4096 + center..4096 + center + 4], &[255, 0, 0, 255]);
}
