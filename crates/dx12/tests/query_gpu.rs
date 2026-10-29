//! Query validation against a real DX12 queue.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

fn query_to_copy(
    e: &mut CommandEncoder<fluxel_rhi_dx12::Dx12>,
    buffer: &Buffer<fluxel_rhi_dx12::Dx12>,
    size: u64,
) {
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer,
            range: 0..size,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
}

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[test]
fn timestamps_and_triangle_statistics() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let caps = d.capabilities();
    assert!(caps.storage_buffers);
    for (kind, reported) in [
        (QueryKind::Timestamp, caps.timestamp),
        (QueryKind::Occlusion, caps.occlusion),
        (QueryKind::PipelineStatistics, caps.pipeline_statistics),
    ] {
        assert_eq!(reported, d.queries(kind).is_some());
    }
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let ts = d.queries(QueryKind::Timestamp).unwrap();
    assert!(ts.timestamp_frequency(lane).unwrap() > 0);
    let stats = d.queries(QueryKind::PipelineStatistics).unwrap();
    let tq = ts
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let sq = stats
        .create(&QueryDesc {
            kind: QueryKind::PipelineStatistics,
            count: 1,
        })
        .unwrap();
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
        include_bytes!("../../rhi/examples/assets/01_triangle/triangle_vs.dxil"),
        "vs_main",
        ShaderStage::Vertex,
    );
    let ps = shader(
        include_bytes!("../../rhi/examples/assets/01_triangle/triangle_ps.dxil"),
        "ps_main",
        ShaderStage::Fragment,
    );
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = block_on(d.raster_pipeline(RasterPipelineDesc {
        label: None,
        layout,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "ps_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                VertexAttribute {
                    location: 1,
                    offset: 12,
                    format: VertexFormat::Float32(3),
                },
            ],
        }],
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
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    }))
    .unwrap();
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
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
    let vertex = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 72,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 104,
            usage: BufferUsage::QUERY_RESOLVE.union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let vertices: [f32; 18] = [
        -0.8, -0.8, 0.0, 1.0, 0.0, 0.0, 0.8, -0.8, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0,
    ];
    let vertex_bytes: Vec<u8> = vertices.into_iter().flat_map(f32::to_ne_bytes).collect();
    e.write_buffer(&vertex, 0, &vertex_bytes).unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
        },
        layout: Some((ImageLayout::Common, ImageLayout::ColorAttachment)),
        ownership: None,
    })])
    .unwrap();
    stats.begin_statistics(&mut e, &sq, 0).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
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
                width: 16,
                height: 16,
            },
            multiview_mask: None,
            timestamp_writes: Some(PassTimestampWrites {
                set: &tq,
                begin: Some(0),
                end: Some(1),
            }),
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
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &vertex, 0..72).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    stats.end_statistics(&mut e).unwrap();
    ts.resolve(&mut e, &tq, 0..2, &dst, 0).unwrap();
    stats.resolve(&mut e, &sq, 0..1, &dst, 16).unwrap();
    query_to_copy(&mut e, &dst, 104);
    let ticket = e.readback_buffer(&dst, 0..104).unwrap();
    let work = e.finish().unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755311,
    };
    // SAFETY: the single step and recorded work share the device and lane.
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
    // SAFETY: work is submitted once in its matching plan.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the awaited point covers this readback ticket.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    let first = u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
    let second = u64::from_ne_bytes(bytes[8..16].try_into().unwrap());
    assert!(first > 0 && second >= first, "{first}, {second}");
    assert_eq!(u64::from_ne_bytes(bytes[16..24].try_into().unwrap()), 3);
    assert_eq!(u64::from_ne_bytes(bytes[24..32].try_into().unwrap()), 1);
    assert_eq!(u64::from_ne_bytes(bytes[32..40].try_into().unwrap()), 3);
    assert_eq!(u64::from_ne_bytes(bytes[56..64].try_into().unwrap()), 1);
    assert_eq!(u64::from_ne_bytes(bytes[64..72].try_into().unwrap()), 1);
    assert!(u64::from_ne_bytes(bytes[72..80].try_into().unwrap()) > 0);
}

#[test]
fn resolve_timestamp_written_by_an_earlier_encoder() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let q = d.queries(QueryKind::Timestamp).unwrap();
    let set = q
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 1,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::QUERY_RESOLVE.union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut writer = d.encoder(lane).unwrap();
    q.timestamp(&mut writer, &set, 0).unwrap();
    let writer = writer.finish().unwrap();
    let mut resolver = d.encoder(lane).unwrap();
    q.resolve(&mut resolver, &set, 0..1, &dst, 0).unwrap();
    query_to_copy(&mut resolver, &dst, 8);
    let ticket = resolver.readback_buffer(&dst, 0..8).unwrap();
    let resolver = resolver.finish().unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755312,
    };
    // SAFETY: the fixed two-step plan runs both recordings on the same lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
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
    let first = plan.step(0);
    let second = plan.step(1);
    // SAFETY: each recording is submitted once to its matching lane.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: first,
                    payload: StepPayload::Commands(vec![writer]),
                },
                BoundStep {
                    step: second,
                    payload: StepPayload::Commands(vec![resolver]),
                },
            ],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the awaited point covers the readback ticket.
    let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
    assert_ne!(u64::from_ne_bytes(bytes[..8].try_into().unwrap()), 0);
}

#[test]
fn unsubmitted_write_cannot_authorize_resolve_and_rejected_work_is_retryable() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let q = d.queries(QueryKind::Timestamp).unwrap();
    let set = q
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 1,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::QUERY_RESOLVE.union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut discarded = d.encoder(lane).unwrap();
    q.timestamp(&mut discarded, &set, 0).unwrap();
    drop(discarded);
    let mut reset_writer = d.encoder(lane).unwrap();
    q.timestamp(&mut reset_writer, &set, 0).unwrap();
    q.reset(&mut reset_writer, &set, 0..1).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755315,
    };
    // SAFETY: this one-step plan matches the recording's lane.
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
    // SAFETY: the recording is consumed once by its own plan.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![reset_writer.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    assert!(matches!(outcome, SubmitOutcome::Accepted(_)));
    let mut resolver = d.encoder(lane).unwrap();
    q.resolve(&mut resolver, &set, 0..1, &dst, 0).unwrap();
    query_to_copy(&mut resolver, &dst, 8);
    let ticket = resolver.readback_buffer(&dst, 0..8).unwrap();
    let resolver = resolver.finish().unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755313,
    };
    // SAFETY: the plan contains one matching command step.
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
    let packet = SubmitPacket {
        plan,
        steps: vec![BoundStep {
            step,
            payload: StepPayload::Commands(vec![resolver]),
        }],
        completion: CompletionRequest::End,
    };
    // SAFETY: the packet owns its one-shot recording and matches the device.
    let retry = match unsafe { d.queue().submit(packet) } {
        SubmitOutcome::Rejected {
            error: Error::InvalidState,
            work: RetryableSubmit::Fresh(packet),
        } => packet,
        _ => panic!("uninitialized query was not cleanly rejected"),
    };
    let mut writer = d.encoder(lane).unwrap();
    q.timestamp(&mut writer, &set, 0).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755314,
    };
    // SAFETY: writer and its one-step plan have the same lane.
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
    // SAFETY: writer is submitted exactly once.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![writer.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    assert!(matches!(outcome, SubmitOutcome::Accepted(_)));
    // SAFETY: rejection returned this exact unsubmitted owning packet.
    let outcome = unsafe { d.queue().submit(retry) };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("retry rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    // SAFETY: the awaited point covers the readback ticket.
    let data = unsafe { block_on(d.read_after(ticket, &point)).unwrap() };
    assert_ne!(u64::from_ne_bytes(data.bytes[..8].try_into().unwrap()), 0);
}
