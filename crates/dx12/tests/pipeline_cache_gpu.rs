//! Primary graphics state after secondary bundle execution.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};
fn block<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
#[test]
fn pipeline_cache_raster_compute_mesh_roundtrip() {
    run(0);
}
fn run(mode: u8) {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|x| x.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let shader = |bytes: &'static [u8], stage| {
        block(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Dxil,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        }))
        .unwrap()
    };
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[
                PushConstantRange {
                    stages: vec![ShaderStage::Vertex],
                    range: 0..4,
                },
                PushConstantRange {
                    stages: vec![ShaderStage::Fragment],
                    range: 8..12,
                },
            ],
        })
        .unwrap();
    let cache = d.create_pipeline_cache(None).unwrap();
    assert!(
        d.create_pipeline_cache(Some(b"corrupt pipeline cache"))
            .is_err()
    );
    let desc = |write_mask, cache: &PipelineCache<_>| RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: shader(
                include_bytes!("assets/push_stage_vertex.dxil"),
                ShaderStage::Vertex,
            ),
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader(
                include_bytes!("assets/secondary_restore_pixel.dxil"),
                ShaderStage::Fragment,
            ),
            entry: "main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask,
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
        cache: Some(cache.clone()),
    };
    let empty_bytes = d.serialize_pipeline_cache(&cache).unwrap();
    let rp = block(d.raster_pipeline(desc(15, &cache))).unwrap();
    assert!(d.serialize_pipeline_cache(&cache).unwrap().len() > empty_bytes.len());
    let empty_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let compute_shader = block(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@compute @workgroup_size(1) fn main() {}"[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let compute_desc = |cache: &PipelineCache<_>| ComputePipelineDesc {
        label: None,
        layout: empty_layout.clone(),
        compute: PipelineStage {
            module: compute_shader.clone(),
            entry: "main".into(),
            constants: vec![],
        },
        cache: Some(cache.clone()),
        required_subgroup_size: None,
    };
    let after_raster = d.serialize_pipeline_cache(&cache).unwrap();
    let _compute = block(d.compute().unwrap().create_pipeline(compute_desc(&cache))).unwrap();
    let after_compute = d.serialize_pipeline_cache(&cache).unwrap();
    assert!(
        after_compute.len() > after_raster.len(),
        "compute cache entry was not stored"
    );
    let mesh_shader = if d.capabilities().mesh {
        Some(shader(
            include_bytes!("assets/mesh/mesh.dxil"),
            ShaderStage::Mesh,
        ))
    } else {
        eprintln!("SKIP: mesh shader tier unavailable");
        None
    };
    let mesh_pixel = if mesh_shader.is_some() {
        Some(shader(
            include_bytes!("assets/mesh/pixel.dxil"),
            ShaderStage::Fragment,
        ))
    } else {
        None
    };
    let mesh_desc = |cache: &PipelineCache<_>| MeshPipelineDesc {
        layout: empty_layout.clone(),
        task: None,
        mesh: PipelineStage {
            module: mesh_shader.as_ref().unwrap().clone(),
            entry: "main".into(),
            constants: vec![],
        },
        fragment: mesh_pixel.as_ref().map(|module| PipelineStage {
            module: module.clone(),
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
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        cache: Some(cache.clone()),
    };
    if mesh_shader.is_some() {
        block(d.mesh().unwrap().create_pipeline(mesh_desc(&cache))).unwrap();
        assert!(
            d.serialize_pipeline_cache(&cache).unwrap().len() > after_compute.len(),
            "mesh cache entry was not stored"
        );
    }
    let bytes = d.serialize_pipeline_cache(&cache).unwrap();
    assert!(!bytes.is_empty());
    let mut corrupted = bytes.clone();
    let middle = corrupted.len() / 2;
    corrupted[middle] ^= 0x80;
    assert!(d.create_pipeline_cache(Some(&corrupted)).is_err());
    assert!(
        d.create_pipeline_cache(Some(&bytes[..bytes.len() - 1]))
            .is_err()
    );
    let reopened = d.create_pipeline_cache(Some(&bytes)).unwrap();
    let rp2 = block(d.raster_pipeline(desc(if mode == 3 { 0 } else { 15 }, &reopened))).unwrap();
    let compute2 = block(
        d.compute()
            .unwrap()
            .create_pipeline(compute_desc(&reopened)),
    )
    .unwrap();
    let mesh2 = if mesh_shader.is_some() {
        Some(block(d.mesh().unwrap().create_pipeline(mesh_desc(&reopened))).unwrap())
    } else {
        None
    };
    assert!(!d.serialize_pipeline_cache(&reopened).unwrap().is_empty());
    let tex = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [16, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let views = (0..1)
        .map(|layer| {
            d.create_view(
                &tex,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
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
            size: 4096,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let sig = SecondaryRasterDesc {
        color_formats: vec![Some(TextureFormat::Rgba8Unorm)],
        depth_stencil_format: None,
        samples: 1,
        multiview_mask: None,
    };
    let domain = d.secondary().unwrap();
    let mut secondary = domain.begin(&sig).unwrap();
    secondary.set_pipeline(&rp2).unwrap();
    secondary
        .viewport(Viewport {
            x: 0.,
            y: 0.,
            width: if mode == 2 { 8. } else { 16. },
            height: 16.,
            min_depth: 0.,
            max_depth: 1.,
        })
        .unwrap();
    secondary
        .scissor(Rect {
            x: 0,
            y: 0,
            width: if mode == 1 { 8 } else { 16 },
            height: 16,
        })
        .unwrap();
    secondary
        .push_constants(
            StageMask::VERTEX,
            0,
            &(if mode != 0 { 12u32 } else { 99 }).to_le_bytes(),
        )
        .unwrap();
    secondary
        .push_constants(
            StageMask::FRAGMENT,
            8,
            &(if mode != 0 { 30u32 } else { 99 }).to_le_bytes(),
        )
        .unwrap();
    secondary.draw(0..3, 0..1).unwrap();

    let bundle = domain.finish(secondary).unwrap();

    let mut e = d.encoder(lane).unwrap();
    {
        let mut compute = d.compute().unwrap().begin(&mut e).unwrap();
        compute.set_pipeline(&compute2).unwrap();
        compute.dispatch([1, 1, 1]).unwrap();
        compute.finish().unwrap();
    }
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
    let barrier = |from, to, layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition { from, to },
            layout: Some(layout),
            ownership: None,
        })
    };
    e.transition_resources(&[barrier(
        empty,
        render,
        (ImageLayout::Common, ImageLayout::ColorAttachment),
    )])
    .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&views[0]),
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
                width: 16,
                height: 16,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: Some(sig),
        })
        .unwrap();
    if let Some(mesh2) = &mesh2 {
        let mesh = d.mesh().unwrap();
        mesh.bind_pipeline(&mut pass, mesh2).unwrap();
        mesh.dispatch(&mut pass, [1, 1, 1]).unwrap();
    }
    pass.set_pipeline(&rp).unwrap();
    pass.viewport(Viewport {
        x: 0.,
        y: 0.,
        width: 16.,
        height: 16.,
        min_depth: 0.,
        max_depth: 1.,
    })
    .unwrap();
    pass.scissor(Rect {
        x: 0,
        y: 0,
        width: 16,
        height: 16,
    })
    .unwrap();
    d.raster_extras()
        .unwrap()
        .push_constants(&mut pass, StageMask::VERTEX, 0, &12u32.to_le_bytes())
        .unwrap();
    d.raster_extras()
        .unwrap()
        .push_constants(&mut pass, StageMask::FRAGMENT, 8, &30u32.to_le_bytes())
        .unwrap();
    domain.execute(&mut pass, &bundle).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    drop(bundle);
    e.transition_resources(&[barrier(
        render,
        copy,
        (ImageLayout::ColorAttachment, ImageLayout::CopySrc),
    )])
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 0],
            extent: [16, 16, 1],
            aspect: TextureAspects::COLOR,
        },
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 16,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..4096).unwrap();
    let work = e.finish().unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 755336,
    };
    // SAFETY: the recorded command list and plan use this device and lane.
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
    // SAFETY: the work is submitted once on its recording lane.
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
    block(d.wait_completion(&point)).unwrap();
    // SAFETY: completion covers the readback ticket.
    let data = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    let left = 8 * 256 + 4 * 4;
    let right = 8 * 256 + 12 * 4;
    assert_eq!(&data[left..left + 4], &[255, 0, 0, 255]);
    assert_eq!(&data[right..right + 4], &[255, 0, 0, 255]);
}
