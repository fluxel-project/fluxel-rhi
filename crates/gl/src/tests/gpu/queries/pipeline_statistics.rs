//! Core pipeline statistics availability, reset and result semantics.
use super::*;

async fn pipeline_statistics(d: &Device<Gl>) {
    let Some(queries) = d.queries(QueryKind::PipelineStatistics) else {
        eprintln!("SKIP: pipeline statistics not available on this context");
        return;
    };
    let source = br#"

@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn first()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }
@fragment fn second()->@location(0) vec4f { return vec4f(0.,1.,0.,1.); }
"#;
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_slice()),
            entries: vec![
                ShaderEntryPoint {
                    name: "vs".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "first".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "second".into(),
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
    let desc = |entry: &str, constants| RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: shader.clone(),
            entry: "vs".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader.clone(),
            entry: entry.into(),
            constants,
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
        primitive: triangle_list_state(),
        depth_stencil: None,
        multisample: single_sample_state(u64::MAX),
        multiview: None,
        cache: None,
    };
    let pipeline = d.raster_pipeline(desc("first", vec![])).await.unwrap();
    let compute = d.compute().expect("desktop compute");
    let cs = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(br"@compute @workgroup_size(1) fn cs() {}".as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        })
        .await
        .unwrap();
    let cp = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout: layout.clone(),
            compute: PipelineStage {
                module: cs,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::PipelineStatistics,
            count: 2,
        })
        .unwrap();
    let out = d
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
    let ov = d
        .create_view(
            &out,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 288,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 160).unwrap();
    // SAFETY: context and resources survive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    e.write_buffer(&buffer, 0, &[0xad; 288]).unwrap();
    // The active native query owns its set through end, even if the caller drops it.
    let transient = queries
        .create(&QueryDesc {
            kind: QueryKind::PipelineStatistics,
            count: 1,
        })
        .unwrap();
    queries.reset(e, &transient, 0..1).unwrap();
    queries.begin_statistics(e, &transient, 0).unwrap();
    drop(transient);
    queries.end_statistics(e).unwrap();
    queries.reset(e, &set, 0..2).unwrap();
    queries.begin_statistics(e, &set, 0).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&ov),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 0.])),
        store: StoreOp::Store,
    })];
    let mut p = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    p.set_pipeline(&pipeline).unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.finish().unwrap();
    queries.end_statistics(e).unwrap();
    queries.begin_statistics(e, &set, 1).unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&cp).unwrap();
    p.dispatch([2, 1, 1]).unwrap();
    p.finish().unwrap();
    queries.end_statistics(e).unwrap();
    queries.resolve(e, &set, 0..2, &buffer, 8).unwrap();
    // Reuse the same slot after reset: old draw values must not survive.
    queries.reset(e, &set, 0..1).unwrap();
    queries.begin_statistics(e, &set, 0).unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&cp).unwrap();
    p.dispatch([3, 1, 1]).unwrap();
    p.finish().unwrap();
    queries.end_statistics(e).unwrap();
    queries.resolve(e, &set, 0..1, &buffer, 192).unwrap();
    let ticket = e.readback_buffer(&buffer, 0..288).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers the query resolve and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(&bytes[..8], &[0xad; 8]);
    assert_eq!(&bytes[184..192], &[0xad; 8]);
    assert_eq!(&bytes[280..], &[0xad; 8]);
    let counters = |offset: usize| {
        (0..11)
            .map(|i| {
                u64::from_ne_bytes(
                    bytes[offset + i * 8..offset + i * 8 + 8]
                        .try_into()
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
    };
    let draw = counters(8);
    let dispatch = counters(96);
    let reused = counters(192);
    eprintln!(
        "draw statistics: {draw:?}; compute statistics: {dispatch:?}; reused slot: {reused:?}"
    );
    assert!(draw[0] >= 3 && draw[1] >= 1 && draw[2] >= 3 && draw[7] > 0);
    for i in [3, 4, 8, 9, 10] {
        assert_eq!(draw[i], 0, "draw counter {i}");
    }
    assert_eq!(dispatch[10], 2);
    for (i, value) in dispatch[..10].iter().enumerate() {
        assert_eq!(*value, 0, "compute counter {i}");
    }
    assert_eq!(reused[10], 3);
    for (i, value) in reused[..10].iter().enumerate() {
        assert_eq!(*value, 0, "reused compute counter {i}");
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_pipeline_statistics() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL Core feature fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(pipeline_statistics(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_pipeline_statistics() {
    let native = GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer");
    let device = block_on(open(native));
    assert!(device.queries(QueryKind::PipelineStatistics).is_none());
    block_on(pipeline_statistics(&device));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_pipeline_statistics() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    assert!(device.queries(QueryKind::PipelineStatistics).is_none());
    pipeline_statistics(&device).await;
}
