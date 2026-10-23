//! Core stage-scoped push constants on real GPU contexts.
use super::*;

async fn push_constants(d: &Device<Gl>) {
    let source = br#"
struct P { padding: f32, x: f32, y: f32 }
var<push_constant> pc: P;
struct O { @builtin(position) p: vec4f, @location(0) v: f32 }
@vertex fn vs(@builtin(vertex_index) i:u32)->O {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));
 var o:O; o.p=vec4f(p[i],0.5,1.); o.v=pc.x; return o;
}
fn read_y() -> f32 { return pc.y; }
@fragment fn first(o:O)->@location(0) vec4f { return vec4f(o.v,pc.x,read_y(),1.); }
@fragment fn second(o:O)->@location(0) vec4f { return vec4f(o.v,pc.x,pc.y,1.); }
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
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Vertex, ShaderStage::Fragment],
                range: 4..12,
            }],
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
    };
    // Active members must be covered for each stage; unused struct padding need not be.
    for ranges in [
        vec![PushConstantRange {
            stages: vec![ShaderStage::Fragment],
            range: 4..12,
        }],
        vec![PushConstantRange {
            stages: vec![ShaderStage::Vertex],
            range: 4..8,
        }],
        vec![PushConstantRange {
            stages: vec![ShaderStage::Vertex, ShaderStage::Fragment],
            range: 4..8,
        }],
    ] {
        let invalid_layout = d
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: None,
                groups: &[],
                push_constants: &ranges,
            })
            .unwrap();
        let mut invalid_desc = desc("first", vec![]);
        invalid_desc.layout = invalid_layout;
        assert!(
            d.raster_pipeline(invalid_desc).await.is_err(),
            "uncovered active push member accepted"
        );
    }
    let a = d.raster_pipeline(desc("first", vec![])).await.unwrap();
    let b = d.raster_pipeline(desc("second", vec![])).await.unwrap();
    let extras = d.raster_extras().expect("push constants capability");
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
            size: 512,
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
    for (index, pipeline) in [&a, &b].into_iter().enumerate() {
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
        p.set_pipeline(pipeline).unwrap();
        extras
            .push_constants(&mut p, StageMask::VERTEX, 4, &0.25f32.to_le_bytes())
            .unwrap();
        extras
            .push_constants(&mut p, StageMask::FRAGMENT, 4, &0.5f32.to_le_bytes())
            .unwrap();
        extras
            .push_constants(&mut p, StageMask::FRAGMENT, 8, &0.75f32.to_le_bytes())
            .unwrap();
        p.set_pipeline(&b).unwrap();
        p.draw(0..3, 0..1).unwrap();
        p.finish().unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &out,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            &buffer,
            TexelCopyBufferLayout {
                offset: (index * 256) as u64,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&buffer, 0..512).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers the readback commands.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let offset = 4 * 32 + 4 * 4;
    assert_eq!(&bytes[offset..offset + 4], &[64, 128, 191, 255]);
    assert_eq!(&bytes[offset + 256..offset + 260], &[64, 128, 191, 255]);
    // Exercise the last advertised byte range through a real fragment shader.
    let limit = d.capabilities().limits.max_push_constant_bytes;
    assert!(limit >= 8 && limit % 4 == 0);
    let boundary_source = format!(
        r#"
struct P {{ @size({}) padding: u32, value: f32 }}
var<push_constant> pc: P;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {{
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}}
@fragment fn first()->@location(0) vec4f {{ return vec4f(pc.value,0.,0.,1.); }}
"#,
        limit - 4
    );
    let boundary_shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(boundary_source.as_bytes()),
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
            ],
            required_features: vec![],
        })
        .await
        .unwrap();
    let boundary_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Fragment],
                range: limit - 4..limit,
            }],
        })
        .unwrap();
    assert!(
        d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Fragment],
                range: limit - 4..limit.checked_add(4).unwrap(),
            }],
        })
        .is_err()
    );
    let mut boundary_desc = desc("first", vec![]);
    boundary_desc.layout = boundary_layout;
    boundary_desc.vertex.module = boundary_shader.clone();
    boundary_desc.fragment.as_mut().unwrap().module = boundary_shader;
    let boundary_pipeline = d.raster_pipeline(boundary_desc).await.unwrap();
    let plan = common::single_commands_plan(&fixture, 169).unwrap();
    // SAFETY: all referenced resources remain alive until completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin boundary fixture"));
    let e = run.commands();
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
    p.set_pipeline(&boundary_pipeline).unwrap();
    extras
        .push_constants(
            &mut p,
            StageMask::FRAGMENT,
            limit - 4,
            &0.75f32.to_le_bytes(),
        )
        .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &out,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &buffer,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&buffer, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish boundary fixture"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the draw and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    assert_eq!(
        &bytes[offset..offset + 4],
        &[191, 0, 0, 255],
        "push limit {limit}"
    );
    for (i, bad_offset) in [5, 12].into_iter().enumerate() {
        let plan = common::single_commands_plan(&fixture, 170 + i as u64).unwrap();
        // SAFETY: all referenced resources outlive this abandoned invalid session.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin invalid fixture"));
        let e = run.commands();
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&ov),
            resolve_target: None,
            load: LoadOp::Load,
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
        p.set_pipeline(&a).unwrap();
        assert!(
            extras
                .push_constants(
                    &mut p,
                    StageMask::FRAGMENT,
                    bad_offset,
                    &0.5f32.to_le_bytes()
                )
                .is_err()
        );
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_push_constants() {
    use fluxel_host::{Window, WindowConfig};
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let window =
        Window::new(WindowConfig::new("GL push constants fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(push_constants(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_push_constants() {
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(push_constants(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_push_constants() {
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
    let d = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    push_constants(&d).await;
}
