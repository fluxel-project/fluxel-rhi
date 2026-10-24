//! WGL MSAA texture-copy regressions.
//!
//! Distinct sample-mask draws give the source an observable per-sample pattern.
//! A later resolve proves that `copy_texture` retained that sample storage.
use super::*;

const SHADER: &[u8] = br#"
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn red()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }
@fragment fn green()->@location(0) vec4f { return vec4f(0.,1.,0.,1.); }
@fragment fn blue()->@location(0) vec4f { return vec4f(0.,0.,1.,1.); }
"#;

fn texture_desc(samples: u32, layers: u32) -> TextureDesc {
    TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: layers,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    }
}

fn region<'a>(texture: &'a Texture<Gl>, layer: u32) -> TextureRegion<'a, Gl> {
    TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, layer],
        extent: [4, 4, 1],
        aspect: TextureAspects::COLOR,
    }
}

fn view(device: &Device<Gl>, texture: &Texture<Gl>, layer: u32) -> TextureView<Gl> {
    device
        .create_view(
            texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: layer..layer + 1,
            },
        )
        .unwrap()
}

fn attachment<'a>(
    view: &'a TextureView<Gl>,
    color: [f32; 4],
) -> [Option<ColorAttachment<'a, Gl>>; 1] {
    [Some(ColorAttachment {
        view: AttachmentView::Texture(view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float(color)),
        store: StoreOp::Store,
    })]
}

async fn pipelines(device: &Device<Gl>) -> [RasterPipeline<Gl>; 3] {
    let shader = device
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(SHADER),
            entries: ["vs", "red", "green", "blue"]
                .into_iter()
                .map(|name| ShaderEntryPoint {
                    name: name.into(),
                    stage: if name == "vs" {
                        ShaderStage::Vertex
                    } else {
                        ShaderStage::Fragment
                    },
                    workgroup_size: None,
                })
                .collect(),
            required_features: vec![],
        })
        .await
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let make = |entry: &str, mask: u64| {
        device.raster_pipeline(RasterPipelineDesc {
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
                count: 4,
                mask,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    };
    [
        make("red", 1).await.unwrap(),
        make("green", 2).await.unwrap(),
        make("blue", 4).await.unwrap(),
    ]
}

async fn copy_msaa_samples(device: &Device<Gl>, layers: u32) {
    let [red, green, blue] = pipelines(device).await;
    let ms_desc = texture_desc(4, layers);
    if !matches!(device.texture_support(&ms_desc), Support::Supported(_)) {
        eprintln!(
            "SKIP WGL MSAA {}copy: profile rejects 4x Rgba8Unorm{}",
            if layers == 1 { "" } else { "array " },
            if layers == 1 { "" } else { " array" }
        );
        return;
    }
    let source = device.create_texture(&ms_desc).unwrap();
    let copied = device.create_texture(&ms_desc).unwrap();
    let resolved_desc = texture_desc(1, 1);
    let source_resolved = device.create_texture(&resolved_desc).unwrap();
    let copied_layer0 = device.create_texture(&resolved_desc).unwrap();
    let copied_layer1 = device.create_texture(&resolved_desc).unwrap();
    let copied_layer0_view = view(device, &copied, 0);
    let copied_layer1_view = view(device, &copied, layers - 1);
    let source_view = view(device, &source, layers - 1);
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 768,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, if layers == 1 { 286 } else { 287 }).unwrap();
    // SAFETY: device, WGL context, and all fixture resources outlive completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin MSAA copy commands"));
    let encoder = run.commands();
    let area = Rect {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };
    let colors = attachment(&source_view, [0., 0., 0., 1.]);
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: area,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    for pipeline in [&red, &green, &blue] {
        pass.set_pipeline(pipeline).unwrap();
        pass.draw(0..3, 0..1).unwrap();
    }
    pass.finish().unwrap();
    let clear0 = attachment(&copied_layer0_view, [1., 0., 1., 1.]);
    encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &clear0,
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
    if layers > 1 {
        let clear1 = attachment(&copied_layer1_view, [0., 0., 1., 1.]);
        encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &clear1,
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
    let copied_layer = layers - 1;
    encoder
        .copy_texture(region(&source, copied_layer), region(&copied, copied_layer))
        .unwrap();
    encoder
        .resolve(region(&source, copied_layer), region(&source_resolved, 0))
        .unwrap();
    encoder
        .resolve(region(&copied, 0), region(&copied_layer0, 0))
        .unwrap();
    encoder
        .resolve(region(&copied, copied_layer), region(&copied_layer1, 0))
        .unwrap();
    for (offset, texture) in [
        (0, &source_resolved),
        (256, &copied_layer0),
        (512, &copied_layer1),
    ] {
        encoder
            .texture_to_buffer(
                region(texture, 0),
                &output,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 16,
                    rows_per_image: 4,
                },
            )
            .unwrap();
    }
    let ticket = encoder.readback_buffer(&output, 0..768).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("MSAA copy did not issue"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers both resolves and the readback.
    let bytes = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    let source_color = &bytes[0..4];
    let layer0 = &bytes[256..260];
    let copied_color = &bytes[512..516];
    assert!(
        (62..=65).contains(&source_color[0])
            && (62..=65).contains(&source_color[1])
            && (62..=65).contains(&source_color[2])
            && source_color[3] == 255,
        "unexpected resolved source color: {source_color:?}"
    );
    assert_eq!(
        copied_color, source_color,
        "copy changed the MSAA sample distribution"
    );
    if layers == 1 {
        assert_eq!(
            layer0, source_color,
            "2D copied resolve differs from source"
        );
    } else {
        assert_eq!(
            layer0,
            &[255, 0, 255, 255],
            "array layer zero sentinel changed"
        );
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
fn with_wgl(test: impl FnOnce(&Device<Gl>)) {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL MSAA copy fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the WGL window outlives every Core object used by the closure.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let device = block_on(open(native));
    test(&device);
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_core_msaa_copy_preserves_samples() {
    with_wgl(|device| block_on(copy_msaa_samples(device, 1)));
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires an explicit WGL hardware-test run"]
fn wgl_core_msaa_copy_array_layer_preserves_samples() {
    with_wgl(|device| block_on(copy_msaa_samples(device, 2)));
}
