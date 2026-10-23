//! Native GL lease exercises Core recording and state invalidation on a real context.
use super::*;
use glow::HasContext;
use std::panic::{AssertUnwindSafe, catch_unwind};

async fn native_interop(d: &Device<Gl>) {
    let source = br#"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
 let p = array<vec2f, 3>(vec2f(-1.,-1.), vec2f(0.,-1.), vec2f(-1.,3.));
 return vec4f(p[i],0.5,1.);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(1.,0.,0.,1.); }
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
    let buffer_desc = BufferDesc {
        label: None,
        size: 512,
        usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: MemoryClass::Device,
    };
    let buffer = d.create_buffer(&buffer_desc).unwrap();
    let retired_in_scope = d.create_buffer(&buffer_desc).unwrap();
    let interop = d.native_interop().expect("native GL interop capability");
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 180).unwrap();
    // SAFETY: the fixture keeps all native resources and context alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
        store: StoreOp::Store,
    })];
    let pass_desc = RasterPassDesc {
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
    };
    let mut pass = e.begin_raster(&pass_desc).unwrap();
    // SAFETY: this test owns the current GL context and does not race other work.
    assert!(
        unsafe { interop.acquire(NativeInteropRequest::Device) }.is_err(),
        "device lease during an active raster pass"
    );
    pass.set_pipeline(&pipeline).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &texture,
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

    // SAFETY: the scope stays on the owning thread with its context current, does not
    // access RHI-owned native objects, and introduces no resource hazards. The pointer
    // is borrowed only as the GL dispatch table and never escapes the guard.
    let (foreign_buffer, pack_alignment, unpack_alignment) = unsafe {
        let guard = interop.acquire(NativeInteropRequest::Device).unwrap();
        assert!(
            interop.acquire(NativeInteropRequest::Device).is_err(),
            "nested native lease"
        );
        assert!(
            d.create_buffer(&buffer_desc).is_err(),
            "RHI operation during lease"
        );
        drop(retired_in_scope);
        let gl = &*guard.raw_pointer().cast::<glow::Context>();
        let pack_alignment = gl.get_parameter_i32(glow::PACK_ALIGNMENT);
        let unpack_alignment = gl.get_parameter_i32(glow::UNPACK_ALIGNMENT);
        let pbo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
        gl.buffer_data_size(glow::PIXEL_PACK_BUFFER, 1024, glow::STREAM_DRAW);
        gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(pbo));
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 8);
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 8);
        gl.use_program(None);
        gl.bind_vertex_array(None);
        gl.enable(glow::BLEND);
        gl.color_mask(false, false, false, false);
        gl.viewport(0, 0, 1, 1);
        gl.enable(glow::SCISSOR_TEST);
        gl.scissor(0, 0, 1, 1);
        gl.enable(glow::RASTERIZER_DISCARD);
        gl.enable(glow::SAMPLE_COVERAGE);
        gl.sample_coverage(0.0, false);
        gl.disable(glow::DITHER);
        if !gl.get_parameter_string(glow::VERSION).contains("OpenGL ES") {
            gl.enable(glow::COLOR_LOGIC_OP);
        }
        (pbo, pack_alignment, unpack_alignment)
    };
    let panic = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: guard remains on context owner thread, with no foreign resource access.
        let guard = unsafe { interop.acquire(NativeInteropRequest::Device) }.unwrap();
        // SAFETY: valid shared dispatch reference, limited to guard lifetime.
        unsafe {
            let gl = &*guard.raw_pointer().cast::<glow::Context>();
            assert!(!gl.is_enabled(glow::RASTERIZER_DISCARD));
            assert!(!gl.is_enabled(glow::SAMPLE_COVERAGE));
            assert_eq!(gl.get_parameter_i32(glow::PIXEL_PACK_BUFFER_BINDING), 0);
            assert_eq!(gl.get_parameter_i32(glow::PIXEL_UNPACK_BUFFER_BINDING), 0);
            assert_eq!(gl.get_parameter_i32(glow::PACK_ALIGNMENT), pack_alignment);
            assert_eq!(
                gl.get_parameter_i32(glow::UNPACK_ALIGNMENT),
                unpack_alignment
            );
            if !gl.get_parameter_string(glow::VERSION).contains("OpenGL ES") {
                assert!(!gl.is_enabled(glow::COLOR_LOGIC_OP));
            }
            gl.color_mask(false, false, false, false);
            gl.enable(glow::RASTERIZER_DISCARD);
        };
        panic!("exercise unwinding native guard");
    }));
    assert!(panic.is_err());
    assert!(
        d.create_buffer(&buffer_desc).is_ok(),
        "lease must be released during unwinding"
    );
    let mut pass = e.begin_raster(&pass_desc).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &buffer,
        TexelCopyBufferLayout {
            offset: 256,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&buffer, 0..512).unwrap();
    if let Some(compute) = d.compute() {
        let pass = compute.begin(e).unwrap();
        // SAFETY: owning thread and context; acquire must fail before any raw access.
        assert!(
            unsafe { interop.acquire(NativeInteropRequest::Device) }.is_err(),
            "device lease during an active compute pass"
        );
        pass.finish().unwrap();
    }
    // SAFETY: native work is confined to the guarded encoder and owns no RHI objects.
    unsafe {
        let guard = e.native_interop().unwrap();
        let gl = &*guard.raw_pointer().cast::<glow::Context>();
        gl.delete_buffer(foreign_buffer);
    }
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers the copy and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for base in [0, 256] {
        assert_eq!(&bytes[base..base + 4], &[255, 0, 0, 255], "triangle");
        let sentinel = base + 4 * 32 + 7 * 4;
        assert_eq!(
            &bytes[sentinel..sentinel + 4],
            &[0, 0, 255, 255],
            "background sentinel"
        );
    }

    let plan = common::single_commands_plan(&fixture, 181).unwrap();
    // SAFETY: the device and context outlive the immediate session.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin native-only commands"));
    // SAFETY: access remains on the owning thread and introduces no resource hazards.
    unsafe {
        let guard = run.commands().native_interop().unwrap();
        let gl = &*guard.raw_pointer().cast::<glow::Context>();
        gl.flush();
    }
    assert!(
        matches!(run.close(), ImmediateRunOutcome::Issued(_)),
        "native-only encoder must be issued"
    );
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::Issued(_)
    ));
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::task::{Context, Poll, Wake, Waker};
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
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

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_native_interop() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL native interop fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: window remains alive on the owning thread until all GL resources are dropped.
    let instance = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(instance));
    block_on(native_interop(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_native_interop() {
    let instance = GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer");
    let d = block_on(open(instance));
    block_on(native_interop(&d));
}
