//! Mixed linear surface and sRGB attachment load/publish readback.
use super::*;

async fn mixed_surface_srgb(d: &Device<Gl>) {
    use crate::backend::gl::native::NativeContextOwner as _;
    // SAFETY: this test owns the current device and context on this thread.
    let native = unsafe { d.native_ref() };
    let observed = native
        .0
        .with_context("observe fixture window", |context| {
            Ok(context.drawable_extent())
        })
        .unwrap();
    eprintln!("WGL_DRAWABLE={observed:?}");
    let mut surface = common::adopted_surface(d).unwrap();
    let caps = surface.capabilities().unwrap();
    let format = TextureFormat::Rgba8Unorm;
    assert!(caps.formats.contains(&format));
    surface
        .configure(&SurfaceConfig {
            extent: [120, 64],
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await
        .unwrap();
    let image = surface.acquire().await.unwrap();
    let source = b"struct F { @location(0) a: vec4f, @location(2) b: vec4f }; @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->F { var o:F; o.a=vec4f(0.5,0.5,0.5,1.); o.b=vec4f(0.5,0.5,0.5,1.); return o; }";
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
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let color = |format| {
        Some(ColorTargetState {
            format,
            blend: None,
            write_mask: 15,
        })
    };
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
            color_targets: vec![color(TextureFormat::Rgba8UnormSrgb), None, color(format)],
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
            extent: [120, 64, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8UnormSrgb,
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
                format: TextureFormat::Rgba8UnormSrgb,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 30720,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 93).unwrap();
    // SAFETY: the context, surface image and all resources remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let full = [Some(ColorAttachment {
        view: AttachmentView::Surface(&image),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &full,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 120,
            height: 64,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    let mixed = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        }),
        None,
        Some(ColorAttachment {
            view: AttachmentView::Surface(&image),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        }),
    ];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &mixed,
            depth_stencil: None,
            render_area: Rect {
                x: 16,
                y: 16,
                width: 32,
                height: 32,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.scissor(Rect {
        x: 16,
        y: 16,
        width: 32,
        height: 32,
    })
    .unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    // A second mixed pass must preload and republish the surface without
    // converting its linear bytes through the sRGB texture's state.
    let loaded = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&view),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        }),
        None,
        Some(ColorAttachment {
            view: AttachmentView::Surface(&image),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        }),
    ];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &loaded,
        depth_stencil: None,
        render_area: Rect {
            x: 16,
            y: 16,
            width: 32,
            height: 32,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [120, 64, 1],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 480,
            rows_per_image: 64,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..30720).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish commands"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers the texture copy above.
    let data = unsafe { d.read_after(ticket, &point).await }.unwrap();
    assert_eq!(
        &data.bytes[32 * 480 + 32 * 4..32 * 480 + 32 * 4 + 4],
        &[188, 188, 188, 255]
    );
    // SAFETY: image remains an acquired lease on the owner thread.
    let target = unsafe { image.native_ref() }.default_target().unwrap();
    // SAFETY: device is alive and the same owner thread retains the context.
    let native = unsafe { d.native_ref() };
    let pixels = native
        .0
        .with_native("inspect fixture surface", |provider| {
            provider.read_default_framebuffer(target)
        })
        .unwrap()
        .bytes;
    assert_eq!(
        &pixels[32 * 480 + 32 * 4..32 * 480 + 32 * 4 + 4],
        &[128, 128, 128, 255]
    );
    assert_eq!(&pixels[0..4], &[0, 0, 255, 255]);
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_mixed_srgb() {
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
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(mixed_surface_srgb(&d));
}
