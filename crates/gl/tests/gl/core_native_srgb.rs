//! Native Core offscreen sRGB conversion checked against linear output.
use super::*;
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
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

async fn check(d: &Device<Gl>) {
    let source = b"@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->@location(0) vec4f { return vec4f(0.5,0.5,0.5,1.); }";
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
    for (format, expected) in [
        (TextureFormat::Rgba8UnormSrgb, 188u8),
        (TextureFormat::Rgba8Unorm, 128u8),
    ] {
        let pipeline = d
            .raster_pipeline(RasterPipelineDesc {
                label: None,
                layout: layout.clone(),
                vertex: PipelineStage {
                    module: vertex.clone(),
                    entry: "vs".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: fragment.clone(),
                    entry: "fs".into(),
                    constants: vec![],
                }),
                vertex_buffers: vec![],
                color_targets: vec![Some(ColorTargetState {
                    format,
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
                format,
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
                    format,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )
            .unwrap();
        let out = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 256,
                usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let fixture = common::ImmediateFixture {
            device: d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 121).unwrap();
        // SAFETY: resources and the current context outlive the completion point.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin commands"));
        let e = run.commands();
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.5, 0.5, 0.5, 1.])),
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
                    width: 8,
                    height: 8,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.scissor(Rect {
            x: 4,
            y: 0,
            width: 4,
            height: 8,
        })
        .unwrap();
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
            &out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&out, 0..256).unwrap();
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
        for (x, operation) in [(1usize, "clear"), (6, "draw")] {
            let pixel = &bytes[4 * 32 + x * 4..4 * 32 + x * 4 + 4];
            eprintln!("{format:?} {operation}: {pixel:?}");
            for channel in &pixel[..3] {
                assert!(
                    (i16::from(*channel) - i16::from(expected)).abs() <= 1,
                    "{format:?} {operation}: {pixel:?}"
                );
            }
            assert_eq!(pixel[3], 255);
        }
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_offscreen_srgb() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL sRGB Core fixture", 8, 8).unwrap()).unwrap();
    // SAFETY: the window outlives the device and all of its resources.
    let native = unsafe { GlInstance::from_wgl_window(&window, [8, 8]) }.unwrap();
    let d = block_on(open(native));
    block_on(check(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES context"]
fn gles_core_offscreen_srgb() {
    let native = GlInstance::gles_pbuffer([8, 8]).unwrap();
    let d = block_on(open(native));
    block_on(check(&d));
}

async fn copy_and_resolve(d: &Device<Gl>) {
    let format = TextureFormat::Rgba8UnormSrgb;
    let texture = |samples, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let msaa = texture(4, TextureUsage::RENDER_ATTACHMENT);
    let resolved = texture(1, TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC);
    let copied = texture(1, TextureUsage::COPY_DST | TextureUsage::COPY_SRC);
    let view = |texture| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let msaa_view = view(&msaa);
    let resolved_view = view(&resolved);
    let output = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let out_resolved = output();
    let out_copied = output();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 122).unwrap();
    // SAFETY: resources and the current context remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&msaa_view),
        resolve_target: Some(AttachmentView::Texture(&resolved_view)),
        load: LoadOp::Clear(ClearColor::Float([0.5, 0.5, 0.5, 1.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&RasterPassDesc {
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
    .unwrap()
    .finish()
    .unwrap();
    let region = |texture| TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [8, 8, 1],
        aspect: TextureAspects::COLOR,
    };
    e.copy_texture(region(&resolved), region(&copied)).unwrap();
    for (texture, out) in [(&resolved, &out_resolved), (&copied, &out_copied)] {
        e.texture_to_buffer(
            region(texture),
            out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let a = e.readback_buffer(&out_resolved, 0..256).unwrap();
    let b = e.readback_buffer(&out_copied, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    for (label, ticket) in [("resolve", a), ("copy", b)] {
        // SAFETY: point covers the resolve, copy and readbacks.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        let pixel = &bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
        eprintln!("sRGB {label}: {pixel:?}");
        for channel in &pixel[..3] {
            assert!((i16::from(*channel) - 188).abs() <= 1);
        }
        assert_eq!(pixel[3], 255);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_srgb_copy_resolve() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL sRGB copy fixture", 8, 8).unwrap()).unwrap();
    // SAFETY: window outlives the device and its resources.
    let native = unsafe { GlInstance::from_wgl_window(&window, [8, 8]) }.unwrap();
    let d = block_on(open(native));
    block_on(copy_and_resolve(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES context"]
fn gles_core_srgb_copy_resolve() {
    let d = block_on(open(GlInstance::gles_pbuffer([8, 8]).unwrap()));
    block_on(copy_and_resolve(&d));
}

async fn copy_only(d: &Device<Gl>) {
    let format = TextureFormat::Rgba8UnormSrgb;
    let texture = |samples, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let resolved = texture(1, TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC);
    let copied = texture(1, TextureUsage::COPY_DST | TextureUsage::COPY_SRC);
    let view = |texture| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let resolved_view = view(&resolved);
    let output = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let out_resolved = output();
    let out_copied = output();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 122).unwrap();
    // SAFETY: resources and the current context remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&resolved_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.5, 0.5, 0.5, 1.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&RasterPassDesc {
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
    .unwrap()
    .finish()
    .unwrap();
    let region = |texture| TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [8, 8, 1],
        aspect: TextureAspects::COLOR,
    };
    e.copy_texture(region(&resolved), region(&copied)).unwrap();
    for (texture, out) in [(&resolved, &out_resolved), (&copied, &out_copied)] {
        e.texture_to_buffer(
            region(texture),
            out,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let a = e.readback_buffer(&out_resolved, 0..256).unwrap();
    let b = e.readback_buffer(&out_copied, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    for (label, ticket) in [("resolve", a), ("copy", b)] {
        // SAFETY: point covers the resolve, copy and readbacks.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        let pixel = &bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
        eprintln!("sRGB clear/copy {label}: {pixel:?}");
        for channel in &pixel[..3] {
            assert!((i16::from(*channel) - 188).abs() <= 1);
        }
        assert_eq!(pixel[3], 255);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_srgb_copy() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL sRGB copy fixture", 8, 8).unwrap()).unwrap();
    // SAFETY: window outlives the device and resources.
    let native = unsafe { GlInstance::from_wgl_window(&window, [8, 8]) }.unwrap();
    let d = block_on(open(native));
    block_on(copy_only(&d));
}
#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES context"]
fn gles_core_srgb_copy() {
    let d = block_on(open(GlInstance::gles_pbuffer([8, 8]).unwrap()));
    block_on(copy_only(&d));
}
