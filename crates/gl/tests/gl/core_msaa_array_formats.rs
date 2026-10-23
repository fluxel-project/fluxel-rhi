//! Multisample array layer attachment and per-sample shader readback.
use super::*;
async fn msaa_array_format_one(d: &Device<Gl>, format: TextureFormat, sample: TextureSampleType) {
    let height = 1;
    let ms_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [1, height, 1],
        array_layers: 2,
        format,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&ms_desc), Support::Supported(_)) {
        eprintln!("SKIP multisample array {format:?} count=4: texture_support rejected descriptor");
        return;
    }
    let ms2 = d.create_texture(&ms_desc).unwrap();
    let scalar = match sample {
        TextureSampleType::Uint => "u32",
        TextureSampleType::Sint => "i32",
        _ => "f32",
    };
    let scale = if matches!(sample, TextureSampleType::Uint | TextureSampleType::Sint) {
        " / 127.0"
    } else {
        ""
    };
    let source = format!(r#"@group(0) @binding(0) var image: texture_multisampled_2d_array<{scalar}>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {{let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);}}
@fragment fn fs()->@location(0) vec4f {{return vec4f(f32(textureLoad(image,vec2i(0,0),0,0).r){scale},f32(textureLoad(image,vec2i(0,0),0,1).r){scale},f32(textureLoad(image,vec2i(0,0),1,2).g){scale},f32(textureLoad(image,vec2i(0,0),1,3).g){scale});}}"#).into_bytes();
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
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let mut desc = BindGroupLayoutDesc {
        label: None,
        entries: vec![entry(
            0,
            BindingType::SampledTexture {
                sample,
                dimension: TextureViewDimension::D2Array,
                multisampled: true,
            },
        )],
    };

    // SAFETY: the private layout is created and wrapped on the same live device.
    let textures = unsafe {
        BindGroupLayout::from_native(
            d.id(),
            Gl::create_bind_group_layout(d.native_ref(), &desc).unwrap(),
            Arc::from(desc.entries.clone()),
        )
    };
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&textures],
            push_constants: &[],
        })
        .unwrap();
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
    let texture = |format, extent, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent,
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
    };
    let out = texture(
        TextureFormat::Rgba8Unorm,
        [8, 8, 1],
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    )
    .unwrap();
    let view = |t, view_format, dimension, layers| {
        d.create_view(
            t,
            &TextureViewDesc {
                format: view_format,
                dimension,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers,
            },
        )
        .unwrap()
    };
    let layer2 = view(&ms2, format, TextureViewDimension::D2, 0..1);
    let layer3 = view(&ms2, format, TextureViewDimension::D2, 1..2);
    let sampled2 = view(&ms2, format, TextureViewDimension::D2Array, 0..2);
    let out_v = view(
        &out,
        TextureFormat::Rgba8Unorm,
        TextureViewDimension::D2,
        0..1,
    );
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&sampled2),
            }],
        })
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 230).unwrap();
    // SAFETY: resources and context survive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let (red, green) = match sample {
        TextureSampleType::Uint => (
            ClearColor::Uint([127, 0, 0, 127]),
            ClearColor::Uint([0, 127, 0, 127]),
        ),
        TextureSampleType::Sint => (
            ClearColor::Sint([127, 0, 0, 127]),
            ClearColor::Sint([0, 127, 0, 127]),
        ),
        _ => (
            ClearColor::Float([1., 0., 0., 1.]),
            ClearColor::Float([0., 1., 0., 1.]),
        ),
    };
    let attachments = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&layer2),
            resolve_target: None,
            load: LoadOp::Clear(red),
            store: StoreOp::Store,
        }),
        Some(ColorAttachment {
            view: AttachmentView::Texture(&layer3),
            resolve_target: None,
            load: LoadOp::Clear(green),
            store: StoreOp::Store,
        }),
    ];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &attachments,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&out_v),
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
    p.set_bind_group(0, &group, &[]).unwrap();
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
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers texture readback.
    let data = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let pixels: Vec<_> = [1usize, 6]
        .into_iter()
        .map(|x| data[4 * 32 + x * 4..4 * 32 + x * 4 + 4].to_vec())
        .collect();
    eprintln!("dynamic msaa array pixels: {pixels:?}");
    assert_eq!(
        pixels,
        vec![vec![255, 255, 255, 255], vec![255, 255, 255, 255]]
    );
}
async fn msaa_array_formats(d: &Device<Gl>) {
    for (format, sample) in [
        (TextureFormat::Rgba8UnormSrgb, TextureSampleType::Float),
        (TextureFormat::Rgba8Uint, TextureSampleType::Uint),
        (TextureFormat::Rgba8Sint, TextureSampleType::Sint),
    ] {
        eprintln!("MSAA MRT format {format:?}");
        msaa_array_format_one(d, format, sample).await;
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_msaa_array_format() {
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
    block_on(msaa_array_formats(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_msaa_array_format() {
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
    block_on(msaa_array_formats(&d));
}
