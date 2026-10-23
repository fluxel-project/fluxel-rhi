//! Core multiview layer routing and shader view indices.
use super::*;

pub(super) async fn multiview(d: &Device<Gl>) {
    if !d.capabilities().multiview {
        eprintln!("SKIP: multiview capability unavailable");
        return;
    }
    let source=br#"
struct VertexOutput { @builtin(position) position: vec4f, @location(0) @interpolate(flat) view: i32 }
@vertex fn vs(@builtin(vertex_index) i:u32, @builtin(view_index) v:i32)->VertexOutput {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); var o:VertexOutput; o.position=vec4f(p[i],0.5,1.); o.view=v; return o;
}
@fragment fn first(input:VertexOutput, @builtin(view_index) v:i32)->@location(0) vec4f { return vec4f(f32(input.view)/4.,f32(v)/4.,0.,1.); }
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
    let desc = |entry: &str, constants, views| RasterPipelineDesc {
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
        multiview: Some(views),
        cache: None,
    };
    let pipeline2 = d.raster_pipeline(desc("first", vec![], 2)).await.unwrap();
    let pipeline1 = d.raster_pipeline(desc("first", vec![], 1)).await.unwrap();
    let out = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 4,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT
                | TextureUsage::COPY_SRC
                | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let ov = d
        .create_view(
            &out,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..4,
            },
        )
        .unwrap();
    // Attachment-only views must never expose the parent's omitted layer to a
    // sampled binding when no native texture-view alias was created.
    let sampled_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let sampled = d.create_bind_group(&BindGroupDesc {
        label: None,
        layout: &sampled_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&ov),
        }],
    });
    // SAFETY: ov is a live view from this device, retained throughout the test.
    if !unsafe { ov.native_ref() }.full_sampling_view {
        eprintln!("attachment-only subrange: sampled binding rejected");
        assert!(
            sampled.is_err(),
            "attachment-only subrange must reject sampling"
        );
    } else {
        assert!(sampled.is_ok(), "a native view alias supports sampling");
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if d.capabilities().storage_textures {
        let image = d
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [4, 4, 1],
                array_layers: 4,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::STORAGE_BINDING | TextureUsage::RENDER_ATTACHMENT,
                view_formats: vec![],
            })
            .unwrap();
        let image_view = d
            .create_view(
                &image,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2Array,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 1..4,
                },
            )
            .unwrap();
        let image_layout = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2Array,
                        access: StorageAccess::Write,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        let image_group = d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &image_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&image_view),
            }],
        });
        // SAFETY: the view remains live and belongs to this device.
        if unsafe { image_view.native_ref() }.view_texture.is_none() {
            eprintln!("attachment-only subrange: layered storage binding rejected");
            assert!(
                image_group.is_err(),
                "partial layered image must require a native alias"
            );
        } else {
            assert!(image_group.is_ok(), "native image alias must be bindable");
        }
    }
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    let occlusion = d.queries(QueryKind::Occlusion).map(|queries| {
        let set = queries
            .create(&QueryDesc {
                kind: QueryKind::Occlusion,
                count: 1,
            })
            .unwrap();
        (queries, set)
    });
    // Invalid passes each use a fresh session. The following positive session
    // then proves that validation failures did not leave native pass state active.
    for (index, mask) in [0u32, 5, 8].into_iter().enumerate() {
        let plan = common::single_commands_plan(&fixture, 170 + index as u64).unwrap();
        // SAFETY: all resources and the context outlive this session.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("negative begin"));
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&ov),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let desc = RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: Some(mask),
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        };
        let result = run.commands().begin_raster(&desc);
        match mask {
            0 | 8 => assert!(matches!(result, Err(Error::InvalidRange))),
            5 => assert!(matches!(result, Err(Error::Unsupported(_)))),
            _ => unreachable!(),
        }
        drop(result);
        let _ = run.close();
        let _ = session.abort(Error::InvalidState);
    }
    {
        let plan = common::single_commands_plan(&fixture, 174).unwrap();
        // SAFETY: resources remain live for this immediate session.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("mismatch begin"));
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&ov),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let desc = RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            multiview_mask: Some(3),
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        };
        let mut pass = run.commands().begin_raster(&desc).unwrap();
        assert!(matches!(
            pass.set_pipeline(&pipeline1),
            Err(Error::InvalidState)
        ));
        drop(pass);
        let _ = run.close();
        let _ = session.abort(Error::InvalidState);
    }
    let plan = common::single_commands_plan(&fixture, 160).unwrap();
    // SAFETY: context and resources survive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if let Some((queries, set)) = &occlusion {
        queries.reset(e, set, 0..1).unwrap();
    }
    for (case, mask) in [3u32, 6, 2].into_iter().enumerate() {
        e.clear_texture(
            &out,
            0..1,
            0..4,
            TextureClearValue::Color(ClearColor::Float([0., 0., 1., 1.])),
        )
        .unwrap();
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
                    width: 4,
                    height: 4,
                },
                multiview_mask: Some(mask),
                timestamp_writes: None,
                occlusion_query_set: {
                    #[cfg(any(
                        all(windows, feature = "native-gl-wgl"),
                        feature = "native-gles-egl"
                    ))]
                    {
                        occlusion.as_ref().map(|(_, set)| set)
                    }
                    #[cfg(not(any(
                        all(windows, feature = "native-gl-wgl"),
                        feature = "native-gles-egl"
                    )))]
                    {
                        None
                    }
                },
                secondary_inheritance: None,
            })
            .unwrap();
        p.set_pipeline(if mask == 2 { &pipeline1 } else { &pipeline2 })
            .unwrap();
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if case == 0 {
            if let Some((queries, set)) = &occlusion {
                queries.begin_occlusion(&mut p, set, 0).unwrap();
            }
        }
        p.draw(0..3, 0..1).unwrap();
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if case == 0 {
            if let Some((queries, _)) = &occlusion {
                queries.end_occlusion(&mut p).unwrap();
            }
        }
        p.finish().unwrap();
        for layer in 0..4u32 {
            e.texture_to_buffer(
                TextureRegion {
                    texture: &out,
                    mip: 0,
                    origin: [0, 0, layer],
                    extent: [4, 4, 1],
                    aspect: TextureAspects::COLOR,
                },
                &buffer,
                TexelCopyBufferLayout {
                    offset: (case as u32 * 4 + layer) as u64 * 64,
                    bytes_per_row: 16,
                    rows_per_image: 4,
                },
            )
            .unwrap();
        }
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    let after_completion = occlusion
        .as_ref()
        .is_some_and(|(queries, _)| queries.resolve_mode() == QueryResolveMode::AfterCompletion);
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    let after_completion = false;
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if !after_completion {
        if let Some((queries, set)) = &occlusion {
            queries.resolve(e, set, 0..1, &buffer, 768).unwrap();
        }
    }
    let ticket = if after_completion {
        None
    } else {
        Some(e.readback_buffer(&buffer, 0..1024).unwrap())
    };
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    let bytes = if after_completion {
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        {
            let plan = common::single_commands_plan(&fixture, 190).unwrap();
            // SAFETY: producer completion is established and resources remain live.
            let mut session =
                unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
            let mut run = session
                .begin_commands()
                .unwrap_or_else(|_| panic!("begin query resolve"));
            let e = run.commands();
            let (queries, set) = occlusion.as_ref().unwrap();
            queries.resolve(e, set, 0..1, &buffer, 768).unwrap();
            let ticket = e.readback_buffer(&buffer, 0..1024).unwrap();
            assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
            let point = match session.finish() {
                ImmediateFinishOutcome::Issued(prefix) => {
                    prefix.checkpoints.into_iter().next().unwrap().point
                }
                _ => panic!("finish query resolve"),
            };
            d.wait_completion(&point).await.unwrap();
            // SAFETY: the point covers the result readback.
            unsafe { d.read_after(ticket, &point).await }.unwrap().bytes
        }
        #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
        {
            unreachable!()
        }
    } else {
        // SAFETY: the producer point covers the ordered result readback.
        unsafe { d.read_after(ticket.unwrap(), &point).await }
            .unwrap()
            .bytes
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if occlusion.is_some() {
        let samples = u64::from_ne_bytes(bytes[768..776].try_into().unwrap());
        eprintln!("multiview occlusion: {samples}");
        assert!(samples > 0);
    }
    let actual: Vec<Vec<u8>> = (0..12usize)
        .map(|slot| bytes[slot * 64..slot * 64 + 4].to_vec())
        .collect();
    eprintln!("multiview layers: {actual:?}");
    let blue = vec![0, 0, 255, 255];
    let v0 = vec![0, 0, 0, 255];
    let v1 = vec![64, 64, 0, 255];
    let v2 = vec![128, 128, 0, 255];
    assert_eq!(
        actual,
        vec![
            blue.clone(),
            v0,
            v1.clone(),
            blue.clone(),
            blue.clone(),
            blue.clone(),
            v1.clone(),
            v2,
            blue.clone(),
            blue.clone(),
            v1,
            blue
        ]
    );
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_multiview() {
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
    block_on(multiview(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_multiview() {
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
    block_on(multiview(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_multiview() {
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
    multiview(&d).await;
}
