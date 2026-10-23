//! GLES depth/stencil uploads and plane copies, observed via raster tests.
use super::*;

async fn gles_packed_transfer(d: &Device<Gl>, format: TextureFormat) {
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 2,
        format,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&desc), Support::Supported(_)) {
        eprintln!("SKIP: {format:?} packed texture unavailable");
        return;
    }
    let a = d.create_texture(&desc).unwrap();
    let b = d.create_texture(&desc).unwrap();
    let view = |t| {
        d.create_view(
            t,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                mips: 1..2,
                layers: 1..2,
            },
        )
        .unwrap()
    };
    let av = view(&a);
    let bv = view(&b);
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let cv = d
        .create_view(
            &color,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let buf = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 384,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let shader = d.shader(ShaderDesc { label: None, format: ShaderFormat::Wgsl, bytes: Arc::from(br"@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }".as_slice()), entries: vec![ShaderEntryPoint { name: "vs".into(), stage: ShaderStage::Vertex, workgroup_size: None }, ShaderEntryPoint { name: "fs".into(), stage: ShaderStage::Fragment, workgroup_size: None }], required_features: vec![] }).await.unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let face = StencilFaceState {
        compare: CompareFunction::Equal,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let desc = |compare| RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: shader.clone(),
            entry: "vs".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader.clone(),
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
        depth_stencil: Some(DepthStencilState {
            format,
            depth_write_enabled: false,
            depth_compare: compare,
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 255,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.,
            depth_bias_clamp: 0.,
        }),
        multisample: MultisampleState {
            count: 1,
            mask: u64::MAX,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    };
    let greater = d
        .raster_pipeline(desc(CompareFunction::Greater))
        .await
        .unwrap();
    let less = d
        .raster_pipeline(desc(CompareFunction::Less))
        .await
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 490).unwrap();
    // SAFETY: the device and resources remain alive until completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin GLES plane transfer"));
    let e = run.commands();
    for (v, depth, stencil) in [(&av, 0.25, 17), (&bv, 0.75, 91)] {
        let pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(v),
                    depth: Some((LoadOp::Clear(depth), StoreOp::Store)),
                    stencil: Some((LoadOp::Clear(stencil), StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.finish().unwrap();
    }
    let region = |texture, aspect| TextureRegion {
        texture,
        mip: 1,
        origin: [1, 1, 1],
        extent: [2, 2, 1],
        aspect,
    };
    let stencil_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 4,
        rows_per_image: 2,
    };
    let depth_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 8,
        rows_per_image: 2,
    };
    e.copy_texture(
        region(&a, TextureAspects::DEPTH),
        region(&b, TextureAspects::DEPTH),
    )
    .unwrap();
    let cpu = [0, 0, 0, 0, 17, 17, 0, 0, 17, 17];
    e.write_texture(region(&b, TextureAspects::STENCIL), stencil_layout(4), &cpu)
        .unwrap();
    let draw = |e: &mut CommandEncoder<Gl>, pipeline: &RasterPipeline<Gl>, stencil, output| {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&cv),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
            store: StoreOp::Store,
        })];
        let mut pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &colors,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&bv),
                    depth: Some((LoadOp::Load, StoreOp::Store)),
                    stencil: Some((LoadOp::Load, StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(pipeline).unwrap();
        pass.stencil_reference(stencil).unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: &color,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &buf,
            TexelCopyBufferLayout {
                offset: output,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
        )
        .unwrap();
    };
    draw(e, &greater, 17, 0);
    let compute = d.compute().is_some();
    if compute {
        e.write_buffer(&buf, 320, &[0, 0, 0, 0, 23, 23, 0, 0, 23, 23])
            .unwrap();
        e.buffer_to_texture(
            &buf,
            stencil_layout(324),
            region(&b, TextureAspects::STENCIL),
        )
        .unwrap();
        draw(e, &greater, 23, 64);
    }
    if format == TextureFormat::Depth32FloatStencil8 {
        let depths = [0.75f32.to_le_bytes(); 4].concat();
        e.write_texture(region(&b, TextureAspects::DEPTH), depth_layout(0), &depths)
            .unwrap();
        draw(e, &less, if compute { 23 } else { 17 }, 128);
        if compute {
            let depths = [0.25f32.to_le_bytes(); 4].concat();
            e.write_buffer(&buf, 352, &depths).unwrap();
            e.buffer_to_texture(&buf, depth_layout(352), region(&b, TextureAspects::DEPTH))
                .unwrap();
            draw(e, &greater, 23, 192);
        }
    }
    let ticket = e.readback_buffer(&buf, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish GLES transfer"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers all draws and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let mut cases = vec![0];
    if compute {
        cases.push(64);
    }
    if format == TextureFormat::Depth32FloatStencil8 {
        cases.push(128);
        if compute {
            cases.push(192);
        }
    }
    for off in cases {
        let p = off + 16 + 4;
        assert_eq!(
            &bytes[p..p + 4],
            &[255, 0, 0, 255],
            "selected plane result at {off}"
        );
        assert_eq!(
            &bytes[off..off + 4],
            &[0, 0, 255, 255],
            "other pixel at {off}"
        );
    }
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_packed_transfer() {
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
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("EGL GLES pbuffer"),
    ));
    for format in [
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ] {
        block_on(gles_packed_transfer(&d, format));
    }
}
