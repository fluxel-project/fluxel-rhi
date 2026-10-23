//! Private SPI fixed texture/sampler arrays, cross-group pairs, and rebind.
use super::*;

async fn binding_dynamic_sampled_float_one(d: &Device<Gl>, filterable: bool) {
    let sample = if filterable {
        TextureSampleType::Float
    } else {
        TextureSampleType::UnfilterableFloat
    };
    let format = TextureFormat::Rgba8Unorm;
    let scalar = "f32";
    let source = format!(r#"@group(0) @binding(0) var a: binding_array<texture_2d<{scalar}>,2>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {{ let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); }}
@fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f {{ let i=select(0u,1u,pos.x>=5.); let x=textureLoad(a[i],vec2i(0),0); let y=textureLoad(a[1],vec2i(0),0); let z=textureLoad(a[i+2u],vec2i(0),0); return vec4f(f32(x.r),f32(y.b),f32(z.r),1.); }}"#).into_bytes();
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
    let texture_type = || BindingType::SampledTexture {
        sample,
        dimension: TextureViewDimension::D2,
        multisampled: false,
    };
    let make_layout = |ty| {
        let mut desc = BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(0, ty)],
        };
        desc.entries[0].count = BindingCount::Fixed(2);

        // SAFETY: private SPI layout and wrapper belong to the same live device;
        // production device capabilities remain unchanged.
        unsafe {
            BindGroupLayout::from_native(
                d.id(),
                Gl::create_bind_group_layout(d.native_ref(), &desc).unwrap(),
                Arc::from(desc.entries.clone()),
            )
        }
    };
    let textures = make_layout(texture_type());
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
    let tex = |w, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [w, 1, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let a = tex(2, TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST);
    let b = tex(1, TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST);
    let view = |t| {
        d.create_view(
            t,
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
    let av = view(&a);
    let bv = view(&b);
    let tg = |first, second| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![
                    Some(BindingResource::TextureView(first)),
                    Some(BindingResource::TextureView(second)),
                ]),
            }],
        })
        .unwrap()
    };
    let tg1 = tg(&av, &bv);
    let tg2 = tg(&bv, &av);
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
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 151).unwrap();
    // SAFETY: context and resources remain alive through the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for (texture, width, data) in [
        (&a, 2u32, &[255u8, 0, 0, 255, 0, 255, 0, 255][..]),
        (&b, 1, &[0u8, 0, 255, 255][..]),
    ] {
        e.write_texture(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [width, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: width * 4,
                rows_per_image: 1,
            },
            data,
        )
        .unwrap();
    }
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&ov),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
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
    p.set_bind_group(0, &tg1, &[]).unwrap();
    p.scissor(Rect {
        x: 0,
        y: 0,
        width: 2,
        height: 8,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.scissor(Rect {
        x: 2,
        y: 0,
        width: 3,
        height: 8,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.set_bind_group(0, &tg2, &[]).unwrap();
    p.scissor(Rect {
        x: 5,
        y: 0,
        width: 3,
        height: 8,
    })
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
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: this completion covers both draws and readback.
    let data = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (x, expected) in [
        (1usize, [255, 255, 0, 255]),
        (3, [255, 255, 0, 255]),
        (6, [255, 0, 0, 255]),
    ] {
        let actual = &data[4 * 32 + x * 4..4 * 32 + x * 4 + 4];
        eprintln!("integer partial {format:?} pixel {x}: {actual:?}");
        assert_eq!(actual, &expected);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_dynamic_sampled_float() {
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
    block_on(binding_dynamic_sampled_float_one(&d, false));
    block_on(binding_dynamic_sampled_float_one(&d, true));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_dynamic_sampled_float() {
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
    block_on(binding_dynamic_sampled_float_one(&d, false));
    block_on(binding_dynamic_sampled_float_one(&d, true));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_dynamic_sampled_float() {
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
    binding_dynamic_sampled_float_one(&d, false).await;
    binding_dynamic_sampled_float_one(&d, true).await;
}
