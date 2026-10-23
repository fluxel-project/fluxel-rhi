//! Private SPI fixed texture/sampler arrays, cross-group pairs, and rebind.
use super::*;

async fn binding_dynamic_cube_shadow(d: &Device<Gl>) {
    let format = TextureFormat::Depth32Float;
    let source = br#"@group(0) @binding(0) var a: binding_array<texture_depth_cube_array,2>;
@group(0) @binding(1) var s: binding_array<sampler_comparison,2>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); }
@fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f {
 let x=u32(pos.x); let ti=select(select(x/2u, 2u, x>=4u), 0u, x>=6u);
 let si=select(select(x%2u, 2u, x>=4u), 0u, x>=6u);
 let v=textureSampleCompare(a[ti],s[si],vec3f(1.,0.999,0.),0,0.5);
 return vec4f(v,0.,0.,1.);
}"#.to_vec();
    let tex = || {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 12,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
    };
    let a = match tex() {
        Ok(v) => v,
        Err(Error::Unsupported(reason)) => {
            eprintln!("SKIP CubeArrayShadow texture: {reason}");
            return;
        }
        Err(e) => panic!("cube depth texture: {e:?}"),
    };
    let b = tex().unwrap();
    let view_desc = TextureViewDesc {
        format,
        dimension: TextureViewDimension::CubeArray,
        aspects: TextureAspects::DEPTH,
        mips: 0..1,
        layers: 0..12,
    };
    let av = match d.create_view(&a, &view_desc) {
        Ok(v) => v,
        Err(Error::Unsupported(reason)) => {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            assert_eq!(reason, "GL texture view subrange");
            eprintln!("SKIP CubeArrayShadow view: {reason}");
            return;
        }
        Err(e) => panic!("cube array view: {e:?}"),
    };
    let bv = d.create_view(&b, &view_desc).unwrap();
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
        entries: vec![
            entry(
                0,
                BindingType::SampledTexture {
                    sample: TextureSampleType::Depth,
                    dimension: TextureViewDimension::CubeArray,
                    multisampled: false,
                },
            ),
            entry(
                1,
                BindingType::Sampler {
                    filtering: false,
                    comparison: true,
                },
            ),
        ],
    };
    for e in &mut desc.entries {
        e.count = BindingCount::Fixed(2);
        e.partially_bound = true;
    }
    // SAFETY: both SPI layout and Core wrapper belong to this live device.
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
    let sampler = |compare| {
        d.create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: Some(compare),
            border: None,
        })
        .unwrap()
    };
    let less = sampler(CompareFunction::Less);
    let greater = sampler(CompareFunction::Greater);
    let tg = |texture_hole: bool, sampler_hole: bool| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(vec![
                        Some(BindingResource::TextureView(&av)),
                        (!texture_hole).then_some(BindingResource::TextureView(&bv)),
                    ]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Array(vec![
                        Some(BindingResource::Sampler(&less)),
                        (!sampler_hole).then_some(BindingResource::Sampler(&greater)),
                    ]),
                },
            ],
        })
        .unwrap()
    };
    let full = tg(false, false);
    let texture_hole = tg(true, false);
    let sampler_hole = tg(false, true);
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
    for (texture, value) in [(&a, 0.25), (&b, 0.75)] {
        e.clear_texture(texture, 0..1, 0..12, TextureClearValue::Depth(value))
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
    p.set_bind_group(0, &full, &[]).unwrap();
    p.scissor(Rect {
        x: 0,
        y: 0,
        width: 6,
        height: 8,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.set_bind_group(0, &texture_hole, &[]).unwrap();
    p.scissor(Rect {
        x: 6,
        y: 0,
        width: 1,
        height: 8,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.set_bind_group(0, &sampler_hole, &[]).unwrap();
    p.scissor(Rect {
        x: 7,
        y: 0,
        width: 1,
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
    for (x, red) in [0u8, 255, 255, 0, 0, 0, 0, 0].into_iter().enumerate() {
        let actual = &data[4 * 32 + x * 4..4 * 32 + x * 4 + 4];
        eprintln!("CubeArrayShadow pixel {x}: {actual:?}");
        assert_eq!(actual, &[red, 0, 0, 255]);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_dynamic_cube_shadow() {
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
    block_on(binding_dynamic_cube_shadow(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_dynamic_cube_shadow() {
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
    block_on(binding_dynamic_cube_shadow(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_dynamic_cube_shadow() {
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
    binding_dynamic_cube_shadow(&d).await;
}
