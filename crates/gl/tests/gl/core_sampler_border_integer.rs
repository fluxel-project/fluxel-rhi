//! Core direct GLSL integer border sampler variants.
use super::*;

async fn sampler_border_integer(d: &Device<Gl>, es: bool) {
    let vertex_source = b"#version 400 core\nvoid main(){ vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3)); gl_Position=vec4(p[gl_VertexID],0.5,1); }";
    let fragment_source = b"#version 400 core\nuniform sampler2D binding_0; uniform usampler2D group_1_binding_0; uniform isampler2D group_2_binding_0; out vec4 result; void main(){ result=vec4(texture(binding_0,vec2(-0.25,0.5)).r,float(texture(group_1_binding_0,vec2(-0.25,0.5)).r==0u && texture(group_1_binding_0,vec2(0.5)).r==7u),float(texture(group_2_binding_0,vec2(-0.25,0.5)).r==0 && texture(group_2_binding_0,vec2(0.5)).r==7),texture(binding_0,vec2(0.5)).r); }";
    let convert = |bytes: &[u8]| {
        if es {
            String::from_utf8(bytes.to_vec()).unwrap().replacen("#version 400 core", "#version 310 es\nprecision highp float; precision highp int; precision highp sampler2D; precision highp usampler2D; precision highp isampler2D;", 1).into_bytes()
        } else {
            bytes.to_vec()
        }
    };
    let shader = |bytes: &[u8], stage| ShaderDesc {
        label: None,
        format: if es {
            ShaderFormat::GlslEs(310)
        } else {
            ShaderFormat::Glsl(400)
        },
        bytes: Arc::from(convert(bytes)),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d
        .shader(shader(vertex_source, ShaderStage::Vertex))
        .await
        .unwrap();
    let fragment = d
        .shader(shader(fragment_source, ShaderStage::Fragment))
        .await
        .unwrap();
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let texture_type = |sample| BindingType::SampledTexture {
        sample,
        dimension: TextureViewDimension::D2,
        multisampled: false,
    };
    let make_layout = |sample| {
        d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(0, texture_type(sample)),
                entry(
                    1,
                    BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                ),
            ],
        })
        .unwrap()
    };
    let float_layout = make_layout(TextureSampleType::Float);
    let uint_layout = make_layout(TextureSampleType::Uint);
    let sint_layout = make_layout(TextureSampleType::Sint);
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&float_layout, &uint_layout, &sint_layout],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "main".into(),
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
    let texture = |format, usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap_or_else(|error| panic!("create {format:?}: {error:?}"))
    };
    let color = texture(
        TextureFormat::Rgba8Unorm,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );
    let uint = texture(
        TextureFormat::Rgba8Uint,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );
    let sint = texture(
        TextureFormat::Rgba8Sint,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );
    let out = texture(
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let view = |t, format, aspects| {
        d.create_view(
            t,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap()
    };
    let cv = view(&color, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let uv = view(&uint, TextureFormat::Rgba8Uint, TextureAspects::COLOR);
    let sv = view(&sint, TextureFormat::Rgba8Sint, TextureAspects::COLOR);
    let ov = view(&out, TextureFormat::Rgba8Unorm, TextureAspects::COLOR);
    let zero = d
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToZero; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let group = |layout, view| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&zero),
                },
            ],
        })
        .unwrap()
    };
    let fg = group(&float_layout, &cv);
    let ug = group(&uint_layout, &uv);
    let sg = group(&sint_layout, &sv);
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
    let plan = common::single_commands_plan(&fixture, 153).unwrap();
    // SAFETY: resources and context remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for (t, bytes) in [
        (&color, [64u8, 0, 0, 255]),
        (&uint, [7u8, 0, 0, 255]),
        (&sint, [7u8, 0, 0, 127]),
    ] {
        e.write_texture(
            TextureRegion {
                texture: t,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 8, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
            &bytes.repeat(64),
        )
        .unwrap();
    }
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
    p.set_pipeline(&pipeline).unwrap();
    p.set_bind_group(0, &fg, &[]).unwrap();
    p.set_bind_group(1, &ug, &[]).unwrap();
    p.set_bind_group(2, &sg, &[]).unwrap();
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
    // SAFETY: point covers all commands including readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let pixel = &bytes[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
    eprintln!("integer border GLSL: {pixel:?}");
    assert_eq!(pixel, &[0, 255, 255, 64]);
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_sampler_border_integer() {
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
    block_on(sampler_border_integer(&d, false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_sampler_border_integer() {
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
    use crate::backend::gl::api::{GlFamilyApi as _, GlFamilyProfile};
    // SAFETY: the fixture owns the current context for this inspection.
    let native = unsafe { d.native_ref() };
    let available = native.0.with_native("inspect GLES border clamp", |provider| {
        let discovery = provider.discovery();
        let core = matches!(discovery.context().profile(), GlFamilyProfile::Embedded { major, minor } if major > 3 || (major == 3 && minor >= 2));
        let names: Vec<_> = discovery.extensions().raw_reported_names().filter(|name| *name == "GL_OES_texture_border_clamp" || *name == "GL_EXT_texture_border_clamp").collect();
        eprintln!("GLES border clamp extensions: {names:?}");
        Ok(core || !names.is_empty())
    }).unwrap();
    if !available {
        eprintln!("SKIP: GLES border clamp unavailable");
        return;
    }
    block_on(sampler_border_integer(&d, true));
}
