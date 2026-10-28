//! Real Core immediate texture/sampler pairing and group update checks.
use super::*;

async fn binding_shared_sampler(d: &Device<Gl>) {
    let source = br#"
@group(0) @binding(0) var a: texture_2d<f32>;
@group(0) @binding(1) var b: texture_2d<f32>;
@group(1) @binding(0) var repeat_s: sampler;
@group(1) @binding(1) var clamp_s: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
  let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f {
  let red=textureSample(a,repeat_s,vec2f(1.25,0.5)).r;
  let green=textureSample(b,repeat_s,vec2f(0.25,0.5)).b;
  let blue=textureSample(b,repeat_s,vec2f(0.25,0.5)).b * textureLoad(b,vec2i(0,0),0).b;
  return vec4f(red,green,blue,1.);
}
"#;
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
        sample: TextureSampleType::Float,
        dimension: TextureViewDimension::D2,
        multisampled: false,
    };
    let sampler_type = || BindingType::Sampler {
        filtering: false,
        comparison: false,
    };
    let textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(0, texture_type()), entry(1, texture_type())],
        })
        .unwrap();
    let samplers = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(0, sampler_type()), entry(1, sampler_type())],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&textures, &samplers],
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
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
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
            format: TextureFormat::Rgba8Unorm,
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
                format: TextureFormat::Rgba8Unorm,
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
    let sampler = |mode| {
        d.create_sampler(&SamplerDesc {
            label: None,
            address: [mode; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap()
    };
    let repeat = sampler(AddressMode::Repeat);
    let clamp = sampler(AddressMode::ClampToEdge);
    let tg = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&av),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&bv),
                },
            ],
        })
        .unwrap();
    let sg = |first, second| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &samplers,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(first),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(second),
                },
            ],
        })
        .unwrap()
    };
    let sg1 = sg(&repeat, &clamp);
    let sg2 = sg(&clamp, &repeat);
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
    let ov = view(&out);
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
    p.set_bind_group(1, &sg1, &[]).unwrap();
    p.set_bind_group(0, &tg, &[]).unwrap();
    p.scissor(Rect {
        x: 0,
        y: 0,
        width: 4,
        height: 8,
    })
    .unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.set_bind_group(1, &sg2, &[]).unwrap();
    p.scissor(Rect {
        x: 4,
        y: 0,
        width: 4,
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
    for (x, expected) in [(1usize, [255, 255, 255, 255]), (6, [0, 255, 255, 255])] {
        let actual = &data[4 * 32 + x * 4..4 * 32 + x * 4 + 4];
        eprintln!("pair pixel {x}: {actual:?}");
        assert_eq!(actual, &expected);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_shared_sampler() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_shared_sampler(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_shared_sampler() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_shared_sampler(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_shared_sampler() {
    let d = webgl2_device([8, 8]).await;
    binding_shared_sampler(&d).await;
}
