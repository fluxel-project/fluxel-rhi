//! Per-format MSAA clear, resolve and sampling readback.
use super::*;
async fn format_msaa_one(
    d: &Device<Gl>,
    format: TextureFormat,
    sample: TextureSampleType,
    _pixel: &[u8],
    has_g: bool,
) {
    let ms_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 1,
        format,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&ms_desc), Support::Supported(_)) {
        eprintln!("SKIP {format:?} sample_count=4 unsupported");
        return;
    }
    let ms = d.create_texture(&ms_desc).unwrap();
    let ty = match sample {
        TextureSampleType::Uint => "u32",
        TextureSampleType::Sint => "i32",
        _ => "f32",
    };
    let source = format!(
        r#"@group(0) @binding(0) var image: texture_2d<{ty}>;
@group(0) @binding(1) var nearest: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {{let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);}}
@fragment fn fs()->@location(0) vec4f {{let a=textureLoad(image,vec2i(0,0),0); let b=textureLoad(image,vec2i(0,1),0);return vec4f(f32(a.r),f32(a.g),f32(b.r),f32(b.g));}}"#
    );
    let source = source.into_bytes();
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
    let textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(
                    0,
                    BindingType::SampledTexture {
                        sample,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    1,
                    BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                ),
            ],
        })
        .unwrap();
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
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
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
    let compressed = match texture(
        format,
        [2, 2, 1],
        TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
    ) {
        Ok(t) => t,
        Err(Error::Unsupported(reason)) => {
            eprintln!("SKIP {format:?}: {reason}");
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            web_sys::console::log_1(&format!("SKIP {format:?}: {reason}").into());
            return;
        }
        Err(e) => panic!("create {format:?}: {e:?}"),
    };
    let plain = texture(
        format,
        [2, 2, 1],
        TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
    )
    .unwrap();
    let out = texture(
        TextureFormat::Rgba8Unorm,
        [8, 8, 1],
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    )
    .unwrap();
    let view = |t, format| {
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
    let ms_v = view(&ms, format);
    let plain_v = view(&plain, format);
    let compressed_v = view(&compressed, format);
    let out_v = view(&out, TextureFormat::Rgba8Unorm);
    let sampler = d
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let group = |v| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(v),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        })
        .unwrap()
    };
    let plain_g = group(&plain_v);
    let compressed_g = group(&compressed_v);
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
    let clear = match sample {
        TextureSampleType::Uint => ClearColor::Uint([1, 1, 1, 1]),
        TextureSampleType::Sint => ClearColor::Sint([1, 1, 1, 1]),
        _ => ClearColor::Float([1., 1., 1., 1.]),
    };
    for v in [&plain_v, &compressed_v] {
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(&ms_v),
            resolve_target: Some(AttachmentView::Texture(v)),
            load: LoadOp::Clear(clear),
            store: StoreOp::Store,
        })];
        e.begin_raster(&RasterPassDesc {
            label: None,
            colors: &attachments,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap()
        .finish()
        .unwrap();
    }
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
    for (x, g) in [(0, &plain_g), (4, &compressed_g)] {
        p.set_bind_group(0, g, &[]).unwrap();
        p.scissor(Rect {
            x,
            y: 0,
            width: 4,
            height: 8,
        })
        .unwrap();
        p.draw(0..3, 0..1).unwrap();
    }
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
    let actual: Vec<_> = [1usize, 6]
        .into_iter()
        .map(|x| data[4 * 32 + x * 4..4 * 32 + x * 4 + 4].to_vec())
        .collect();
    eprintln!("{format:?} clear pair: {actual:?}");
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    web_sys::console::log_1(&format!("{format:?} clear pair: {actual:?}").into());
    let expected = if has_g {
        vec![255, 255, 255, 255]
    } else {
        vec![255, 0, 255, 0]
    };
    assert_eq!(actual, vec![expected.clone(), expected]);
}
async fn format_msaa(d: &Device<Gl>) {
    use TextureFormat::*;
    use TextureSampleType::*;
    let cases: &[(TextureFormat, TextureSampleType, &[u8], bool)] = &[
        (Rgba8Unorm, Float, &[], true),
        (Rgba8Uint, Uint, &[], true),
        (Rgba8Sint, Sint, &[], true),
        (Rgba16Float, Float, &[], true),
    ];
    for &(format, sample, pixel, has_g) in cases {
        format_msaa_one(d, format, sample, pixel, has_g).await;
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_format_msaa() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(format_msaa(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_format_msaa() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(format_msaa(&d));
}
