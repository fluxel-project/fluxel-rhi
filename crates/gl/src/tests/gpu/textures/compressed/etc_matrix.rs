//! ETC2/EAC format upload and sample decoding through Core.
use super::*;
async fn compressed_etc_matrix(d: &Device<Gl>) {
    for (format, color, alpha, srgb, two_channels) in [
        (TextureFormat::Etc2Rgb8Unorm, true, false, false, false),
        (TextureFormat::Etc2Rgb8UnormSrgb, true, false, true, false),
        (TextureFormat::Etc2Rgba8Unorm, true, true, false, false),
        (TextureFormat::Etc2Rgba8UnormSrgb, true, true, true, false),
        (TextureFormat::Etc2Rgb8A1Unorm, true, false, false, false),
        (TextureFormat::Etc2Rgb8A1UnormSrgb, true, false, true, false),
        (TextureFormat::EacR11Unorm, false, false, false, false),
        (TextureFormat::EacR11Snorm, false, false, false, false),
        (TextureFormat::EacRg11Unorm, false, false, false, true),
        (TextureFormat::EacRg11Snorm, false, false, false, true),
    ] {
        compressed_etc_matrix_case(d, format, color, alpha, srgb, two_channels).await;
    }
}
async fn compressed_etc_matrix_case(
    d: &Device<Gl>,
    format: TextureFormat,
    color: bool,
    alpha: bool,
    srgb: bool,
    two_channels: bool,
) {
    let signed = matches!(
        format,
        TextureFormat::EacR11Snorm | TextureFormat::EacRg11Snorm
    );
    let source: &[u8] = if !signed {
        br#"
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f {
 return vec4f(textureLoad(image,vec2i(0,0),0).r,textureLoad(image,vec2i(1,0),0).g,
  textureSampleLevel(image,nearest,vec2f(0.375,0.875),0.,vec2i(-1,0)).r,
  textureSampleGrad(image,nearest,vec2f(0.375,0.875),vec2f(0.02,0.03),vec2f(0.04,0.01)).g);
}
"#
    } else {
        br#"
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f {
 let v = textureLoad(image,vec2i(0,0),0);
 return vec4f(v.r * 0.5 + 0.5, v.g * 0.5 + 0.5, v.r * 0.5 + 0.5, v.g * 0.5 + 0.5);
}
"#
    };
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source),
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
                        sample: if signed {
                            TextureSampleType::UnfilterableFloat
                        } else {
                            TextureSampleType::Float
                        },
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
    let descriptor = || RasterPipelineDesc {
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
        color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
        primitive: triangle_list_state(),
        depth_stencil: None,
        multisample: single_sample_state(u64::MAX),
        multiview: None,
        cache: None,
    };
    let pipeline = d.raster_pipeline(descriptor()).await.unwrap();
    let second_pipeline = d.raster_pipeline(descriptor()).await.unwrap();
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
        [4, 4, 1],
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    ) {
        Ok(t) => t,
        Err(Error::Unsupported(reason)) => {
            eprintln!("SKIP {format:?}: {reason}");
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&format!("SKIP {format:?}: {reason}").into());
            return;
        }
        Err(e) => panic!("create {format:?}: {e:?}"),
    };
    let plain = texture(
        if signed {
            TextureFormat::Rgba32Float
        } else {
            TextureFormat::Rgba8Unorm
        },
        [4, 4, 1],
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
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
    let plain_v = view(
        &plain,
        if signed {
            TextureFormat::Rgba32Float
        } else {
            TextureFormat::Rgba8Unorm
        },
    );
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
    // ETC2 zero base, table zero and selector zero decode to 2/255;
    // EAC uses distinct nonzero bases in each channel to check decoding.
    let mut bc = vec![0u8; if alpha || two_channels { 16 } else { 8 }];
    if matches!(
        format,
        TextureFormat::Etc2Rgb8A1Unorm | TextureFormat::Etc2Rgb8A1UnormSrgb
    ) {
        bc[3] = 2;
    }
    if !color {
        bc[0] = if signed { 192 } else { 128 };
        if two_channels {
            bc[8] = if signed { 32 } else { 192 };
        }
    }
    let red = if signed {
        64
    } else if color {
        if srgb { 0 } else { 2 }
    } else {
        128
    };
    let green = if signed {
        if two_channels { 159 } else { 128 }
    } else if color {
        red
    } else if two_channels {
        192
    } else {
        0
    };
    let mut rgba = Vec::new();
    for _ in 0..16 {
        if signed {
            for v in [-0.5f32, if two_channels { 0.25 } else { 0.0 }, 0.0, 1.0] {
                rgba.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            rgba.extend_from_slice(&[red, green, 0, 255]);
        }
    }
    for (t, data, bpr, rows) in [
        (&plain, rgba.as_slice(), if signed { 64 } else { 16 }, 4),
        (&compressed, bc.as_slice(), bc.len() as u32, 1),
    ] {
        e.write_texture(
            TextureRegion {
                texture: t,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: bpr,
                rows_per_image: rows,
            },
            data,
        )
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
        p.set_pipeline(&second_pipeline).unwrap();
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
    for x in [1usize, 6] {
        let actual = &data[4 * 32 + x * 4..4 * 32 + x * 4 + 4];
        eprintln!("{format:?} compressed rebind pixel {x}: {actual:?}");
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("EXECUTED {format:?} pixel {x}: {actual:?}").into());
        for (actual, expected) in actual.iter().zip([red, green, red, green]) {
            assert!(
                (i16::from(*actual) - i16::from(expected)).abs() <= 1,
                "{format:?}: {actual} != {expected}"
            );
        }
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_compressed_etc_matrix() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(compressed_etc_matrix(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_compressed_etc_matrix() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(compressed_etc_matrix(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_compressed_etc_matrix() {
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
    compressed_etc_matrix(&d).await;
}
