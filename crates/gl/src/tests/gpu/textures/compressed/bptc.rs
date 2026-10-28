//! BPTC upload and decode through the public Core texture path.
use super::*;
async fn compressed_bptc(d: &Device<Gl>) {
    for (format, is_bc7) in [
        (TextureFormat::Bc6hRgbUfloat, false),
        (TextureFormat::Bc6hRgbFloat, false),
        (TextureFormat::Bc7RgbaUnorm, true),
        (TextureFormat::Bc7RgbaUnormSrgb, true),
    ] {
        compressed_bptc_case(d, format, is_bc7).await;
    }
}

async fn compressed_bptc_case(d: &Device<Gl>, format: TextureFormat, is_bc7: bool) {
    let source: &[u8] = if is_bc7 {
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
 return vec4f(v.r, v.g * 0.25 + 0.5, v.r, v.g * 0.25 + 0.5);
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
                        sample: if is_bc7 {
                            TextureSampleType::Float
                        } else {
                            TextureSampleType::UnfilterableFloat
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
        if is_bc7 {
            TextureFormat::Rgba8Unorm
        } else {
            TextureFormat::Rgba32Float
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
        if is_bc7 {
            TextureFormat::Rgba8Unorm
        } else {
            TextureFormat::Rgba32Float
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
    // BC6H mode 3 stores six 10-bit endpoints consecutively. Constant
    // endpoints plus zero selectors produce the same value in each texel.
    // Values below follow the BC6H unquantization and FP16 conversion rules.
    let signed = format == TextureFormat::Bc6hRgbFloat;
    let (red, green, rcode, gcode) = if signed {
        (0.413330078125f32, -0.1424560546875f32, 225u128, 824u128)
    } else {
        (0.409423828125f32, 0.1405029296875f32, 450u128, 400u128)
    };
    let bc = if is_bc7 {
        ((1u128 << 6) | (127u128 << 7) | (127u128 << 14)).to_le_bytes()
    } else {
        (3u128 | (rcode << 5) | (gcode << 15) | (rcode << 35) | (gcode << 45)).to_le_bytes()
    };
    let expected = if is_bc7 {
        let red = if format == TextureFormat::Bc7RgbaUnormSrgb {
            253
        } else {
            254
        };
        [red, 0, red, 0]
    } else {
        let r = (red * 255.0).round() as u8;
        let g = ((green * 0.25 + 0.5) * 255.0).round() as u8;
        [r, g, r, g]
    };
    let mut rgba = Vec::new();
    for _ in 0..16 {
        if is_bc7 {
            rgba.extend_from_slice(&[expected[0], 0, 0, 255]);
        } else {
            for v in [red, green, 0.0, 1.0] {
                rgba.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    for (t, data, bpr, rows) in [
        (&plain, rgba.as_slice(), if is_bc7 { 16 } else { 64 }, 4),
        (&compressed, bc.as_slice(), 16, 1),
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
        for (actual, expected) in actual.iter().zip(expected) {
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
fn wgl_core_compressed_bptc() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(compressed_bptc(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_compressed_bptc() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(compressed_bptc(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_compressed_bptc() {
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
    compressed_bptc(&d).await;
}
