//! WebGL2 packed depth/stencil writes preserve depth and restore raster state.
use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
async fn packed_stencil_keeps_depth(
    device: &Device<Gl>,
    format: TextureFormat,
    serial: u64,
    depth_writes: bool,
    array: bool,
) {
    let layers = if array { 2 } else { 1 };
    let packed = device
        .create_texture(&TextureDesc {
            label: None,
            // Core represents 2D arrays as a D2 texture with more than one
            // layer; only the view chooses the D2Array dimension.
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: layers,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC
                | TextureUsage::COPY_DST
                | TextureUsage::TEXTURE_BINDING
                | TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let output = device
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
    let packed_view = device
        .create_view(
            &packed,
            &TextureViewDesc {
                format,
                dimension: if array {
                    TextureViewDimension::D2Array
                } else {
                    TextureViewDimension::D2
                },
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..layers,
            },
        )
        .unwrap();
    let output_view = device
        .create_view(
            &output,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let source = if array {
        r#"
@group(0) @binding(0) var image: texture_depth_2d_array;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  return vec4f(p[i], 0.5, 1.);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(image, vec2i(i32(p.x), i32(p.y)), 1, 0), 0., 0., 1.);
}
"#
    } else {
        r#"
@group(0) @binding(0) var image: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  return vec4f(p[i], 0.5, 1.);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(image, vec2i(i32(p.x), i32(p.y)), 0), 0., 0., 1.);
}
"#
    };
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = device
        .shader(shader("vs", ShaderStage::Vertex))
        .await
        .unwrap();
    let fragment = device
        .shader(shader("fs", ShaderStage::Fragment))
        .await
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Depth,
                    dimension: if array {
                        TextureViewDimension::D2Array
                    } else {
                        TextureViewDimension::D2
                    },
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = device
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
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&packed_view),
            }],
        })
        .unwrap();
    let transfer = device
        .create_buffer(&BufferDesc {
            label: None,
            // Array depth source uses offset 4 plus two 3-row images at a
            // 12-byte row stride, so it needs more than the stencil staging.
            size: 96,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let result = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut cpu_stencil = [0x9d; 16];
    cpu_stencil[3..5].copy_from_slice(&[0x11, 0x22]);
    cpu_stencil[7..9].copy_from_slice(&[0x33, 0x44]);
    let mut pbo_stencil = [0x9d; 16];
    pbo_stencil[3..5].copy_from_slice(&[0xaa, 0xbb]);
    pbo_stencil[7..9].copy_from_slice(&[0xcc, 0xdd]);
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, serial).unwrap();
    // SAFETY: every GPU object remains live through the returned completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin packed stencil preservation"));
    let commands = run.commands();
    commands
        .clear_texture(&packed, 0..1, 0..layers, TextureClearValue::Depth(0.25))
        .unwrap();
    let stencil_layout = TexelCopyBufferLayout {
        offset: 3,
        bytes_per_row: 4,
        rows_per_image: 2,
    };
    let cpu_region = TextureRegion {
        texture: &packed,
        mip: 0,
        origin: [1, 1, u32::from(array)],
        extent: [2, 2, 1],
        aspect: TextureAspects::STENCIL,
    };
    let pbo_region = TextureRegion {
        texture: &packed,
        mip: 0,
        origin: [0, 0, u32::from(array)],
        extent: [2, 2, 1],
        aspect: TextureAspects::STENCIL,
    };
    commands
        .write_texture(cpu_region, stencil_layout, &cpu_stencil)
        .unwrap();
    commands.write_buffer(&transfer, 0, &pbo_stencil).unwrap();
    commands
        .buffer_to_texture(&transfer, stencil_layout, pbo_region)
        .unwrap();
    if depth_writes {
        assert_eq!(format, TextureFormat::Depth32FloatStencil8);
        // CPU and PBO write disjoint depth rectangles after the stencil work.
        // The source layouts carry nonzero offsets and four bytes of row padding.
        let rows_per_image = if array { 3 } else { 2 };
        let mut cpu_depth = [0x8a; 72];
        let mut pbo_depth = [0x8a; 72];
        for image in 0..layers as usize {
            for row in 0..2usize {
                let offset = 4 + image * rows_per_image as usize * 12 + row * 12;
                cpu_depth[offset..offset + 4].copy_from_slice(&0.5f32.to_ne_bytes());
                cpu_depth[offset + 4..offset + 8].copy_from_slice(&0.5f32.to_ne_bytes());
                pbo_depth[offset..offset + 4].copy_from_slice(&0.75f32.to_ne_bytes());
                pbo_depth[offset + 4..offset + 8].copy_from_slice(&0.75f32.to_ne_bytes());
            }
        }
        let depth_layout = TexelCopyBufferLayout {
            offset: 4,
            bytes_per_row: 12,
            rows_per_image,
        };
        commands
            .write_texture(
                TextureRegion {
                    texture: &packed,
                    mip: 0,
                    origin: [2, 0, 0],
                    extent: [2, 2, layers],
                    aspect: TextureAspects::DEPTH,
                },
                depth_layout,
                &cpu_depth,
            )
            .unwrap();
        commands.write_buffer(&transfer, 0, &pbo_depth).unwrap();
        commands
            .buffer_to_texture(
                &transfer,
                depth_layout,
                TextureRegion {
                    texture: &packed,
                    mip: 0,
                    origin: [0, 2, 0],
                    extent: [2, 2, layers],
                    aspect: TextureAspects::DEPTH,
                },
            )
            .unwrap();
    }
    commands.write_buffer(&result, 0, &[0xe3; 256]).unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &packed,
                mip: 0,
                origin: [0, 0, u32::from(array)],
                extent: [4, 4, 1],
                aspect: TextureAspects::STENCIL,
            },
            &result,
            TexelCopyBufferLayout {
                offset: 3,
                bytes_per_row: 8,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.; 4])),
        store: StoreOp::Store,
    })];
    let mut pass = commands
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
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &output,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &result,
            TexelCopyBufferLayout {
                offset: 128,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = commands.readback_buffer(&result, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish packed stencil preservation"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the clear, two stencil writes, readback and draw.
    let bytes = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    for pixel in 0..16usize {
        let row = pixel / 4;
        let column = pixel % 4;
        let expected_depth = if depth_writes && column >= 2 && row < 2 {
            128
        } else if depth_writes && column < 2 && row >= 2 {
            191
        } else {
            64
        };
        assert_eq!(
            &bytes[128 + pixel * 4..128 + pixel * 4 + 4],
            &[expected_depth, 0, 0, 255],
            "{format:?} depth changed at {column},{row}"
        );
    }
    // Depth-only clear leaves stencil's other pixels unspecified. Assert only
    // the two rectangles whose plane values this fixture wrote, including the
    // PBO write that intentionally overwrites CPU pixel (1, 1).
    for ((column, row), expected) in [
        ((0, 0), 0xaa),
        ((1, 0), 0xbb),
        ((0, 1), 0xcc),
        ((1, 1), 0xdd),
        ((2, 1), 0x22),
        ((1, 2), 0x33),
        ((2, 2), 0x44),
    ] {
        assert_eq!(
            bytes[3 + row * 8 + column],
            expected,
            "{format:?} stencil {column},{row}"
        );
    }
    for index in [0, 1, 2, 7, 15, 31, 35, 127, 192, 255] {
        assert_eq!(bytes[index], 0xe3, "{format:?} untouched byte {index}");
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_packed_stencil_writes_keep_depth() {
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
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    for (index, format) in [
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ]
    .into_iter()
    .enumerate()
    {
        packed_stencil_keeps_depth(&device, format, 985 + index as u64, false, false).await;
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_packed_depth_writes_keep_stencil() {
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
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    packed_stencil_keeps_depth(
        &device,
        TextureFormat::Depth32FloatStencil8,
        987,
        true,
        false,
    )
    .await;
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_packed_depth_array_writes_keep_stencil() {
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
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    packed_stencil_keeps_depth(
        &device,
        TextureFormat::Depth32FloatStencil8,
        988,
        true,
        true,
    )
    .await;
}
