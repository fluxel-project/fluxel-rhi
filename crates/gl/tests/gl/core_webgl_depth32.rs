//! WebGL2 D32 upload validation through numeric depth sampling.
//!
//! WebGL2 does not promise a raw floating-point depth readback, so this test
//! validates CPU and PBO upload separately by sampling both mip-1 depth images
//! into an RGBA8 target and reading ordinary color bytes through a PBO.
use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_depth32_cpu_and_pbo_upload_sample() {
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
    let depth = || {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [4, 4, 1],
                array_layers: 1,
                format: TextureFormat::Depth32Float,
                mip_levels: 2,
                samples: 1,
                usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
                view_formats: vec![],
            })
            .unwrap()
    };
    let cpu_depth = depth();
    let pbo_depth = depth();
    let shader_source = br#"
@group(0) @binding(0) var image: texture_depth_2d;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let positions = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  return vec4f(positions[i], 0.5, 1.);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  // WebGL2 has no native texture-view object. Keep the bind group on the
  // complete mip chain and select the uploaded mip in the shader instead.
  let value = textureLoad(image, vec2i(i32(p.x), i32(p.y)), 1);
  return vec4f(value, 0., 0., 1.);
}
"#;
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(shader_source.as_slice()),
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
                    dimension: TextureViewDimension::D2,
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
    let depth_view = |texture| {
        device
            .create_view(
                texture,
                &TextureViewDesc {
                    format: TextureFormat::Depth32Float,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::DEPTH,
                    mips: 0..2,
                    layers: 0..1,
                },
            )
            .unwrap()
    };
    let cpu_view = depth_view(&cpu_depth);
    let pbo_view = depth_view(&pbo_depth);
    let group = |view| {
        device
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                }],
            })
            .unwrap()
    };
    let cpu_group = group(&cpu_view);
    let pbo_group = group(&pbo_view);
    let output = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
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
    let upload = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let result = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 56,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut bytes = vec![0xc5; 32];
    for (row, values) in [[0.0f32, 0.25], [0.75, 1.0]].into_iter().enumerate() {
        let row_start = 4 + row * 12;
        bytes[row_start..row_start + 4].copy_from_slice(&values[0].to_ne_bytes());
        bytes[row_start + 4..row_start + 8].copy_from_slice(&values[1].to_ne_bytes());
    }
    let fixture = common::ImmediateFixture {
        device: &device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 977).unwrap();
    // SAFETY: all resources outlive the issued commands and readback ticket.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin WebGL D32 fixture"));
    let commands = run.commands();
    let depth_layout = TexelCopyBufferLayout {
        offset: 4,
        bytes_per_row: 12,
        rows_per_image: 2,
    };
    let depth_region = |texture| TextureRegion {
        texture,
        mip: 1,
        origin: [0, 0, 0],
        extent: [2, 2, 1],
        aspect: TextureAspects::DEPTH,
    };
    commands.write_buffer(&upload, 0, &bytes).unwrap();
    commands
        .write_texture(depth_region(&cpu_depth), depth_layout, &bytes)
        .unwrap();
    commands
        .buffer_to_texture(&upload, depth_layout, depth_region(&pbo_depth))
        .unwrap();
    commands.write_buffer(&result, 0, &[0xa3; 56]).unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.; 4])),
        store: StoreOp::Store,
    })];
    let draw = |commands: &mut CommandEncoder<Gl>, group: &BindGroup<Gl>| {
        let mut pass = commands
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &colors,
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
            .unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, group, &[]).unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
    };
    let result_layout = |offset| TexelCopyBufferLayout {
        offset,
        bytes_per_row: 12,
        rows_per_image: 2,
    };
    draw(commands, &cpu_group);
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &output,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            &result,
            result_layout(2),
        )
        .unwrap();
    draw(commands, &pbo_group);
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &output,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            &result,
            result_layout(30),
        )
        .unwrap();
    let ticket = commands.readback_buffer(&result, 0..56).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish WebGL D32 fixture"),
    };
    device.wait_completion(&point).await.unwrap();
    // SAFETY: point covers both upload paths, draws, color copies and ticket.
    let actual = unsafe { device.read_after(ticket, &point).await }
        .unwrap()
        .bytes;
    for base in [2usize, 30] {
        assert_eq!(&actual[base..base + 4], &[0, 0, 0, 255]);
        assert_eq!(&actual[base + 4..base + 8], &[64, 0, 0, 255]);
        assert_eq!(&actual[base + 12..base + 16], &[191, 0, 0, 255]);
        assert_eq!(&actual[base + 16..base + 20], &[255, 0, 0, 255]);
        assert_eq!(&actual[base + 8..base + 12], &[0xa3; 4]);
    }
    for index in [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 50, 55] {
        assert_eq!(actual[index], 0xa3, "padding byte {index} changed");
    }
}
