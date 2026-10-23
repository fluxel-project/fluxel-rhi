//! WebGL2 bounded runtime texture/sampler arrays through ordinary slots.
//!
//! This is deliberately a uniform selector: WebGL2 cannot promise the Core
//! nonuniform indexing feature (nor its SSBO half), while a bounded layout can
//! still specialize to legal ESSL sampler arrays. It covers a partial hole, a
//! short prefix, and group rebinds without relying on an unbounded native ABI.

use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
async fn runtime_webgl_bounded_arrays(d: &Device<Gl>) {
    let source = br#"
struct Pick { texture_index: u32, sampler_index: u32 }
@group(0) @binding(0) var images: binding_array<texture_2d<f32>>;
@group(1) @binding(0) var samplers: binding_array<sampler>;
@group(2) @binding(0) var<uniform> pick: Pick;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let points = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
  return vec4f(points[i], 0.5, 1.);
}
@fragment fn fs() -> @location(0) vec4f {
  return textureSampleLevel(images[pick.texture_index], samplers[pick.sampler_index], vec2f(0.5), 0.0);
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
    let runtime = |ty, partial| {
        let layout = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Fragment],
                    ty,
                    count: BindingCount::RuntimeSized { max_count: 3 },
                    partially_bound: partial,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        layout
    };
    let images = runtime(
        BindingType::SampledTexture {
            sample: TextureSampleType::Float,
            dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        true,
    );
    let samplers = runtime(
        BindingType::Sampler {
            filtering: false,
            comparison: false,
        },
        false,
    );
    let picks = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    // WGSL uniform-address-space structs have a 16-byte
                    // struct alignment even though `Pick` stores two u32s.
                    // The backing range must cover that linked block size.
                    min_size: 16,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let too_large = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Fragment],
            ty: BindingType::SampledTexture {
                sample: TextureSampleType::Float,
                dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: BindingCount::RuntimeSized {
                max_count: u32::MAX,
            },
            partially_bound: false,
            nonuniform_indexing: false,
        }],
    };
    assert!(matches!(
        d.create_bind_group_layout(&too_large),
        Err(Error::Unsupported(_))
    ));
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&images, &samplers, &picks],
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
    let texture = || {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap()
    };
    let red = texture();
    let blue = texture();
    let view = |texture| {
        d.create_view(
            texture,
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
    let red_view = view(&red);
    let blue_view = view(&blue);
    let sampler = |address| {
        d.create_sampler(&SamplerDesc {
            label: None,
            address: [address; 3],
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
    let image_group = |values| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &images,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(values),
            }],
        })
        .unwrap()
    };
    let images_with_hole = image_group(vec![
        Some(BindingResource::TextureView(&red_view)),
        None,
        Some(BindingResource::TextureView(&blue_view)),
    ]);
    let short_images = image_group(vec![Some(BindingResource::TextureView(&blue_view))]);
    let sampler_group = |values| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &samplers,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(values),
            }],
        })
        .unwrap()
    };
    let both_samplers = sampler_group(vec![
        Some(BindingResource::Sampler(&repeat)),
        Some(BindingResource::Sampler(&clamp)),
    ]);
    let short_samplers = sampler_group(vec![Some(BindingResource::Sampler(&clamp))]);
    let pick_buffer = |values: [u32; 2]| {
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 16,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &picks,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &buffer,
                        range: 0..16,
                    },
                }],
            })
            .unwrap();
        (buffer, group, values)
    };
    let pick_hole = pick_buffer([2, 1]);
    let pick_red = pick_buffer([0, 0]);
    let pick_short = pick_buffer([0, 0]);
    let out = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [6, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let out_view = view(&out);
    let readback = d
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
    // SAFETY: all resources and the adopted browser context remain live through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = match session.begin_commands() {
        Ok(run) => run,
        Err(_) => panic!("begin commands"),
    };
    let commands = run.commands();
    for (texture, color) in [(&red, [255, 0, 0, 255]), (&blue, [0, 0, 255, 255])] {
        commands
            .write_texture(
                TextureRegion {
                    texture,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [1, 1, 1],
                    aspect: TextureAspects::COLOR,
                },
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: 4,
                    rows_per_image: 1,
                },
                &color,
            )
            .unwrap();
    }
    for (buffer, _, values) in [&pick_hole, &pick_red, &pick_short] {
        let mut bytes = Vec::with_capacity(16);
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes.resize(16, 0);
        commands.write_buffer(buffer, 0, &bytes).unwrap();
    }
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&out_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
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
                width: 6,
                height: 1,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    for (x, image, sampler, pick) in [
        (0, &images_with_hole, &both_samplers, &pick_hole.1),
        (2, &images_with_hole, &both_samplers, &pick_red.1),
        (4, &short_images, &short_samplers, &pick_short.1),
    ] {
        pass.set_bind_group(0, image, &[]).unwrap();
        pass.set_bind_group(1, sampler, &[]).unwrap();
        pass.set_bind_group(2, pick, &[]).unwrap();
        pass.scissor(Rect {
            x,
            y: 0,
            width: 2,
            height: 1,
        })
        .unwrap();
        pass.draw(0..3, 0..1).unwrap();
    }
    pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &out,
                mip: 0,
                origin: [0, 0, 0],
                extent: [6, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 24,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let ticket = commands.readback_buffer(&readback, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers the readback copy.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (x, expected) in [
        (0, [0, 0, 255, 255]),
        (2, [255, 0, 0, 255]),
        (4, [0, 0, 255, 255]),
    ] {
        assert_eq!(&bytes[x * 4..x * 4 + 4], expected);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_binding_runtime_bounded_arrays() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(6);
    canvas.set_height(1);
    let device = open(GlInstance::from_webgl2_canvas(canvas).unwrap()).await;
    assert!(device.capabilities().runtime_sized_binding_arrays);
    assert!(
        !device
            .capabilities()
            .nonuniform_sampled_texture_and_storage_buffer_indexing
    );
    runtime_webgl_bounded_arrays(&device).await;
}
