//! Dynamic storage-buffer array holes and rebinding through real GPU draws.
use super::*;

async fn binding_partial_ssbo(d: &Device<Gl>, public: bool) {
    if public && !d.capabilities().binding_arrays {
        eprintln!("SKIP public SSBO hole: binding_arrays unavailable");
        return;
    }
    let source = br#"
struct Data { color: vec4f, tail: array<vec4f, 16> };
@group(0) @binding(0) var<storage, read> buffers: binding_array<Data, 4>;
@group(0) @binding(1) var<uniform> selector: vec4u;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f {
 let index = selector.x;
 if pos.x > 100. { return buffers[index].tail[15]; }
 return buffers[index].color;
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
    let mut desc = BindGroupLayoutDesc {
        label: None,
        entries: vec![entry(
            0,
            BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: 16,
                access: StorageAccess::Read,
            },
        )],
    };
    desc.entries[0].count = BindingCount::Fixed(4);
    desc.entries.push(entry(
        1,
        BindingType::UniformBuffer {
            dynamic_offset: false,
            min_size: 16,
        },
    ));
    desc.entries[0].partially_bound = true;
    // SAFETY: the layout and wrapper belong to this live device.
    let uniforms = if public {
        d.create_bind_group_layout(&desc).unwrap()
    } else {
        unsafe {
            BindGroupLayout::from_native(
                d.id(),
                Gl::create_bind_group_layout(d.native_ref(), &desc).unwrap(),
                Arc::from(desc.entries.clone()),
            )
        }
    };
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&uniforms],
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
    let uniform = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 272,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let other = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 272,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let extras: Vec<_> = (0..3)
        .map(|_| {
            d.create_buffer(&BufferDesc {
                label: None,
                size: 272,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap()
        })
        .collect();
    let selector = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let group = |full: bool| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &uniforms,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(vec![
                        Some(BindingResource::Buffer {
                            buffer: if full { &uniform } else { &other },
                            range: 0..272,
                        }),
                        full.then_some(BindingResource::Buffer {
                            buffer: &extras[0],
                            range: 0..272,
                        }),
                        full.then_some(BindingResource::Buffer {
                            buffer: &extras[1],
                            range: 0..272,
                        }),
                        full.then_some(BindingResource::Buffer {
                            buffer: &extras[2],
                            range: 0..272,
                        }),
                    ]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer {
                        buffer: &selector,
                        range: 0..16,
                    },
                },
            ],
        })
        .unwrap()
    };
    let complete = group(true);
    let partial = group(false);
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
            size: 768,
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
    let color: Vec<u8> = [1.0f32, 0., 0., 1.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    e.write_buffer(&uniform, 0, &color).unwrap();
    let green: Vec<u8> = [0.0f32, 1., 0., 1.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    e.write_buffer(&other, 0, &green).unwrap();
    e.write_buffer(&selector, 0, &[0; 16]).unwrap();
    for (step, group) in [&complete, &partial, &complete].into_iter().enumerate() {
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
        p.set_bind_group(0, group, &[]).unwrap();
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
                offset: (step * 256) as u64,
                bytes_per_row: 32,
                rows_per_image: 8,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&buffer, 0..768).unwrap();
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
    let center = 4 * 32 + 4 * 4;
    for (i, expected) in [[255, 0, 0, 255], [0, 255, 0, 255], [255, 0, 0, 255]]
        .iter()
        .enumerate()
    {
        assert_eq!(&data[i * 256 + center..i * 256 + center + 4], expected);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_partial_ssbo() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_partial_ssbo(&d, false));
    block_on(binding_partial_ssbo(&d, true));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_partial_ssbo() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_partial_ssbo(&d, false));
    block_on(binding_partial_ssbo(&d, true));
}
