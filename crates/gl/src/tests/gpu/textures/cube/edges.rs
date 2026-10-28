//! Cube edge filtering, mip interpolation, gradients, and shared attachment alias.
use super::*;

async fn cube_edges(d: &Device<Gl>) {
    let source = br#"
@group(0) @binding(0) var a: texture_cube<f32>;
@group(0) @binding(1) var s: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);}
@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f {
 let dirs=array<vec3f,24>(vec3f(1.000,0.999,0.000),vec3f(0.999,1.000,0.000),vec3f(1.000,-0.999,0.000),vec3f(0.999,-1.000,0.000),vec3f(-1.000,0.999,0.000),vec3f(-0.999,1.000,0.000),vec3f(-1.000,-0.999,0.000),vec3f(-0.999,-1.000,0.000),vec3f(1.000,0.000,0.999),vec3f(0.999,0.000,1.000),vec3f(1.000,0.000,-0.999),vec3f(0.999,0.000,-1.000),vec3f(-1.000,0.000,0.999),vec3f(-0.999,0.000,1.000),vec3f(-1.000,0.000,-0.999),vec3f(-0.999,0.000,-1.000),vec3f(0.000,1.000,0.999),vec3f(0.000,0.999,1.000),vec3f(0.000,1.000,-0.999),vec3f(0.000,0.999,-1.000),vec3f(0.000,-1.000,0.999),vec3f(0.000,-0.999,1.000),vec3f(0.000,-1.000,-0.999),vec3f(0.000,-0.999,-1.000));let v=dirs[min(u32(p.x),23u)];
 if p.y < 1.0 {return textureSampleLevel(a,s,v,0.);}
 if p.y < 2.0 {return textureSampleLevel(a,s,v,0.5);}
 return textureSampleGrad(a,s,v,vec3f(10.,0.,0.),vec3f(0.,10.,0.));
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
    let entry = BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::SampledTexture {
            sample: TextureSampleType::Float,
            dimension: TextureViewDimension::Cube,
            multisampled: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry,
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
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
    let input = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 6,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 3,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING
                | TextureUsage::COPY_DST
                | TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let out = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [24, 3, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let iv = d
        .create_view(
            &input,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::Cube,
                aspects: TextureAspects::COLOR,
                mips: 0..3,
                layers: 0..6,
            },
        )
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
    let sampler = d
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..2.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&iv),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        })
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 288,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 155).unwrap();
    // SAFETY: resources and context remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for mip in 0..3u32 {
        let n = 4 >> mip;
        for face in 0..6u32 {
            let color = [20 + (face as u8) * 30, (mip as u8) * 100, 0, 255];
            let mut data = Vec::new();
            for _ in 0..n * n {
                data.extend_from_slice(&color);
            }
            e.write_texture(
                TextureRegion {
                    texture: &input,
                    mip,
                    origin: [0, 0, face],
                    extent: [n, n, 1],
                    aspect: TextureAspects::COLOR,
                },
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: n * 4,
                    rows_per_image: n,
                },
                &data,
            )
            .unwrap();
        }
    }
    // Render through a separate array-layer attachment alias, then sample the cube view.
    let alias = d
        .create_view(
            &input,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 5..6,
            },
        )
        .unwrap();
    let alias_colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&alias),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([200.0 / 255.0, 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let alias_pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &alias_colors,
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
    alias_pass.finish().unwrap();
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
                width: 24,
                height: 3,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    p.set_pipeline(&pipeline).unwrap();
    p.set_bind_group(0, &group, &[]).unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &out,
            mip: 0,
            origin: [0, 0, 0],
            extent: [24, 3, 1],
            aspect: TextureAspects::COLOR,
        },
        &buffer,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 96,
            rows_per_image: 3,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&buffer, 0..288).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: point covers draw and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let face_red = [20i32, 50, 80, 110, 140, 200];
    let pairs = [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (0, 4),
        (0, 5),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ];
    for row in 0..3usize {
        for x in 0..24usize {
            let (a, b) = pairs[x / 2];
            let expected = (face_red[a] + face_red[b]) / 2
                - if a == 5 || b == 5 {
                    if row == 1 {
                        8
                    } else if row == 2 {
                        15
                    } else {
                        0
                    }
                } else {
                    0
                };
            let pixel = &bytes[row * 96 + x * 4..row * 96 + x * 4 + 4];
            assert!(
                (i32::from(pixel[0]) - expected).abs() <= 3,
                "row {row}, edge {}, side {}: {pixel:?} expected red {expected}",
                x / 2,
                x % 2
            );
            let green = if row == 1 {
                50i32
            } else if row == 2 {
                200
            } else {
                0
            };
            assert!(
                (i32::from(pixel[1]) - green).abs() <= 2,
                "row {row}, edge {}: {pixel:?}",
                x / 2
            );
        }
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_cube_edges() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(cube_edges(&d));
}
