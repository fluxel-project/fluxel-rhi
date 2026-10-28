//! Core cube view face selection and upper-left orientation.
use super::*;

async fn binding_cube(d: &Device<Gl>) {
    let source = br#"
@group(0) @binding(0) var a: texture_cube<f32>;
@group(0) @binding(1) var s: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f {
 let face = min(u32(p.x), 5u); let v = select(0.5,-0.5,p.y < 4.0);
 let dirs = array<vec3f,6>(vec3f(1.,-v,-0.5),vec3f(-1.,-v,0.5),vec3f(-0.5,1.,v),vec3f(-0.5,-1.,-v),vec3f(-0.5,-v,1.),vec3f(0.5,-v,-1.));
 if p.x >= 6.0 { return textureSampleLevel(a,s,vec3f(1.,0.999,0.),0.); }
 return textureSampleLevel(a,s,dirs[face],0.);
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
            extent: [2, 2, 1],
            array_layers: 6,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
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
    let iv = d
        .create_view(
            &input,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::Cube,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
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
            mip_filter: FilterMode::Nearest,
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
            size: 256,
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
    let mut data = [0u8; 96];
    for face in 0..6 {
        for y in 0..2 {
            for x in 0..2 {
                let i = face * 16 + y * 8 + x * 4;
                data[i..i + 4].copy_from_slice(&[
                    (face as u8 + 1) * 30,
                    if y == 0 { 64 } else { 192 },
                    if x == 0 { 32 } else { 224 },
                    255,
                ]);
            }
        }
    }
    e.write_texture(
        TextureRegion {
            texture: &input,
            mip: 0,
            origin: [0, 0, 0],
            extent: [2, 2, 6],
            aspect: TextureAspects::COLOR,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 8,
            rows_per_image: 2,
        },
        &data,
    )
    .unwrap();
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
    p.set_bind_group(0, &group, &[]).unwrap();
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
    // SAFETY: point covers draw and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for face in 0..6 {
        for (row, green) in [(1, 64), (6, 192)] {
            let i = row * 32 + face * 4;
            assert_eq!(
                &bytes[i..i + 4],
                &[
                    (face as u8 + 1) * 30,
                    green,
                    if face < 2 { 224 } else { 32 },
                    255
                ],
                "face {face} row {row}"
            );
        }
    }
    let seam = &bytes[1 * 32 + 6 * 4..1 * 32 + 6 * 4 + 4];
    eprintln!("cube +X/+Y linear seam: {seam:?}");
    // Equal weighting at the +X/+Y boundary gives red approximately 60.
    assert!(
        (58..=62).contains(&seam[0]),
        "unexpected cube seam {seam:?}"
    );
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_cube() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_cube(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_cube() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_cube(&d));
}
