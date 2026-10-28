//! Core cube view face selection and upper-left orientation.
use super::*;

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
async fn cube_mirror_mips(d: &Device<Gl>) {
    // SAFETY: this fixture owns the current native context for its full lifetime.
    let native = unsafe { d.native_ref() };
    native
        .0
        .with_native("force cube mirror fixture", |provider| {
            provider.disable_texture_views_for_cube_mirror_test();
            Ok(())
        })
        .unwrap();
    let source = br#"
@group(0) @binding(0) var a: texture_cube<f32>;
@group(0) @binding(1) var s: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f {
 let face = min(u32(p.x), 5u); let v = 0.0;
 let dirs = array<vec3f,6>(vec3f(1.,-v,-0.5),vec3f(-1.,-v,0.5),vec3f(-0.5,1.,v),vec3f(-0.5,-1.,-v),vec3f(-0.5,-v,1.),vec3f(0.5,-v,-1.));
 if p.x >= 6.0 { return textureSampleLevel(a,s,vec3f(1.,0.999,0.),0.); }
 return textureSampleLevel(a,s,dirs[face],select(0.,1.,p.y >= 4.0));
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
            array_layers: 12,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
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
                mips: 0..2,
                layers: 0..6,
            },
        )
        .unwrap();
    let iv_b = d
        .create_view(
            &input,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::Cube,
                aspects: TextureAspects::COLOR,
                mips: 0..2,
                layers: 6..12,
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
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
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
    let group_b = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&iv_b),
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
            size: 768,
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
    for mip in 0..2u32 {
        let size = 2u32 >> mip;
        let mut data = vec![0u8; (size * size * 12 * 4) as usize];
        for face in 0..12u32 {
            for y in 0..size {
                for x in 0..size {
                    let i = ((face * size * size + y * size + x) * 4) as usize;
                    data[i..i + 4].copy_from_slice(&[
                        ((face + 1) * 12) as u8,
                        (40 + mip * 100) as u8,
                        0,
                        255,
                    ]);
                }
            }
        }
        e.write_texture(
            TextureRegion {
                texture: &input,
                mip,
                origin: [0, 0, 0],
                extent: [size, size, 12],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: size * 4,
                rows_per_image: size,
            },
            &data,
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
    for (group, offset) in [(&group_b, 256), (&group, 512)] {
        if offset == 512 {
            for mip in 0..2u32 {
                let size = 2u32 >> mip;
                let mut data = vec![0u8; (size * size * 6 * 4) as usize];
                for face in 0..6u32 {
                    for pixel in 0..size * size {
                        let i = ((face * size * size + pixel) * 4) as usize;
                        data[i..i + 4].copy_from_slice(&[
                            ((face + 1) * 12) as u8,
                            (80 + mip * 100) as u8,
                            0,
                            255,
                        ]);
                    }
                }
                e.write_texture(
                    TextureRegion {
                        texture: &input,
                        mip,
                        origin: [0, 0, 0],
                        extent: [size, size, 6],
                        aspect: TextureAspects::COLOR,
                    },
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: size * 4,
                        rows_per_image: size,
                    },
                    &data,
                )
                .unwrap();
            }
        }
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
                offset,
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
    // SAFETY: point covers draw and readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for (pass, first_face) in [(0, 0), (1, 6), (2, 0)] {
        for mip in 0..2 {
            for face in 0..6 {
                let row = if mip == 0 { 1 } else { 6 };
                let i = pass * 256 + row * 32 + face * 4;
                let expected = [
                    ((first_face + face + 1) as u8) * 12,
                    (if pass == 2 { 80 } else { 40 }) + mip as u8 * 100,
                    0,
                    255,
                ];
                assert_eq!(
                    &bytes[i..i + 4],
                    &expected,
                    "pass {pass} mip {mip} face {face}"
                );
            }
        }
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_cube_mirror_mips() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(cube_mirror_mips(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_cube_mirror_mips() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(cube_mirror_mips(&d));
}
