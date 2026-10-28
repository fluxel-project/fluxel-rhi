//! Compressed and uncompressed texture gather rebinding on a single Core pipeline.
use super::*;
async fn compressed_gather(d: &Device<Gl>) {
    let source = br#"
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f {
 return textureGather(0,image,nearest,vec2f(0.25,0.25));
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
    let textures = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(
                    0,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
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
        TextureFormat::Bc1RgbaUnorm,
        [4, 4, 1],
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    ) {
        Ok(t) => t,
        Err(Error::Unsupported(reason)) => {
            eprintln!("SKIP BC1: {reason}");
            return;
        }
        Err(e) => panic!("create BC1: {e:?}"),
    };
    let plain = texture(
        TextureFormat::Rgba8Unorm,
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
    let plain_v = view(&plain, TextureFormat::Rgba8Unorm);
    let compressed_v = view(&compressed, TextureFormat::Bc1RgbaUnorm);
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
    // BC1 endpoints 0xf800 (red), 0x001f (blue); 2-bit selector 0=red, 1=blue.
    // Four rows differ, including the first two pixels in each row.
    let bc = [0x00, 0xf8, 0x1f, 0x00, 0x44, 0x11, 0x55, 0x11];
    let mut rgba = [0u8; 64];
    for y in 0..4 {
        for x in 0..4 {
            let sel = (bc[4 + y] >> (x * 2)) & 3;
            let off = (y * 4 + x) * 4;
            rgba[off..off + 4].copy_from_slice(if sel == 0 {
                &[255, 0, 0, 255]
            } else {
                &[0, 0, 255, 255]
            });
        }
    }
    for (t, data, bpr, rows) in [
        (&plain, rgba.as_slice(), 16, 4),
        (&compressed, bc.as_slice(), 8, 1),
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
        eprintln!("compressed rebind pixel {x}: {actual:?}");
        assert_eq!(actual, &[0, 255, 0, 255]);
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_compressed_gather() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(compressed_gather(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_compressed_gather() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(compressed_gather(&d));
}
