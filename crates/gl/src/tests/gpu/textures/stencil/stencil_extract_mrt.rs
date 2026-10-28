//! Indexed color state survives stencil mirror extraction.
use super::*;

async fn stencil_extract_mrt(d: &Device<Gl>) {
    let source=br#"
@group(0) @binding(0) var tex: texture_2d_array<u32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.,1.); }
struct O { @location(0) a:vec4f, @location(1) b:vec4f };
@fragment fn fs()->O { return O(vec4f(f32(textureLoad(tex,vec2i(0),1,1).x)/255.,0.8,0.8,0.8),vec4f(0.8)); }
"#;
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_slice()),
            entries: vec![
                ShaderEntryPoint {
                    name: "vs".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                },
                ShaderEntryPoint {
                    name: "fs".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                },
            ],
            required_features: vec![],
        })
        .await
        .unwrap();
    let bgl = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Uint,
                    dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&bgl],
            push_constants: &[],
        })
        .unwrap();
    let add = BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    };
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![
                Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: Some(BlendState {
                        color: add,
                        alpha: add,
                    }),
                    write_mask: 1,
                }),
                Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: 2,
                }),
            ],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 2,
            format: TextureFormat::Depth24PlusStencil8,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let attachment = d
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth24PlusStencil8,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::STENCIL,
                mips: 1..2,
                layers: 1..2,
            },
        )
        .unwrap();
    let sampled = d
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth24PlusStencil8,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::STENCIL,
                mips: 0..2,
                layers: 0..2,
            },
        )
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&sampled),
            }],
        })
        .unwrap();
    let color = || {
        d.create_texture(&TextureDesc {
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
        .unwrap()
    };
    let c0 = color();
    let c1 = color();
    let color_view = |c| {
        d.create_view(
            c,
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
    let v0 = color_view(&c0);
    let v1 = color_view(&c1);
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 128,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 180).unwrap();
    // SAFETY: resources are alive through the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    let p = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&attachment),
                depth: None,
                stencil: Some((LoadOp::Clear(41), StoreOp::Store)),
                depth_read_only: false,
                stencil_read_only: false,
            }),
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
    p.finish().unwrap();
    let colors = [
        Some(ColorAttachment {
            view: AttachmentView::Texture(&v0),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.2; 4])),
            store: StoreOp::Store,
        }),
        Some(ColorAttachment {
            view: AttachmentView::Texture(&v1),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.2; 4])),
            store: StoreOp::Store,
        }),
    ];
    let mut p = e
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
    if let Err(e) = p.set_pipeline(&pipeline) {
        if matches!(
            e,
            Error::Unsupported("independent blend requires indexed draw-buffer state")
        ) {
            eprintln!("SKIP indexed draw-buffer state unavailable");
            return;
        }
        panic!("indexed pipeline: {e:?}");
    }
    p.set_bind_group(0, &group, &[]).unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.finish().unwrap();
    for (i, c) in [&c0, &c1].iter().enumerate() {
        e.texture_to_buffer(
            TextureRegion {
                texture: c,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &output,
            TexelCopyBufferLayout {
                offset: (i * 64) as u64,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&output, 0..128).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let a = &bytes[0..4];
    let b = &bytes[64..68];
    eprintln!("indexed stencil MRT: {a:?} {b:?}");
    assert_eq!(a, &[92, 51, 51, 51]);
    assert_eq!(b, &[51, 204, 51, 51]);
}

#[test]
#[ignore = "requires GLES 3.1 hardware context"]
fn gles_core_stencil_extract_mrt() {
    let native = GlInstance::gles_pbuffer([8, 8]).expect("GLES pbuffer");
    let d = block_on(open(native));
    block_on(stencil_extract_mrt(&d));
}
