//! Stencil mirror extraction following a zero sample mask.
use super::*;

async fn stencil_extract_mask(d: &Device<Gl>) {
    let compute = d.compute().expect("GLES 3.1 compute required");
    let source = br#"
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.,1.);
}
@fragment fn fs()->@location(0) vec4f { return vec4f(1.); }
@group(0) @binding(0) var tex: texture_2d_array<u32>;
@group(0) @binding(1) var<storage,read_write> out: array<u32>;
@compute @workgroup_size(1) fn cs() {
 out[0]=textureLoad(tex,vec2i(0),0,0).x;
 out[1]=textureLoad(tex,vec2i(0),1,0).x;
 out[2]=textureLoad(tex,vec2i(0),0,1).x;
 out[3]=textureLoad(tex,vec2i(0),1,1).x;
}
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
                ShaderEntryPoint {
                    name: "cs".into(),
                    stage: ShaderStage::Compute,
                    workgroup_size: Some([1, 1, 1]),
                },
            ],
            required_features: vec![],
        })
        .await
        .unwrap();
    let empty = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let raster = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout: empty,
            vertex: PipelineStage {
                module: shader.clone(),
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader.clone(),
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: 0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let bgl = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(
                    0,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Uint,
                        dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                ),
                entry(
                    1,
                    BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 16,
                        access: StorageAccess::ReadWrite,
                    },
                ),
            ],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&bgl],
            push_constants: &[],
        })
        .unwrap();
    let cp = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
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
    let view = |m, l| {
        d.create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth24PlusStencil8,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::STENCIL,
                mips: m..m + 1,
                layers: l..l + 1,
            },
        )
        .unwrap()
    };
    let cells = [
        (view(0, 0), 11, 4),
        (view(0, 1), 18, 4),
        (view(1, 0), 41, 2),
        (view(1, 1), 48, 2),
    ];
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
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&sampled),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer {
                        buffer: &output,
                        range: 0..16,
                    },
                },
            ],
        })
        .unwrap();
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })
        .unwrap();
    let cv = d
        .create_view(
            &color,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 179).unwrap();
    // SAFETY: resources survive through the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for (v, value, size) in &cells {
        let p = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(v),
                    depth: None,
                    stencil: Some((LoadOp::Clear(*value), StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: *size,
                    height: *size,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        p.finish().unwrap();
    }
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&cv),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.; 4])),
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
                width: 4,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    p.set_pipeline(&raster).unwrap();
    p.draw(0..3, 0..1).unwrap();
    p.finish().unwrap();
    e.write_buffer(&output, 0, &[0; 16]).unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&cp).unwrap();
    p.set_bind_group(0, &group, &[]).unwrap();
    p.dispatch([1, 1, 1]).unwrap();
    p.finish().unwrap();
    let ticket = e.readback_buffer(&output, 0..16).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers readback.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let actual: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    eprintln!("stencil extract after zero sample mask: {actual:?}");
    assert_eq!(actual, [11, 18, 41, 48]);
}

#[test]
#[ignore = "requires GLES 3.1 hardware context"]
fn gles_core_stencil_extract_mask() {
    let native = GlInstance::gles_pbuffer([8, 8]).expect("GLES pbuffer");
    let d = block_on(open(native));
    block_on(stencil_extract_mask(&d));
}
