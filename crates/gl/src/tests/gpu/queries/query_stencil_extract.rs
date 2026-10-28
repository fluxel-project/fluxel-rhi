//! Occlusion query segments exclude draws used for stencil mirror extraction.
use super::*;

async fn query_stencil_extract(d: &Device<Gl>) {
    let shader=d.shader(ShaderDesc { label:None,format:ShaderFormat::Wgsl,bytes:Arc::from(br#"
@group(0) @binding(0) var stencil: texture_2d_array<u32>;
@vertex fn clipped(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(2.,2.),vec2f(3.,2.),vec2f(2.,3.)); return vec4f(p[i],0.,1.);
}
@vertex fn visible(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.,1.);
}
@fragment fn fs()->@location(0) vec4f { return vec4f(f32(textureLoad(stencil,vec2i(0),0,0).x)/255.,0.,0.,1.); }
"#.as_slice()), entries:vec![ShaderEntryPoint {name:"clipped".into(),stage:ShaderStage::Vertex,workgroup_size:None},ShaderEntryPoint {name:"visible".into(),stage:ShaderStage::Vertex,workgroup_size:None},ShaderEntryPoint {name:"fs".into(),stage:ShaderStage::Fragment,workgroup_size:None}],required_features:vec![] }).await.unwrap();
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
    let make_pipeline = |entry: &'static str| RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: shader.clone(),
            entry: entry.into(),
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
        multisample: single_sample_state(u64::MAX),
        multiview: None,
        cache: None,
    };
    let clipped = d.raster_pipeline(make_pipeline("clipped")).await.unwrap();
    let visible = d.raster_pipeline(make_pipeline("visible")).await.unwrap();
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 2,
            format: TextureFormat::Depth24PlusStencil8,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let stencil = d
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth24PlusStencil8,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::STENCIL,
                mips: 0..1,
                layers: 0..2,
            },
        )
        .unwrap();
    // SAFETY: the view remains live and this reads its backend-owned metadata only.
    let native_view = unsafe { stencil.native_ref() };
    assert!(
        native_view.stencil_mirror.is_some(),
        "fixture must exercise stencil extraction rather than a native view"
    );
    let dv = d
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth24PlusStencil8,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::STENCIL,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&stencil),
            }],
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
    let queries = d.queries(QueryKind::Occlusion).unwrap();
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 1,
        })
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    for (round, (first_visible, last_visible)) in
        [(false, false), (true, false), (false, true), (false, false)]
            .into_iter()
            .enumerate()
    {
        let expected = first_visible || last_visible;
        let plan = common::single_commands_plan(&fixture, 9000 + round as u64).unwrap();
        // SAFETY: all resources remain live through completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
        let e = run.commands();
        queries.reset(e, &set, 0..1).unwrap();
        let p = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&dv),
                    depth: None,
                    stencil: Some((LoadOp::Clear(7), StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
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
        p.finish().unwrap();
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
                occlusion_query_set: Some(&set),
                secondary_inheritance: None,
            })
            .unwrap();
        queries.begin_occlusion(&mut p, &set, 0).unwrap();
        p.set_pipeline(&clipped).unwrap();
        p.set_bind_group(0, &group, &[]).unwrap();
        if first_visible {
            p.set_pipeline(&visible).unwrap();
            p.draw(0..3, 0..1).unwrap();
            p.set_pipeline(&clipped).unwrap();
        }
        p.draw(0..3, 0..1).unwrap();
        p.draw(0..3, 0..1).unwrap();
        if last_visible {
            p.set_pipeline(&visible).unwrap();
            p.draw(0..3, 0..1).unwrap();
        }
        queries.end_occlusion(&mut p).unwrap();
        p.finish().unwrap();
        assert_eq!(queries.resolve_mode(), QueryResolveMode::AfterCompletion);
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish"),
        };
        d.wait_completion(&point).await.unwrap();
        let plan = common::single_commands_plan(&fixture, 9100 + round as u64).unwrap();
        // SAFETY: the producer is complete and all resources remain live.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin resolve"));
        let e = run.commands();
        queries.resolve(e, &set, 0..1, &buffer, 0).unwrap();
        let ticket = e.readback_buffer(&buffer, 0..8).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish resolve"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: completion covers the result readback.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        let value = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
        eprintln!("round {round}: query={value}, expected={expected}");
        assert_eq!(value != 0, expected);
    }
}

#[test]
#[ignore = "requires GLES hardware context"]
fn gles_core_query_stencil_extract() {
    let d = block_on(open(
        GlInstance::gles_pbuffer([8, 8]).expect("GLES pbuffer"),
    ));
    block_on(query_stencil_extract(&d));
}
