//! Focused real-GPU queries output assertions.
use super::*;

#[wasm_bindgen_test(async)]
async fn gpu_timestamp_pass_boundaries_resolve_nanoseconds() {
    let d = open().await;
    if !d.capabilities().timestamp {
        console_log!("SKIP timestamp-query: browser does not expose the feature");
        return;
    }
    let queries = d.queries(QueryKind::Timestamp).unwrap();
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let output = buffer(
        &d,
        256,
        BufferUsage::QUERY_RESOLVE | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    queries.timestamp(&mut e, &set, 0).unwrap();
    queries.timestamp(&mut e, &set, 1).unwrap();
    queries.resolve(&mut e, &set, 0..2, &output, 0).unwrap();
    let ticket = e.readback_buffer(&output, 0..16).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: both query writes and their resolve/readback are in the awaited recording.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let a = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let b = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    assert!(b >= a, "timestamp order {a} -> {b}");
    assert!(b != 0, "timestamps must not resolve as unwritten slots");
}

#[wasm_bindgen_test(async)]
async fn gpu_attachmentless_sample_switch_and_reused_occlusion_slots() {
    let d = open().await;
    let source = "@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);} @fragment fn fs(){}";
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
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
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let mut pipelines = Vec::new();
    for count in [1, 4] {
        pipelines.push(
            d.raster_pipeline(RasterPipelineDesc {
                label: None,
                layout: layout.clone(),
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
                color_targets: vec![],
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
                    count,
                    mask: u64::MAX,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await
            .unwrap(),
        );
    }
    let queries = d.queries(QueryKind::Occlusion).unwrap();
    let set = queries
        .create(&QueryDesc {
            kind: QueryKind::Occlusion,
            count: 3,
        })
        .unwrap();
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    queries.reset(&mut e, &set, 0..3).unwrap();
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: Some(&set),
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipelines[0]).unwrap();
    queries.begin_occlusion(&mut pass, &set, 0).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.set_pipeline(&pipelines[1]).unwrap();
    queries.end_occlusion(&mut pass).unwrap();
    queries.begin_occlusion(&mut pass, &set, 1).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    queries.end_occlusion(&mut pass).unwrap();
    queries.begin_occlusion(&mut pass, &set, 1).unwrap();
    queries.end_occlusion(&mut pass).unwrap();
    queries.begin_occlusion(&mut pass, &set, 2).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    queries.end_occlusion(&mut pass).unwrap();
    pass.finish().unwrap();
    let out = buffer(
        &d,
        256,
        BufferUsage::QUERY_RESOLVE | BufferUsage::COPY_SRC,
        MemoryClass::Device,
    );
    queries.resolve(&mut e, &set, 0..3, &out, 0).unwrap();
    let ticket = e.readback_buffer(&out, 0..24).unwrap();
    let point = submit(&d, e).await;
    // SAFETY: the point covers every native query segment and GPU reduction.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let values: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(
        values[0] > 0,
        "first query lost its pre-split samples: {:?}",
        values
    );
    assert_eq!(
        values[1], 0,
        "reused slot must contain the latest empty query"
    );
    assert!(values[2] > 0);
}
