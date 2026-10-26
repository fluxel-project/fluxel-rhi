//! Raster output assertions for MRT, stage-local immediates and partial resolves.
use super::*;

fn rgba(d: &Device<WebGpu>, samples: u32) -> Texture<WebGpu> {
    d.create_texture(&TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples,
        usage: TextureUsage::RENDER_ATTACHMENT
            | if samples == 1 {
                TextureUsage::COPY_SRC | TextureUsage::COPY_DST
            } else {
                TextureUsage(0)
            },
        view_formats: vec![],
    })
    .unwrap()
}
fn view(d: &Device<WebGpu>, texture: &Texture<WebGpu>) -> TextureView<WebGpu> {
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
}
fn read_texture(
    d: &Device<WebGpu>,
    e: &mut CommandEncoder<WebGpu>,
    texture: &Texture<WebGpu>,
) -> ReadbackTicket<WebGpu> {
    let out = buffer(
        d,
        8 * 256,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    e.texture_to_buffer(
        TextureRegion {
            texture,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        },
        &out,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 8,
        },
    )
    .unwrap();
    e.readback_buffer(&out, 0..8 * 256).unwrap()
}

#[wasm_bindgen_test(async)]
async fn gpu_mrt_preserves_vertex_and_fragment_immediate_values_independently() {
    let d = open().await;
    if !d.capabilities().push_constants {
        console_log!("SKIP native immediate_address_space");
        return;
    }
    let source = "var<push_constant> pc:vec4f; struct V { @builtin(position) position:vec4f,@location(0) value:vec4f } struct F { @location(0) a:vec4f,@location(1) b:vec4f } @vertex fn vs(@builtin(vertex_index) i:u32)->V{let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));var o:V;o.position=vec4f(p[i],0.5,1.);o.value=pc;return o;} @fragment fn fs(v:V)->F{var o:F;o.a=v.value;o.b=pc;return o;}";
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
            push_constants: &[
                PushConstantRange {
                    stages: vec![ShaderStage::Vertex],
                    range: 0..16,
                },
                PushConstantRange {
                    stages: vec![ShaderStage::Fragment],
                    range: 0..16,
                },
            ],
        })
        .unwrap();
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
                    blend: None,
                    write_mask: 15
                });
                2
            ],
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
    let a = rgba(&d, 1);
    let b = rgba(&d, 1);
    let av = view(&d, &a);
    let bv = view(&d, &b);
    let colors = [&av, &bv].map(|v| {
        Some(ColorAttachment {
            view: AttachmentView::Texture(v),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })
    });
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    let mut pass = e
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
    pass.set_pipeline(&pipeline).unwrap();
    d.raster_extras()
        .unwrap()
        .push_constants(
            &mut pass,
            StageMask::VERTEX,
            0,
            &[1f32, 0., 0., 1.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    d.raster_extras()
        .unwrap()
        .push_constants(
            &mut pass,
            StageMask::FRAGMENT,
            0,
            &[0f32, 1., 0., 1.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    let at = read_texture(&d, &mut e, &a);
    let bt = read_texture(&d, &mut e, &b);
    let point = submit(&d, e).await;
    for (ticket, expected) in [(at, [255, 0, 0, 255]), (bt, [0, 255, 0, 255])] {
        // SAFETY: the MRT draw and both readback copies precede this point.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        assert_eq!(&bytes[4 * 256 + 4 * 4..4 * 256 + 4 * 4 + 4], &expected);
    }
}

#[wasm_bindgen_test(async)]
async fn gpu_partial_msaa_resolve_preserves_destination_outside_area() {
    let d = open().await;
    let src = rgba(&d, 4);
    let dst = rgba(&d, 1);
    let sv = view(&d, &src);
    let dv = view(&d, &dst);
    let mut e = d.encoder(d.lanes()[0].id).unwrap();
    e.clear_texture(
        &src,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([0., 0., 1., 1.])),
    )
    .unwrap();
    e.clear_texture(
        &dst,
        0..1,
        0..1,
        TextureClearValue::Color(ClearColor::Float([0., 0., 1., 1.])),
    )
    .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&sv),
        resolve_target: Some(AttachmentView::Texture(&dv)),
        load: LoadOp::Clear(ClearColor::Float([1., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    e.begin_raster(&RasterPassDesc {
        label: None,
        colors: &colors,
        depth_stencil: None,
        render_area: Rect {
            x: 2,
            y: 3,
            width: 3,
            height: 2,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    })
    .unwrap()
    .finish()
    .unwrap();
    let ticket = read_texture(&d, &mut e, &dst);
    let point = submit(&d, e).await;
    // SAFETY: the native resolve and destination copy precede this point.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for y in 0..8 {
        for x in 0..8 {
            let expected = if (2..5).contains(&x) && (3..5).contains(&y) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            assert_eq!(
                &bytes[y * 256 + x * 4..y * 256 + x * 4 + 4],
                &expected,
                "pixel {x},{y}"
            );
        }
    }
}
