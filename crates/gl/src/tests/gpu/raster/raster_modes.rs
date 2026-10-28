//! Native polygon modes and depth-clamp state changes across a raster pass.
use super::*;

async fn raster_modes(d: &Device<Gl>) {
    let source = b"fn pos(i:u32,z:f32)->vec4f { let p=array<vec2f,3>(vec2f(-0.8,-0.8),vec2f(0.8,-0.8),vec2f(0.,0.8)); return vec4f(p[i],z,1.); } @vertex fn normal(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { return pos(i,0.5); } @vertex fn near(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { return pos(i,-0.5); } @vertex fn far(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { return pos(i,1.5); } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }";
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
    let normal = d
        .shader(shader("normal", ShaderStage::Vertex))
        .await
        .unwrap();
    let near = d.shader(shader("near", ShaderStage::Vertex)).await.unwrap();
    let far = d.shader(shader("far", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let cases = [
        ("normal", &normal, PolygonMode::Fill, false, false),
        ("normal", &normal, PolygonMode::Line, false, false),
        ("normal", &normal, PolygonMode::Point, false, false),
        ("normal", &normal, PolygonMode::Fill, false, false),
        ("near", &near, PolygonMode::Fill, false, false),
        ("near", &near, PolygonMode::Fill, true, false),
        ("far", &far, PolygonMode::Fill, false, false),
        ("far", &far, PolygonMode::Fill, true, false),
        ("normal", &normal, PolygonMode::Fill, false, true),
    ];
    let mut pipelines = Vec::new();
    for (name, vertex, mode, unclipped, conservative) in cases {
        let desc = RasterPipelineDesc {
            label: None,
            layout: layout.clone(),
            vertex: PipelineStage {
                module: vertex.clone(),
                entry: name.into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment.clone(),
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: unclipped,
                polygon_mode: mode,
                conservative,
            },
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        };
        match d.raster_pipeline(desc).await {
            Ok(p) => pipelines.push(Some(p)),
            Err(Error::Unsupported(reason)) => {
                eprintln!(
                    "SKIP {name} {mode:?} unclipped={unclipped} conservative={conservative}: {reason}"
                );
                pipelines.push(None);
            }
            Err(e) => panic!("pipeline: {e:?}"),
        }
    }
    let color = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [144, 16, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
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
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 9216,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 319).unwrap();
    // SAFETY: resources remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&cv),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 144,
                height: 16,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    for (i, p) in pipelines.iter().enumerate() {
        if let Some(p) = p {
            pass.set_pipeline(p).unwrap();
            pass.viewport(Viewport {
                x: (i * 16) as f32,
                y: 0.,
                width: 16.,
                height: 16.,
                min_depth: 0.,
                max_depth: 1.,
            })
            .unwrap();
            pass.draw(0..3, 0..1).unwrap();
        }
    }
    pass.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &color,
            mip: 0,
            origin: [0, 0, 0],
            extent: [144, 16, 1],
            aspect: TextureAspects::COLOR,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 576,
            rows_per_image: 16,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&output, 0..9216).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the copy and readback.
    let pixels = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let mut count = [0usize; 9];
    for y in 0..16 {
        for x in 0..144 {
            if pixels[(y * 144 + x) * 4] != 0 {
                count[x / 16] += 1;
            }
        }
    }
    eprintln!("raster modes pixel counts: {count:?}");
    assert!(count[0] > 40);
    assert_eq!(count[3], count[0], "fill restored");
    if pipelines[1].is_some() {
        assert!(count[1] > 3 && count[1] < count[0], "line coverage");
    }
    if pipelines[2].is_some() {
        assert!(count[2] > 0 && count[2] < count[1], "point coverage");
    }
    assert_eq!(count[4], 0, "near clipped");
    assert_eq!(count[6], 0, "far clipped");
    if pipelines[5].is_some() {
        assert_eq!(count[5], count[0], "near clamp");
    }
    if pipelines[7].is_some() {
        assert_eq!(count[7], count[0], "far clamp");
    }
    if pipelines[8].is_some() {
        assert!(count[8] > count[0], "conservative coverage");
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_raster_modes() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL raster feature fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(raster_modes(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_raster_modes() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(raster_modes(&d));
}
