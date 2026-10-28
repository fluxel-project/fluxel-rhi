//! Program binaries survive serialization and are loaded for actual GPU draws.
use super::*;

async fn cache_gpu(d: &Device<Gl>) {
    let cache = d
        .create_pipeline_cache(None)
        .expect("program binary formats");
    let source = b"@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f { let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.); } @fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f { if p.y < 3.0 { return vec4f(1.,0.,0.,1.); } return vec4f(0.,1.,0.,1.); }";
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
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let desc = |cache| RasterPipelineDesc {
        label: None,
        layout: layout.clone(),
        vertex: PipelineStage {
            module: vertex.clone(),
            entry: "vs".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: fragment.clone(),
            entry: "fs".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
        primitive: triangle_list_state(),
        depth_stencil: None,
        multisample: single_sample_state(u64::MAX),
        multiview: None,
        cache: Some(cache),
    };
    let _first = d.raster_pipeline(desc(cache.clone())).await.unwrap();
    let serialized = d.serialize_pipeline_cache(&cache).unwrap();
    let mut corrupted = serialized.clone();
    *corrupted.last_mut().unwrap() ^= 1;
    assert!(matches!(
        d.create_pipeline_cache(Some(&corrupted)),
        Err(Error::InvalidRange)
    ));
    let loaded = d.create_pipeline_cache(Some(&serialized)).unwrap();
    assert!(matches!(
        d.create_pipeline_cache(Some(b"invalid")),
        Err(Error::InvalidRange)
    ));
    let damaged = d.create_pipeline_cache(Some(&serialized)).unwrap();
    unsafe { damaged.native_ref() }.0.invalidate_binary_format();
    let _fallback = d.raster_pipeline(desc(damaged.clone())).await.unwrap();
    assert_eq!(
        unsafe { damaged.native_ref() }.0.hits.get(),
        0,
        "invalid binary must compile from source"
    );
    let _recovered = d.raster_pipeline(desc(damaged.clone())).await.unwrap();
    assert_eq!(
        unsafe { damaged.native_ref() }.0.hits.get(),
        1,
        "fallback must replace the rejected binary"
    );
    let pipeline = d.raster_pipeline(desc(loaded.clone())).await.unwrap();
    assert_eq!(
        unsafe { loaded.native_ref() }.0.hits.get(),
        1,
        "actual glProgramBinary hit"
    );
    let texture = |height| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, height, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap()
    };
    let a = texture(8);
    let b = texture(12);
    let view = |t: &Texture<Gl>| {
        d.create_view(
            t,
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
    let av = view(&a);
    let bv = view(&b);
    let output = |size| {
        d.create_buffer(&BufferDesc {
            label: None,
            size,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let ab = output(256);
    let bb = output(384);
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 761).unwrap();
    // SAFETY: the resources and device outlive completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for (view, texture, output, height) in [(&av, &a, &ab, 8), (&bv, &b, &bb, 12)] {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 1., 1.])),
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
                    width: 8,
                    height,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.viewport(Viewport {
            x: 0.,
            y: 2.,
            width: 8.,
            height: 4.,
            min_depth: 0.,
            max_depth: 1.,
        })
        .unwrap();
        pass.draw(0..3, 0..1).unwrap();
        pass.finish().unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, height, 1],
                aspect: TextureAspects::COLOR,
            },
            output,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: height,
            },
        )
        .unwrap();
    }
    let at = e.readback_buffer(&ab, 0..256).unwrap();
    let bt = e.readback_buffer(&bb, 0..384).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    for ticket in [at, bt] {
        // SAFETY: point covers both copies and readbacks.
        let pixels = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        let pixel = |y: usize| &pixels[y * 32 + 4 * 4..y * 32 + 4 * 4 + 4];
        assert_eq!(pixel(0), &[0, 0, 255, 255]);
        assert_eq!(pixel(2), &[255, 0, 0, 255]);
        assert_eq!(pixel(3), &[0, 255, 0, 255]);
        assert_eq!(pixel(6), &[0, 0, 255, 255]);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_program_cache() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL Core feature fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(cache_gpu(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_program_cache() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(cache_gpu(&d));
}
