//! Multisample array layer attachment and per-sample shader readback.
use super::*;
async fn msaa_array(d: &Device<Gl>) {
    let format = TextureFormat::Rgba8Unorm;
    let ms_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 2,
        format,
        mip_levels: 1,
        samples: 4,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        view_formats: vec![],
    };
    if !matches!(d.texture_support(&ms_desc), Support::Supported(_)) {
        eprintln!("SKIP multisample array Rgba8Unorm count=4: texture_support rejected descriptor");
        return;
    }
    let ms = d.create_texture(&ms_desc).unwrap();
    let source = r#"@group(0) @binding(0) var image: texture_multisampled_2d_array<f32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));return vec4f(p[i],0.5,1.);}
@fragment fn fs()->@location(0) vec4f {let a=textureLoad(image,vec2i(0,0),0,0);let b=textureLoad(image,vec2i(0,0),1,3);return vec4f(a.r,b.g,a.b,b.a);}"#.to_owned();
    let source = source.into_bytes();
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
            entries: vec![entry(
                0,
                BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2Array,
                    multisampled: true,
                },
            )],
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
    let out = texture(
        TextureFormat::Rgba8Unorm,
        [8, 8, 1],
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    )
    .unwrap();
    let view = |t, dimension, layers| {
        d.create_view(
            t,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers,
            },
        )
        .unwrap()
    };
    let layer0 = view(&ms, TextureViewDimension::D2, 0..1);
    let layer1 = view(&ms, TextureViewDimension::D2, 1..2);
    let sampled = view(&ms, TextureViewDimension::D2Array, 0..2);
    let out_v = view(&out, TextureViewDimension::D2, 0..1);
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &textures,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&sampled),
            }],
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
    let plan = common::single_commands_plan(&fixture, 230).unwrap();
    // SAFETY: resources and context survive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    for (view, color) in [(&layer0, [1., 0., 0., 1.]), (&layer1, [0., 1., 0., 1.])] {
        let attachments = [Some(ColorAttachment {
            view: AttachmentView::Texture(view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float(color)),
            store: StoreOp::Store,
        })];
        e.begin_raster(&RasterPassDesc {
            label: None,
            colors: &attachments,
            depth_stencil: None,
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
        .unwrap()
        .finish()
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
    // SAFETY: completion covers texture readback.
    let data = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    let actual = &data[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4];
    eprintln!("multisample array layer0 sample0 / layer1 sample3: {actual:?}");
    assert_eq!(actual, &[255, 255, 0, 255]);
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_msaa_array_sampling() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(msaa_array(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_msaa_array_sampling() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(msaa_array(&d));
}
