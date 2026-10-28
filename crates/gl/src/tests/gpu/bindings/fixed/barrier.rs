//! Core compute image barriers observed by copy and vertex sampling.
use super::*;
async fn binding_barrier(d: &Device<Gl>) {
    let Some(compute) = d.compute() else {
        eprintln!("SKIP: compute unavailable");
        return;
    };
    let compute_src=br#"@group(0) @binding(0) var img: texture_storage_2d<rgba8unorm, write>; @compute @workgroup_size(1) fn cs(){textureStore(img,vec2i(0,0),vec4f(0.25,0.5,0.75,1.));}"#;
    let raster_src=br#"
 @group(0) @binding(0) var img:texture_2d<f32>;
 struct O { @builtin(position) p:vec4f, @location(0) c:vec4f };
 @vertex fn vs(@builtin(vertex_index) i:u32)->O {let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.));var o:O;o.p=vec4f(p[i],0.5,1.);o.c=textureLoad(img,vec2i(0,0),0);return o;}
 @fragment fn fs(i:O)->@location(0) vec4f {return i.c;}
 "#;
    let shader = |src: &[u8], name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(src),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: (stage == ShaderStage::Compute).then_some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let cs = d
        .shader(shader(compute_src, "cs", ShaderStage::Compute))
        .await
        .unwrap();
    let vertex = d
        .shader(shader(raster_src, "vs", ShaderStage::Vertex))
        .await
        .unwrap();
    let fragment = d
        .shader(shader(raster_src, "fs", ShaderStage::Fragment))
        .await
        .unwrap();
    let entry = |stage, ty| BindGroupLayoutEntry {
        binding: 0,
        visibility: vec![stage],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let write_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(
                ShaderStage::Compute,
                BindingType::StorageTexture {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    access: StorageAccess::Write,
                },
            )],
        })
        .unwrap();
    let read_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(
                ShaderStage::Vertex,
                BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            )],
        })
        .unwrap();
    let cl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&write_layout],
            push_constants: &[],
        })
        .unwrap();
    let cp = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout: cl,
            compute: PipelineStage {
                module: cs,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&read_layout],
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

    let make = |usage| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let image = make(
        TextureUsage::STORAGE_BINDING | TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_SRC,
    );
    let out = make(TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC);
    let view = |t| {
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
    let iv = view(&image);
    let ov = view(&out);
    let group = |layout: &BindGroupLayout<Gl>| {
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&iv),
            }],
        })
        .unwrap()
    };
    let write = group(&write_layout);
    let read = group(&read_layout);
    let make_buffer = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let copybuf = make_buffer();
    let drawbuf = make_buffer();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 157).unwrap();
    // SAFETY: context and resources stay alive until completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    let transition = |from_layout, to_layout, from_stages, from_access, to_stages, to_access| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &image,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: from_stages,
                    accesses: from_access,
                },
                to: TextureUses {
                    stages: to_stages,
                    accesses: to_access,
                },
            },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })
    };
    e.transition_resources(&[transition(
        ImageLayout::Undefined,
        ImageLayout::General,
        StageMask::HOST,
        AccessMask::EMPTY,
        StageMask::COMPUTE,
        AccessMask::SHADER_WRITE,
    )])
    .unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&cp).unwrap();
    p.set_bind_group(0, &write, &[]).unwrap();
    p.dispatch([1, 1, 1]).unwrap();
    p.finish().unwrap();
    e.transition_resources(&[transition(
        ImageLayout::General,
        ImageLayout::CopySrc,
        StageMask::COMPUTE,
        AccessMask::SHADER_WRITE,
        StageMask::COPY,
        AccessMask::COPY_READ,
    )])
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &image,
            mip: 0,
            origin: [0, 0, 0],
            extent: [1, 1, 1],
            aspect: TextureAspects::COLOR,
        },
        &copybuf,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 4,
            rows_per_image: 1,
        },
    )
    .unwrap();
    e.transition_resources(&[transition(
        ImageLayout::CopySrc,
        ImageLayout::General,
        StageMask::COPY,
        AccessMask::COPY_READ,
        StageMask::COMPUTE,
        AccessMask::SHADER_WRITE,
    )])
    .unwrap();
    let mut p = compute.begin(e).unwrap();
    p.set_pipeline(&cp).unwrap();
    p.set_bind_group(0, &write, &[]).unwrap();
    p.dispatch([1, 1, 1]).unwrap();
    p.finish().unwrap();
    e.transition_resources(&[transition(
        ImageLayout::General,
        ImageLayout::ShaderReadOnly,
        StageMask::COMPUTE,
        AccessMask::SHADER_WRITE,
        StageMask::VERTEX,
        AccessMask::SHADER_READ,
    )])
    .unwrap();
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
    p.set_bind_group(0, &read, &[]).unwrap();
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
        &drawbuf,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 32,
            rows_per_image: 8,
        },
    )
    .unwrap();
    let a = e.readback_buffer(&copybuf, 0..4).unwrap();
    let b = e.readback_buffer(&drawbuf, 0..256).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers writes and both readbacks.
    let copy = unsafe { d.read_after(a, &point).await }.unwrap().bytes;
    let draw = unsafe { d.read_after(b, &point).await }.unwrap().bytes;
    for pixel in [&copy[..4], &draw[4 * 32 + 4 * 4..4 * 32 + 4 * 4 + 4]] {
        eprintln!("image barrier: {pixel:?}");
        assert_eq!(pixel, &[64, 128, 191, 255]);
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_barrier() {
    use fluxel_host::{Window, WindowConfig};
    let window = Window::new(WindowConfig::new("GL MRT Core fixture", 120, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [120, 64]) }.unwrap();
    let d = block_on(open(native));
    block_on(binding_barrier(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_binding_barrier() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(binding_barrier(&d));
}
