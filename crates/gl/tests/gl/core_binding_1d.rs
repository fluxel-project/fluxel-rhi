//! Desktop GL 1D texture upload, mip view, sampling, and readback through Core.
use super::*;

async fn binding_1d(d: &Device<Gl>) {
    let source = br#"
@group(0) @binding(0) var image: texture_1d<f32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs()->@location(0) vec4f { return textureLoad(image,0,0); }
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
    let texture_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D1,
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
            groups: &[&texture_layout],
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
            color_targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 15,
            })],
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
    let input = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D1,
            extent: [8, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let copied = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D1,
            extent: [8, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let copied_view = d
        .create_view(
            &copied,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D1,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: 0..1,
            },
        )
        .unwrap();
    let output = d
        .create_texture(&TextureDesc {
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
        .unwrap();
    let output_view = d
        .create_view(
            &output,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let copied_group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &texture_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&copied_view),
            }],
        })
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let upload = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let input_readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let copied_readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 1).unwrap();
    // SAFETY: fixture resources and the context remain alive through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let commands = run.commands();
    let mut upload_bytes = [0xadu8; 64];
    for texel in upload_bytes[16..32].chunks_exact_mut(4) {
        texel.copy_from_slice(&[32, 96, 160, 255]);
    }
    commands.write_buffer(&upload, 0, &upload_bytes).unwrap();
    commands
        .write_texture(
            TextureRegion {
                texture: &input,
                mip: 1,
                origin: [0, 0, 0],
                extent: [4, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 16,
                rows_per_image: 1,
            },
            &[
                32, 96, 160, 255, 32, 96, 160, 255, 32, 96, 160, 255, 32, 96, 160, 255,
            ],
        )
        .unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &input,
                mip: 1,
                origin: [0, 0, 0],
                extent: [4, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &input_readback,
            TexelCopyBufferLayout {
                offset: 8,
                bytes_per_row: 24,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let input_ticket = commands.readback_buffer(&input_readback, 0..32).unwrap();
    commands
        .buffer_to_texture(
            &upload,
            TexelCopyBufferLayout {
                offset: 16,
                bytes_per_row: 24,
                rows_per_image: 1,
            },
            TextureRegion {
                texture: &input,
                mip: 1,
                origin: [0, 0, 0],
                extent: [4, 1, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    commands
        .copy_texture(
            TextureRegion {
                texture: &input,
                mip: 1,
                origin: [0, 0, 0],
                extent: [4, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TextureRegion {
                texture: &copied,
                mip: 1,
                origin: [0, 0, 0],
                extent: [4, 1, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &copied,
                mip: 1,
                origin: [0, 0, 0],
                extent: [4, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &copied_readback,
            TexelCopyBufferLayout {
                offset: 8,
                bytes_per_row: 24,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let copied_ticket = commands.readback_buffer(&copied_readback, 0..32).unwrap();
    commands
        .clear_texture(
            &copied,
            1..2,
            0..1,
            TextureClearValue::Color(ClearColor::Float([0.25, 0.5, 0.75, 1.])),
        )
        .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
        store: StoreOp::Store,
    })];
    let mut pass = commands
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
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &copied_group, &[]).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &output,
                mip: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &buffer,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 16,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = commands.readback_buffer(&buffer, 0..64).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the render and readback.
    let pixels = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    // SAFETY: the same completion point covers the 1D texture PBO readback.
    let input_bytes = unsafe { d.read_after(input_ticket, &point).await }
        .unwrap()
        .bytes;
    assert_eq!(&input_bytes[8..24], &[32, 96, 160, 255].repeat(4));
    // SAFETY: completion covers the copy and its 1D texture PBO readback.
    let copied_bytes = unsafe { d.read_after(copied_ticket, &point).await }
        .unwrap()
        .bytes;
    assert_eq!(&copied_bytes[8..24], &[32, 96, 160, 255].repeat(4));
    for pixel in pixels.chunks_exact(4) {
        assert_eq!(pixel, &[64, 128, 191, 255]);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_binding_1d() {
    use fluxel_host::{Window, WindowConfig};
    use std::{
        future::Future,
        task::{Context, Poll, Wake, Waker},
    };
    struct WakeThread(std::thread::Thread);
    impl Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let window = Window::new(WindowConfig::new("GL 1D Core fixture", 64, 64).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [64, 64]) }.unwrap();
    let device = block_on(open(native));
    block_on(binding_1d(&device));
}
