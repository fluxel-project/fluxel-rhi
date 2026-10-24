//! Depth-only direct SPIR-V vertex pipeline readback on WGL.
use super::*;
async fn direct_spirv_depth_only(d: &Device<Gl>) {
    let vertex = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::SpirV,
            bytes: Arc::from(super::direct_shader_artifacts::base64_words(
                super::direct_shader_artifacts::SPIRV_VERTEX_BASE64,
            )),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            }],
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
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "main".into(),
                constants: vec![],
            },
            fragment: None,
            vertex_buffers: vec![VertexBufferLayout {
                stride: 8,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32(2),
                }],
            }],
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
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil_front: face,
                stencil_back: face,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.,
                depth_bias_clamp: 0.,
            }),
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
    let depth = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 4, 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let vertices = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 24,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = d
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
    let plan = common::single_commands_plan(&fixture, 948).unwrap();
    // SAFETY: the fixture owns the context and retains all resources through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let commands = run.commands();
    let positions: [f32; 6] = [-1., -1., 3., -1., -1., 3.];
    let bytes: Vec<u8> = positions.into_iter().flat_map(f32::to_le_bytes).collect();
    commands.write_buffer(&vertices, 0, &bytes).unwrap();
    commands
        .transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &vertices,
                    range: 0..24,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    to: BufferUses {
                        stages: StageMask::VERTEX_INPUT,
                        accesses: AccessMask::VERTEX_READ,
                    },
                },
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &depth,
                    aspects: TextureAspects::DEPTH,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask(0),
                        accesses: AccessMask(0),
                    },
                    to: TextureUses {
                        stages: StageMask::ALL_COMMANDS,
                        accesses: AccessMask::DEPTH_READ | AccessMask::DEPTH_WRITE,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
                ownership: None,
            }),
        ])
        .unwrap();
    let mut pass = commands
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &[],
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&view),
                depth: Some((LoadOp::Clear(1.), StoreOp::Store)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &vertices, 0..24).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    commands
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &depth,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::ALL_COMMANDS,
                    accesses: AccessMask::DEPTH_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::DepthStencilAttachment, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();

    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &depth,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 4, 1],
                aspect: TextureAspects::DEPTH,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = commands.readback_buffer(&readback, 0..128).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: completion covers the draw and depth transfer.
    let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for value in bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
    {
        assert!((value - 0.5).abs() < 1e-6, "depth {value}");
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context with GL_ARB_gl_spirv"]
fn wgl_core_direct_spirv_depth_only() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL direct SPIR-V depth", 1024, 768).unwrap()).unwrap(); // SAFETY: window outlives its adopted device.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let device = super::direct_shader_artifacts::artifact_block_on(open(native));
    super::direct_shader_artifacts::artifact_block_on(direct_spirv_depth_only(&device));
}
