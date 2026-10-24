//! Core indirect draw and GPU count-buffer output checks.
use super::*;
async fn indirect(d: &Device<Gl>, builtin: bool, broken_driver: bool, gpu_shared: bool) {
    let caps = d.capabilities();
    if !caps.indirect_draw {
        return;
    }
    let shader_source = br#"@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
      let p=array<vec2f,6>(vec2f(-1.,-1.),vec2f(0.,-1.),vec2f(-1.,1.),vec2f(0.,-1.),vec2f(1.,-1.),vec2f(1.,1.)); return vec4f(p[i],0.5,1.);
    } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }"#;
    let attribute_source = br#"@vertex fn vs(@location(0) pos:vec2f)->@builtin(position) vec4f { return vec4f(pos,0.5,1.); } @fragment fn fs()->@location(0) vec4f { return vec4f(1.,0.,0.,1.); }"#;
    let attribute_storage_source = br#"@group(0) @binding(0) var<storage,read> tint:vec4f; @vertex fn vs(@location(0) pos:vec2f)->@builtin(position) vec4f { return vec4f(pos,0.5,1.); } @fragment fn fs()->@location(0) vec4f { return tint; }"#;
    let shader_source = if gpu_shared {
        attribute_storage_source.as_slice()
    } else if builtin {
        shader_source.as_slice()
    } else {
        attribute_source.as_slice()
    };
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(shader_source),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let shade = if gpu_shared {
        let layout = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 16,
                        access: StorageAccess::Read,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        let buffer = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 16,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &buffer,
                        range: 0..16,
                    },
                }],
            })
            .unwrap();
        Some((layout, buffer, group))
    } else {
        None
    };
    let groups = shade
        .as_ref()
        .map(|(layout, _, _)| vec![layout])
        .unwrap_or_default();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &groups,
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
            vertex_buffers: if builtin {
                vec![]
            } else {
                vec![VertexBufferLayout {
                    stride: 8,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32(2),
                    }],
                }]
            },
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
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let buffer = |size, usage| {
        d.create_buffer(&BufferDesc {
            label: None,
            size,
            usage,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let args = buffer(
        80,
        BufferUsage::INDIRECT
            | BufferUsage::COPY_DST
            | if gpu_shared {
                BufferUsage::STORAGE
            } else {
                BufferUsage(0)
            },
    );
    let positions = buffer(48, BufferUsage::VERTEX | BufferUsage::COPY_DST);
    let separate_count = buffer(16, BufferUsage::INDIRECT | BufferUsage::COPY_DST);
    let count_buffer = if gpu_shared { &args } else { &separate_count };
    let indices = buffer(16, BufferUsage::INDEX | BufferUsage::COPY_DST);
    // Native WGL exercises explicit tight and padded record strides, and
    // both native indexed batching and nonzero-index-offset GPU rebasing.
    // The GPU-shared count case keeps its
    // count word at byte 4, so it deliberately retains the non-overlapping
    // padded command layout.
    let out: Vec<_> = (0..24)
        .map(|_| buffer(256, BufferUsage::COPY_DST | BufferUsage::COPY_SRC))
        .collect();
    let producer = if gpu_shared {
        let compute = d.compute().unwrap();
        let shader = d.shader(ShaderDesc { label: None, format: ShaderFormat::Wgsl, bytes: Arc::from(b"@group(0) @binding(0) var<storage,read_write> out:array<u32>; @group(0) @binding(1) var<uniform> input:vec4u; @compute @workgroup_size(1) fn main(){out[1]=input.x;}".as_slice()), entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1,1,1]) }], required_features: vec![] }).await.unwrap();
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
                        BindingType::StorageBuffer {
                            dynamic_offset: false,
                            min_size: 80,
                            access: StorageAccess::ReadWrite,
                        },
                    ),
                    entry(
                        1,
                        BindingType::UniformBuffer {
                            dynamic_offset: false,
                            min_size: 16,
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
        let pipeline = compute
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                required_subgroup_size: None,
                cache: None,
            })
            .await
            .unwrap();
        let uniform = buffer(16, BufferUsage::UNIFORM | BufferUsage::COPY_DST);
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: &args,
                            range: 0..80,
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer {
                            buffer: &uniform,
                            range: 0..16,
                        },
                    },
                ],
            })
            .unwrap();
        Some((compute, pipeline, group, uniform))
    } else {
        None
    };
    let query = if gpu_shared {
        let q = d.queries(QueryKind::Occlusion).unwrap();
        let set = q
            .create(&QueryDesc {
                kind: QueryKind::Occlusion,
                count: 8,
            })
            .unwrap();
        Some((q, set))
    } else {
        None
    };
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 280).unwrap();
    // SAFETY: the device and all resources outlive the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let e = run.commands();
    if let Some((_, buffer, _)) = &shade {
        e.write_buffer(
            buffer,
            0,
            &[0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63],
        )
        .unwrap();
    }
    if let Some((queries, set)) = &query {
        queries.reset(e, set, 0..8).unwrap();
    }
    e.write_buffer(
        &indices,
        0,
        &[99, 0, 99, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    )
    .unwrap();
    let mut vertices = Vec::new();
    for p in [
        [-1f32, -1.],
        [0., -1.],
        [-1., 1.],
        [0., -1.],
        [1., -1.],
        [1., 1.],
    ] {
        for component in p {
            vertices.extend_from_slice(&component.to_le_bytes());
        }
    }
    e.write_buffer(&positions, 0, &vertices).unwrap();
    e.write_buffer(count_buffer, 4, &1u32.to_le_bytes())
        .unwrap();
    let extras = d.raster_extras().unwrap();
    let mut tickets = Vec::new();
    let index_modes: &[(bool, u64)] = if gpu_shared {
        &[(true, 4), (false, 0)]
    } else {
        &[(true, 4), (true, 0), (false, 0)]
    };
    for &(indexed, index_offset) in index_modes {
        let layouts: &[(u64, u32)] = if gpu_shared {
            &[(16, 32)]
        } else {
            &[(0, 0), (16, 32)]
        };
        for &(command_offset, stride) in layouts {
            // Core uses an explicit record stride; native GL's zero-stride
            // spelling is only accepted below that public contract.
            let stride = if stride == 0 {
                if indexed { 20 } else { 16 }
            } else {
                stride
            };
            let record_stride = stride as usize;
            let mut words = vec![0u8; 80];
            for (base, first) in [
                (command_offset as usize, 0u32),
                (command_offset as usize + record_stride, 3),
            ] {
                let values = if indexed {
                    vec![3u32, 1, first + if index_offset == 0 { 2 } else { 0 }, 0, 0]
                } else {
                    vec![3u32, 1, first, 0]
                };
                for (i, value) in values.iter().enumerate() {
                    words[base + i * 4..base + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            e.write_buffer(&args, 0, &words).unwrap();
            if gpu_shared {
                e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                    target: BufferSlice {
                        buffer: &args,
                        range: 0..80,
                    },
                    usage: StateTransition {
                        from: BufferUses {
                            stages: StageMask::COPY,
                            accesses: AccessMask::COPY_WRITE,
                        },
                        to: BufferUses {
                            stages: StageMask::INDIRECT,
                            accesses: AccessMask::INDIRECT_READ,
                        },
                    },
                    ownership: None,
                })])
                .unwrap();
            }
            for gpu_count in [None, Some(0u32), Some(1), Some(3)] {
                let counted = gpu_count.is_some();
                if counted && !caps.multi_draw_count {
                    continue;
                }
                if let Some(value) = gpu_count {
                    if let Some((compute, pipeline, group, uniform)) = &producer {
                        let mut bytes = [0u8; 16];
                        bytes[..4].copy_from_slice(&value.to_le_bytes());
                        e.write_buffer(uniform, 0, &bytes).unwrap();
                        e.transition_resources(&[
                            ResourceTransition::Buffer(BufferBarrier {
                                target: BufferSlice {
                                    buffer: &args,
                                    range: 0..80,
                                },
                                usage: StateTransition {
                                    from: BufferUses {
                                        stages: StageMask::INDIRECT,
                                        accesses: AccessMask::INDIRECT_READ,
                                    },
                                    to: BufferUses {
                                        stages: StageMask::COMPUTE,
                                        accesses: AccessMask::SHADER_WRITE,
                                    },
                                },
                                ownership: None,
                            }),
                            ResourceTransition::Buffer(BufferBarrier {
                                target: BufferSlice {
                                    buffer: uniform,
                                    range: 0..16,
                                },
                                usage: StateTransition {
                                    from: BufferUses {
                                        stages: StageMask::COPY,
                                        accesses: AccessMask::COPY_WRITE,
                                    },
                                    to: BufferUses {
                                        stages: StageMask::COMPUTE,
                                        accesses: AccessMask::UNIFORM_READ,
                                    },
                                },
                                ownership: None,
                            }),
                        ])
                        .unwrap();
                        let mut cp = compute.begin(e).unwrap();
                        cp.set_pipeline(pipeline).unwrap();
                        cp.set_bind_group(0, group, &[]).unwrap();
                        cp.dispatch([1, 1, 1]).unwrap();
                        cp.finish().unwrap();
                        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                            target: BufferSlice {
                                buffer: &args,
                                range: 0..80,
                            },
                            usage: StateTransition {
                                from: BufferUses {
                                    stages: StageMask::COMPUTE,
                                    accesses: AccessMask::SHADER_WRITE,
                                },
                                to: BufferUses {
                                    stages: StageMask::INDIRECT,
                                    accesses: AccessMask::INDIRECT_READ,
                                },
                            },
                            ownership: None,
                        })])
                        .unwrap();
                    } else {
                        e.write_buffer(count_buffer, 4, &value.to_le_bytes())
                            .unwrap();
                    }
                }
                let colors = [Some(ColorAttachment {
                    view: AttachmentView::Texture(&view),
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
                            width: 8,
                            height: 8,
                        },
                        multiview_mask: None,
                        timestamp_writes: None,
                        occlusion_query_set: query.as_ref().map(|(_, set)| set),
                        secondary_inheritance: None,
                    })
                    .unwrap();
                pass.set_pipeline(&pipeline).unwrap();
                if let Some((_, _, group)) = &shade {
                    pass.set_bind_group(0, group, &[]).unwrap();
                }
                if !builtin {
                    pass.set_vertex_buffer(0, &positions, 0..48).unwrap();
                }
                if indexed {
                    pass.set_index_buffer(&indices, index_offset..16, IndexFormat::Uint16)
                        .unwrap();
                }
                if let Some((queries, set)) = &query {
                    queries
                        .begin_occlusion(&mut pass, set, tickets.len() as u32)
                        .unwrap();
                }
                let result = if counted {
                    extras.multi_draw_count(
                        &mut pass,
                        &args,
                        command_offset,
                        count_buffer,
                        4,
                        2,
                        stride,
                        indexed,
                    )
                } else {
                    extras.multi_draw(&mut pass, &args, command_offset, 2, stride, indexed)
                };
                if broken_driver && builtin {
                    assert!(matches!(
                        result,
                        Err(Error::Unsupported(
                            "ANGLE D3D11 indirect gl_VertexID driver defect"
                        ))
                    ));
                    drop(pass);
                    let _ = run.close();
                    let _ = session.abort(Error::InvalidState);
                    return;
                }
                result.unwrap();
                if let Some((queries, _)) = &query {
                    queries.end_occlusion(&mut pass).unwrap();
                }
                pass.finish().unwrap();
                let buffer = &out[tickets.len()];
                e.texture_to_buffer(
                    TextureRegion {
                        texture: &texture,
                        mip: 0,
                        origin: [0, 0, 0],
                        extent: [8, 8, 1],
                        aspect: TextureAspects::COLOR,
                    },
                    buffer,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: 32,
                        rows_per_image: 8,
                    },
                )
                .unwrap();
                tickets.push((
                    e.readback_buffer(buffer, 0..256).unwrap(),
                    gpu_count,
                    indexed,
                    stride,
                ));
            }
        }
    }
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    for (ticket, gpu_count, indexed, stride) in tickets {
        let counted = gpu_count.is_some();
        // SAFETY: the point covers the draw and readback.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        eprintln!(
            "builtin={builtin} indexed={indexed} stride={stride} counted={counted} rows={:?}",
            (0..8)
                .map(|y| (0..8)
                    .map(|x| if bytes[y * 32 + x * 4] > 0 { "#" } else { "." })
                    .collect::<String>())
                .collect::<Vec<_>>()
        );
        // Both selected pixels lie inside the corresponding triangles.
        assert_eq!(
            &bytes[3 * 32..3 * 32 + 4],
            if gpu_count == Some(0) {
                &[0, 0, 0, 255]
            } else {
                &[255, 0, 0, 255]
            }
        );
        assert_eq!(
            &bytes[3 * 32 + 7 * 4..3 * 32 + 7 * 4 + 4],
            if matches!(gpu_count, Some(0 | 1)) {
                &[0, 0, 0, 255]
            } else {
                &[255, 0, 0, 255]
            }
        );
    }
    if let Some((queries, set)) = &query {
        let result_buffer = buffer(64, BufferUsage::COPY_SRC | BufferUsage::COPY_DST);
        let plan = common::single_commands_plan(&fixture, 281).unwrap();
        // SAFETY: producer completed above and all resources remain live.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin query resolve"));
        let e = run.commands();
        queries.resolve(e, set, 0..8, &result_buffer, 0).unwrap();
        let ticket = e.readback_buffer(&result_buffer, 0..64).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish query resolve"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: this completion covers the query result readback.
        let bytes = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
        for (index, expected) in [true, false, true, true, true, false, true, true]
            .into_iter()
            .enumerate()
        {
            let value = u64::from_ne_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap());
            assert_eq!(value != 0, expected, "query {index}");
        }
    }
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_indirect() {
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
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let window =
        Window::new(WindowConfig::new("GL Core feature fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let name = native.0.adapter_name().unwrap();
    let broken = name.contains("ANGLE 2.1.22762")
        && name.contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
        && name.contains("D3D11-32.0.21028.2002");
    let d = block_on(open(native));
    block_on(indirect(&d, false, broken, false));
    block_on(indirect(&d, true, broken, false));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_indirect() {
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
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let native =
        GlInstance::gles_pbuffer([1024, 768]).expect("open EGL GLES pbuffer for hardware fixture");
    let name = native.0.adapter_name().unwrap();
    let broken = name.contains("ANGLE 2.1.22762")
        && name.contains("AMD Radeon 780M Graphics (0x00001900) Direct3D11")
        && name.contains("D3D11-32.0.21028.2002");
    let d = block_on(open(native));
    block_on(indirect(&d, false, broken, false));
    // A second count lowering on the same GLES context must keep the first
    // draw's GPU-written count ordering while reusing its private scratch.
    block_on(indirect(&d, false, broken, false));
    block_on(indirect(&d, true, broken, false));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_indirect() {
    use wasm_bindgen::JsCast;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(8);
    canvas.set_height(8);
    let native = GlInstance::from_webgl2_canvas(canvas.clone()).unwrap();
    let d = open(native).await;
    indirect(&d, false, false, false).await;
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_indirect_gpu_shared_occlusion() {
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
    fn block_on<F: Future>(f: F) -> F::Output {
        let waker = Waker::from(Arc::new(WakeThread(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }
    let native = GlInstance::gles_pbuffer([1024, 768]).expect("open EGL GLES pbuffer");
    let d = block_on(open(native));
    crate::backend::gl::native::indirect_count_emulated::set_test_chunk_capacity(1);
    block_on(indirect(&d, false, false, true));
    crate::backend::gl::native::indirect_count_emulated::set_test_chunk_capacity(u32::MAX);
}
