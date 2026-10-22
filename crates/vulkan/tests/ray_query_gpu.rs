//! Real GPU inline ray query: the first ray hits and the second misses.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn inline_ray_query_hits_and_misses() {
    run_ray_query(false);
}

#[test]
fn extended_vertex_format_ray_query_hits_and_misses() {
    run_ray_query(true);
}

fn run_ray_query(extended: bool) {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    if !device.capabilities().ray_query {
        eprintln!("SKIP: Vulkan ray query unavailable");
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let ray = device
        .ray()
        .expect("ray query requires acceleration structure support");
    let buffer = |size, usage| {
        device
            .create_buffer(&BufferDesc {
                label: None,
                size,
                usage,
                memory: MemoryClass::Device,
            })
            .unwrap()
    };
    let vertices = buffer(48, BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST);
    let descriptor = |format| AccelerationDesc {
        input: AccelerationInput::Blas(vec![AccelerationGeometry::Triangles {
            vertices: vertices.clone(),
            vertex_offset: 0,
            vertex_count: 3,
            vertex_format: format,
            stride: 16,
            indices: None,
            opaque: true,
        }]),
        fast_trace: true,
        allow_update: false,
    };
    let format = if extended {
        let candidates = [
            VertexFormat::Float32(4),
            VertexFormat::Snorm16(4),
            VertexFormat::Snorm8(4),
        ];
        let mut selected = None;
        for format in candidates {
            let supported = match ray.sizes(&descriptor(format)) {
                Ok(_) => true,
                Err(Error::Unsupported(_)) => false,
                Err(error) => panic!("unexpected ray geometry probe failure: {error:?}"),
            };
            eprintln!("Vulkan ray geometry {format:?}: {supported}");
            if supported && selected.is_none() {
                selected = Some(format);
            }
        }
        let Some(format) = selected else {
            eprintln!("SKIP extended ray geometry formats: device capability");
            return;
        };
        format
    } else {
        VertexFormat::Float32(3)
    };
    let blas_desc = descriptor(format);
    let blas = ray.create_acceleration(&blas_desc).unwrap();
    let tlas_desc = AccelerationDesc {
        input: AccelerationInput::Tlas(vec![AccelerationInstance {
            blas: blas.clone(),
            transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            mask: 255,
            custom_index: 0,
            shader_table_offset: 0,
            flags: AccelerationInstanceFlags::TRIANGLE_CULL_DISABLE,
        }]),
        fast_trace: true,
        allow_update: false,
    };
    let tlas = ray.create_acceleration(&tlas_desc).unwrap();
    let bs = ray.sizes(&blas_desc).unwrap();
    let ts = ray.sizes(&tlas_desc).unwrap();
    let bscratch = buffer(bs.scratch_bytes, BufferUsage::ACCEL_SCRATCH);
    let tscratch = buffer(ts.scratch_bytes, BufferUsage::ACCEL_SCRATCH);
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = device
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
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::AccelerationStructure,
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2,
                        access: StorageAccess::Write,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Acceleration(&tlas),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&view),
                },
            ],
        })
        .unwrap();
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block_on(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(&include_bytes!("assets/ray_query/query.spv")[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([2, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_RAY_QUERY.into()],
    }))
    .unwrap();
    let compute = device.compute().unwrap();
    let pipeline = block_on(compute.create_pipeline(ComputePipelineDesc {
        label: None,
        layout: pipeline_layout,
        compute: PipelineStage {
            module: shader,
            entry: "main".into(),
            constants: vec![],
        },
        cache: None,
        required_subgroup_size: None,
    }))
    .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    let mut bytes = vec![0; 48];
    let xyz = [[-1.0f32, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    for (i, position) in xyz.into_iter().enumerate() {
        for (j, component) in position.into_iter().enumerate() {
            match format {
                VertexFormat::Float32(_) => bytes[i * 16 + j * 4..i * 16 + j * 4 + 4]
                    .copy_from_slice(&component.to_ne_bytes()),
                VertexFormat::Snorm16(_) => bytes[i * 16 + j * 2..i * 16 + j * 2 + 2]
                    .copy_from_slice(&((component * 32767.0) as i16).to_ne_bytes()),
                VertexFormat::Snorm8(_) => bytes[i * 16 + j] = ((component * 127.0) as i8) as u8,
                _ => unreachable!(),
            }
        }
    }
    encoder.write_buffer(&vertices, 0, &bytes).unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let barrier = |buffer, size, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer,
                range: 0..size,
            },
            usage: StateTransition { from: empty, to },
            ownership: None,
        })
    };
    encoder
        .transition_resources(&[
            barrier(
                &vertices,
                48,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_READ,
                },
            ),
            barrier(
                &bscratch,
                bs.scratch_bytes,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
            ),
            barrier(
                &tscratch,
                ts.scratch_bytes,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
            ),
        ])
        .unwrap();
    ray.build(&mut encoder, &blas, &bscratch, &blas_desc)
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Acceleration {
            target: &blas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            },
        }])
        .unwrap();
    ray.build(&mut encoder, &tlas, &tscratch, &tlas_desc)
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Acceleration {
            target: &tlas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::ACCEL_READ,
            },
        }])
        .unwrap();
    let range = || Subresources {
        texture: &texture,
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
    };
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::General)),
            ownership: None,
        })])
        .unwrap();
    let mut pass = compute.begin(&mut encoder).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::General, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..256).unwrap();
    let work = encoder.finish().unwrap();
    // SAFETY: one submitted command step matches the recording device and lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 755499,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the submitted recording and retained resources remain valid until completion.
    let result = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: the waited completion covers the readback operation.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..8], &[255, 0, 0, 255, 0, 255, 0, 255]);
}
