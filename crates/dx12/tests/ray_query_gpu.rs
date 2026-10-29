//! Real GPU inline ray query: the first ray hits and the second misses.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[test]
fn inline_ray_query_hits_and_misses() {
    run_query(false);
}

#[test]
fn inline_ray_query_consumes_acceleration_array() {
    run_query(true);
}

fn run_query(array: bool) {
    let device = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !device.capabilities().ray_query {
        eprintln!("SKIP: DXR 1.1 and shader model 6.5 unavailable");
        return;
    }
    if array && !device.capabilities().binding_arrays {
        return;
    }
    if array {
        assert!(device.capabilities().acceleration_structure_binding_arrays);
        let layout = |count, partially_bound| BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::AccelerationStructure,
                count,
                partially_bound,
                nonuniform_indexing: false,
            }],
        };
        assert!(matches!(
            device.create_bind_group_layout(&layout(
                BindingCount::RuntimeSized { max_count: 2 },
                false,
            )),
            Err(Error::Unsupported(
                "runtime-sized acceleration structure arrays"
            ))
        ));
        assert!(matches!(
            device.create_bind_group_layout(&layout(BindingCount::Fixed(2), true)),
            Err(Error::Unsupported(
                "partially-bound acceleration structure arrays"
            ))
        ));
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
    let vertices = buffer(36, BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST);
    let blas_desc = AccelerationDesc {
        input: AccelerationInput::Blas(vec![AccelerationGeometry::Triangles {
            vertices: vertices.clone(),
            vertex_offset: 0,
            vertex_count: 3,
            vertex_format: VertexFormat::Float32(3),
            stride: 12,
            indices: None,
            opaque: true,
        }]),
        fast_trace: true,
        allow_update: false,
    };
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
    let tlas2_desc = AccelerationDesc {
        input: AccelerationInput::Tlas(vec![AccelerationInstance {
            blas: blas.clone(),
            transform: [[1., 0., 0., 4.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            mask: 255,
            custom_index: 0,
            shader_table_offset: 0,
            flags: AccelerationInstanceFlags::TRIANGLE_CULL_DISABLE,
        }]),
        fast_trace: true,
        allow_update: false,
    };
    let tlas2 = ray.create_acceleration(&tlas2_desc).unwrap();
    let bs = ray.sizes(&blas_desc).unwrap();
    let ts = ray.sizes(&tlas_desc).unwrap();
    let bscratch = buffer(bs.scratch_bytes, BufferUsage::ACCEL_SCRATCH);
    let tscratch = buffer(ts.scratch_bytes, BufferUsage::ACCEL_SCRATCH);
    let t2sizes = ray.sizes(&tlas2_desc).unwrap();
    let t2scratch = buffer(t2sizes.scratch_bytes, BufferUsage::ACCEL_SCRATCH);
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
                    count: if array {
                        BindingCount::Fixed(2)
                    } else {
                        BindingCount::One
                    },
                    partially_bound: false,
                    nonuniform_indexing: array,
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
                    resource: if array {
                        BindingResource::Array(vec![
                            Some(BindingResource::Acceleration(&tlas)),
                            Some(BindingResource::Acceleration(&tlas2)),
                        ])
                    } else {
                        BindingResource::Acceleration(&tlas)
                    },
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
        format: ShaderFormat::Dxil,
        bytes: if array {
            Arc::from(&include_bytes!("assets/ray_query/array.dxil")[..])
        } else {
            Arc::from(&include_bytes!("assets/ray_query/query.dxil")[..])
        },
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
        layout: pipeline_layout.clone(),
        compute: PipelineStage {
            module: shader.clone(),
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
    let bytes: Vec<_> = [-1f32, -1., 0., 1., -1., 0., 0., 1., 0.]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
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
                36,
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
            barrier(
                &t2scratch,
                t2sizes.scratch_bytes,
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
    ray.build(&mut encoder, &tlas2, &t2scratch, &tlas2_desc)
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
    encoder
        .transition_resources(&[ResourceTransition::Acceleration {
            target: &tlas2,
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
    if array {
        // The recorded work and bind group must retain both distinct TLAS
        // values plus the BLAS/build inputs after caller handles are released.
        drop(group);
        drop(pipeline);
        drop(shader);
        drop(pipeline_layout);
        drop(layout);
        drop(view);
        drop(texture);
        drop(tlas);
        drop(tlas2);
        drop(blas);
        drop(bscratch);
        drop(tscratch);
        drop(t2scratch);
        drop(vertices);
    }
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
