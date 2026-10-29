//! A storage array view addresses its selected native array slice.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

#[test]
fn enhanced_global_copy_scope_records_or_runtime_skips() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let graphics = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let run = |lane, global, ordinal| -> Result<Vec<u8>> {
        let buffer = d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })?;
        let readback = d.create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        let copy_write = BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        };
        let copy_read = BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        };
        let mut e = d.encoder(lane)?;
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..256,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: copy_write,
            },
            ownership: None,
        })])?;
        // Both clears execute in unchanged COPY_DEST state. The Global
        // transition supplies only their memory dependency.
        e.clear_buffer(&buffer, 0..256, 0x1122_3344)?;
        e.transition_resources(&[global])?;
        e.clear_buffer(&buffer, 0..256, 0xa1b2_c3d4)?;
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..256,
            },
            usage: StateTransition {
                from: copy_write,
                to: copy_read,
            },
            ownership: None,
        })])?;
        e.copy_buffer(&buffer, 0, &readback, 0, 256)?;
        let ticket = e.readback_buffer(&readback, 0..256)?;
        let work = e.finish()?;
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let point = match unsafe {
            d.queue().submit(SubmitPacket {
                steps: vec![BoundStep {
                    step: plan.step(0),
                    payload: StepPayload::Commands(vec![work]),
                }],
                plan,
                completion: CompletionRequest::End,
            })
        } {
            SubmitOutcome::Accepted(receipt) => {
                receipt.checkpoints.into_iter().next().unwrap().point
            }
            _ => return Err(Error::InvalidState),
        };
        block(d.wait_completion(&point))?;
        Ok(unsafe { block(d.read_after(ticket, &point))?.bytes })
    };

    let copy_global = ResourceTransition::Global {
        before: AccessScope {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
        after: AccessScope {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        },
    };
    match run(graphics, copy_global, 0x67_10_01) {
        Ok(bytes) => {
            assert_eq!(&bytes[..4], &0xa1b2_c3d4_u32.to_le_bytes());
            eprintln!("Enhanced global Copy write->write: submitted and read back");
        }
        Err(Error::Unsupported(reason)) => {
            eprintln!("SKIP: enhanced global copy barrier unavailable at runtime: {reason}");
            return;
        }
        Err(error) => panic!("unexpected global barrier error: {error:?}"),
    }

    let Some(compute) = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Compute)
        .map(|v| v.id)
    else {
        eprintln!("SKIP: no DX12 compute lane for Global ALL/ALL acceptance");
        return;
    };
    let all = ResourceTransition::Global {
        before: AccessScope {
            stages: StageMask::ALL_COMMANDS,
            accesses: AccessMask::ALL,
        },
        after: AccessScope {
            stages: StageMask::ALL_COMMANDS,
            accesses: AccessMask::ALL,
        },
    };
    let bytes = run(compute, all, 0x67_10_02).expect("compute Global ALL/ALL submission");
    assert_eq!(&bytes[..4], &0xa1b2_c3d4_u32.to_le_bytes());
    eprintln!("Enhanced global ALL/ALL on compute lane: submitted and read back");
}

#[test]
fn uint_storage_view_writes_selected_array_slice() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let tex = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 3,
            format: TextureFormat::R32Uint,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format: TextureFormat::R32Uint,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
        )
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::R32Uint,
                    dimension: TextureViewDimension::D2Array,
                    access: StorageAccess::Write,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        })
        .unwrap();
    let pl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(include_bytes!("assets/uint_uav.dxil").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let pipeline = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
        label: None,
        layout: pl,
        compute: PipelineStage {
            module: shader,
            entry: "main".into(),
            constants: vec![],
        },
        required_subgroup_size: None,
        cache: None,
    }))
    .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let barrier = |from, to, from_layout, to_layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition { from, to },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })
    };
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[barrier(
        empty,
        write,
        ImageLayout::Common,
        ImageLayout::General,
    )])
    .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    // The texture stays in UAV/General state: this is solely a memory
    // dependency between a producer and consumer, not a layout transition.
    // On enhanced-barrier runtimes it records a global barrier; older DX12
    // runtimes use the valid null-UAV subset.
    e.transition_resources(&[ResourceTransition::Global {
        before: AccessScope {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_WRITE,
        },
        after: AccessScope {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_WRITE,
        },
    }])
    .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[barrier(
        write,
        read,
        ImageLayout::General,
        ImageLayout::CopySrc,
    )])
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 1],
            extent: [1, 1, 1],
            aspect: TextureAspects::COLOR,
        },
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 1,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..256).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99853,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submit"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..4], &[0x78, 0x56, 0x34, 0x12]);
}
