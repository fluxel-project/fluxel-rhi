//! Sparse volume storage mip executes through a mapped packed tail.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

fn block<F: Future>(f: F) -> F::Output {
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
fn sparse_storage_volume_mip_writes() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let sparse = d.sparse().unwrap();
    let tex = sparse
        .texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D3,
            extent: [4, 4, 4],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let tiling = sparse.tiling(&tex).unwrap();
    let mut binds = Vec::new();
    for (mip, grid) in tiling.standard_mip_grids.iter().enumerate() {
        let count = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
        binds.push(SparseBind {
            resource: tex.clone(),
            tiles: vec![SparseTile {
                mip: mip as u32,
                layer: 0,
                origin: [0, 0, 0],
                count: *grid,
            }],
            allocation: Some(sparse.allocation(count * tiling.tile_bytes).unwrap()),
            allocation_offset: 0,
        });
    }
    if let Some(tail) = tiling.packed_tail {
        binds.push(SparseBind {
            resource: tex.clone(),
            tiles: vec![SparseTile {
                mip: tail.first_mip,
                layer: 0,
                origin: [0, 0, 0],
                count: [tail.tile_count, 1, 1],
            }],
            allocation: Some(
                sparse
                    .allocation(u64::from(tail.tile_count) * tiling.tile_bytes)
                    .unwrap(),
            ),
            allocation_offset: 0,
        });
    }
    let view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D3,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: 0..1,
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
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D3,
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
        bytes: Arc::from(include_bytes!("assets/volume_uav.dxil").as_slice()),
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
                mips: 1..2,
                layers: 0..1,
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
            mip: 1,
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
                ordinal: 99851,
            },
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::SparseBind,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let first = plan.step(0);
    let step = plan.step(1);
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: first,
                    payload: StepPayload::SparseBind(binds),
                },
                BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![e.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submit"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..4], &[255, 0, 0, 255]);
}
