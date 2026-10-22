//! Real GPU test of restricted sparse mapping, remapping and readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};
fn block_on<F: Future>(f: F) -> F::Output {
    let w = std::task::Waker::noop();
    let mut c = std::task::Context::from_waker(w);
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut c) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
#[test]
fn bc_mips_multiple_heaps_and_packed_tail() {
    all_mips_case(1, 0);
}

fn all_mips_case(layers: u32, layer: u32) {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let sparse = d.sparse().expect("sparse capability");
    let texture = match sparse.texture(&TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [256, 256, 1],
        array_layers: layers,
        format: TextureFormat::Bc1RgbaUnorm,
        mip_levels: 9,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    }) {
        Ok(texture) => texture,
        Err(Error::Unsupported(_)) if layers > 1 => return,
        Err(error) => panic!("sparse texture creation failed: {error:?}"),
    };
    let tiling = sparse.tiling(&texture).unwrap();
    assert_eq!(
        tiling.standard_mip_grids.len() + tiling.packed_tail.map_or(0, |t| t.mip_count as usize),
        9
    );
    eprintln!("tiling: {tiling:?}");
    let mut binds = Vec::new();
    for (mip, grid) in tiling.standard_mip_grids.iter().enumerate() {
        let count = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
        let heap = sparse.allocation((count + 1) * tiling.tile_bytes).unwrap();
        binds.push(SparseBind {
            resource: texture.clone(),
            tiles: vec![SparseTile {
                aspect: None,
                mip: mip as u32,
                layer,
                origin: [0, 0, 0],
                count: *grid,
            }],
            allocation: Some(heap),
            allocation_offset: tiling.tile_bytes,
        });
    }
    if let Some(tail) = tiling.packed_tail {
        let heap = sparse
            .allocation((u64::from(tail.tile_count) + 1) * tiling.tile_bytes)
            .unwrap();
        binds.push(SparseBind {
            resource: texture.clone(),
            tiles: vec![SparseTile {
                aspect: None,
                mip: tail.first_mip,
                layer,
                origin: [0, 0, 0],
                count: [tail.tile_count, 1, 1],
            }],
            allocation: Some(heap),
            allocation_offset: tiling.tile_bytes,
        });
    }
    let mut e = d.encoder(lane).unwrap();
    let mut tickets = Vec::new();
    for mip in 0..9u32 {
        let width = (256u32 >> mip).max(1);
        let blocks = width.div_ceil(4);
        let pitch = (blocks * 8 + 255) & !255;
        let size = u64::from(pitch) * u64::from(blocks);
        let source = d
            .create_buffer(&BufferDesc {
                label: None,
                size,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let dest = d
            .create_buffer(&BufferDesc {
                label: None,
                size,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let color = [mip as u8 * 23; 8];
        let mut bytes = vec![0u8; size as usize];
        for row in 0..blocks as usize {
            for x in 0..blocks as usize {
                let at = row * pitch as usize + x * 8;
                bytes[at..at + 8].copy_from_slice(&color);
            }
        }
        e.write_buffer(&source, 0, &bytes).unwrap();
        let empty = TextureUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        let write = TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        };
        let read = TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        };
        let barrier = |from, to, layout| {
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &texture,
                    aspects: TextureAspects::COLOR,
                    mips: mip..mip + 1,
                    layers: layer..layer + 1,
                },
                usage: StateTransition { from, to },
                layout: Some(layout),
                ownership: None,
            })
        };
        e.transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &source,
                    range: 0..size,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            }),
            barrier(empty, write, (ImageLayout::Common, ImageLayout::CopyDst)),
        ])
        .unwrap();
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: pitch,
            rows_per_image: blocks,
        };
        let region = || TextureRegion {
            texture: &texture,
            mip,
            origin: [0, 0, layer],
            extent: [width, width, 1],
            aspect: TextureAspects::COLOR,
        };
        e.buffer_to_texture(&source, layout, region()).unwrap();
        e.transition_resources(&[barrier(
            write,
            read,
            (ImageLayout::CopyDst, ImageLayout::CopySrc),
        )])
        .unwrap();
        e.texture_to_buffer(region(), &dest, layout).unwrap();
        tickets.push((
            e.readback_buffer(&dest, 0..size).unwrap(),
            width,
            pitch,
            color,
        ));
    }
    // SAFETY: the mapping step and commands are ordered on their owning queue.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 876010,
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
    let steps = vec![
        BoundStep {
            step: plan.step(0),
            payload: StepPayload::SparseBind(binds),
        },
        BoundStep {
            step: plan.step(1),
            payload: StepPayload::Commands(vec![e.finish().unwrap()]),
        },
    ];
    // SAFETY: work is submitted once and mapping makes every accessed mip resident.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps,
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("sparse mapping rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
    for (ticket, width, pitch, color) in tickets {
        // SAFETY: the waited point covers each readback.
        let data = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
        for row in 0..width.div_ceil(4) as usize {
            for x in 0..width.div_ceil(4) as usize {
                let at = row * pitch as usize + x * 8;
                assert_eq!(&data[at..at + 8], &color);
            }
        }
    }
    if let Some(tail) = tiling.packed_tail {
        let make_plan = |ordinal| unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::SparseBind,
                    waits: vec![],
                }]),
            )
        };
        let plan = make_plan(876011);
        let step = plan.step(0);
        let partial = SparseBind {
            resource: texture.clone(),
            tiles: vec![SparseTile {
                aspect: None,
                mip: tail.first_mip,
                layer,
                origin: [0, 0, 0],
                count: [tail.tile_count.saturating_sub(1), 1, 1],
            }],
            allocation: Some(sparse.allocation(tiling.tile_bytes).unwrap()),
            allocation_offset: 0,
        };
        // SAFETY: malformed tail mappings must be rejected before native handoff.
        let result = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::SparseBind(vec![partial]),
                }],
                completion: CompletionRequest::End,
            })
        };
        assert!(matches!(
            result,
            SubmitOutcome::Rejected {
                error: Error::InvalidRange,
                work: RetryableSubmit::Fresh(_)
            }
        ));
        let region = || SparseTile {
            aspect: None,
            mip: tail.first_mip,
            layer,
            origin: [0, 0, 0],
            count: [tail.tile_count, 1, 1],
        };
        let plan = make_plan(876012);
        let step = plan.step(0);
        let replacement = sparse
            .allocation(u64::from(tail.tile_count) * tiling.tile_bytes)
            .unwrap();
        // SAFETY: the earlier submission completed before the mapping changes.
        let result = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::SparseBind(vec![
                        SparseBind {
                            resource: texture.clone(),
                            tiles: vec![region()],
                            allocation: None,
                            allocation_offset: 0,
                        },
                        SparseBind {
                            resource: texture.clone(),
                            tiles: vec![region()],
                            allocation: Some(replacement),
                            allocation_offset: 0,
                        },
                    ]),
                }],
                completion: CompletionRequest::End,
            })
        };
        let point = match result {
            SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
            _ => panic!("tail remap rejected"),
        };
        block_on(d.wait_completion(&point)).unwrap();
    }
}
