//! Sparse 3D tile mapping and copy roundtrip.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block as block_on;
#[test]
fn sparse_3d_map_remap_and_readback() {
    map_remap_case(TextureFormat::Rgba8Unorm, 1, 0);
}

fn map_remap_case(format: TextureFormat, layers: u32, layer: u32) {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let sparse = d
        .sparse()
        .expect("device must support tiled resources for this fixture");
    let texture = sparse
        .texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D3,
            extent: [64, 64, 16],
            array_layers: layers,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST),
            view_formats: vec![],
        })
        .unwrap();
    let tiling = sparse.tiling(&texture).unwrap();
    let size = u64::from(tiling.tile_grid[0])
        * u64::from(tiling.tile_grid[1])
        * u64::from(tiling.tile_grid[2])
        * tiling.tile_bytes;
    let allocation_a = sparse.allocation(size).unwrap();
    let allocation_b = sparse.allocation(size).unwrap();
    let (pitch, rows) = (256u64, 64u64);
    let buffer_size = pitch * rows * 16;
    for (iteration, allocation) in [allocation_a, allocation_b].into_iter().enumerate() {
        let source = d
            .create_buffer(&BufferDesc {
                label: None,
                size: buffer_size,
                usage: BufferUsage::COPY_SRC.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
            .unwrap();
        let dest = d
            .create_buffer(&BufferDesc {
                label: None,
                size: buffer_size,
                usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let expected = vec![if iteration == 0 { 37u8 } else { 193u8 }; buffer_size as usize];
        let mut e = d.encoder(lane).unwrap();
        e.write_buffer(&source, 0, &expected).unwrap();
        let empty = TextureUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        let read = TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        };
        let write = TextureUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_WRITE,
        };
        let barrier = |from, to, layouts| {
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &texture,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: layer..layer + 1,
                },
                usage: StateTransition { from, to },
                layout: Some(layouts),
                ownership: None,
            })
        };
        e.transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &source,
                    range: 0..buffer_size,
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
            barrier(
                if iteration == 0 { empty } else { read },
                write,
                (
                    if iteration == 0 {
                        ImageLayout::Common
                    } else {
                        ImageLayout::CopySrc
                    },
                    ImageLayout::CopyDst,
                ),
            ),
        ])
        .unwrap();
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: pitch as u32,
            rows_per_image: rows as u32,
        };
        let region = || TextureRegion {
            texture: &texture,
            mip: 0,
            origin: [0, 0, layer],
            extent: [64, 64, 16],
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
        let ticket = e.readback_buffer(&dest, 0..buffer_size).unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 876000 + iteration as u64,
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
                payload: StepPayload::SparseBind(vec![SparseBind {
                    resource: texture.clone(),
                    tiles: vec![SparseTile {
                        aspect: None,
                        mip: 0,
                        layer,
                        origin: [0, 0, 0],
                        count: tiling.tile_grid,
                    }],
                    allocation: Some(allocation),
                    allocation_offset: 0,
                }]),
            },
            BoundStep {
                step: plan.step(1),
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            },
        ];
        // SAFETY: plan is unique and work is ordered on its owning queue; mappings cover the entire texture.
        let outcome = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps,
                completion: CompletionRequest::End,
            })
        };
        let point = match outcome {
            SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
            _ => panic!("submission rejected"),
        };
        block_on(d.wait_completion(&point)).unwrap();
        // SAFETY: the point covers the readback.
        let bytes = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
        assert_eq!(&*bytes, expected.as_slice());
    }
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
    let plan = make_plan(876002);
    let invalid = SparseBind {
        resource: texture.clone(),
        tiles: vec![SparseTile {
            aspect: None,
            mip: 0,
            layer,
            origin: [0, 0, 0],
            count: tiling.tile_grid,
        }],
        allocation: Some(sparse.allocation(size).unwrap()),
        allocation_offset: 1,
    };
    let step = plan.step(0);
    // SAFETY: the invalid offset must be rejected before native queue handoff.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::SparseBind(vec![invalid]),
            }],
            completion: CompletionRequest::End,
        })
    };
    assert!(matches!(
        outcome,
        SubmitOutcome::Rejected {
            error: Error::InvalidRange,
            work: RetryableSubmit::Fresh(_)
        }
    ));
    let plan = make_plan(876003);
    let step = plan.step(0);
    let unmap = SparseBind {
        resource: texture.clone(),
        tiles: vec![SparseTile {
            aspect: None,
            mip: 0,
            layer,
            origin: [0, 0, 0],
            count: tiling.tile_grid,
        }],
        allocation: None,
        allocation_offset: 0,
    };
    // SAFETY: preceding readback completed, so no command uses the unmapped tiles.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::SparseBind(vec![unmap]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("unmap rejected"),
    };
    block_on(d.wait_completion(&point)).unwrap();
}
