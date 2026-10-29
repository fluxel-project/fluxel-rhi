//! Real GPU coverage for changing sparse mip residency across submissions.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;

#[path = "support/mod.rs"]
mod support;
use support::block as block_on;

#[test]
fn switch_standard_mips_and_packed_tail() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let sparse = d.sparse().expect("sparse capability required");
    let texture = sparse
        .texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [256, 256, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 9,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let tiling = sparse.tiling(&texture).unwrap();
    assert_eq!(
        tiling.standard_mip_grids.len() + tiling.packed_tail.map_or(0, |t| t.mip_count as usize),
        9
    );
    eprintln!("sparse switch tiling: {tiling:?}");
    let standard = tiling.standard_mip_grids.len() as u32;
    let group_for = |mip: u32| if mip < standard { mip } else { standard };
    let region_for = |group: u32| {
        if group < standard {
            SparseTile {
                aspect: None,
                mip: group,
                layer: 0,
                origin: [0, 0, 0],
                count: tiling.standard_mip_grids[group as usize],
            }
        } else {
            let tail = tiling.packed_tail.expect("packed tail");
            SparseTile {
                aspect: None,
                mip: tail.first_mip,
                layer: 0,
                origin: [0, 0, 0],
                count: [tail.tile_count, 1, 1],
            }
        }
    };
    let mut current = None;
    let mut initialized = [false; 9];
    for (iteration, mip) in (0..9u32).chain((0..9u32).rev()).enumerate() {
        let group = group_for(mip);
        let mut binds = Vec::new();
        if current != Some(group) {
            if let Some(old) = current {
                binds.push(SparseBind {
                    resource: texture.clone(),
                    tiles: vec![region_for(old)],
                    allocation: None,
                    allocation_offset: 0,
                });
            }
            let region = region_for(group);
            let tiles = u64::from(region.count[0])
                * u64::from(region.count[1])
                * u64::from(region.count[2]);
            // The only public allocation handle moves into the submitted mapping. The backend
            // must retain it beyond completion while that mapping remains installed.
            let heap = sparse.allocation(tiles * tiling.tile_bytes).unwrap();
            binds.push(SparseBind {
                resource: texture.clone(),
                tiles: vec![region],
                allocation: Some(heap),
                allocation_offset: 0,
            });
            current = Some(group);
        }
        let width = 256u32 >> mip;
        let pitch = (width * 4 + 255) & !255;
        let size = u64::from(pitch) * u64::from(width);
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
        let color = [mip as u8 * 23, 255 - mip as u8 * 17, mip as u8 * 11, 255];
        let mut bytes = vec![0u8; size as usize];
        for row in 0..width as usize {
            for x in 0..width as usize {
                let at = row * pitch as usize + x * 4;
                bytes[at..at + 4].copy_from_slice(&color);
            }
        }
        let mut e = d.encoder(lane).unwrap();
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
                    layers: 0..1,
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
            barrier(
                if initialized[mip as usize] {
                    read
                } else {
                    empty
                },
                write,
                (
                    if initialized[mip as usize] {
                        ImageLayout::CopySrc
                    } else {
                        ImageLayout::Common
                    },
                    ImageLayout::CopyDst,
                ),
            ),
        ])
        .unwrap();
        initialized[mip as usize] = true;
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: pitch,
            rows_per_image: width,
        };
        let region = || TextureRegion {
            texture: &texture,
            mip,
            origin: [0, 0, 0],
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
        let ticket = e.readback_buffer(&dest, 0..size).unwrap();
        let mut descriptions = Vec::new();
        if !binds.is_empty() {
            descriptions.push(StepDesc {
                lane,
                kind: StepKind::SparseBind,
                waits: vec![],
            });
        }
        descriptions.push(StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![],
        });
        // SAFETY: each unique plan describes work ordered on the graphics queue.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 877000 + iteration as u64,
                },
                descriptions.into(),
            )
        };
        let mut steps = Vec::new();
        if !binds.is_empty() {
            steps.push(BoundStep {
                step: plan.step(0),
                payload: StepPayload::SparseBind(binds),
            });
        }
        steps.push(BoundStep {
            step: plan.step(steps.len()),
            payload: StepPayload::Commands(vec![e.finish().unwrap()]),
        });
        // SAFETY: previous submission completed, mapping precedes its use, and the plan is unique.
        let result = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps,
                completion: CompletionRequest::End,
            })
        };
        let point = match result {
            SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
            _ => panic!("submission rejected at mip {mip}"),
        };
        block_on(d.wait_completion(&point)).unwrap();
        // SAFETY: the waited completion covers this readback.
        let data = unsafe { block_on(d.read_after(ticket, &point)).unwrap().bytes };
        for row in 0..width as usize {
            for x in 0..width as usize {
                let at = row * pitch as usize + x * 4;
                assert_eq!(
                    &data[at..at + 4],
                    &color,
                    "iteration {iteration}, mip {mip}, pixel ({x}, {row})"
                );
            }
        }
    }
}
