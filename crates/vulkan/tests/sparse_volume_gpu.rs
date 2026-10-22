//! Real GPU sparse array layer and 3D volume residency readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::future::Future;
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
fn sparse_array_and_volume_readback_when_supported() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let Some(sparse) = d.sparse() else {
        eprintln!("sparse unavailable");
        return;
    };
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    for (dimension, base_extent, layers, target_layer, mip_levels) in [
        (TextureDimension::D2, [128, 128, 1], 2, 1, 1),
        (TextureDimension::D2, [256, 256, 1], 2, 1, 9),
        (TextureDimension::D3, [32, 32, 8], 1, 0, 1),
    ] {
        let desc = TextureDesc {
            label: None,
            dimension,
            extent: base_extent,
            array_layers: layers,
            format: TextureFormat::Rgba8Unorm,
            mip_levels,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        };
        let texture = match sparse.texture(&desc) {
            Ok(t) => t,
            Err(Error::Unsupported(reason)) => {
                eprintln!("skip {dimension:?}: {reason}");
                continue;
            }
            Err(e) => panic!("sparse creation failed: {e:?}"),
        };
        let tiling = sparse.tiling(&texture).unwrap();
        eprintln!("{dimension:?} tiling: {tiling:?}");
        let mip = if mip_levels > 1 {
            let Some(tail) = tiling.packed_tail else {
                eprintln!("skip: no packed tail");
                continue;
            };
            tail.first_mip
        } else {
            0
        };
        let extent = [
            (base_extent[0] >> mip).max(1),
            (base_extent[1] >> mip).max(1),
            (base_extent[2] >> mip).max(1),
        ];
        let tile = if let Some(grid) = tiling.standard_mip_grids.get(mip as usize) {
            SparseTile {
                aspect: None,
                mip,
                layer: target_layer,
                origin: [0; 3],
                count: *grid,
            }
        } else {
            let tail = tiling.packed_tail.unwrap();
            SparseTile {
                aspect: None,
                mip: tail.first_mip,
                layer: if tiling.packed_tail_shared_by_layers {
                    0
                } else {
                    target_layer
                },
                origin: [0; 3],
                count: [tail.tile_count, 1, 1],
            }
        };
        let count = u64::from(tile.count[0]) * u64::from(tile.count[1]) * u64::from(tile.count[2]);
        let allocation = sparse
            .allocation_for(&texture, count * tiling.tile_bytes)
            .unwrap();
        let pitch = extent[0] * 4;
        let size = u64::from(pitch) * u64::from(extent[1]) * u64::from(extent[2]);
        let source = d
            .create_buffer(&BufferDesc {
                label: None,
                size,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let output = d
            .create_buffer(&BufferDesc {
                label: None,
                size,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let bytes: Vec<u8> = (0..size).map(|i| ((i * 17 + 31) % 251) as u8).collect();
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
                    layers: target_layer..target_layer + 1,
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
            barrier(empty, write, (ImageLayout::Undefined, ImageLayout::CopyDst)),
        ])
        .unwrap();
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: pitch,
            rows_per_image: extent[1],
        };
        let region = || TextureRegion {
            texture: &texture,
            mip,
            origin: [0, 0, target_layer],
            extent,
            aspect: TextureAspects::COLOR,
        };
        e.buffer_to_texture(&source, layout, region()).unwrap();
        e.transition_resources(&[barrier(
            write,
            read,
            (ImageLayout::CopyDst, ImageLayout::CopySrc),
        )])
        .unwrap();
        e.texture_to_buffer(region(), &output, layout).unwrap();
        let ticket = e.readback_buffer(&output, 0..size).unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 879001 + u64::from(target_layer) + u64::from(mip),
                },
                vec![
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
                ]
                .into(),
            )
        };
        let steps = vec![
            BoundStep {
                step: plan.step(0),
                payload: StepPayload::SparseBind(vec![SparseBind {
                    resource: texture.clone(),
                    tiles: vec![tile],
                    allocation: Some(allocation),
                    allocation_offset: 0,
                }]),
            },
            BoundStep {
                step: plan.step(1),
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            },
        ];
        let receipt = match unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps,
                completion: CompletionRequest::End,
            })
        } {
            SubmitOutcome::Accepted(r) => r,
            _ => panic!("sparse {dimension:?} submit failed"),
        };
        let point = &receipt.checkpoints[0].point;
        block_on(d.wait_completion(point)).unwrap();
        let actual = unsafe { block_on(d.read_after(ticket, point)).unwrap().bytes };
        assert_eq!(actual, bytes, "sparse {dimension:?} layer {target_layer}");
        eprintln!("sparse {dimension:?} mip {mip} readback passed");
    }
}

#[test]
fn probe_array_packed_tail_layout() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let Some(sparse) = d.sparse() else { return };
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [256, 256, 1],
        array_layers: 2,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 9,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let image = match sparse.texture(&desc) {
        Ok(image) => image,
        Err(Error::Unsupported(reason)) => {
            eprintln!("skip: {reason}");
            return;
        }
        Err(error) => panic!("sparse creation failed: {error:?}"),
    };
    let tiling = sparse.tiling(&image).unwrap();
    eprintln!("array packed tail: {tiling:?}");
    assert!(!tiling.packed_tail_shared_by_layers || tiling.packed_tail.is_some());
}

#[test]
fn probe_sparse_depth_residency() {
    let d = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let Some(sparse) = d.sparse() else { return };
    for format in [
        TextureFormat::Depth16Unorm,
        TextureFormat::Depth32Float,
        TextureFormat::Depth24PlusStencil8,
    ] {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [256, 256, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        };
        match sparse.texture(&desc) {
            Ok(texture) => eprintln!(
                "sparse {format:?} available: {:?}",
                sparse.tiling(&texture).unwrap()
            ),
            Err(Error::Unsupported(reason)) => eprintln!("sparse {format:?} unavailable: {reason}"),
            Err(error) => panic!("sparse {format:?} creation failed: {error:?}"),
        }
    }
}
