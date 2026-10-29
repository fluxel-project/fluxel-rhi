//! D3 texture writes preserve distinct slices across GPU readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn d3_upload_readback() {
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
            dimension: TextureDimension::D3,
            extent: [4, 4, 4],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 2048,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let copied = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D3,
            extent: [4, 4, 4],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let mut data = vec![0u8; 128];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as u8).wrapping_mul(3).wrapping_add(1);
    }
    let region = || TextureRegion {
        texture: &tex,
        mip: 0,
        origin: [0, 0, 1],
        extent: [4, 4, 2],
        aspect: TextureAspects::COLOR,
    };
    let mut e = d.encoder(lane).unwrap();
    e.write_texture(
        region(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 4,
        },
        &data,
    )
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    let copy_region = || TextureRegion {
        texture: &copied,
        mip: 0,
        origin: [0, 0, 0],
        extent: [4, 4, 2],
        aspect: TextureAspects::COLOR,
    };
    let copy_barrier = |from, to, from_layout, to_layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &copied,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
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
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[copy_barrier(
        empty,
        write,
        ImageLayout::Common,
        ImageLayout::CopyDst,
    )])
    .unwrap();
    e.copy_texture(region(), copy_region()).unwrap();
    e.transition_resources(&[copy_barrier(
        write,
        read,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    )])
    .unwrap();
    e.texture_to_buffer(
        copy_region(),
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 4,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..2048).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99801,
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
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for y in 0..8 {
        assert_eq!(&bytes[y * 256..y * 256 + 16], &data[y * 16..y * 16 + 16]);
    }
}

#[test]
fn d3_buffer_copy_readback() {
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
            dimension: TextureDimension::D3,
            extent: [4, 4, 4],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 2048,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let copied = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D3,
            extent: [4, 4, 4],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let mut data = vec![0u8; 128];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as u8).wrapping_mul(3).wrapping_add(1);
    }
    let region = || TextureRegion {
        texture: &tex,
        mip: 0,
        origin: [0, 0, 1],
        extent: [4, 4, 2],
        aspect: TextureAspects::COLOR,
    };
    let mut e = d.encoder(lane).unwrap();
    let source = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 2048,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let mut staged = vec![0u8; 2048];
    for y in 0..8 {
        staged[y * 256..y * 256 + 16].copy_from_slice(&data[y * 16..y * 16 + 16]);
    }
    e.write_buffer(&source, 0, &staged).unwrap();
    e.transition_resources(&[
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &source,
                range: 0..2048,
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
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::CopyDst)),
            ownership: None,
        }),
    ])
    .unwrap();
    e.buffer_to_texture(
        &source,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 4,
        },
        region(),
    )
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            to: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    let copy_region = || TextureRegion {
        texture: &copied,
        mip: 0,
        origin: [0, 0, 0],
        extent: [4, 4, 2],
        aspect: TextureAspects::COLOR,
    };
    let copy_barrier = |from, to, from_layout, to_layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &copied,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
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
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[copy_barrier(
        empty,
        write,
        ImageLayout::Common,
        ImageLayout::CopyDst,
    )])
    .unwrap();
    e.copy_texture(region(), copy_region()).unwrap();
    e.transition_resources(&[copy_barrier(
        write,
        read,
        ImageLayout::CopyDst,
        ImageLayout::CopySrc,
    )])
    .unwrap();
    e.texture_to_buffer(
        copy_region(),
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 4,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..2048).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99802,
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
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for y in 0..8 {
        assert_eq!(&bytes[y * 256..y * 256 + 16], &data[y * 16..y * 16 + 16]);
    }
}
