//! CPU upload and texture copy across array layers.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn bc_array_cpu_and_texture_roundtrip() {
    check(TextureFormat::Bc1RgbaUnorm, TextureAspects::COLOR, 0);
}
fn check(format: TextureFormat, aspect: TextureAspects, index: u64) {
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
            extent: [8, 8, 1],
            array_layers: 3,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let mut data = vec![0u8; 2304];
    for y in 0..2 {
        for x in 0..16 {
            data[y * 256 + x] = (x + y * 16 + 1) as u8;
        }
    }
    let (first, rest) = data.split_at_mut(768);
    rest[..768].copy_from_slice(first);
    rest[768..].copy_from_slice(first);
    for layer in 0..3 {
        for row in 0..2 {
            data[layer * 768 + row * 256] ^= layer as u8 * 31;
        }
    }
    e.write_texture(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 3],
            aspect,
        },
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 3,
        },
        &data,
    )
    .unwrap();
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 2304,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: aspect,
            mips: 0..1,
            layers: 0..3,
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
    let target = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 3,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &target,
            aspects: aspect,
            mips: 0..1,
            layers: 0..3,
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
        layout: Some((ImageLayout::Undefined, ImageLayout::CopyDst)),
        ownership: None,
    })])
    .unwrap();
    e.copy_texture(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 3],
            aspect,
        },
        TextureRegion {
            texture: &target,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 3],
            aspect,
        },
    )
    .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &target,
            aspects: aspect,
            mips: 0..1,
            layers: 0..3,
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
    e.texture_to_buffer(
        TextureRegion {
            texture: &target,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 3],
            aspect,
        },
        &output,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 3,
        },
    )
    .unwrap();
    let sample_ticket = e.readback_buffer(&output, 0..2304).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99530 + index as u64,
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
    let sample = unsafe { block(d.read_after(sample_ticket, &point)).unwrap().bytes };
    let row_bytes = 16;
    for layer in 0..3 {
        for row in 0..2 {
            let offset = layer * 768 + row * 256;
            assert_eq!(
                &sample[offset..offset + row_bytes],
                &data[offset..offset + row_bytes],
                "layer {layer}, row {row}"
            );
        }
    }
}
