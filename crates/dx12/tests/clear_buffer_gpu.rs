//! GPU readback coverage for standalone patterned buffer clear.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

#[test]
fn patterned_clear_reuses_upload_data_and_preserves_explicit_states() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let size = 65_552_u64;
    let buffer = device
        .create_buffer(&BufferDesc {
            label: None,
            size,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    // Establish nonzero sentinels through the public upload path before the
    // explicitly transitioned clear, so the boundary assertions do not rely
    // on driver allocation initialization.
    encoder
        .write_buffer(&buffer, 0, &vec![0x5a; size as usize])
        .unwrap();
    let copy_write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..size,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: copy_write,
            },
            ownership: None,
        })])
        .unwrap();
    encoder
        .clear_buffer(&buffer, 4..65_548, 0x6a3b_19e4)
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..size,
            },
            usage: StateTransition {
                from: copy_write,
                to: BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            ownership: None,
        })])
        .unwrap();
    encoder.copy_buffer(&buffer, 0, &readback, 0, size).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..size).unwrap();
    let work = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 6_120_021,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![work]),
            }],
            plan,
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("DX12 submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..4], &[0x5a; 4]);
    assert_eq!(&bytes[4..8], &0x6a3b_19e4_u32.to_le_bytes());
    assert_eq!(&bytes[65_540..65_544], &0x6a3b_19e4_u32.to_le_bytes());
    assert_eq!(&bytes[65_548..], &[0x5a; 4]);
}
