//! Real-device coverage for resource-scoped same-state UAV dependencies.
//!
//! Both transitions below lower to `UNORDERED_ACCESS`.  They must still emit a
//! resource UAV barrier because the preceding dispatch writes data consumed by
//! the following dispatch.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

fn uses(accesses: AccessMask) -> BufferUses {
    BufferUses {
        stages: StageMask::COMPUTE,
        accesses,
    }
}

fn barrier(
    buffer: &Buffer<Dx12>,
    from: BufferUses,
    to: BufferUses,
) -> ResourceTransition<'_, Dx12> {
    ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer,
            range: 0..16,
        },
        usage: StateTransition { from, to },
        ownership: None,
    })
}

fn pipeline(
    device: &Device<Dx12>,
    layout: PipelineLayout<Dx12>,
    source: &'static [u8],
) -> ComputePipeline<Dx12> {
    let shader = block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap()
}

fn run_same_state_dependency(after: BufferUses, consumer_source: &'static [u8]) {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let storage = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 4,
                    access: StorageAccess::ReadWrite,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &storage,
                    range: 0..16,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let producer = pipeline(
        &device,
        layout.clone(),
        b"@group(0) @binding(0) var<storage, read_write> values: array<u32>; @compute @workgroup_size(1) fn main() { values[0] = 7u; }",
    );
    let consumer = pipeline(&device, layout, consumer_source);

    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = uses(AccessMask::SHADER_WRITE);
    let copy = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[barrier(&storage, empty, write)])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&producer).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    // The second dispatch has a dependent UAV access while the native state
    // remains UNORDERED_ACCESS.
    encoder
        .transition_resources(&[barrier(&storage, write, after)])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&consumer).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(&storage, after, copy)])
        .unwrap();
    encoder.copy_buffer(&storage, 0, &readback, 0, 4).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..4).unwrap();
    let commands = encoder.finish().unwrap();
    // SAFETY: the single command step uses this device's graphics lane and
    // receives the matching, one-shot recorded work below.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 704,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the plan, lane and recorded work all belong to this device;
    // the only plan step is bound once and has no external dependencies.
    let point = match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![commands]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission was not accepted"),
    };
    block(device.wait_completion(&point)).unwrap();
    // SAFETY: this completion follows the buffer copy and its readback ticket.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 42);
}

#[test]
fn same_state_uav_write_after_write_reaches_readback() {
    run_same_state_dependency(
        uses(AccessMask::SHADER_WRITE),
        b"@group(0) @binding(0) var<storage, read_write> values: array<u32>; @compute @workgroup_size(1) fn main() { values[0] = 42u; }",
    );
}

#[test]
fn same_state_uav_read_write_after_write_reaches_readback() {
    run_same_state_dependency(
        uses(AccessMask::SHADER_READ | AccessMask::SHADER_WRITE),
        b"@group(0) @binding(0) var<storage, read_write> values: array<u32>; @compute @workgroup_size(1) fn main() { values[0] = values[0] * 6u; }",
    );
}
