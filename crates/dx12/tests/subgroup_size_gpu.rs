//! SM 6.6 required wave size reaches real DX12 execution when available.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use std::{future::Future, sync::Arc};

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn submit(device: &Device<Dx12>, lane: LaneId, work: RecordedWork<Dx12>) -> CompletionPoint<Dx12> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 49_161,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(mut receipt) => receipt.checkpoints.remove(0).point,
        SubmitOutcome::Rejected { error, .. } => {
            panic!("DX12 subgroup submission rejected: {error:?}")
        }
        SubmitOutcome::PartiallyAccepted { error, .. } => {
            panic!("DX12 subgroup submission partially accepted: {error:?}")
        }
        SubmitOutcome::Indeterminate { error, .. } => {
            panic!("DX12 subgroup submission indeterminate: {error:?}")
        }
    }
}

#[test]
fn required_wave_size_matches_subgroup_size_builtin() {
    const SIZE: u32 = 32;
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let layout = device
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
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let output = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &output,
                    range: 0..4,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let source = format!(
        "@group(0) @binding(0) var<storage,read_write> out:array<u32>; @compute @workgroup_size({SIZE}) fn main(@builtin(subgroup_size) size:u32,@builtin(subgroup_invocation_id) lane:u32){{ if lane==0u {{ out[0]=size; }} }}"
    );
    let shader = match block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([SIZE, 1, 1]),
        }],
        required_features: vec!["subgroups".into()],
    })) {
        Ok(shader) => shader,
        Err(Error::Unsupported(_)) => return,
        Err(error) => panic!("DX12 subgroup shader creation failed: {error:?}"),
    };
    let compute = device.compute().unwrap();
    let pipeline = match block(compute.create_pipeline(ComputePipelineDesc {
        label: None,
        layout: pipeline_layout,
        compute: PipelineStage {
            module: shader,
            entry: "main".into(),
            constants: vec![],
        },
        required_subgroup_size: Some(SIZE),
        cache: None,
    })) {
        Ok(pipeline) => pipeline,
        Err(Error::Unsupported(_)) => return,
        Err(error) => panic!("DX12 required wave-size pipeline failed: {error:?}"),
    };
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition {
                from: empty,
                to: write,
            },
            ownership: None,
        })])
        .unwrap();
    let mut pass = compute.begin(&mut encoder).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition {
                from: write,
                to: copy,
            },
            ownership: None,
        })])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 4).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..4).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block(device.wait_completion(&point)).unwrap();
    assert_eq!(
        unsafe { block(device.read_after(ticket, &point)).unwrap().bytes },
        SIZE.to_le_bytes()
    );
}
