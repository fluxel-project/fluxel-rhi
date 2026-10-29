//! Direct HLSL is compiled per selected DX12 pipeline entry and read back.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

fn submit(device: &Device<Dx12>, lane: LaneId, work: RecordedWork<Dx12>) -> CompletionPoint<Dx12> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 49_163,
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
        SubmitOutcome::Rejected { error, .. } => panic!("HLSL submission rejected: {error:?}"),
        SubmitOutcome::PartiallyAccepted { error, .. } => {
            panic!("HLSL submission partially accepted: {error:?}")
        }
        SubmitOutcome::Indeterminate { error, .. } => {
            panic!("HLSL submission indeterminate: {error:?}")
        }
    }
}

#[test]
fn direct_hlsl_compute_specializes_macro_and_required_wave_size() {
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
    let shader = match block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Hlsl(66),
        bytes: Arc::from(
            &b"RWByteAddressBuffer output : register(u0, space0);\nstruct Decoy { void main() {} };\n// void main() { output.Store(0, 0); }\n#if 0\nvoid main() { output.Store(0, 0); }\n#endif\n[numthreads(64, 1, 1)] void main(uint lane : SV_GroupIndex) { if (WaveGetLaneIndex() == 0) { double value = DOUBLE_VALUE; output.Store(0, (uint)value + WaveGetLaneCount()); } }\n"[..],
        ),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([64, 1, 1]),
        }],
        required_features: vec![],
    })) {
        Ok(shader) => shader,
        Err(Error::Unsupported(_)) => return,
        Err(error) => panic!("DX12 HLSL shader creation failed: {error:?}"),
    };
    let compute = device.compute().unwrap();
    let mut selected = None;
    for size in [64, 32] {
        match block(compute.create_pipeline(ComputePipelineDesc {
            label: None,
            layout: pipeline_layout.clone(),
            compute: PipelineStage {
                module: shader.clone(),
                entry: "main".into(),
                constants: vec![(
                    "DOUBLE_VALUE".into(),
                    SpecializationValue::F64(16_777_217.0),
                )],
            },
            required_subgroup_size: Some(size),
            cache: None,
        })) {
            Ok(pipeline) => {
                selected = Some((pipeline, size));
                break;
            }
            Err(Error::Unsupported(_)) => continue,
            Err(error) => panic!("DX12 HLSL required wave-size pipeline failed: {error:?}"),
        }
    }
    let Some((pipeline, wave_size)) = selected else {
        return;
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
        (16_777_217u32 + wave_size).to_le_bytes()
    );
}
