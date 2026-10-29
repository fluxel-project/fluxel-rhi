//! Real Vulkan subgroup builtin admission and execution.
#![cfg(all(windows, feature = "hardware-tests"))]

#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::{Vulkan, VulkanInstance};
use std::sync::Arc;
use support::block;

fn submit(
    device: &Device<Vulkan>,
    lane: LaneId,
    work: RecordedWork<Vulkan>,
) -> CompletionPoint<Vulkan> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 83_291,
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
            panic!("Vulkan subgroup submission rejected: {error:?}")
        }
        SubmitOutcome::PartiallyAccepted { error, .. } => {
            panic!("Vulkan subgroup submission partially accepted: {error:?}")
        }
        SubmitOutcome::Indeterminate { error, .. } => {
            panic!("Vulkan subgroup submission indeterminate: {error:?}")
        }
    }
}

#[test]
fn wgsl_subgroup_size_builtin_reaches_gpu() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
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
    // Naga 27 parses the builtins without `enable subgroups;`; that directive
    // itself remains unavailable in this pinned frontend.
    let shader = match block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(
            b"@group(0) @binding(0) var<storage,read_write> output: array<u32>; @compute @workgroup_size(1) fn main(@builtin(subgroup_size) size: u32, @builtin(subgroup_invocation_id) lane: u32) { if lane == 0u { output[0] = size; } }"
                .as_slice(),
        ),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec!["subgroups".into()],
    })) {
        Ok(shader) => shader,
        Err(Error::Unsupported(_)) => return,
        Err(error) => panic!("Vulkan subgroup shader creation failed: {error:?}"),
    };
    let pipeline = block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout: pipeline_layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                required_subgroup_size: None,
                cache: None,
            }),
    )
    .unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let shader_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
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
                to: shader_write,
            },
            ownership: None,
        })])
        .unwrap();
    let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
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
                from: shader_write,
                to: copy_read,
            },
            ownership: None,
        })])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 4).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..4).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert!(u32::from_le_bytes(bytes[..4].try_into().unwrap()) > 0);
}

#[test]
fn native_spirv_f16_i8_specialization_and_storage_reach_gpu() {
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device.lanes()[0].id;
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 12,
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
            size: 12,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 12,
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
                    range: 0..12,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let shader = match block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(include_bytes!("assets/precision/precision.spv").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    })) {
        Ok(shader) => shader,
        Err(Error::Unsupported(_)) => return,
        Err(error) => panic!("Vulkan precision shader creation failed: {error:?}"),
    };
    let pipeline = block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout: pipeline_layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![
                        ("half_value".into(), SpecializationValue::F16Bits(0x4200)),
                        ("byte_value".into(), SpecializationValue::I8(4)),
                    ],
                },
                required_subgroup_size: None,
                cache: None,
            }),
    )
    .unwrap();
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
    let barrier = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..12,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    encoder
        .transition_resources(&[barrier(empty, write)])
        .unwrap();
    let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[barrier(write, copy)])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 12).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..12).unwrap();
    let point = submit(&device, lane, encoder.finish().unwrap());
    block(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..4], &7u32.to_le_bytes());
    assert_eq!(&bytes[4..6], &0x4200u16.to_le_bytes());
    assert_eq!(&bytes[6..8], &4i16.to_le_bytes());
    assert_eq!(bytes[8], 4);
}
