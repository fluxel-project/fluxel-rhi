//! Direct typed-RHI coverage for divergent DX12 storage-buffer array indexing.
//!
//! The two tests deliberately exercise the read-only SRV and read-write UAV
//! descriptor classes separately.  Their readbacks distinguish descriptor
//! selection from an accidental in-buffer offset calculation.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
const VALUE_BYTES: u64 = 32;
const OUTPUT_BYTES: u64 = 8;
const COMPUTE_READ: BufferUses = BufferUses {
    stages: StageMask::COMPUTE,
    accesses: AccessMask::SHADER_READ,
};
const COMPUTE_READ_WRITE: BufferUses = BufferUses {
    stages: StageMask::COMPUTE,
    accesses: AccessMask::SHADER_READ.union(AccessMask::SHADER_WRITE),
};
const COMPUTE_WRITE: BufferUses = BufferUses {
    stages: StageMask::COMPUTE,
    accesses: AccessMask::SHADER_WRITE,
};
const COPY_READ: BufferUses = BufferUses {
    stages: StageMask::COPY,
    accesses: AccessMask::COPY_READ,
};
type Dx12Device = Device<fluxel_rhi_dx12::Dx12>;
type Dx12Buffer = Buffer<fluxel_rhi_dx12::Dx12>;
type Dx12Shader = Shader<fluxel_rhi_dx12::Dx12>;
type Dx12Work = RecordedWork<fluxel_rhi_dx12::Dx12>;
type Dx12Completion = CompletionPoint<fluxel_rhi_dx12::Dx12>;
#[path = "support/mod.rs"]
mod support;
use support::block as block_on;
fn graphics_lane(device: &Dx12Device) -> LaneId {
    device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id
}
fn hardware_device() -> (
    fluxel_rhi_core::api::Instance<fluxel_rhi_dx12::Dx12>,
    AdapterInfo,
    Dx12Device,
) {
    let instance = Dx12Instance::new().unwrap().instance().unwrap();
    let adapters = instance.enumerate().unwrap();
    eprintln!(
        "DXGI adapters: {:?}",
        adapters
            .iter()
            .map(|adapter| adapter.name.as_str())
            .collect::<Vec<_>>()
    );
    let adapter = adapters
        .into_iter()
        .next()
        .expect("DXGI hardware adapter list is empty");
    eprintln!("using DXGI adapter: {}", adapter.name);
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: vec![],
    }))
    .unwrap();
    (instance, adapter, device)
}
fn storage_layout(
    device: &Dx12Device,
    access: StorageAccess,
) -> BindGroupLayout<fluxel_rhi_dx12::Dx12> {
    device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("nonuniform storage-buffer array layout".into()),
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: VALUE_BYTES,
                        access,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: false,
                    nonuniform_indexing: true,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: OUTPUT_BYTES,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap()
}
fn device_buffer(device: &Dx12Device, label: &str, size: u64, usage: BufferUsage) -> Dx12Buffer {
    device
        .create_buffer(&BufferDesc {
            label: Some(label.into()),
            size,
            usage,
            memory: MemoryClass::Device,
        })
        .unwrap()
}
fn buffer_transition(
    buffer: &Dx12Buffer,
    size: u64,
    from: BufferUses,
    to: BufferUses,
) -> ResourceTransition<'_, fluxel_rhi_dx12::Dx12> {
    ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer,
            range: 0..size,
        },
        usage: StateTransition { from, to },
        ownership: None,
    })
}
fn assert_pre_native_rejections(
    instance: &Instance<fluxel_rhi_dx12::Dx12>,
    adapter: &AdapterInfo,
    device: &Dx12Device,
    layout: &BindGroupLayout<fluxel_rhi_dx12::Dx12>,
    local: &Dx12Buffer,
    output: &Dx12Buffer,
) {
    let local_view = || BindingResource::StorageBufferView {
        buffer: local,
        range: 0..VALUE_BYTES,
        view: StorageBufferView::Raw,
    };
    let output_view = || BindingResource::StorageBufferView {
        buffer: output,
        range: 0..OUTPUT_BYTES,
        view: StorageBufferView::Raw,
    };
    let missing = device.create_bind_group(&BindGroupDesc {
        label: Some("missing nonuniform array element".into()),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![Some(local_view()), None]),
            },
            BindGroupEntry {
                binding: 2,
                resource: output_view(),
            },
        ],
    });
    assert!(matches!(missing, Err(Error::InvalidState)));
    let foreign = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: vec![],
    }))
    .unwrap();
    let foreign_input = device_buffer(
        &foreign,
        "foreign nonuniform storage input",
        VALUE_BYTES,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
    );
    let foreign_group = device.create_bind_group(&BindGroupDesc {
        label: Some("foreign nonuniform array element".into()),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![
                    Some(local_view()),
                    Some(BindingResource::StorageBufferView {
                        buffer: &foreign_input,
                        range: 0..VALUE_BYTES,
                        view: StorageBufferView::Raw,
                    }),
                ]),
            },
            BindGroupEntry {
                binding: 2,
                resource: output_view(),
            },
        ],
    });
    assert!(matches!(foreign_group, Err(Error::WrongDevice)));
}
fn storage_group<'a>(
    device: &Dx12Device,
    layout: &'a BindGroupLayout<fluxel_rhi_dx12::Dx12>,
    inputs: [&'a Dx12Buffer; 2],
    output: &'a Dx12Buffer,
    label: &str,
) -> BindGroup<fluxel_rhi_dx12::Dx12> {
    device
        .create_bind_group(&BindGroupDesc {
            label: Some(label.into()),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(
                        inputs
                            .into_iter()
                            .map(|buffer| {
                                Some(BindingResource::StorageBufferView {
                                    buffer,
                                    range: 0..VALUE_BYTES,
                                    view: StorageBufferView::Raw,
                                })
                            })
                            .collect(),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::StorageBufferView {
                        buffer: output,
                        range: 0..OUTPUT_BYTES,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })
        .unwrap()
}
fn shader(device: &Dx12Device, label: &str, source: &'static [u8]) -> Dx12Shader {
    block_on(device.shader(ShaderDesc {
        label: Some(label.into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([2, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap()
}
fn submit(device: &Dx12Device, lane: LaneId, ordinal: u64, work: Dx12Work) -> Dx12Completion {
    // SAFETY: this plan has one command step for this device-owned lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: `work` was recorded by this device; encoder retention owns all
    // native dependencies after caller handles are dropped below.
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
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("nonuniform storage-buffer submission rejected"),
    }
}
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
#[test]
fn divergent_storage_buffer_array_reads_distinct_values() {
    let (instance, adapter, device) = hardware_device();
    assert!(device.capabilities().nonuniform_storage_buffer_indexing);
    let lane = graphics_lane(&device);
    let inputs = [
        device_buffer(
            &device,
            "nonuniform storage read input 0",
            VALUE_BYTES,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        ),
        device_buffer(
            &device,
            "nonuniform storage read input 1",
            VALUE_BYTES,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        ),
    ];
    let output = device_buffer(
        &device,
        "nonuniform storage read output",
        OUTPUT_BYTES,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    );
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("nonuniform storage readback".into()),
            size: OUTPUT_BYTES,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = storage_layout(&device, StorageAccess::Read);
    assert_pre_native_rejections(&instance, &adapter, &device, &layout, &inputs[0], &output);
    let group = storage_group(
        &device,
        &layout,
        [&inputs[0], &inputs[1]],
        &output,
        "nonuniform storage read group",
    );
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("nonuniform storage read pipeline layout".into()),
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let module = shader(
        &device,
        "nonuniform storage read",
        include_bytes!("assets/storage_buffer_nonuniform_read.wgsl"),
    );
    let pipeline = block_on(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: Some("nonuniform storage read pipeline".into()),
                layout: pipeline_layout,
                compute: PipelineStage {
                    module,
                    entry: "main".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    for (input, value) in inputs.iter().zip([11u32, 22]) {
        encoder
            .write_buffer(input, 16, &value.to_le_bytes())
            .unwrap();
    }
    encoder
        .transition_resources(&[
            buffer_transition(
                &inputs[0],
                VALUE_BYTES,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                COMPUTE_READ,
            ),
            buffer_transition(
                &inputs[1],
                VALUE_BYTES,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                COMPUTE_READ,
            ),
            buffer_transition(
                &output,
                OUTPUT_BYTES,
                BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                COMPUTE_WRITE,
            ),
        ])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[buffer_transition(
            &output,
            OUTPUT_BYTES,
            COMPUTE_WRITE,
            COPY_READ,
        )])
        .unwrap();
    encoder
        .copy_buffer(&output, 0, &readback, 0, OUTPUT_BYTES)
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..OUTPUT_BYTES).unwrap();
    let work = encoder.finish().unwrap();
    drop((group, pipeline, inputs, output));
    let point = submit(&device, lane, 0x5353_424f_0001, work);
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: `point` covers the ticket recorded above.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!([read_u32(&bytes, 0), read_u32(&bytes, 4)], [11, 22]);
}
#[test]
fn divergent_storage_buffer_array_writes_selected_elements() {
    let (instance, adapter, device) = hardware_device();
    assert!(device.capabilities().nonuniform_storage_buffer_indexing);
    let lane = graphics_lane(&device);
    let inputs = [
        device_buffer(
            &device,
            "nonuniform storage write input 0",
            VALUE_BYTES,
            BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        ),
        device_buffer(
            &device,
            "nonuniform storage write input 1",
            VALUE_BYTES,
            BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        ),
    ];
    let output = device_buffer(
        &device,
        "nonuniform storage write output",
        OUTPUT_BYTES,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    );
    let input_readbacks: Vec<_> = (0..2)
        .map(|index| {
            device
                .create_buffer(&BufferDesc {
                    label: Some(format!("nonuniform storage write readback {index}")),
                    size: VALUE_BYTES,
                    usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                    memory: MemoryClass::Readback,
                })
                .unwrap()
        })
        .collect();
    let output_readback = device
        .create_buffer(&BufferDesc {
            label: Some("nonuniform storage write output readback".into()),
            size: OUTPUT_BYTES,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = storage_layout(&device, StorageAccess::ReadWrite);
    assert_pre_native_rejections(&instance, &adapter, &device, &layout, &inputs[0], &output);
    let group = storage_group(
        &device,
        &layout,
        [&inputs[0], &inputs[1]],
        &output,
        "nonuniform storage write group",
    );
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("nonuniform storage write pipeline layout".into()),
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let module = shader(
        &device,
        "nonuniform storage write",
        include_bytes!("assets/storage_buffer_nonuniform_write.wgsl"),
    );
    let pipeline = block_on(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: Some("nonuniform storage write pipeline".into()),
                layout: pipeline_layout,
                compute: PipelineStage {
                    module,
                    entry: "main".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    for (input, value) in inputs.iter().zip([11u32, 22]) {
        let mut initial = [0xa5u8; VALUE_BYTES as usize];
        initial[16..20].copy_from_slice(&value.to_le_bytes());
        initial[20..24].copy_from_slice(&0xdddd_dddd_u32.to_le_bytes());
        initial[24..28].copy_from_slice(&0u32.to_le_bytes());
        initial[28..32].copy_from_slice(&0x5a5a_5a5a_u32.to_le_bytes());
        encoder.write_buffer(input, 0, &initial).unwrap();
    }
    encoder
        .transition_resources(&[
            buffer_transition(
                &inputs[0],
                VALUE_BYTES,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                COMPUTE_READ_WRITE,
            ),
            buffer_transition(
                &inputs[1],
                VALUE_BYTES,
                BufferUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                COMPUTE_READ_WRITE,
            ),
            buffer_transition(
                &output,
                OUTPUT_BYTES,
                BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                COMPUTE_WRITE,
            ),
        ])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    let mut copies = vec![buffer_transition(
        &output,
        OUTPUT_BYTES,
        COMPUTE_WRITE,
        COPY_READ,
    )];
    copies.extend(
        inputs
            .iter()
            .map(|input| buffer_transition(input, VALUE_BYTES, COMPUTE_READ_WRITE, COPY_READ)),
    );
    encoder.transition_resources(&copies).unwrap();
    encoder
        .copy_buffer(&output, 0, &output_readback, 0, OUTPUT_BYTES)
        .unwrap();
    for (input, readback) in inputs.iter().zip(&input_readbacks) {
        encoder
            .copy_buffer(input, 0, readback, 0, VALUE_BYTES)
            .unwrap();
    }
    let output_ticket = encoder
        .readback_buffer(&output_readback, 0..OUTPUT_BYTES)
        .unwrap();
    let input_tickets: Vec<_> = input_readbacks
        .iter()
        .map(|readback| encoder.readback_buffer(readback, 0..VALUE_BYTES).unwrap())
        .collect();
    let work = encoder.finish().unwrap();
    drop((group, pipeline, inputs, output));
    let point = submit(&device, lane, 0x5353_424f_0002, work);
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: this completion point covers all three tickets from the one encoder.
    let output_bytes = unsafe {
        block_on(device.read_after(output_ticket, &point))
            .unwrap()
            .bytes
    };
    assert_eq!(
        [read_u32(&output_bytes, 0), read_u32(&output_bytes, 4)],
        [11, 22]
    );
    for (index, ticket) in input_tickets.into_iter().enumerate() {
        // SAFETY: this ticket was recorded before the completed submission.
        let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
        assert_eq!(&bytes[..16], &[0xa5; 16]);
        assert_eq!(read_u32(&bytes, 16), [11, 22][index]);
        assert_eq!(read_u32(&bytes, 20), [111, 122][index]);
        assert_eq!(read_u32(&bytes, 24), 1);
        assert_eq!(read_u32(&bytes, 28), 0x5a5a_5a5a);
    }
}
