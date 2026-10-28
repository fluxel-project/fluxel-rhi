//! Direct typed-RHI coverage for fixed DX12 uniform-buffer descriptor arrays.
//!
//! The WGSL test is a reconstructed Stage 7 diagnostic, not a claim that it
//! reproduces the unavailable historical Direct-DXIL artifact or GPU result.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::{future::Future, sync::Arc};

type Dx12Device = Device<fluxel_rhi_dx12::Dx12>;

fn block<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

fn graphics_lane(device: &Dx12Device) -> LaneId {
    device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .expect("DX12 graphics lane is unavailable")
        .id
}

fn transition(
    buffer: &Buffer<fluxel_rhi_dx12::Dx12>,
    size: u64,
    from: BufferUses,
    to: BufferUses,
) -> ResourceTransition<'_, fluxel_rhi_dx12::Dx12> {
    ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice { buffer, range: 0..size },
        usage: StateTransition { from, to },
        ownership: None,
    })
}

fn check(device: Dx12Device, nonuniform: bool) {
    let lane = graphics_lane(&device);
    let output_size = if nonuniform { 8 } else { 4 };
    let expected = if nonuniform { [11, 22] } else { [42, 0] };
    let uniform = || {
        device
            .create_buffer(&BufferDesc {
                label: Some("uniform array value".into()),
                size: 256,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap()
    };
    let values = [uniform(), uniform()];
    let output = device
        .create_buffer(&BufferDesc {
            label: Some("uniform array output".into()),
            size: output_size,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("uniform array readback".into()),
            size: output_size,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("uniform array layout".into()),
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: if nonuniform { 32 } else { 4 },
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: false,
                    nonuniform_indexing: nonuniform,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: output_size,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    if nonuniform {
        let uniform_view = |buffer| BindingResource::Buffer { buffer, range: 0..256 };
        let output_view = || BindingResource::StorageBufferView {
            buffer: &output,
            range: 0..output_size,
            view: StorageBufferView::Raw,
        };
        let missing = device.create_bind_group(&BindGroupDesc {
            label: Some("missing nonuniform uniform array element".into()),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(vec![Some(uniform_view(&values[0])), None]),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_view(),
                },
            ],
        });
        assert!(matches!(missing, Err(Error::InvalidState)));

        let foreign = block(Dx12Instance::new().unwrap().open_default()).unwrap();
        let foreign_value = foreign
            .create_buffer(&BufferDesc {
                label: Some("foreign nonuniform uniform value".into()),
                size: 256,
                usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let wrong_device = device.create_bind_group(&BindGroupDesc {
            label: Some("foreign nonuniform uniform array element".into()),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(vec![
                        Some(uniform_view(&values[0])),
                        Some(uniform_view(&foreign_value)),
                    ]),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_view(),
                },
            ],
        });
        assert!(matches!(wrong_device, Err(Error::WrongDevice)));
    }
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: Some("uniform array group".into()),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Array(
                        values
                            .iter()
                            .map(|buffer| Some(BindingResource::Buffer { buffer, range: 0..256 }))
                            .collect(),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..output_size,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })
        .unwrap();
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("uniform array pipeline layout".into()),
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(device.shader(ShaderDesc {
        label: Some("uniform array shader".into()),
        format: if nonuniform {
            ShaderFormat::Wgsl
        } else {
            ShaderFormat::Dxil
        },
        bytes: if nonuniform {
            Arc::from(include_bytes!("assets/uniform_buffer_nonuniform.wgsl").as_slice())
        } else {
            Arc::from(include_bytes!("assets/descriptor_uniform.dxil").as_slice())
        },
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: nonuniform.then_some([2, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let pipeline = block(device.compute().unwrap().create_pipeline(ComputePipelineDesc {
        label: Some("uniform array pipeline".into()),
        layout: pipeline_layout,
        compute: PipelineStage {
            module: shader,
            entry: "main".into(),
            constants: vec![],
        },
        cache: None,
        required_subgroup_size: None,
    }))
    .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    if nonuniform {
        for (value, number) in values.iter().zip([11u32, 22]) {
            let mut bytes = [0xa5u8; 32];
            bytes[16..20].copy_from_slice(&number.to_le_bytes());
            encoder.write_buffer(value, 0, &bytes).unwrap();
        }
    } else {
        encoder.write_buffer(&values[0], 0, &12u32.to_le_bytes()).unwrap();
        encoder.write_buffer(&values[1], 0, &30u32.to_le_bytes()).unwrap();
    }
    let copy_write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let uniform_read = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::UNIFORM_READ,
    };
    let shader_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    encoder
        .transition_resources(&[
            transition(&values[0], 256, copy_write, uniform_read),
            transition(&values[1], 256, copy_write, uniform_read),
            transition(
                &output,
                output_size,
                BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                shader_write,
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
        .transition_resources(&[transition(&output, output_size, shader_write, copy_read)])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, output_size).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..output_size).unwrap();
    let work = encoder.finish().unwrap();
    drop((group, pipeline, values, output));
    // SAFETY: this one-step plan uses the device-owned graphics lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: if nonuniform { 0x5542_4f00_0002 } else { 0x5542_4f00_0001 },
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the recorded work belongs to this device and retains native inputs.
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("uniform array submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    // SAFETY: the completion point covers this encoder's readback ticket.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), expected[0]);
    if nonuniform {
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), expected[1]);
    }
}

#[test]
fn uniform_array_is_consumed_by_static_dxil_shader() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    if device.capabilities().binding_arrays {
        check(device, false);
    }
}

#[test]
fn reconstructed_stage7_nonuniform_uniform_array_reads_distinct_values() {
    let instance = Dx12Instance::new().unwrap().instance().unwrap();
    let adapters = instance.enumerate().unwrap();
    eprintln!(
        "DXGI adapters: {:?}",
        adapters.iter().map(|adapter| adapter.name.as_str()).collect::<Vec<_>>()
    );
    let adapter = adapters
        .into_iter()
        .next()
        .expect("DXGI hardware adapter list is empty");
    eprintln!("using DXGI adapter: {}", adapter.name);
    let device = block(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: vec![],
    }))
    .unwrap();
    assert!(device.capabilities().nonuniform_uniform_buffer_indexing);
    check(device, true);
}
