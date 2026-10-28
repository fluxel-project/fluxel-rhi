//! Real-device readback for one exact Vulkan cooperative-matrix tuple.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{future::Future, sync::Arc};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

#[test]
fn f16_f16_f32_subgroup_matrix_multiply_reads_back() {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let config = CooperativeMatrixConfiguration {
        m: 16,
        n: 16,
        k: 16,
        a_type: CooperativeMatrixComponentType::Float16,
        b_type: CooperativeMatrixComponentType::Float16,
        c_type: CooperativeMatrixComponentType::Float32,
        result_type: CooperativeMatrixComponentType::Float32,
        saturating_accumulation: false,
        scope: CooperativeMatrixScope::Subgroup,
    };
    if !device
        .capabilities()
        .cooperative_matrix_configurations
        .contains(&config)
    {
        eprintln!("SKIP: Vulkan F16/F16/F32 subgroup cooperative matrix tuple unavailable");
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let a = device
        .create_buffer(&BufferDesc {
            label: Some("CoopMat A".into()),
            size: 512,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let b = device
        .create_buffer(&BufferDesc {
            label: Some("CoopMat B".into()),
            size: 512,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let out = device
        .create_buffer(&BufferDesc {
            label: Some("CoopMat output".into()),
            size: 1024,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("CoopMat readback".into()),
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut a_data = vec![0u8; 512];
    let mut b_data = vec![0u8; 512];
    let f16_integer = |value: u32| -> u16 {
        match value {
            0 => 0x0000,
            1 => 0x3c00,
            2 => 0x4000,
            3 => 0x4200,
            4 => 0x4400,
            5 => 0x4500,
            6 => 0x4600,
            _ => unreachable!("fixture values are exact small integers"),
        }
    };
    let mut a_values = [[0.0f32; 16]; 16];
    let mut b_values = [[0.0f32; 16]; 16];
    for row in 0..16 {
        for column in 0..16 {
            let a = ((row * 2 + column) % 5) as u32;
            let b = ((row * 3 + column * 2) % 7) as u32;
            a_values[row][column] = a as f32;
            b_values[row][column] = b as f32;
            let a_offset = (row * 16 + column) * 2;
            let b_offset = (row * 16 + column) * 2;
            a_data[a_offset..a_offset + 2].copy_from_slice(&f16_integer(a).to_le_bytes());
            b_data[b_offset..b_offset + 2].copy_from_slice(&f16_integer(b).to_le_bytes());
        }
    }
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("CoopMat layout".into()),
            entries: (0..3)
                .map(|binding| BindGroupLayoutEntry {
                    binding,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: if binding == 2 { 1024 } else { 512 },
                        access: if binding == 2 {
                            StorageAccess::Write
                        } else {
                            StorageAccess::Read
                        },
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                })
                .collect(),
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: Some("CoopMat buffers".into()),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &a,
                        range: 0..512,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &b,
                        range: 0..512,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::StorageBufferView {
                        buffer: &out,
                        range: 0..1024,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })
        .unwrap();
    let pipeline_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader_desc = ShaderDesc {
        label: Some("CoopMat direct SPIR-V".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(include_bytes!("assets/cooperative_matrix_f16f16f32.comp.spv").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([64, 1, 1]),
        }],
        required_features: vec![SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX.into()],
    };
    let mut undeclared = shader_desc.clone();
    undeclared.required_features.clear();
    assert!(matches!(
        block_on(device.shader(undeclared)),
        Err(Error::Shader(_))
    ));
    assert!(matches!(
        device.shader_route(&shader_desc),
        ShaderRoute::Direct(_)
    ));
    let shader = block_on(device.shader(shader_desc)).unwrap();
    let pipeline = block_on(
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
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    encoder.write_buffer(&a, 0, &a_data).unwrap();
    encoder.write_buffer(&b, 0, &b_data).unwrap();
    let transition = |buffer, range, from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice { buffer, range },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let copy_write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let compute_read = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ,
    };
    let compute_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    encoder
        .transition_resources(&[
            transition(&a, 0..512, copy_write, compute_read),
            transition(&b, 0..512, copy_write, compute_read),
            transition(
                &out,
                0..1024,
                BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                compute_write,
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
        .transition_resources(&[transition(&out, 0..1024, compute_write, copy_read)])
        .unwrap();
    encoder.copy_buffer(&out, 0, &readback, 0, 1024).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..1024).unwrap();
    let commands = encoder.finish().unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 0xc00f_0001,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
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
        _ => panic!("cooperative matrix submission rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    for row in 0..16 {
        for column in 0..16 {
            let expected = (0..16)
                .map(|k| a_values[row][k] * b_values[k][column])
                .sum::<f32>();
            let offset = (row * 16 + column) * 4;
            let actual = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert_eq!(
                actual, expected,
                "cooperative matrix output ({row}, {column})"
            );
        }
    }
}
