//! Real-device readback for direct-SPIR-V GLSL450 StorageBuffer memory decorations.
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

fn run_memory_decoration_route(
    feature: &str,
    shader_bytes: &[u8],
    mismatched_feature: &str,
    ordinal: u64,
) {
    let native = VulkanInstance::new().unwrap();
    let instance = native.instance().unwrap();
    let adapter = instance
        .enumerate()
        .unwrap()
        .into_iter()
        .next()
        .expect("a graphics-capable Vulkan adapter");
    eprintln!("Vulkan memory-decoration adapter: {}", adapter.name);
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: vec![],
    }))
    .unwrap();
    let enabled = match feature {
        SHADER_FEATURE_MEMORY_COHERENT_SPIRV => device.capabilities().shader_memory_coherent_spirv,
        SHADER_FEATURE_MEMORY_VOLATILE_SPIRV => device.capabilities().shader_memory_volatile_spirv,
        _ => unreachable!(),
    };
    assert!(
        enabled,
        "Vulkan did not report the validated {feature} route"
    );
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let data = device
        .create_buffer(&BufferDesc {
            label: Some("memory decoration data".into()),
            size: 4,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("memory decoration readback".into()),
            size: 4,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("memory decoration layout".into()),
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
            label: Some("memory decoration group".into()),
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &data,
                    range: 0..4,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("memory decoration pipeline layout".into()),
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let shader_desc = ShaderDesc {
        label: Some(format!("direct SPIR-V {feature}")),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(shader_bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![feature.into()],
    };
    let mut undeclared = shader_desc.clone();
    undeclared.required_features.clear();
    assert!(matches!(
        block_on(device.shader(undeclared)),
        Err(Error::Shader(_)) | Err(Error::Unsupported(_))
    ));
    let mut mismatched = shader_desc.clone();
    mismatched.required_features = vec![mismatched_feature.into()];
    assert!(matches!(
        block_on(device.shader(mismatched)),
        Err(Error::Shader(_)) | Err(Error::Unsupported(_))
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
                label: Some("memory decoration compute pipeline".into()),
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
    .unwrap();
    let copy_write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let compute_read_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let transition = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &data,
                range: 0..4,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_buffer(&data, 0, &40u32.to_ne_bytes())
        .unwrap();
    encoder
        .transition_resources(&[transition(copy_write, compute_read_write)])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[transition(compute_read_write, copy_read)])
        .unwrap();
    encoder.copy_buffer(&data, 0, &readback, 0, 4).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..4).unwrap();
    let commands = encoder.finish().unwrap();
    // SAFETY: the one-step plan and recorded commands belong to this device and lane.
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
    // SAFETY: the plan references the matching device-owned command payload exactly once.
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
        _ => panic!("memory decoration submission was not accepted"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: completion follows the copy and the ticket belongs to this readback buffer.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    let actual = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
    eprintln!("Vulkan {feature} StorageBuffer readback: expected=41 actual={actual}");
    assert_eq!(actual, 41);
}

#[test]
fn coherent_and_volatile_spirv_storage_buffer_routes_execute() {
    run_memory_decoration_route(
        SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        include_bytes!("assets/memory_decorations/coherent.spv"),
        SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        0x4d45_4d01,
    );
    run_memory_decoration_route(
        SHADER_FEATURE_MEMORY_VOLATILE_SPIRV,
        include_bytes!("assets/memory_decorations/volatile_only.spv"),
        SHADER_FEATURE_MEMORY_COHERENT_SPIRV,
        0x4d45_4d02,
    );
}
