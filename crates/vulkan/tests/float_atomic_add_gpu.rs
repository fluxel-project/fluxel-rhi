//! Real-device coverage for Direct SPIR-V Float32 storage-buffer atomic add.
#![cfg(all(windows, feature = "hardware-tests"))]

#[path = "support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block_on;

fn run_float32_atomic_route(
    feature: &str,
    shader_bytes: &[u8],
    init_source: &[u8],
    expected_value: f32,
    ordinal: u64,
    workgroup_size: [u32; 3],
    dispatch_size: [u32; 3],
) {
    let device = block_on(VulkanInstance::new().unwrap().open_default()).unwrap();
    let capability = match feature {
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD => {
            device.capabilities().shader_float32_atomic_buffer_add
        }
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN => {
            device.capabilities().shader_float32_atomic_buffer_min
        }
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX => {
            device.capabilities().shader_float32_atomic_buffer_max
        }
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD => {
            device.capabilities().shader_float32_atomic_workgroup_add
        }
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN => {
            device.capabilities().shader_float32_atomic_workgroup_min
        }
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX => {
            device.capabilities().shader_float32_atomic_workgroup_max
        }
        _ => false,
    };
    if !capability {
        eprintln!("SKIP: Vulkan Float32 atomic route unavailable: {feature}");
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let output = device
        .create_buffer(&BufferDesc {
            label: Some("Float32 atomic output".into()),
            size: 4,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("Float32 atomic readback".into()),
            size: 4,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("Float32 atomic buffer layout".into()),
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
            label: Some("Float32 atomic buffer group".into()),
            layout: &group_layout,
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
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("Float32 atomic pipeline layout".into()),
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let init_shader = block_on(device.shader(ShaderDesc {
        label: Some("Float32 atomic WGSL initializer".into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(init_source),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let atomic_desc = ShaderDesc {
        label: Some("Float32 atomic direct SPIR-V".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(shader_bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some(workgroup_size),
        }],
        required_features: vec![feature.into()],
    };
    let undeclared_atomic = ShaderDesc {
        label: Some("undeclared Float32 atomic direct SPIR-V".into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(shader_bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some(workgroup_size),
        }],
        required_features: vec![],
    };
    assert!(matches!(
        block_on(device.shader(undeclared_atomic)),
        Err(Error::Shader(_))
    ));
    let (cross_route_bytes, cross_route_feature, cross_route_name, cross_route_wg) = match feature {
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD => (
            include_bytes!("assets/float_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD,
            "workgroup add under buffer route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN => (
            include_bytes!("assets/float_atomic_workgroup_min.comp.spv").as_slice(),
            SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN,
            "workgroup min under buffer route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX => (
            include_bytes!("assets/float_atomic_workgroup_max.comp.spv").as_slice(),
            SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX,
            "workgroup max under buffer route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD => (
            include_bytes!("assets/float_atomic_add.comp.spv").as_slice(),
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD,
            "buffer add under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN => (
            include_bytes!("assets/float_atomic_min.comp.spv").as_slice(),
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN,
            "buffer min under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX => (
            include_bytes!("assets/float_atomic_max.comp.spv").as_slice(),
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX,
            "buffer max under workgroup route",
            [1, 1, 1],
        ),
        _ => unreachable!(),
    };
    let cross_route_shader = ShaderDesc {
        label: Some(cross_route_name.into()),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(cross_route_bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some(cross_route_wg),
        }],
        required_features: vec![cross_route_feature.into()],
    };
    assert!(matches!(
        block_on(device.shader(cross_route_shader)),
        Err(Error::Shader(_))
    ));

    if matches!(
        feature,
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN
            | SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX
            | SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN
            | SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX
    ) {
        let (wrong_bytes, wrong_feature, wrong_name, wrong_wg) = match feature {
            SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN => (
                include_bytes!("assets/float_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN,
                "add under min",
                [1, 1, 1],
            ),
            SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX => (
                include_bytes!("assets/float_atomic_min.comp.spv").as_slice(),
                SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX,
                "min under max",
                [1, 1, 1],
            ),
            SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN => (
                include_bytes!("assets/float_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN,
                "workgroup add under min",
                [37, 1, 1],
            ),
            _ => (
                include_bytes!("assets/float_atomic_workgroup_min.comp.spv").as_slice(),
                SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX,
                "workgroup min under max",
                [37, 1, 1],
            ),
        };
        let wrong_operation = ShaderDesc {
            label: Some(format!("Float32 atomic {wrong_name}")),
            format: ShaderFormat::SpirV,
            bytes: Arc::from(wrong_bytes),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some(wrong_wg),
            }],
            required_features: vec![wrong_feature.into()],
        };
        assert!(matches!(
            block_on(device.shader(wrong_operation)),
            Err(Error::Unsupported(_))
        ));
    }
    assert!(matches!(
        device.shader_route(&atomic_desc),
        ShaderRoute::Direct(_)
    ));
    let atomic_shader = block_on(device.shader(atomic_desc)).unwrap();
    let make_pipeline = |shader| {
        block_on(
            device
                .compute()
                .unwrap()
                .create_pipeline(ComputePipelineDesc {
                    label: None,
                    layout: layout.clone(),
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
    };
    let initializer = make_pipeline(init_shader);
    let atomic = make_pipeline(atomic_shader);
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let read_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[barrier(empty, write)])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&initializer).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(write, read_write)])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&atomic).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch(dispatch_size).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(read_write, copy_read)])
        .unwrap();
    encoder.copy_buffer(&output, 0, &readback, 0, 4).unwrap();
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
        _ => panic!("Float32 atomic submission was not accepted"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: completion follows the copy and the ticket belongs to this readback buffer.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(
        f32::from_le_bytes(bytes[..4].try_into().unwrap()),
        expected_value
    );
}

#[test]
fn float32_storage_buffer_atomic_add_accumulates_and_reads_back() {
    run_float32_atomic_route(
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD,
        include_bytes!("assets/float_atomic_add.comp.spv"),
        b"@group(0) @binding(0) var<storage, read_write> value: f32; @compute @workgroup_size(1) fn main() { value = 0.0; }",
        37.0,
        0xf10a_3201,
        [1, 1, 1],
        [37, 1, 1],
    );
}

#[test]
fn float32_storage_buffer_atomic_min_accumulates_and_reads_back() {
    run_float32_atomic_route(
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN,
        include_bytes!("assets/float_atomic_min.comp.spv"),
        b"@group(0) @binding(0) var<storage, read_write> value: f32; @compute @workgroup_size(1) fn main() { value = 37.0; }",
        1.0,
        0xf10a_3202,
        [1, 1, 1],
        [37, 1, 1],
    );
}

#[test]
fn float32_storage_buffer_atomic_max_accumulates_and_reads_back() {
    run_float32_atomic_route(
        SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX,
        include_bytes!("assets/float_atomic_max.comp.spv"),
        b"@group(0) @binding(0) var<storage, read_write> value: f32; @compute @workgroup_size(1) fn main() { value = -37.0; }",
        1.0,
        0xf10a_3203,
        [1, 1, 1],
        [37, 1, 1],
    );
}

#[test]
fn float32_workgroup_atomic_add_accumulates_and_reads_back() {
    run_float32_atomic_route(
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD,
        include_bytes!("assets/float_atomic_workgroup_add.comp.spv"),
        b"@group(0) @binding(0) var<storage, read_write> value: f32; @compute @workgroup_size(1) fn main() { value = 0.0; }",
        37.0,
        0xf10a_3204,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn float32_workgroup_atomic_min_accumulates_and_reads_back() {
    run_float32_atomic_route(
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN,
        include_bytes!("assets/float_atomic_workgroup_min.comp.spv"),
        b"@group(0) @binding(0) var<storage, read_write> value: f32; @compute @workgroup_size(1) fn main() { value = 37.0; }",
        1.0,
        0xf10a_3205,
        [37, 1, 1],
        [1, 1, 1],
    );
}

#[test]
fn float32_workgroup_atomic_max_accumulates_and_reads_back() {
    run_float32_atomic_route(
        SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX,
        include_bytes!("assets/float_atomic_workgroup_max.comp.spv"),
        b"@group(0) @binding(0) var<storage, read_write> value: f32; @compute @workgroup_size(1) fn main() { value = -37.0; }",
        37.0,
        0xf10a_3206,
        [37, 1, 1],
        [1, 1, 1],
    );
}
