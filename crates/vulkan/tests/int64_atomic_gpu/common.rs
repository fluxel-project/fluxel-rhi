//! Shared Int64 atomic feature routing, submission, and readback checks.
#![cfg(all(windows, feature = "hardware-tests"))]

#[path = "../support/mod.rs"]
mod support;
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::sync::Arc;
use support::block_on;

pub(super) fn run_int64_atomic_route(
    feature: &str,
    shader_bytes: &[u8],
    initial_value: i64,
    expected_value: i64,
    ordinal: u64,
    workgroup_size: [u32; 3],
    dispatch_size: [u32; 3],
) {
    let native_instance = VulkanInstance::new().unwrap();
    let instance = native_instance.instance().unwrap();
    let adapter = instance
        .enumerate()
        .unwrap()
        .into_iter()
        .next()
        .expect("a graphics-capable Vulkan adapter");
    eprintln!("Vulkan Int64 atomic fixture adapter: {}", adapter.name);
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    }))
    .unwrap();
    if !match feature {
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD => {
            device.capabilities().shader_int64_atomic_buffer_add
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN => {
            device.capabilities().shader_int64_atomic_buffer_min
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX => {
            device.capabilities().shader_int64_atomic_buffer_max
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN => {
            device.capabilities().shader_int64_atomic_buffer_umin
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX => {
            device.capabilities().shader_int64_atomic_buffer_umax
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR => {
            device.capabilities().shader_int64_atomic_buffer_xor
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE => {
            device
                .capabilities()
                .shader_int64_atomic_buffer_compare_exchange
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE => {
            device.capabilities().shader_int64_atomic_buffer_exchange
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND => {
            device.capabilities().shader_int64_atomic_buffer_and
        }
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR => {
            device.capabilities().shader_int64_atomic_buffer_or
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD => {
            device.capabilities().shader_int64_atomic_workgroup_add
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN => {
            device.capabilities().shader_int64_atomic_workgroup_min
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX => {
            device.capabilities().shader_int64_atomic_workgroup_max
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR => {
            device.capabilities().shader_int64_atomic_workgroup_xor
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND => {
            device.capabilities().shader_int64_atomic_workgroup_and
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR => {
            device.capabilities().shader_int64_atomic_workgroup_or
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN => {
            device.capabilities().shader_int64_atomic_workgroup_umin
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX => {
            device.capabilities().shader_int64_atomic_workgroup_umax
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE => {
            device.capabilities().shader_int64_atomic_workgroup_exchange
        }
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE => {
            device
                .capabilities()
                .shader_int64_atomic_workgroup_compare_exchange
        }
        _ => false,
    } {
        eprintln!("SKIP: Vulkan Int64 atomic route unavailable: {feature}");
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let output_size = if matches!(
        feature,
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
    ) {
        8 + 37 * 8
    } else {
        8
    };
    let output = device
        .create_buffer(&BufferDesc {
            label: Some("Int64 atomic output".into()),
            size: output_size,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("Int64 atomic readback".into()),
            size: output_size,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("Int64 atomic buffer layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: output_size,
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
            label: Some("Int64 atomic buffer group".into()),
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageBufferView {
                    buffer: &output,
                    range: 0..output_size,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("Int64 atomic pipeline layout".into()),
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let atomic_desc = ShaderDesc {
        label: Some("Int64 atomic direct SPIR-V".into()),
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
        label: Some("undeclared Int64 atomic direct SPIR-V".into()),
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
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD,
            "workgroup add under buffer route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN => (
            include_bytes!("../assets/int64_atomic_workgroup_min.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN,
            "workgroup min under buffer route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX => (
            include_bytes!("../assets/int64_atomic_workgroup_max.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX,
            "workgroup max under buffer route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN,
            "workgroup add under buffer unsigned min route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX,
            "workgroup add under buffer unsigned max route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR,
            "workgroup add under buffer xor route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE,
            "workgroup add under buffer compare-exchange route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND,
            "workgroup add under buffer and route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR,
            "workgroup add under buffer or route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE => (
            include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE,
            "workgroup add under buffer exchange route",
            [37, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD => (
            include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD,
            "buffer add under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN => (
            include_bytes!("../assets/int64_atomic_min.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN,
            "buffer min under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX => (
            include_bytes!("../assets/int64_atomic_max.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX,
            "buffer max under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR => (
            include_bytes!("../assets/int64_atomic_xor.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR,
            "buffer xor under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND => (
            include_bytes!("../assets/int64_atomic_and.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND,
            "buffer and under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR => (
            include_bytes!("../assets/int64_atomic_or.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR,
            "buffer or under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN => (
            include_bytes!("../assets/int64_atomic_umin.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN,
            "buffer unsigned min under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX => (
            include_bytes!("../assets/int64_atomic_umax.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX,
            "buffer unsigned max under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE => (
            include_bytes!("../assets/int64_atomic_exchange.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE,
            "buffer exchange under workgroup route",
            [1, 1, 1],
        ),
        SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE => (
            include_bytes!("../assets/int64_atomic_compare_exchange.comp.spv").as_slice(),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE,
            "buffer compare-exchange under workgroup route",
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
        Err(Error::Shader(_)) | Err(Error::Unsupported(_))
    ));
    if matches!(
        feature,
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND
            | SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
    ) {
        let (wrong_bytes, wrong_feature, wrong_name, wrong_wg) = match feature {
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN,
                "add under min",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX => (
                include_bytes!("../assets/int64_atomic_min.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX,
                "min under max",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN,
                "signed add under unsigned min",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX,
                "signed add under unsigned max",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR,
                "add under xor",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE,
                "add under compare-exchange",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE,
                "add under exchange route",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND,
                "add under and route",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR => (
                include_bytes!("../assets/int64_atomic_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR,
                "add under or route",
                [1, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN,
                "workgroup add under min",
                [37, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR,
                "workgroup add under workgroup xor",
                [37, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND,
                "workgroup add under workgroup and",
                [37, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR,
                "workgroup add under workgroup or",
                [37, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN,
                "signed workgroup add under unsigned min",
                [37, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX,
                "signed workgroup add under unsigned max",
                [37, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE,
                "workgroup add under workgroup exchange",
                [37, 1, 1],
            ),
            SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE => (
                include_bytes!("../assets/int64_atomic_workgroup_add.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE,
                "workgroup add under workgroup compare-exchange",
                [37, 1, 1],
            ),
            _ => (
                include_bytes!("../assets/int64_atomic_workgroup_min.comp.spv").as_slice(),
                SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX,
                "workgroup min under max",
                [37, 1, 1],
            ),
        };
        let unsupported_operation = ShaderDesc {
            label: Some(format!(
                "unsupported Int64 atomic {wrong_name} direct SPIR-V"
            )),
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
            block_on(device.shader(unsupported_operation)),
            Err(Error::Unsupported(_))
        ));
    }
    assert!(matches!(
        device.shader_route(&atomic_desc),
        ShaderRoute::Direct(_)
    ));
    let shader = block_on(device.shader(atomic_desc)).unwrap();
    let pipeline = block_on(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: Some("Int64 atomic pipeline".into()),
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
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_buffer(&output, 0, &initial_value.to_ne_bytes())
        .unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let atomic = BufferUses {
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
                range: 0..output_size,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    encoder
        .transition_resources(&[barrier(empty, atomic)])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch(dispatch_size).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(atomic, copy_read)])
        .unwrap();
    encoder
        .copy_buffer(&output, 0, &readback, 0, output_size)
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..output_size).unwrap();
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
        _ => panic!("Int64 atomic submission was not accepted"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: completion follows the copy and the ticket belongs to this readback buffer.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    if matches!(
        feature,
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
    ) {
        let final_value = i64::from_ne_bytes(bytes[..8].try_into().unwrap());
        assert!((1..=37).contains(&final_value));
        let returned = bytes[8..]
            .chunks_exact(8)
            .map(|word| i64::from_ne_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(returned.len(), 37);
        assert_eq!(returned.iter().filter(|&&old| old == 0).count(), 1);
        assert!(returned.iter().all(|&old| old == 0 || old == final_value));
    } else if matches!(
        feature,
        SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE
            | SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
    ) {
        let final_value = i64::from_ne_bytes(bytes[..8].try_into().unwrap());
        assert!((1..=37).contains(&final_value));
        let mut returned = bytes[8..]
            .chunks_exact(8)
            .map(|word| i64::from_ne_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        returned.sort_unstable();
        let mut expected = (0..=37).collect::<Vec<_>>();
        expected.remove(final_value as usize);
        assert_eq!(returned, expected);
    } else {
        assert_eq!(
            i64::from_ne_bytes(bytes[..8].try_into().unwrap()),
            expected_value
        );
    }
}
