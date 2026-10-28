//! Real-device proof for the Vulkan R64Uint image Int64 atomic-add route.
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

fn mutate_spirv_operand(bytes: &[u8], opcode: u32, operand: usize, from: u32, to: u32) -> Vec<u8> {
    let mut words = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut offset = 5;
    let mut changed = false;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(count > 0 && offset + count <= words.len());
        if words[offset] & 0xffff == opcode && operand < count && words[offset + operand] == from {
            words[offset + operand] = to;
            changed = true;
            break;
        }
        offset += count;
    }
    assert!(changed, "selected SPIR-V operand was not found");
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn run_r64uint_image_atomic_route(
    feature: &'static str,
    shader_bytes: &[u8],
    initial_value: u64,
    expected_value: u64,
) {
    let native = VulkanInstance::new().unwrap();
    let instance = native.instance().unwrap();
    let adapter = instance
        .enumerate()
        .unwrap()
        .into_iter()
        .next()
        .expect("a graphics-capable Vulkan adapter");
    eprintln!("Vulkan R64Uint image atomic adapter: {}", adapter.name);
    let device = block_on(instance.open(DeviceRequest {
        adapter: adapter.id,
        required_features: vec![],
    }))
    .unwrap();
    assert!(match feature {
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD => {
            device.capabilities().shader_int64_atomic_image_r64uint_add
        }
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN => {
            device.capabilities().shader_int64_atomic_image_r64uint_umin
        }
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX => {
            device.capabilities().shader_int64_atomic_image_r64uint_umax
        }
        _ => false,
    });
    let make_shader = |required_features: Vec<String>| ShaderDesc {
        label: Some(format!("R64Uint image Int64 atomic: {feature}")),
        format: ShaderFormat::SpirV,
        bytes: Arc::from(shader_bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([37, 1, 1]),
        }],
        required_features,
    };
    assert!(matches!(
        block_on(device.shader(make_shader(vec![]))),
        Err(Error::Shader(_))
    ));
    assert!(matches!(
        block_on(device.shader(make_shader(vec![
            SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD.into()
        ]))),
        Err(Error::Shader(_)) | Err(Error::Unsupported(_))
    ));
    for malformed in [
        mutate_spirv_operand(shader_bytes, 25, 8, 40, 41), // R64i instead of R64ui.
        mutate_spirv_operand(shader_bytes, 25, 3, 1, 2),   // 3D instead of 2D.
        mutate_spirv_operand(shader_bytes, 32, 2, 11, 12), // StorageBuffer instead of Image.
    ] {
        let mut desc = make_shader(vec![feature.into()]);
        desc.bytes = Arc::from(malformed);
        assert!(matches!(
            block_on(device.shader(desc)),
            Err(Error::Shader(_))
        ));
    }
    let wrong_operation_bytes: &[u8] = match feature {
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD => {
            include_bytes!("assets/int64_atomic_image_r64uint_umin.comp.spv")
        }
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN => {
            include_bytes!("assets/int64_atomic_image_r64uint_add.comp.spv")
        }
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX => {
            include_bytes!("assets/int64_atomic_image_r64uint_add.comp.spv")
        }
        _ => unreachable!(),
    };
    let mut wrong_operation = make_shader(vec![feature.into()]);
    wrong_operation.bytes = Arc::from(wrong_operation_bytes);
    assert!(matches!(
        block_on(device.shader(wrong_operation)),
        Err(Error::Shader(_)) | Err(Error::Unsupported(_))
    ));
    let shader = block_on(device.shader(make_shader(vec![feature.into()]))).unwrap();

    let texture = device
        .create_texture(&TextureDesc {
            label: Some("R64Uint atomic image".into()),
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::R64Uint,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::R64Uint,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let binding_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("R64Uint atomic image layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::R64Uint,
                    dimension: TextureViewDimension::D2,
                    access: StorageAccess::Atomic,
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
            layout: &binding_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&binding_layout],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = block_on(
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
    .unwrap();

    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let upload = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Upload,
        })
        .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_buffer(&upload, 0, &initial_value.to_le_bytes())
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let copy_write = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let atomic = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
    };
    let copy_read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to, from_layout, to_layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition { from, to },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })
    };
    encoder
        .transition_resources(&[barrier(
            empty,
            copy_write,
            ImageLayout::Undefined,
            ImageLayout::CopyDst,
        )])
        .unwrap();
    encoder
        .buffer_to_texture(
            &upload,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
        )
        .unwrap();
    encoder
        .transition_resources(&[barrier(
            copy_write,
            atomic,
            ImageLayout::CopyDst,
            ImageLayout::General,
        )])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(
            atomic,
            atomic,
            ImageLayout::General,
            ImageLayout::General,
        )])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(
            atomic,
            copy_read,
            ImageLayout::General,
            ImageLayout::CopySrc,
        )])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..256).unwrap();
    let work = encoder.finish().unwrap();
    // SAFETY: this one-step plan submits work recorded on the selected lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 0x4936_3401,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the plan and recorded work belong to this device and lane.
    let result = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(result) => result.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("R64Uint image atomic submission rejected"),
    };
    block_on(device.wait_completion(&point)).unwrap();
    // SAFETY: this completion covers the image-to-buffer copy and readback ticket.
    let bytes = unsafe { block_on(device.read_after(ticket, &point)).unwrap().bytes };
    let actual = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    eprintln!("R64Uint image {feature} readback: expected={expected_value} actual={actual}");
    assert_eq!(actual, expected_value);
}

#[test]
fn r64uint_image_atomic_add_reads_back_competing_updates() {
    run_r64uint_image_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD,
        include_bytes!("assets/int64_atomic_image_r64uint_add.comp.spv"),
        0,
        74,
    );
}

#[test]
fn r64uint_image_atomic_unsigned_min_reads_back_competing_updates() {
    run_r64uint_image_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN,
        include_bytes!("assets/int64_atomic_image_r64uint_umin.comp.spv"),
        u64::MAX,
        0,
    );
}

#[test]
fn r64uint_image_atomic_unsigned_max_reads_back_competing_updates() {
    run_r64uint_image_atomic_route(
        SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX,
        include_bytes!("assets/int64_atomic_image_r64uint_umax.comp.spv"),
        0,
        u64::MAX,
    );
}
