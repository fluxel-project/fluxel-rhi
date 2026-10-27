//! Real-device coverage for resource-scoped same-state texture UAV ordering.
//!
//! The `General` transitions cover six texture subresources, but D3D12 UAV
//! ordering is resource-wide.  The two dependent dispatches exercise a write
//! followed by a read/write access while the native state remains UAV.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
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

#[test]
fn same_state_texture_uav_dependency_covers_multi_mip_multi_layer_range() {
    let instance = Dx12Instance::new().unwrap();
    eprintln!(
        "DXGI adapters: {:?}",
        instance
            .instance()
            .unwrap()
            .enumerate()
            .unwrap()
            .iter()
            .map(|adapter| &adapter.name)
            .collect::<Vec<_>>()
    );
    let device = block(instance.open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 2, 1],
            array_layers: 3,
            format: TextureFormat::R32Uint,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::R32Uint,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 1..2,
                layers: 1..2,
            },
        )
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::R32Uint,
                    dimension: TextureViewDimension::D2Array,
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
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = |source: &'static [u8]| {
        let shader = block(device.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        }))
        .unwrap();
        block(
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
    let producer = pipeline(b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, read_write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(7u)); }");
    let consumer = pipeline(b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, read_write>; @compute @workgroup_size(1) fn main() { let v = textureLoad(values, vec2<i32>(0, 0), 0); textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(v.x * 6u)); }");
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
    let write = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let read_write = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ | AccessMask::SHADER_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to, from_layout, to_layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..2,
                layers: 0..3,
            },
            usage: StateTransition { from, to },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })
    };
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .transition_resources(&[barrier(
            empty,
            write,
            ImageLayout::Common,
            ImageLayout::General,
        )])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&producer).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    // This remains UAV/General and must emit one resource-level UAV barrier,
    // despite the six selected subresources.
    let before = fluxel_rhi_dx12::diagnostics::snapshot();
    encoder
        .transition_resources(&[barrier(
            write,
            read_write,
            ImageLayout::General,
            ImageLayout::General,
        )])
        .unwrap();
    assert_eq!(
        fluxel_rhi_dx12::diagnostics::snapshot().uav_barriers - before.uav_barriers,
        1
    );
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&consumer).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    let before = fluxel_rhi_dx12::diagnostics::snapshot();
    encoder
        .transition_resources(&[barrier(
            read_write,
            read,
            ImageLayout::General,
            ImageLayout::General,
        )])
        .unwrap();
    assert_eq!(
        fluxel_rhi_dx12::diagnostics::snapshot().uav_barriers - before.uav_barriers,
        1
    );
    let before = fluxel_rhi_dx12::diagnostics::snapshot();
    encoder
        .transition_resources(&[barrier(
            read,
            read,
            ImageLayout::General,
            ImageLayout::General,
        )])
        .unwrap();
    assert_eq!(
        fluxel_rhi_dx12::diagnostics::snapshot().uav_barriers - before.uav_barriers,
        0
    );
    encoder
        .transition_resources(&[barrier(
            read,
            copy,
            ImageLayout::General,
            ImageLayout::CopySrc,
        )])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &texture,
                mip: 1,
                origin: [0, 0, 1],
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
    let commands = encoder.finish().unwrap();
    // SAFETY: this single command step uses this device's graphics lane and
    // receives the matching one-shot recorded work below.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 0x5541_5601,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: plan, lane and recorded work belong to this device; the only
    // step is bound once and has no external dependencies.
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
        _ => panic!("submission was not accepted"),
    };
    block(device.wait_completion(&point)).unwrap();
    // SAFETY: this completion follows the texture copy and its readback ticket.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 42);
}
