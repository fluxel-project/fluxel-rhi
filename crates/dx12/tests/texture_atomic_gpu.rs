//! Real-device coverage for integer texture atomic accumulation and ordering.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

fn run_texture_atomic_add(
    format: TextureFormat,
    feature: &'static str,
    init_source: &'static [u8],
    atomic_source: &'static [u8],
    expected: u32,
) {
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
    assert!(match format {
        TextureFormat::R32Uint if feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN => {
            device.capabilities().texture_atomic_r32uint_min_compute
        }
        TextureFormat::R32Uint if feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX => {
            device.capabilities().texture_atomic_r32uint_max_compute
        }
        TextureFormat::R32Uint if feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR => {
            device.capabilities().texture_atomic_r32uint_xor_compute
        }
        TextureFormat::R32Uint if feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND => {
            device.capabilities().texture_atomic_r32uint_and_compute
        }
        TextureFormat::R32Uint if feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR => {
            device.capabilities().texture_atomic_r32uint_or_compute
        }
        TextureFormat::R32Uint => device.capabilities().texture_atomic_r32uint_compute,
        TextureFormat::R32Sint if feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN => {
            device.capabilities().texture_atomic_r32sint_min_compute
        }
        TextureFormat::R32Sint if feature == SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX => {
            device.capabilities().texture_atomic_r32sint_max_compute
        }
        TextureFormat::R32Sint => device.capabilities().texture_atomic_r32sint_compute,
        _ => false,
    });
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
            format: format,
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
                format: format,
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
                    format: format,
                    dimension: TextureViewDimension::D2Array,
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
    let init_group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: format,
                    dimension: TextureViewDimension::D2Array,
                    access: StorageAccess::Write,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let init_group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &init_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        })
        .unwrap();
    let init_layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&init_group_layout],
            push_constants: &[],
        })
        .unwrap();
    let undeclared_atomic = block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(atomic_source),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }));
    assert!(
        undeclared_atomic.is_err(),
        "atomic shader must declare its capability"
    );
    let pipeline = |layout: PipelineLayout<_>, source: &'static [u8], atomic: bool| {
        let shader = block(device.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: if atomic {
                vec![feature.to_owned()]
            } else {
                vec![]
            },
        }))
        .unwrap();
        block(
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
        .unwrap_or_else(|error| panic!("atomic={atomic} pipeline failed: {error:?}"))
    };
    let init = pipeline(init_layout, init_source, false);
    let producer = pipeline(layout.clone(), atomic_source, true);
    let consumer = pipeline(layout.clone(), atomic_source, true);
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
        pass.set_pipeline(&init).unwrap();
        pass.set_bind_group(0, &init_group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(
            write,
            read_write,
            ImageLayout::General,
            ImageLayout::General,
        )])
        .unwrap();
    {
        let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
        pass.set_pipeline(&producer).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([7, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    // Atomic read-modify-write accesses remain UAV/General and need a
    // resource-level dependency before the next dispatch.
    let before = fluxel_rhi_dx12::diagnostics::snapshot();
    encoder
        .transition_resources(&[barrier(
            read_write,
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
        pass.dispatch([5, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    encoder
        .transition_resources(&[barrier(
            read_write,
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
    assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), expected);
}

#[test]
fn r32uint_texture_atomic_add_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Uint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_ADD,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(0u)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, atomic>; @compute @workgroup_size(1) fn main() { textureAtomicAdd(values, vec2<i32>(0, 0), 0, 1u); }",
        12,
    );
}

#[test]
fn r32sint_texture_atomic_add_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Sint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_ADD,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32sint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<i32>(0)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32sint, atomic>; @compute @workgroup_size(1) fn main() { textureAtomicAdd(values, vec2<i32>(0, 0), 0, 1); }",
        12,
    );
}

#[test]
fn r32sint_texture_atomic_min_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Sint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MIN,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32sint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<i32>(0)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32sint, atomic>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureAtomicMin(values, vec2<i32>(0, 0), 0, i32(id.x) - 7); }",
        (-7i32) as u32,
    );
}

#[test]
fn r32sint_texture_atomic_max_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Sint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32SINT_MAX,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32sint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<i32>(-100)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32sint, atomic>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureAtomicMax(values, vec2<i32>(0, 0), 0, i32(id.x) - 3); }",
        3,
    );
}

#[test]
fn r32uint_texture_atomic_min_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Uint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MIN,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(0xffffffffu)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, atomic>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureAtomicMin(values, vec2<i32>(0, 0), 0, 0xffffffffu - id.x); }",
        0xfffffff9,
    );
}

#[test]
fn r32uint_texture_atomic_max_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Uint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_MAX,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(0u)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, atomic>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureAtomicMax(values, vec2<i32>(0, 0), 0, 7u - id.x); }",
        7,
    );
}

#[test]
fn r32uint_texture_atomic_xor_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Uint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_XOR,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(0u)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, atomic>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureAtomicXor(values, vec2<i32>(0, 0), 0, 1u << id.x); }",
        96,
    );
}

#[test]
fn r32uint_texture_atomic_and_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Uint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_AND,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(0xffffffffu)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, atomic>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureAtomicAnd(values, vec2<i32>(0, 0), 0, 0xffffffffu ^ (1u << id.x)); }",
        0xffff_ff80,
    );
}

#[test]
fn r32uint_texture_atomic_or_accumulates_across_dispatches() {
    run_texture_atomic_add(
        TextureFormat::R32Uint,
        SHADER_FEATURE_TEXTURE_ATOMIC_R32UINT_OR,
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, write>; @compute @workgroup_size(1) fn main() { textureStore(values, vec2<i32>(0, 0), 0, vec4<u32>(0u)); }",
        b"@group(0) @binding(0) var values: texture_storage_2d_array<r32uint, atomic>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { textureAtomicOr(values, vec2<i32>(0, 0), 0, 1u << id.x); }",
        0x7f,
    );
}
