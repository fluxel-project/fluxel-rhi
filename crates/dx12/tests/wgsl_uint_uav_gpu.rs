//! A storage array view addresses its selected native array slice.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

#[test]
fn uint_storage_view_writes_selected_array_slice() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let tex = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 3,
            format: TextureFormat::R32Uint,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format: TextureFormat::R32Uint,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
        )
        .unwrap();
    let layout = d
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
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        })
        .unwrap();
    let pl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@group(0) @binding(0) var dest: texture_storage_2d_array<r32uint, read_write>; @compute @workgroup_size(1) fn main() { textureStore(dest, vec2<i32>(0, 0), 0, vec4<u32>(0x12345678u, 0u, 0u, 0u)); let v = textureLoad(dest, vec2<i32>(0, 0), 0); textureStore(dest, vec2<i32>(0, 0), 0, v); }"[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let pipeline = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
        label: None,
        layout: pl,
        compute: PipelineStage {
            module: shader,
            entry: "main".into(),
            constants: vec![],
        },
        required_subgroup_size: None,
        cache: None,
    }))
    .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let barrier = |from, to, from_layout, to_layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition { from, to },
            layout: Some((from_layout, to_layout)),
            ownership: None,
        })
    };
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[barrier(
        empty,
        write,
        ImageLayout::Common,
        ImageLayout::General,
    )])
    .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[barrier(
        write,
        read,
        ImageLayout::General,
        ImageLayout::CopySrc,
    )])
    .unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 1],
            extent: [1, 1, 1],
            aspect: TextureAspects::COLOR,
        },
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: 1,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&dst, 0..256).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99853,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submit"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..4], &[0x78, 0x56, 0x34, 0x12]);
}
