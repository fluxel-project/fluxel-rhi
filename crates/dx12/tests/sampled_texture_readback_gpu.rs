//! Sampled texture views preserve uploaded texels through a compute readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;

#[derive(Clone, Copy)]
enum ShaderKind {
    LineDxil,
    VolumeDxil,
    UintDxil,
    VolumeWgsl,
}

struct SampleCase {
    format: TextureFormat,
    expected: [f32; 4],
    texture_dimension: TextureDimension,
    extent: [u32; 3],
    view_dimension: TextureViewDimension,
    sample_type: TextureSampleType,
    origin: [u32; 3],
    pixel: &'static [u8],
    texel_count: usize,
    bytes_per_row: u32,
    rows: u32,
    shader: ShaderKind,
    ordinal: u64,
}

#[test]
fn line_sample_readback() {
    run(SampleCase {
        format: TextureFormat::Rgba8Unorm,
        expected: [1.0, 0.0, 0.0, 1.0],
        texture_dimension: TextureDimension::D1,
        extent: [8, 1, 1],
        view_dimension: TextureViewDimension::D1,
        sample_type: TextureSampleType::Float,
        origin: [0, 0, 0],
        pixel: &[255, 0, 0, 255],
        texel_count: 8,
        bytes_per_row: 32,
        rows: 1,
        shader: ShaderKind::LineDxil,
        ordinal: 99501,
    });
}

#[test]
fn volume_sample_readback() {
    run(SampleCase {
        format: TextureFormat::Rgba8Unorm,
        expected: [1.0, 0.0, 0.0, 1.0],
        texture_dimension: TextureDimension::D3,
        extent: [8, 8, 2],
        view_dimension: TextureViewDimension::D3,
        sample_type: TextureSampleType::Float,
        origin: [0, 0, 1],
        pixel: &[255, 0, 0, 255],
        texel_count: 64,
        bytes_per_row: 32,
        rows: 8,
        shader: ShaderKind::VolumeDxil,
        ordinal: 99801,
    });
}

#[test]
fn uint_sample_readback() {
    run(SampleCase {
        format: TextureFormat::R8Uint,
        expected: [255.0, 0.0, 0.0, 1.0],
        texture_dimension: TextureDimension::D2,
        extent: [8, 8, 1],
        view_dimension: TextureViewDimension::D2,
        sample_type: TextureSampleType::Uint,
        origin: [0, 0, 0],
        pixel: &[255],
        texel_count: 64,
        bytes_per_row: 8,
        rows: 8,
        shader: ShaderKind::UintDxil,
        ordinal: 99520,
    });
}

#[test]
fn wgsl_volume_sample_readback() {
    run(SampleCase {
        format: TextureFormat::Rgba8Unorm,
        expected: [1.0, 0.0, 0.0, 1.0],
        texture_dimension: TextureDimension::D3,
        extent: [8, 8, 2],
        view_dimension: TextureViewDimension::D3,
        sample_type: TextureSampleType::Float,
        origin: [0, 0, 1],
        pixel: &[255, 0, 0, 255],
        texel_count: 64,
        bytes_per_row: 32,
        rows: 8,
        shader: ShaderKind::VolumeWgsl,
        ordinal: 99801,
    });
}

fn run(case: SampleCase) {
    let (shader_format, shader_bytes): (ShaderFormat, Arc<[u8]>) = match case.shader {
        ShaderKind::LineDxil => (
            ShaderFormat::Dxil,
            Arc::from(include_bytes!("assets/line_sample.dxil").as_slice()),
        ),
        ShaderKind::VolumeDxil => (
            ShaderFormat::Dxil,
            Arc::from(include_bytes!("assets/volume_sample.dxil").as_slice()),
        ),
        ShaderKind::UintDxil => (
            ShaderFormat::Dxil,
            Arc::from(include_bytes!("assets/uint_sample.dxil").as_slice()),
        ),
        ShaderKind::VolumeWgsl => (
            ShaderFormat::Wgsl,
            Arc::from(&b"@group(0) @binding(0) var image: texture_3d<f32>; @group(0) @binding(1) var<storage, read_write> result: array<vec4<u32>>; @compute @workgroup_size(1) fn main() { result[0] = bitcast<vec4<u32>>(textureLoad(image, vec3<i32>(1, 1, 1), 0)); }"[..]),
        ),
    };
    let format = case.format;
    let expected = case.expected;
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
            dimension: case.texture_dimension,
            extent: case.extent,
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: u64::from(case.rows) * 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let data = case.pixel.repeat(case.texel_count);
    let row = case.bytes_per_row as usize;
    let region = || TextureRegion {
        texture: &tex,
        mip: 0,
        origin: case.origin,
        extent: [case.extent[0], case.extent[1], 1],
        aspect: TextureAspects::COLOR,
    };
    let mut e = d.encoder(lane).unwrap();
    e.write_texture(
        region(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: case.bytes_per_row,
            rows_per_image: case.rows,
        },
        &data,
    )
    .unwrap();
    let view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format,
                dimension: case.view_dimension,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let output_read = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: case.sample_type,
                        dimension: case.view_dimension,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 16,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
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
        format: shader_format,
        bytes: shader_bytes,
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
    e.transition_resources(&[
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
            ownership: None,
        }),
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..16,
            },
            usage: StateTransition {
                from: BufferUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: BufferUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
            },
            ownership: None,
        }),
    ])
    .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &output,
            range: 0..16,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            },
            to: BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        ownership: None,
    })])
    .unwrap();
    e.copy_buffer(&output, 0, &output_read, 0, 16).unwrap();
    let sample_ticket = e.readback_buffer(&output_read, 0..16).unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_READ,
            },
            to: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        },
        layout: Some((ImageLayout::ShaderReadOnly, ImageLayout::CopySrc)),
        ownership: None,
    })])
    .unwrap();
    e.texture_to_buffer(
        region(),
        &dst,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 256,
            rows_per_image: case.rows,
        },
    )
    .unwrap();
    let ticket = e
        .readback_buffer(&dst, 0..u64::from(case.rows) * 256)
        .unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: case.ordinal,
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
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    let sample = unsafe { block(d.read_after(sample_ticket, &point)).unwrap().bytes };
    assert_eq!(
        sample
            .chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
        expected,
        "{format:?}"
    );
    for y in 0..case.rows as usize {
        assert_eq!(
            &bytes[y * 256..y * 256 + row],
            &data[y * row..y * row + row]
        );
    }
}
