#![cfg(windows)]
#[path = "../src/backend/shader_naga/mod.rs"]
mod shader_naga;
use fluxel_rhi_core::api::{ShaderDesc, ShaderEntryPoint, ShaderFormat, ShaderStage};
use std::sync::Arc;

#[test]
fn compute_wgsl_compiles_with_mapped_raw_buffers() {
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@group(2) @binding(3) var<storage, read_write> data: array<u32>; @compute @workgroup_size(1) fn main() { data[0] = 42u; }"[..]),
        entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1, 1, 1]) }],
        required_features: vec![],
    };
    let result = shader_naga::compile(&desc).unwrap();
    assert!(
        result.entries[0].hlsl.contains("register(u3, space2)"),
        "{}",
        result.entries[0].hlsl
    );
    assert!(result.entries[0].hlsl.contains("RWByteAddressBuffer"));
    assert_eq!(result.entries[0].raw_storage_bindings, vec![(2, 3)]);
    assert!(!result.entries[0].bytes.is_empty());
    assert_eq!(result.entries[0].translated_name, "main");
}

#[test]
fn unsupported_abi_is_rejected() {
    let make = |source: &'static [u8]| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let huge_space = make(b"@group(256) @binding(0) var<storage,read> data: array<u32>; @compute @workgroup_size(1) fn main() { let x = data[0]; }");
    assert!(matches!(
        shader_naga::compile(&huge_space),
        Err(fluxel_rhi_core::api::Error::Unsupported(_))
    ));
    let num_workgroups = make(b"@compute @workgroup_size(1) fn main(@builtin(num_workgroups) n: vec3<u32>) { let x = n.x; }");
    assert!(shader_naga::compile(&num_workgroups).unwrap().entries[0].num_workgroups_constants);
    let mut wrong_workgroup = make(b"@compute @workgroup_size(2) fn main() {}");
    assert!(matches!(
        shader_naga::compile(&wrong_workgroup),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
    wrong_workgroup
        .entries
        .push(wrong_workgroup.entries[0].clone());
    assert!(matches!(
        shader_naga::compile(&wrong_workgroup),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
}

#[test]
fn sampler_array_uses_descriptor_table() {
    let source = b"@group(0) @binding(1) var s: binding_array<sampler, 2>; @group(0) @binding(4) var t: texture_2d<f32>; @group(0) @binding(5) var<uniform> index: u32; @group(0) @binding(6) var<storage,read_write> output: array<vec4<f32>>; @compute @workgroup_size(1) fn main() { output[0] = textureSampleLevel(t, s[index], vec2<f32>(0.5), 0.0); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let compiled = shader_naga::compile(&desc).unwrap();
    assert!(
        compiled.entries[0]
            .hlsl
            .contains("[2] : register(s1, space0)"),
        "{}",
        compiled.entries[0].hlsl
    );
    assert!(!compiled.entries[0].hlsl.contains("nagaSamplerHeap"));
    let entry = &desc.entries[0];
    let map = [
        (0, 1, 1, 3, 1),
        (0, 4, 4, 1, 1),
        (0, 5, 5, 0, 1),
        (0, 6, 6, 2, 1),
    ];
    assert!(matches!(
        shader_naga::compile_variant_with_map(&desc, entry, &[], &map),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
    let map = [
        (0, 1, 1, 1, 2),
        (0, 4, 4, 1, 1),
        (0, 5, 5, 0, 1),
        (0, 6, 6, 2, 1),
    ];
    assert!(matches!(
        shader_naga::compile_variant_with_map(&desc, entry, &[], &map),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
}

#[test]
fn runtime_binding_arrays_specialize_to_the_layout_capacity() {
    use fluxel_rhi_core::api::{
        BindingType, StorageAccess, TextureSampleType, TextureViewDimension,
    };
    let source = b"@group(0) @binding(0) var images: binding_array<texture_2d<f32>>; @group(0) @binding(1) var samplers: binding_array<sampler>; @group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>; @compute @workgroup_size(1) fn main() { out[0] = textureSampleLevel(images[1u], samplers[1u], vec2f(0.5), 0.0); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let map = [(0, 0, 0, 1, 3), (0, 1, 1, 3, 3), (0, 2, 2, 2, 1)];
    let types = [
        (
            0,
            0,
            BindingType::SampledTexture {
                sample: TextureSampleType::Float,
                dimension: TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        (
            0,
            1,
            BindingType::Sampler {
                filtering: true,
                comparison: false,
            },
        ),
        (
            0,
            2,
            BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: 16,
                access: StorageAccess::ReadWrite,
            },
        ),
    ];
    let compiled = shader_naga::compile_variant_with_typed_map(
        &desc,
        &desc.entries[0],
        &[],
        &map,
        &types,
        &[(0, 0, 3), (0, 1, 3)],
        None,
    )
    .unwrap();
    assert!(compiled.hlsl.contains("images[3]"), "{}", compiled.hlsl);
    assert!(
        compiled.hlsl.contains("[3] : register(s1"),
        "{}",
        compiled.hlsl
    );
    assert!(compiled.hlsl.contains("register(t0"), "{}", compiled.hlsl);
    assert!(matches!(
        shader_naga::compile_variant_with_typed_map(
            &desc,
            &desc.entries[0],
            &[],
            &map,
            &types,
            &[],
            None,
        ),
        Err(fluxel_rhi_core::api::Error::Shader(_))
    ));
}

#[test]
fn required_wave_size_is_baked_into_wgsl_derived_dxil() {
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&b"@compute @workgroup_size(32) fn main() {}"[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([32, 1, 1]),
        }],
        required_features: vec![],
    };
    let compiled = shader_naga::compile_variant_with_typed_map(
        &desc,
        &desc.entries[0],
        &[],
        &[],
        &[],
        &[],
        Some(32),
    )
    .unwrap();
    assert!(
        compiled.hlsl.contains("[WaveSize(32)]"),
        "{}",
        compiled.hlsl
    );
    assert!(compiled.bytes.starts_with(b"DXBC"));
}

#[test]
fn comparison_sampler_uses_regular_root_table() {
    let source = b"@group(0) @binding(1) var s: sampler_comparison; @group(0) @binding(2) var t: texture_depth_2d; @group(0) @binding(3) var<storage, read_write> out: array<f32>; @compute @workgroup_size(1) fn main() { out[0] = textureSampleCompareLevel(t, s, vec2<f32>(0.5), 0.5); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let compiled = shader_naga::compile(&desc).unwrap();
    assert!(
        compiled.entries[0]
            .hlsl
            .contains("SamplerComparisonState s : register(s1, space0)")
    );
}

#[test]
fn vertex_fragment_compute_compile_to_independent_dxil() {
    let source = br#"
        @vertex fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(pos, 0.0, 1.0);
        }
        @fragment fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(0.25, 0.5, 0.75, 1.0);
        }
        @compute @workgroup_size(1) fn cs_main() {}
    "#;
    let entries = [
        ("vs_main", ShaderStage::Vertex),
        ("fs_main", ShaderStage::Fragment),
        ("cs_main", ShaderStage::Compute),
    ]
    .into_iter()
    .map(|(name, stage)| ShaderEntryPoint {
        name: name.into(),
        stage,
        workgroup_size: None,
    })
    .collect();
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries,
        required_features: vec![],
    };
    let compiled = shader_naga::compile(&desc).unwrap();
    assert_eq!(compiled.entries.len(), 3);
    for (entry, expected) in compiled.entries.iter().zip(&desc.entries) {
        assert_eq!(entry.name, expected.name);
        assert_eq!(entry.stage, expected.stage);
        assert!(entry.bytes.starts_with(b"DXBC"));
    }
}

#[test]
fn float_texture_load_uses_sampled_texture_register() {
    let source = b"@group(1) @binding(4) var image: texture_2d<f32>; @group(0) @binding(0) var<storage,read_write> data: array<u32>; @compute @workgroup_size(1) fn main() { data[0] = u32(textureLoad(image, vec2<i32>(0), 0).x); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let result = shader_naga::compile(&desc).unwrap();
    assert!(
        result.entries[0].hlsl.contains("register(t4, space1)"),
        "{}",
        result.entries[0].hlsl
    );
}

#[test]
fn storage_texture_write_uses_uav_register() {
    let source = b"@group(2) @binding(5) var image: texture_storage_2d<rgba8unorm,write>; @compute @workgroup_size(1) fn main() { textureStore(image, vec2<i32>(0), vec4<f32>(1.0)); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let result = shader_naga::compile(&desc).unwrap();
    assert!(
        result.entries[0].hlsl.contains("register(u5, space2)"),
        "{}",
        result.entries[0].hlsl
    );
}

#[test]
fn specialization_variants_validate_values_and_produce_distinct_dxil() {
    use fluxel_rhi_core::api::{Error, SpecializationValue};
    let source = b"@id(7) override value: u32 = 1u; @group(0) @binding(0) var<storage,read_write> out: array<u32>; @compute @workgroup_size(1) fn main() { out[0] = value; }";
    let entry = ShaderEntryPoint {
        name: "main".into(),
        stage: ShaderStage::Compute,
        workgroup_size: Some([1, 1, 1]),
    };
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![entry.clone()],
        required_features: vec![],
    };
    let a =
        shader_naga::compile_variant(&desc, &entry, &[("7".into(), SpecializationValue::U32(2))])
            .unwrap();
    let b =
        shader_naga::compile_variant(&desc, &entry, &[("7".into(), SpecializationValue::U32(3))])
            .unwrap();
    assert_ne!(a.bytes, b.bytes);
    assert!(matches!(
        shader_naga::compile_variant(&desc, &entry, &[("7".into(), SpecializationValue::I32(2))]),
        Err(Error::Shader(_))
    ));
    assert!(matches!(
        shader_naga::compile_variant(
            &desc,
            &entry,
            &[("missing".into(), SpecializationValue::U32(2))]
        ),
        Err(Error::Shader(_))
    ));
    assert!(matches!(
        shader_naga::compile_variant(
            &desc,
            &entry,
            &[
                ("7".into(), SpecializationValue::U32(2)),
                ("7".into(), SpecializationValue::U32(3))
            ]
        ),
        Err(Error::Shader(_))
    ));
}

#[test]
fn scalar_samplers_keep_group_and_binding() {
    let source = b"@group(0) @binding(0) var tex: texture_2d<f32>; @group(0) @binding(2) var a: sampler; @group(1) @binding(2) var b: sampler; @fragment fn main() -> @location(0) vec4<f32> { return textureSample(tex, a, vec2<f32>(0.5)) + textureSample(tex, b, vec2<f32>(0.5)); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Fragment,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let result = shader_naga::compile(&desc).unwrap();
    let hlsl = &result.entries[0].hlsl;
    assert!(hlsl.contains("register(s2, space0)"), "{hlsl}");
    assert!(hlsl.contains("register(s2, space1)"), "{hlsl}");
    assert!(!hlsl.contains("SamplerIndexArray"));
}
#[cfg(feature = "hardware-tests")]
#[test]
fn generated_dxil_executes_with_raw_views_on_dx12() {
    use fluxel_rhi_core::api::*;
    use fluxel_rhi_dx12::Dx12Instance;
    use std::future::Future;
    fn block<F: Future>(f: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let mut f = std::pin::pin!(f);
        loop {
            if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    let source = b"@group(0) @binding(1) var samp: sampler; @group(0) @binding(2) var image: texture_2d<f32>; @group(0) @binding(3) var<storage, read_write> data: array<u32>; @group(0) @binding(4) var output_image: texture_storage_2d<rgba8unorm, write>; override extra: u32 = 0u; @compute @workgroup_size(1) fn main() { data[0] = u32(round(textureSampleLevel(image, samp, vec2<f32>(0.5), 0.0).x * 255.0)) + extra; textureStore(output_image, vec2<i32>(0), vec4<f32>(1.0, 0.0, 0.0, 1.0)); }";
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(&source[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let sampler = device
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let texture_view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let output_image = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::STORAGE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let output_view = device
        .create_view(
            &output_image,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let image_readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 256,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let data = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 3,
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
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2,
                        access: StorageAccess::Write,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::StorageBufferView {
                        buffer: &data,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&output_view),
                },
            ],
        })
        .unwrap();
    let structured_group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::StorageBufferView {
                        buffer: &data,
                        range: 0..16,
                        view: StorageBufferView::Structured { stride: 4 },
                    },
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&output_view),
                },
            ],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(device.shader(desc)).unwrap();
    let pipeline = block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![("extra".into(), SpecializationValue::U32(40))],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_texture(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 4,
                rows_per_image: 1,
            },
            &[2, 0, 0, 255],
        )
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
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
        })])
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &output_image,
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
                    accesses: AccessMask::SHADER_WRITE,
                },
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::General)),
            ownership: None,
        })])
        .unwrap();
    let transition = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &data,
                range: 0..16,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    encoder
        .transition_resources(&[transition(empty, write)])
        .unwrap();
    {
        let mut rejected_encoder = device.encoder(lane).unwrap();
        let mut rejected = device
            .compute()
            .unwrap()
            .begin(&mut rejected_encoder)
            .unwrap();
        rejected.set_pipeline(&pipeline).unwrap();
        assert!(rejected.set_bind_group(0, &structured_group, &[]).is_err());
    }
    {
        let mut rejected_encoder = device.encoder(lane).unwrap();
        let mut rejected = device
            .compute()
            .unwrap()
            .begin(&mut rejected_encoder)
            .unwrap();
        rejected.set_pipeline(&pipeline).unwrap();
        rejected.set_bind_group(0, &group, &[]).unwrap();
        rejected.set_pipeline(&pipeline).unwrap();
        assert!(rejected.dispatch([1, 1, 1]).is_err());
    }
    let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[transition(write, read)])
        .unwrap();
    encoder.copy_buffer(&data, 0, &readback, 0, 16).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..16).unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &output_image,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::General, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .texture_to_buffer(
            TextureRegion {
                texture: &output_image,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &image_readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
        )
        .unwrap();
    let image_ticket = encoder.readback_buffer(&image_readback, 0..256).unwrap();
    let work = encoder.finish().unwrap();
    // SAFETY: the plan and recorded work use the same device and lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 19991,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: resources and work are retained until completion below.
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
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    // SAFETY: the completion point covers the readback ticket.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_ne_bytes(bytes[..4].try_into().unwrap()), 42);
    let image_bytes = unsafe {
        block(device.read_after(image_ticket, &point))
            .unwrap()
            .bytes
    };
    assert_eq!(&image_bytes[..4], &[255, 0, 0, 255]);
}
