//! Desktop compressed volume uploads, image copies, sampling, and raw block readback.
use super::*;

async fn compressed_3d(d: &Device<Gl>) {
    let compute = d.compute().expect("desktop fixture requires compute");
    let source = br#"
@group(0) @binding(0) var image: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f,4>;
@compute @workgroup_size(1) fn cs() {
 output[0] = textureLoad(image,vec3i(0,0,1),0);
 output[1] = textureLoad(image,vec3i(0,0,2),0);
 output[2] = textureLoad(image,vec3i(0,0,0),1);
 output[3] = textureLoad(image,vec3i(0,0,1),1);
}
"#;
    let shader = d
        .shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_slice()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        })
        .await
        .unwrap();
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let bgl = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(
                    0,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                ),
                entry(
                    1,
                    BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 64,
                        access: StorageAccess::ReadWrite,
                    },
                ),
            ],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&bgl],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = compute
        .create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "cs".into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
        .unwrap();
    let desc = |format| TextureDesc {
        label: None,
        dimension: TextureDimension::D3,
        extent: [8, 8, 4],
        array_layers: 1,
        format,
        mip_levels: 2,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
        view_formats: vec![],
    };
    // These codecs do not acquire a 3D target merely by exposing 2D formats.
    for format in [
        TextureFormat::Bc1RgbaUnorm,
        TextureFormat::Bc4RUnorm,
        TextureFormat::Etc2Rgb8Unorm,
    ] {
        assert!(matches!(
            d.texture_support(&desc(format)),
            Support::Unsupported(_)
        ));
    }
    for (format, red, green, block) in [
        (
            TextureFormat::Bc6hRgbUfloat,
            0.409423828125f32,
            0.1405029296875f32,
            (3u128 | (450u128 << 5) | (400u128 << 15) | (450u128 << 35) | (400u128 << 45))
                .to_le_bytes(),
        ),
        (
            TextureFormat::Bc6hRgbFloat,
            0.413330078125,
            -0.1424560546875,
            (3u128 | (225u128 << 5) | (824u128 << 15) | (225u128 << 35) | (824u128 << 45))
                .to_le_bytes(),
        ),
        (
            TextureFormat::Bc7RgbaUnorm,
            254.0 / 255.0,
            0.0,
            ((1u128 << 6) | (127u128 << 7) | (127u128 << 14)).to_le_bytes(),
        ),
        (
            TextureFormat::Bc7RgbaUnormSrgb,
            ((254.0f32 / 255.0 + 0.055) / 1.055).powf(2.4),
            0.0,
            ((1u128 << 6) | (127u128 << 7) | (127u128 << 14)).to_le_bytes(),
        ),
        (
            TextureFormat::Astc {
                block: [4, 4],
                srgb: false,
                hdr: false,
            },
            1.0,
            0.0,
            [
                0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff,
            ],
        ),
        (
            TextureFormat::Astc {
                block: [4, 4],
                srgb: true,
                hdr: false,
            },
            1.0,
            0.0,
            [
                0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff,
            ],
        ),
        (
            TextureFormat::Astc {
                block: [4, 4],
                srgb: false,
                hdr: true,
            },
            1.0,
            0.0,
            [
                0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0x3c, 0, 0, 0, 0, 0, 0x3c,
            ],
        ),
    ] {
        if let Support::Unsupported(reason) = d.texture_support(&desc(format)) {
            eprintln!("SKIP compressed 3D {format:?}: {reason}");
            continue;
        }
        let texture = d.create_texture(&desc(format)).unwrap();
        let copied = d.create_texture(&desc(format)).unwrap();
        let view = d
            .create_view(
                &copied,
                &TextureViewDesc {
                    format,
                    dimension: TextureViewDimension::D3,
                    aspects: TextureAspects::COLOR,
                    mips: 0..2,
                    layers: 0..1,
                },
            )
            .unwrap();
        let buffer = |size, usage| {
            d.create_buffer(&BufferDesc {
                label: None,
                size,
                usage,
                memory: MemoryClass::Device,
            })
            .unwrap()
        };
        let upload = buffer(128, BufferUsage::COPY_SRC | BufferUsage::COPY_DST);
        let raw = buffer(256, BufferUsage::COPY_SRC | BufferUsage::COPY_DST);
        let sampled = buffer(64, BufferUsage::STORAGE | BufferUsage::COPY_SRC);
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer {
                            buffer: &sampled,
                            range: 0..64,
                        },
                    },
                ],
            })
            .unwrap();
        let black = match format {
            TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc6hRgbFloat => 3u128.to_le_bytes(),
            TextureFormat::Astc { hdr: false, .. } => [
                0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0xff, 0xff,
            ],
            TextureFormat::Astc { hdr: true, .. } => [
                0xfc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0x3c,
            ],
            _ => (1u128 << 6).to_le_bytes(),
        };
        let region = |mip, origin, extent| TextureRegion {
            texture: &texture,
            mip,
            origin,
            extent,
            aspect: TextureAspects::COLOR,
        };
        let copy_region = |mip, origin, extent| TextureRegion {
            texture: &copied,
            mip,
            origin,
            extent,
            aspect: TextureAspects::COLOR,
        };
        let fixture = common::ImmediateFixture {
            device: d,
            lane: d.lanes()[0].id,
        };
        let plan = common::single_commands_plan(&fixture, 232).unwrap();
        // SAFETY: every resource and the current context survive until completion.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
        let e = run.commands();
        let mut cpu = vec![0xcc; 544];
        for z in 0..4usize {
            for row in 0..2 {
                for x in 0..2 {
                    let offset = 16 + z * 144 + row * 48 + x * 16;
                    cpu[offset..offset + 16].copy_from_slice(if z % 2 == 1 {
                        &block
                    } else {
                        &black
                    });
                }
            }
        }
        e.write_texture(
            region(0, [0, 0, 0], [8, 8, 4]),
            TexelCopyBufferLayout {
                offset: 16,
                bytes_per_row: 48,
                rows_per_image: 3,
            },
            &cpu,
        )
        .unwrap();
        let mut pbo = [0xcc; 128];
        pbo[16..32].copy_from_slice(&block);
        pbo[80..96].copy_from_slice(&black);
        e.write_buffer(&upload, 0, &pbo).unwrap();
        e.buffer_to_texture(
            &upload,
            TexelCopyBufferLayout {
                offset: 16,
                bytes_per_row: 32,
                rows_per_image: 2,
            },
            region(1, [0, 0, 0], [4, 4, 2]),
        )
        .unwrap();
        for (mip, origin, extent) in [(0, [0, 0, 1], [8, 8, 2]), (1, [0, 0, 0], [4, 4, 2])] {
            e.copy_texture(
                region(mip, origin, extent),
                copy_region(mip, origin, extent),
            )
            .unwrap();
        }
        e.write_buffer(&raw, 0, &[0xcc; 256]).unwrap();
        for (mip, origin, extent, offset, rows) in [
            (0, [4, 0, 1], [4, 8, 2], 8, 3),
            (1, [0, 0, 0], [4, 4, 2], 184, 1),
        ] {
            e.texture_to_buffer(
                copy_region(mip, origin, extent),
                &raw,
                TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: 32,
                    rows_per_image: rows,
                },
            )
            .unwrap();
        }
        let mut pass = compute.begin(e).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
        e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &sampled,
                range: 0..64,
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
        let raw_ticket = e.readback_buffer(&raw, 0..256).unwrap();
        let sample_ticket = e.readback_buffer(&sampled, 0..64).unwrap();
        assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
        let point = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => {
                prefix.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("finish"),
        };
        d.wait_completion(&point).await.unwrap();
        // SAFETY: completion covers all transfers and the storage-buffer writes.
        let actual = unsafe { d.read_after(raw_ticket, &point).await }
            .unwrap()
            .bytes;
        let mut expected = [0xcc; 256];
        for (offset, value) in [
            (8, &block),
            (40, &block),
            (104, &black),
            (136, &black),
            (184, &block),
            (216, &black),
        ] {
            expected[offset..offset + 16].copy_from_slice(value);
        }
        assert_eq!(
            actual.as_slice(),
            expected,
            "compressed volume raw blocks {format:?}"
        );
        // SAFETY: same completion covers the compute readback.
        let actual = unsafe { d.read_after(sample_ticket, &point).await }
            .unwrap()
            .bytes;
        for (i, pixel) in actual.chunks_exact(16).enumerate() {
            let r = f32::from_le_bytes(pixel[0..4].try_into().unwrap());
            let g = f32::from_le_bytes(pixel[4..8].try_into().unwrap());
            let expected = if i % 2 == 0 { [red, green] } else { [0.0, 0.0] };
            assert!(
                (r - expected[0]).abs() < 0.002 && (g - expected[1]).abs() < 0.002,
                "{format:?} sample {i}: [{r},{g}] vs {expected:?}"
            );
        }
        eprintln!(
            "EXECUTED {format:?}: 3D CPU/PBO upload, z-offset copy, two mips, padded block readback and sampled values"
        );
    }
}

#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_compressed_3d() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL compressed 3D Core fixture", 1024, 768).unwrap())
            .unwrap();
    // SAFETY: the window outlives the context and all GPU resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    eprintln!("Compressed 3D allocation: immutable storage");
    block_on(compressed_3d(&d));
    // Exercise the GL 4.0/4.1 extension route on this newer context as well:
    // mutable mips must shrink their depth independently.
    // SAFETY: the device remains alive and is used only on its context thread.
    let native = unsafe { d.native_ref() };
    native
        .0
        .with_native("force mutable compressed allocation fixture", |provider| {
            provider.disable_texture_storage_for_test();
            Ok(())
        })
        .unwrap();
    eprintln!("Compressed 3D allocation: forced mutable storage");
    block_on(compressed_3d(&d));
}
