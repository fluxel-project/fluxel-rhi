//! Two stripped SPIR-V texture/sampler pairs across Core binding groups.
//!
//! The second native binding is 3 but its allocated sampler unit is 1. This
//! verifies real uniform remapping instead of accidentally relying on defaults.

use super::direct_shader_artifacts::{
    SPIRV_VERTEX_BASE64, artifact_output, artifact_raster_desc, base64_words,
};
use super::*;
// Generated offline with glslang -G and stripped of OpName by spirv-opt.
const FRAGMENT: &str = "AwIjBwAAAQALAAgAGwAAAAAAAAARAAIAAQAAAAsABgABAAAAR0xTTC5zdGQuNDUwAAAAAA4AAwAAAAAAAQAAAA8ABwAEAAAABAAAAG1haW4AAAAACQAAABIAAAAQAAMABAAAAAgAAABHAAQACQAAAB4AAAAAAAAARwAEAA4AAAAeAAAAAAAAAEcABAAOAAAAIQAAAAAAAABHAAQADgAAACIAAAAAAAAARwAEABIAAAAeAAAAAAAAAEcABAAVAAAAHgAAAAEAAABHAAQAFQAAACEAAAADAAAARwAEABUAAAAiAAAAAAAAABMAAgACAAAAIQADAAMAAAACAAAAFgADAAYAAAAgAAAAFwAEAAcAAAAGAAAABAAAACAABAAIAAAAAwAAAAcAAAA7AAQACAAAAAkAAAADAAAAKwAEAAYAAAAKAAAAAAAAPxkACQALAAAABgAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAABsAAwAMAAAACwAAACAABAANAAAAAAAAAAwAAAA7AAQADQAAAA4AAAAAAAAAFwAEABAAAAAGAAAAAgAAACAABAARAAAAAQAAABAAAAA7AAQAEQAAABIAAAABAAAAOwAEAA0AAAAVAAAAAAAAADYABQACAAAABAAAAAAAAAADAAAA+AACAAUAAAA9AAQADAAAAA8AAAAOAAAAPQAEABAAAAATAAAAEgAAAFcABQAHAAAAFAAAAA8AAAATAAAAPQAEAAwAAAAWAAAAFQAAAD0ABAAQAAAAFwAAABIAAABXAAUABwAAABgAAAAWAAAAFwAAAIEABQAHAAAAGQAAABQAAAAYAAAAjgAFAAcAAAAaAAAAGQAAAAoAAAA+AAMACQAAABoAAAD9AAEAOAABAA==";

const SAME_GROUP_FRAGMENT: &str = "AwIjBwAAAQALAAgAHgAAAAAAAAARAAIAAQAAAAsABgABAAAAR0xTTC5zdGQuNDUwAAAAAA4AAwAAAAAAAQAAAA8ABwAEAAAABAAAAG1haW4AAAAACQAAABIAAAAQAAMABAAAAAgAAABHAAQACQAAAB4AAAAAAAAARwAEAA4AAAAeAAAAAAAAAEcABAAOAAAAIQAAAAAAAABHAAQADgAAACIAAAAAAAAARwAEABIAAAAeAAAAAAAAAEcABAAVAAAAHgAAAAEAAABHAAQAFQAAACEAAAACAAAARwAEABUAAAAiAAAAAAAAABMAAgACAAAAIQADAAMAAAACAAAAFgADAAYAAAAgAAAAFwAEAAcAAAAGAAAABAAAACAABAAIAAAAAwAAAAcAAAA7AAQACAAAAAkAAAADAAAAKwAEAAYAAAAKAAAAAAAAPxkACQALAAAABgAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAABsAAwAMAAAACwAAACAABAANAAAAAAAAAAwAAAA7AAQADQAAAA4AAAAAAAAAFwAEABAAAAAGAAAAAgAAACAABAARAAAAAQAAABAAAAA7AAQAEQAAABIAAAABAAAAOwAEAA0AAAAVAAAAAAAAACsABAAGAAAAGAAAAAAAAEAsAAUAEAAAABkAAAAYAAAAGAAAADYABQACAAAABAAAAAAAAAADAAAA+AACAAUAAAA9AAQADAAAAA8AAAAOAAAAPQAEABAAAAATAAAAEgAAAFcABQAHAAAAFAAAAA8AAAATAAAAPQAEAAwAAAAWAAAAFQAAAD0ABAAQAAAAFwAAABIAAACBAAUAEAAAABoAAAAXAAAAGQAAAFcABQAHAAAAGwAAABYAAAAaAAAAgQAFAAcAAAAcAAAAFAAAABsAAACOAAUABwAAAB0AAAAcAAAACgAAAD4AAwAJAAAAHQAAAP0AAQA4AAEA";

async fn pairs(device: &Device<Gl>, single_group: bool) {
    let make_shader = |bytes: Vec<u8>, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = device
        .shader(make_shader(
            base64_words(SPIRV_VERTEX_BASE64),
            ShaderStage::Vertex,
        ))
        .await
        .unwrap();
    let fragment = device
        .shader(make_shader(
            base64_words(if single_group {
                SAME_GROUP_FRAGMENT
            } else {
                FRAGMENT
            }),
            ShaderStage::Fragment,
        ))
        .await
        .unwrap();
    let texture_entry = |binding| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::SampledTexture {
            sample: TextureSampleType::Float,
            dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let sampler_entry = |binding| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty: BindingType::Sampler {
            filtering: false,
            comparison: false,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let entries = if single_group {
        (0..3)
            .flat_map(|index| [texture_entry(index * 2), sampler_entry(index * 2 + 1)])
            .collect()
    } else {
        vec![texture_entry(0), sampler_entry(1)]
    };
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries,
        })
        .unwrap();
    let groups = if single_group {
        vec![&group_layout]
    } else {
        vec![&group_layout; 4]
    };
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &groups,
            push_constants: &[],
        })
        .unwrap();
    let mut descriptor = artifact_raster_desc(layout, vertex, fragment, vec![]);
    descriptor.vertex_buffers = vec![VertexBufferLayout {
        stride: 8,
        step_mode: VertexStepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: VertexFormat::Float32(2),
        }],
    }];
    let pipeline = device.raster_pipeline(descriptor).await.unwrap();
    let positions = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 24,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let make_texture = || {
        device
            .create_texture(&TextureDesc {
                label: None,
                dimension: TextureDimension::D2,
                extent: [1, 1, 1],
                array_layers: 1,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                view_formats: vec![],
            })
            .unwrap()
    };
    let first = make_texture();
    let second = make_texture();
    let make_view = |texture: &Texture<Gl>| {
        device
            .create_view(
                texture,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )
            .unwrap()
    };
    let first_view = make_view(&first);
    let second_view = make_view(&second);
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
    let border_sampler = device
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToBorder; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: Some(BorderColor::TransparentBlack),
        })
        .unwrap();
    let make_group = |view: &TextureView<Gl>| {
        let mut entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        if single_group {
            entries.extend([
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&first_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&second_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Sampler(&border_sampler),
                },
            ]);
        }
        device
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &group_layout,
                entries: &entries,
            })
            .unwrap()
    };
    let first_group = make_group(&first_view);
    let second_group = (!single_group).then(|| make_group(&second_view));
    let (output, output_view, readback) = artifact_output(device);
    let fixture = common::ImmediateFixture {
        device,
        lane: device.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 947).unwrap();
    // SAFETY: the same-thread context and every resource outlive this session.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let commands = run.commands();
    let position_bytes: Vec<u8> = [-1.0_f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    commands
        .write_buffer(&positions, 0, &position_bytes)
        .unwrap();
    for (texture, pixel) in [(&first, [200, 0, 0, 255]), (&second, [0, 100, 0, 255])] {
        commands
            .write_texture(
                TextureRegion {
                    texture,
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
                &pixel,
            )
            .unwrap();
    }
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
        store: StoreOp::Store,
    })];
    let mut pass = commands
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 4,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_vertex_buffer(0, &positions, 0..24).unwrap();
    pass.set_bind_group(0, &first_group, &[]).unwrap();
    if let Some(second_group) = &second_group {
        pass.set_bind_group(3, second_group, &[]).unwrap();
    }
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    commands
        .texture_to_buffer(
            TextureRegion {
                texture: &output,
                mip: 0,
                origin: [0, 0, 0],
                extent: [8, 4, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 32,
                rows_per_image: 4,
            },
        )
        .unwrap();
    let ticket = commands.readback_buffer(&readback, 0..128).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let completion = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    device.wait_completion(&completion).await.unwrap();
    // SAFETY: completion covers the complete output snapshot.
    let bytes = unsafe { device.read_after(ticket, &completion).await }
        .unwrap()
        .bytes;
    for pixel in bytes.chunks_exact(4) {
        let expected = if single_group {
            [100_u8, 0, 0, 128]
        } else {
            [100, 50, 0, 255]
        };
        for (actual, expected) in pixel.iter().zip(expected) {
            assert!(
                actual.abs_diff(expected) <= 1,
                "{pixel:?}, single_group={single_group}"
            );
        }
    }
}

#[test]
#[ignore = "requires WGL GL_ARB_gl_spirv hardware"]
fn wgl_core_direct_spirv_pairs() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("SPIR-V texture pairs", 1024, 768).unwrap()).unwrap();
    // SAFETY: this window outlives the same-thread device and all GPU resources.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let device = block_on(open(native));
    block_on(pairs(&device, false));
    block_on(pairs(&device, true));
}
