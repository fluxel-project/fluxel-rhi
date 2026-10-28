//! Core PBO upload with padded rows and distinct array layers.
use super::*;
#[path = "../depth/depth_upload.rs"]
mod depth_upload;
use depth_upload::depth_pbo_upload;

async fn pbo_upload_impl(d: &Device<Gl>, clear: bool, integer: bool) {
    let source = r#"
@group(0) @binding(0) var image: texture_2d_array<f32>;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let p=array<vec2f,3>(vec2f(-1.,-1.),vec2f(3.,-1.),vec2f(-1.,3.)); return vec4f(p[i],0.5,1.);
}
@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f {
 let x=i32(p.x); return textureLoad(image,vec2i(0,x % 2),1+x/2,0);
}
"#;
    let source = if clear {
        source.replace("vec2i(0,x % 2),1+x/2,0", "vec2i(0,0),1+x/2,x%2")
    } else {
        source.to_owned()
    };
    let source = if integer {
        source
            .replace("texture_2d_array<f32>", "texture_2d_array<u32>")
            .replace(
                "return textureLoad(image,vec2i(0,0),1+x/2,x%2);",
                "return vec4f(textureLoad(image,vec2i(0,0),1+x/2,x%2)) / 255.0;",
            )
    } else {
        source
    };
    let shader = |name: &str, stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(source.as_bytes()),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d.shader(shader("vs", ShaderStage::Vertex)).await.unwrap();
    let fragment = d.shader(shader("fs", ShaderStage::Fragment)).await.unwrap();
    let group_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: if integer {
                        TextureSampleType::Uint
                    } else {
                        TextureSampleType::Float
                    },
                    dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let pipeline = d
        .raster_pipeline(RasterPipelineDesc {
            label: None,
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "fs".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![],
            color_targets: vec![Some(unblended_target(TextureFormat::Rgba8Unorm))],
            primitive: triangle_list_state(),
            depth_stencil: None,
            multisample: single_sample_state(u64::MAX),
            multiview: None,
            cache: None,
        })
        .await
        .unwrap();
    let texture = |extent, layers, usage, format| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent,
            array_layers: layers,
            format,
            mip_levels: if clear && layers > 1 { 2 } else { 1 },
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let input = texture(
        [2, 2, 1],
        3,
        TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        if integer {
            TextureFormat::Rgba8Uint
        } else {
            TextureFormat::Rgba8Unorm
        },
    );
    let output = texture(
        [4, 2, 1],
        1,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        TextureFormat::Rgba8Unorm,
    );
    let view = |texture, dimension, layers, format| {
        d.create_view(
            texture,
            &TextureViewDesc {
                format,
                dimension,
                aspects: TextureAspects::COLOR,
                mips: 0..if clear && dimension == TextureViewDimension::D2Array {
                    2
                } else {
                    1
                },
                layers,
            },
        )
        .unwrap()
    };
    let input_view = view(
        &input,
        TextureViewDimension::D2Array,
        0..3,
        if integer {
            TextureFormat::Rgba8Uint
        } else {
            TextureFormat::Rgba8Unorm
        },
    );
    let output_view = view(
        &output,
        TextureViewDimension::D2,
        0..1,
        TextureFormat::Rgba8Unorm,
    );
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&input_view),
            }],
        })
        .unwrap();
    let buffer = |size| {
        d.create_buffer(&BufferDesc {
            label: None,
            size,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let upload = buffer(128);
    let read = buffer(32);
    let mut data = [0xadu8; 128];
    let colors = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    for layer in 0..2 {
        for row in 0..2 {
            let start = 16 + layer * 48 + row * 16;
            for x in 0..2 {
                data[start + x * 4..start + x * 4 + 4].copy_from_slice(&colors[layer * 2 + row]);
            }
        }
    }
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 211).unwrap();
    // SAFETY: device and all resources outlive the completion point.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session
        .begin_commands()
        .unwrap_or_else(|_| panic!("begin commands"));
    let e = run.commands();
    e.write_buffer(&upload, 0, &data).unwrap();
    e.buffer_to_texture(
        &upload,
        TexelCopyBufferLayout {
            offset: 16,
            bytes_per_row: 16,
            rows_per_image: 3,
        },
        TextureRegion {
            texture: &input,
            mip: 0,
            origin: [0, 0, 1],
            extent: [2, 2, 2],
            aspect: TextureAspects::COLOR,
        },
    )
    .unwrap();
    if clear {
        e.clear_texture(
            &input,
            1..2,
            1..3,
            TextureClearValue::Color(if integer {
                ClearColor::Uint([255, 0, 255, 255])
            } else {
                ClearColor::Float([1.0, 0.0, 1.0, 1.0])
            }),
        )
        .unwrap();
        e.clear_texture(
            &input,
            0..1,
            2..3,
            TextureClearValue::Color(if integer {
                ClearColor::Uint([64, 128, 191, 255])
            } else {
                ClearColor::Float([0.25, 0.5, 0.75, 1.0])
            }),
        )
        .unwrap();
    }
    let attachments = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.; 4])),
        store: StoreOp::Store,
    })];
    let mut pass = e
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &attachments,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 2,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.draw(0..3, 0..1).unwrap();
    pass.finish().unwrap();
    e.texture_to_buffer(
        TextureRegion {
            texture: &output,
            mip: 0,
            origin: [0, 0, 0],
            extent: [4, 2, 1],
            aspect: TextureAspects::COLOR,
        },
        &read,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: 16,
            rows_per_image: 2,
        },
    )
    .unwrap();
    let ticket = e.readback_buffer(&read, 0..32).unwrap();
    assert!(matches!(run.close(), ImmediateRunOutcome::Issued(_)));
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the point covers the upload, draw, and readback.
    let result = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for row in 0..2 {
        for x in 0..4 {
            let actual = &result[row * 16 + x * 4..row * 16 + x * 4 + 4];
            let expected = if clear {
                match x {
                    0 => colors[0],
                    1 | 3 => [255, 0, 255, 255],
                    _ => [64, 128, 191, 255],
                }
            } else {
                colors[x]
            };
            assert_eq!(actual, &expected, "x={x} row={row}");
        }
    }
}
async fn upload_usage_rejections(d: &Device<Gl>) {
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let buffer = |usage| {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let plain = buffer(BufferUsage::VERTEX);
    let copy_src = buffer(BufferUsage::COPY_SRC);
    let copy_dst = buffer(BufferUsage::COPY_DST);
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    for case in 0..4 {
        let plan = common::single_commands_plan(&fixture, 215 + case).unwrap();
        // SAFETY: all resources remain alive for the immediate scope.
        let mut session =
            unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
        let mut run = session
            .begin_commands()
            .unwrap_or_else(|_| panic!("begin invalid upload"));
        let e = run.commands();
        let result = match case {
            0 => e.write_texture(
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
                &[1, 2, 3, 4],
            ),
            1 => e.write_buffer(&plain, 0, &[1, 2, 3, 4]),
            2 => e.copy_buffer(&plain, 0, &copy_dst, 0, 4),
            _ => e.copy_buffer(&copy_src, 0, &plain, 0, 4),
        };
        assert!(
            matches!(result, Err(Error::InvalidState)),
            "case {case}: {result:?}"
        );
    }
}

pub(super) async fn pbo_upload(d: &Device<Gl>, depth_readback: bool) {
    upload_usage_rejections(d).await;
    pbo_upload_impl(d, false, false).await;
    // Depth PBO readback is currently a desktop GL route. The two-row depth
    // sampling behavior differs across ANGLE profiles and is tracked
    // separately rather than weakening the GL4+ assertion.
    if depth_readback {
        for cpu in [false, true] {
            depth_pbo_upload(d, TextureFormat::Depth16Unorm, cpu).await;
            depth_pbo_upload(d, TextureFormat::Depth32Float, cpu).await;
        }
    }
}
async fn standalone_texture_clear(d: &Device<Gl>) {
    pbo_upload_impl(d, true, false).await;
    pbo_upload_impl(d, true, true).await;
}
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_pbo_upload() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL PBO upload fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(pbo_upload(&d, true));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_pbo_upload() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(pbo_upload(&d, false));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_pbo_upload() {
    let d = webgl2_device([8, 8]).await;
    pbo_upload(&d, false).await;
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_standalone_texture_clear() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL PBO upload fixture", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all its resources on this thread.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let d = block_on(open(native));
    block_on(standalone_texture_clear(&d));
}

#[cfg(feature = "native-gles-egl")]
#[test]
#[ignore = "requires an EGL GLES hardware context"]
fn gles_core_standalone_texture_clear() {
    let native =
        GlInstance::gles_pbuffer([8, 8]).expect("open EGL GLES pbuffer for hardware fixture");
    let d = block_on(open(native));
    block_on(standalone_texture_clear(&d));
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn webgl2_core_standalone_texture_clear() {
    let d = webgl2_device([8, 8]).await;
    standalone_texture_clear(&d).await;
}
