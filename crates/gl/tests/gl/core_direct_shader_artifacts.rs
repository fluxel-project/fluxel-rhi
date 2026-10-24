//! WGL real-GPU fixture inputs for direct shader artifact routes.
//!
//! Registration belongs in `core_feature_registry.rs` under the WGL-only
//! condition.  The artifacts are embedded so this fixture never needs a
//! compiler SDK on the machine which executes it.

use super::*;

const GLSL_VERTEX: &[u8] = br"#version 460 core
const vec2 positions[3] = vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
out vec2 uv;
void main() { gl_Position=vec4(positions[gl_VertexID],0.,1.); uv=positions[gl_VertexID]*.5+.5; }
";
const GLSL_FRAGMENT: &[u8] = br"#version 460 core
in vec2 uv; layout(location=0) out vec4 color;
void main() { color=vec4(PIPELINE_GAIN, 0., 0., 1.); }
";
pub(super) const SPIRV_VERTEX_BASE64: &str = "AwIjBwAAAQALAAgAJQAAAAAAAAARAAIAAQAAAAsABgABAAAAR0xTTC5zdGQuNDUwAAAAAA4AAwAAAAAAAQAAAA8ACgAAAAAABAAAAG1haW4AAAAADQAAABIAAAAcAAAAIwAAACQAAAADAAMAAgAAAMwBAAAFAAQABAAAAG1haW4AAAAABQAGAAsAAABnbF9QZXJWZXJ0ZXgAAAAABgAGAAsAAAAAAAAAZ2xfUG9zaXRpb24ABgAHAAsAAAABAAAAZ2xfUG9pbnRTaXplAAAAAAYABwALAAAAAgAAAGdsX0NsaXBEaXN0YW5jZQAGAAcACwAAAAMAAABnbF9DdWxsRGlzdGFuY2UABQADAA0AAAAAAAAABQADABIAAABwb3MABQADABwAAAB1dgAABQAFACMAAABnbF9WZXJ0ZXhJRAAFAAYAJAAAAGdsX0luc3RhbmNlSUQAAABHAAMACwAAAAIAAABIAAUACwAAAAAAAAALAAAAAAAAAEgABQALAAAAAQAAAAsAAAABAAAASAAFAAsAAAACAAAACwAAAAMAAABIAAUACwAAAAMAAAALAAAABAAAAEcABAASAAAAHgAAAAAAAABHAAQAHAAAAB4AAAAAAAAARwAEACMAAAALAAAABQAAAEcABAAkAAAACwAAAAYAAAATAAIAAgAAACEAAwADAAAAAgAAABYAAwAGAAAAIAAAABcABAAHAAAABgAAAAQAAAAVAAQACAAAACAAAAAAAAAAKwAEAAgAAAAJAAAAAQAAABwABAAKAAAABgAAAAkAAAAeAAYACwAAAAcAAAAGAAAACgAAAAoAAAAgAAQADAAAAAMAAAALAAAAOwAEAAwAAAANAAAAAwAAABUABAAOAAAAIAAAAAEAAAArAAQADgAAAA8AAAAAAAAAFwAEABAAAAAGAAAAAgAAACAABAARAAAAAQAAABAAAAA7AAQAEQAAABIAAAABAAAAKwAEAAYAAAAUAAAAAAAAACsABAAGAAAAFQAAAAAAgD8gAAQAGQAAAAMAAAAHAAAAIAAEABsAAAADAAAAEAAAADsABAAbAAAAHAAAAAMAAAArAAQABgAAAB4AAAAAAAA/IAAEACIAAAABAAAADgAAADsABAAiAAAAIwAAAAEAAAA7AAQAIgAAACQAAAABAAAANgAFAAIAAAAEAAAAAAAAAAMAAAD4AAIABQAAAD0ABAAQAAAAEwAAABIAAABRAAUABgAAABYAAAATAAAAAAAAAFEABQAGAAAAFwAAABMAAAABAAAAUAAHAAcAAAAYAAAAFgAAABcAAAAUAAAAFQAAAEEABQAZAAAAGgAAAA0AAAAPAAAAPgADABoAAAAYAAAAPQAEABAAAAAdAAAAEgAAAI4ABQAQAAAAHwAAAB0AAAAeAAAAUAAFABAAAAAgAAAAHgAAAB4AAACBAAUAEAAAACEAAAAfAAAAIAAAAD4AAwAcAAAAIQAAAP0AAQA4AAEA";

// `glslangValidator -G --target-env opengl`, checked into source as bytes.
// The fragment has `binding_0`, a float sampler, and `SpecId 7` named `gain`.
const SPIRV_FRAGMENT_BASE64: &str = "AwIjBwAAAQALAAgAFgAAAAAAAAARAAIAAQAAAAsABgABAAAAR0xTTC5zdGQuNDUwAAAAAA4AAwAAAAAAAQAAAA8ABwAEAAAABAAAAG1haW4AAAAACQAAABEAAAAQAAMABAAAAAgAAAADAAMAAgAAAMwBAAAFAAQABAAAAG1haW4AAAAABQAEAAkAAABjb2xvcgAAAAUABQANAAAAYmluZGluZ18wAAAABQADABEAAAB1dgAABQAEABQAAABnYWluAAAAAEcABAAJAAAAHgAAAAAAAABHAAQADQAAACEAAAAAAAAARwAEAA0AAAAiAAAAAAAAAEcABAARAAAAHgAAAAAAAABHAAQAFAAAAAEAAAAHAAAARwAEAA0AAAAeAAAAAAAAABMAAgACAAAAIQADAAMAAAACAAAAFgADAAYAAAAgAAAAFwAEAAcAAAAGAAAABAAAACAABAAIAAAAAwAAAAcAAAA7AAQACAAAAAkAAAADAAAAGQAJAAoAAAAGAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAGwADAAsAAAAKAAAAIAAEAAwAAAAAAAAACwAAADsABAAMAAAADQAAAAAAAAAXAAQADwAAAAYAAAACAAAAIAAEABAAAAABAAAADwAAADsABAAQAAAAEQAAAAEAAAAyAAQABgAAABQAAAAAAIA/NgAFAAIAAAAEAAAAAAAAAAMAAAD4AAIABQAAAD0ABAALAAAADgAAAA0AAAA9AAQADwAAABIAAAARAAAAVwAFAAcAAAATAAAADgAAABIAAACOAAUABwAAABUAAAATAAAAFAAAAD4AAwAJAAAAFQAAAP0AAQA4AAEA";

pub(super) fn base64_words(encoded: &str) -> Vec<u8> {
    // The fixture avoids a new decoding dependency. This small decoder accepts
    // the canonical no-whitespace form above and is deliberately strict.
    let mut output = Vec::new();
    let mut bits = 0u32;
    let mut count = 0u8;
    for byte in encoded.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => panic!("fixture base64"),
        };
        bits = (bits << 6) | u32::from(value);
        count += 6;
        while count >= 8 {
            count -= 8;
            output.push((bits >> count) as u8);
        }
    }
    assert_eq!(output.len() % 4, 0);
    output
}

#[test]
fn direct_shader_artifact_fixture_inputs_are_stable() {
    assert!(
        GLSL_FRAGMENT
            .windows(b"PIPELINE_GAIN".len())
            .any(|v| v == b"PIPELINE_GAIN")
    );
    let bytes = base64_words(SPIRV_FRAGMENT_BASE64);
    assert_eq!(&bytes[..4], &0x0723_0203u32.to_le_bytes());
    assert!(bytes.windows(b"binding_0".len()).any(|v| v == b"binding_0"));
    let vertex = base64_words(SPIRV_VERTEX_BASE64);
    assert_eq!(&vertex[..4], &0x0723_0203u32.to_le_bytes());
}

pub(super) fn artifact_raster_desc(
    layout: PipelineLayout<Gl>,
    vertex: Shader<Gl>,
    fragment: Shader<Gl>,
    constants: Vec<(String, SpecializationValue)>,
) -> RasterPipelineDesc<Gl> {
    RasterPipelineDesc {
        label: None,
        layout,
        vertex: PipelineStage {
            module: vertex,
            entry: "main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: fragment,
            entry: "main".into(),
            constants,
        }),
        vertex_buffers: vec![],
        color_targets: vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: u64::MAX,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    }
}

pub(super) fn artifact_output(d: &Device<Gl>) -> (Texture<Gl>, TextureView<Gl>, Buffer<Gl>) {
    let texture = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })
        .unwrap();
    let view = d
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
    let readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 128,
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    (texture, view, readback)
}

async fn direct_glsl_pipeline_constants(d: &Device<Gl>) {
    let shader = |bytes: &'static [u8], stage| ShaderDesc {
        label: None,
        format: ShaderFormat::Glsl(460),
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = d
        .shader(shader(GLSL_VERTEX, ShaderStage::Vertex))
        .await
        .unwrap();
    let fragment = d
        .shader(shader(GLSL_FRAGMENT, ShaderStage::Fragment))
        .await
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })
        .unwrap();
    let low = d
        .raster_pipeline(artifact_raster_desc(
            layout.clone(),
            vertex.clone(),
            fragment.clone(),
            vec![("PIPELINE_GAIN".into(), SpecializationValue::F32(0.25))],
        ))
        .await
        .unwrap();
    let high = d
        .raster_pipeline(artifact_raster_desc(
            layout,
            vertex,
            fragment,
            vec![("PIPELINE_GAIN".into(), SpecializationValue::F32(0.75))],
        ))
        .await
        .unwrap();
    let (output, output_view, readback) = artifact_output(d);
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 941).unwrap();
    // SAFETY: the fixture retains the context and every resource through completion.
    let mut session =
        unsafe { common::begin_commands(&fixture, plan, CompletionRequest::End) }.unwrap();
    let mut run = session.begin_commands().unwrap_or_else(|_| panic!("begin"));
    let commands = run.commands();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
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
    for (pipeline, x) in [(&low, 0), (&high, 4)] {
        pass.set_pipeline(pipeline).unwrap();
        pass.scissor(Rect {
            x,
            y: 0,
            width: 4,
            height: 4,
        })
        .unwrap();
        pass.draw(0..3, 0..1).unwrap();
    }
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
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers the draw and copy commands.
    let pixels = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for row in pixels.chunks_exact(32) {
        for pixel in row[..16].chunks_exact(4) {
            assert_eq!(pixel, &[64, 0, 0, 255]);
        }
        for pixel in row[16..].chunks_exact(4) {
            assert_eq!(pixel, &[191, 0, 0, 255]);
        }
    }
}

async fn direct_spirv_texture_specialization(d: &Device<Gl>) {
    let shader = |bytes: Vec<u8>, stage| ShaderDesc {
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
    let vertex = d
        .shader(shader(
            base64_words(SPIRV_VERTEX_BASE64),
            ShaderStage::Vertex,
        ))
        .await
        .unwrap();
    let fragment = d
        .shader(shader(
            base64_words(SPIRV_FRAGMENT_BASE64),
            ShaderStage::Fragment,
        ))
        .await
        .unwrap();
    let bindings = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Fragment],
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
                    binding: 1,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&bindings],
            push_constants: &[],
        })
        .unwrap();
    let mut descriptor = artifact_raster_desc(
        layout,
        vertex,
        fragment,
        vec![("gain".into(), SpecializationValue::F32(0.75))],
    );
    descriptor.vertex_buffers = vec![VertexBufferLayout {
        stride: 8,
        step_mode: VertexStepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: VertexFormat::Float32(2),
        }],
    }];
    let pipeline = d.raster_pipeline(descriptor).await.unwrap();
    let positions = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 24,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let input = d
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
        .unwrap();
    let input_view = d
        .create_view(
            &input,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let sampler = d
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
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &bindings,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&input_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        })
        .unwrap();
    let (output, output_view, readback) = artifact_output(d);
    let fixture = common::ImmediateFixture {
        device: d,
        lane: d.lanes()[0].id,
    };
    let plan = common::single_commands_plan(&fixture, 942).unwrap();
    // SAFETY: the fixture retains the context and every resource through completion.
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
    commands
        .write_texture(
            TextureRegion {
                texture: &input,
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
            &[255, 0, 0, 255],
        )
        .unwrap();
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&output_view),
        resolve_target: None,
        load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 0.0])),
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
    pass.set_bind_group(0, &group, &[]).unwrap();
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
    let point = match session.finish() {
        ImmediateFinishOutcome::Issued(prefix) => {
            prefix.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("finish"),
    };
    d.wait_completion(&point).await.unwrap();
    // SAFETY: the completion point covers upload, draw, and readback.
    let pixels = unsafe { d.read_after(ticket, &point).await }.unwrap().bytes;
    for pixel in pixels.chunks_exact(4) {
        assert_eq!(pixel, &[191, 0, 0, 191]);
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
struct ArtifactWakeThread(std::thread::Thread);

#[cfg(all(windows, feature = "native-gl-wgl"))]
impl std::task::Wake for ArtifactWakeThread {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(super) fn artifact_block_on<F: std::future::Future>(future: F) -> F::Output {
    let waker = std::task::Waker::from(Arc::new(ArtifactWakeThread(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context"]
fn wgl_core_direct_glsl_pipeline_constants() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL direct GLSL constants", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all fixture resources.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let device = artifact_block_on(open(native));
    artifact_block_on(direct_glsl_pipeline_constants(&device));
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[test]
#[ignore = "requires a WGL hardware context with GL_ARB_gl_spirv"]
fn wgl_core_direct_spirv_texture_specialization() {
    use fluxel_host::{Window, WindowConfig};
    let window =
        Window::new(WindowConfig::new("GL direct SPIR-V artifacts", 1024, 768).unwrap()).unwrap();
    // SAFETY: the window outlives the GL device and all fixture resources.
    let native = unsafe { GlInstance::from_wgl_window(&window, [1024, 768]) }.unwrap();
    let device = artifact_block_on(open(native));
    artifact_block_on(direct_spirv_texture_specialization(&device));
}
