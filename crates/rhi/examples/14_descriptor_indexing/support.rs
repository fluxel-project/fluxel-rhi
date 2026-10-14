//! Shared descriptor scene resources, pipeline construction, geometry, and WebGPU entry.

use super::*;

pub(super) fn buffer(label: &str, size: u64, usage: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    }
}
pub(super) fn depth_target<B: DescriptorSceneBackend>(
    d: &Device<B>,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = d.create_texture(&TextureDesc {
        label: Some("14 depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = d.create_view(
        &texture,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((texture, view))
}
pub(super) async fn pipeline<B: DescriptorSceneBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
) -> Result<RasterPipeline<B>> {
    let vs = utils::native_shader_with_wgsl(
        d,
        "14 descriptor vertex",
        "vs_main",
        ShaderStage::Vertex,
        VS,
        VS_SPIRV,
        WGSL,
    )
    .await?;
    let ps = utils::native_shader_with_wgsl(
        d,
        "14 descriptor fragment",
        "ps_main",
        ShaderStage::Fragment,
        PS,
        PS_SPIRV,
        WGSL,
    )
    .await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some("14 descriptor-indexing pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "ps_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: VERTEX_BYTES,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                VertexAttribute {
                    location: 1,
                    offset: 12,
                    format: VertexFormat::Float32(2),
                },
                VertexAttribute {
                    location: 2,
                    offset: 20,
                    format: VertexFormat::Sint32(1),
                },
            ],
        }],
        color_targets: vec![Some(ColorTargetState {
            format,
            blend: None,
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            // The generated vertex order follows the C++ sample's Vulkan
            // coordinate convention.  Keep both windings while this demo
            // intentionally uses a hand-expanded, non-indexed cube stream.
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
            stencil_front: StencilFaceState {
                compare: CompareFunction::Always,
                fail: StencilOperation::Keep,
                depth_fail: StencilOperation::Keep,
                pass: StencilOperation::Keep,
            },
            stencil_back: StencilFaceState {
                compare: CompareFunction::Always,
                fail: StencilOperation::Keep,
                depth_fail: StencilOperation::Keep,
                pass: StencilOperation::Keep,
            },
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
    .await
}
pub(super) fn texture_pixels(n: u8) -> [u8; 36] {
    let mut p = [0; 36];
    // C++ fills every texel independently with a bright random RGB value.
    // A fixed local generator gives the same patchwork character while making
    // screenshot evidence reproducible.
    let mut state = 0x9e37_79b9_u32 ^ u32::from(n).wrapping_mul(0x85eb_ca6b);
    for texel in p.chunks_exact_mut(4) {
        for channel in &mut texel[..3] {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *channel = 50 + (state % 206) as u8;
        }
        texel[3] = 255;
    }
    p
}
pub(super) fn cube_vertices() -> Vec<u8> {
    // Same 24 per-face vertices as the C++ sample.  The indexed stream keeps
    // each face's texture index and winding intact.
    let c = [
        ([-1., -1., 1.], [0., 0.]),
        ([1., -1., 1.], [1., 0.]),
        ([1., 1., 1.], [1., 1.]),
        ([-1., 1., 1.], [0., 1.]),
        ([1., 1., 1.], [0., 0.]),
        ([1., 1., -1.], [1., 0.]),
        ([1., -1., -1.], [1., 1.]),
        ([1., -1., 1.], [0., 1.]),
        ([-1., -1., -1.], [0., 0.]),
        ([1., -1., -1.], [1., 0.]),
        ([1., 1., -1.], [1., 1.]),
        ([-1., 1., -1.], [0., 1.]),
        ([-1., -1., -1.], [0., 0.]),
        ([-1., -1., 1.], [1., 0.]),
        ([-1., 1., 1.], [1., 1.]),
        ([-1., 1., -1.], [0., 1.]),
        ([1., 1., 1.], [0., 0.]),
        ([-1., 1., 1.], [1., 0.]),
        ([-1., 1., -1.], [1., 1.]),
        ([1., 1., -1.], [0., 1.]),
        ([-1., -1., -1.], [0., 0.]),
        ([1., -1., -1.], [1., 0.]),
        ([1., -1., 1.], [1., 1.]),
        ([-1., -1., 1.], [0., 1.]),
    ];
    let mut o = Vec::new();
    for cube in 0..CUBES {
        for (vertex, (mut position, uv)) in c.into_iter().enumerate() {
            position[0] += 2.5 * cube as f32 - (CUBES as f32 * 2.5 / 2.0) + 1.25;
            for value in position.into_iter().chain(uv) {
                o.extend_from_slice(&value.to_ne_bytes());
            }
            let face = vertex / 4;
            let texture = texture_index(cube as u32, face as u32);
            o.extend_from_slice(&texture.to_ne_bytes());
        }
    }
    o
}
pub(super) fn texture_index(cube: u32, face: u32) -> u32 {
    let mut state = (cube + 1).wrapping_mul(0x9e37_79b9) ^ face.wrapping_mul(0x85eb_ca6b);
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state % TEXTURES as u32
}
pub(super) fn cube_indices() -> Vec<u8> {
    let local = [
        0u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    ];
    (0..CUBES)
        .flat_map(|cube| {
            local
                .into_iter()
                .map(move |index| index + cube as u32 * VERTICES_PER_CUBE)
        })
        .flat_map(u32::to_ne_bytes)
        .collect()
}
pub(super) fn matrices(extent: [u32; 2], _time: f32) -> [u8; 256] {
    let a = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1. / (45f32.to_radians() * 0.5).tan();
    // C++ `Camera::lookat`: position (0, 0, -10), rotation (-35, 0, 0).
    let (s, c) = (-35.0_f32.to_radians()).sin_cos();
    let v = [
        f / a,
        0.,
        0.,
        0.,
        0.,
        // Fluxel exposes WebGPU-style Y-up NDC.  The C++ sample uses a
        // positive-height Vulkan viewport, which maps that axis downward;
        // the backend's negative viewport maps it upward instead.
        -f,
        0.,
        0.,
        0.,
        0.,
        -1.001,
        -1.,
        0.,
        0.,
        -0.1001,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        c,
        s,
        0.,
        0.,
        -s,
        c,
        0.,
        0.,
        0.,
        -10.,
        1.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
    ];
    let mut b = [0; 256];
    for (i, x) in v.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&x.to_ne_bytes())
    }
    b
}
/// Browser visual equivalent using one 32-layer texture. This does not claim
/// runtime-sized descriptor indexing support from WebGPU.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_descriptor_indexing(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};
    let run = async {
        if frames == 0 {
            return Err(Error::InvalidRange);
        }
        let device = WebGpuInstance::new()?.open_default().await?;
        let lane = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .map(|lane| lane.id)
            .ok_or(Error::Unsupported("WebGPU graphics lane"))?;
        let extent = [canvas.width(), canvas.height()];
        let mut surface = WebGpu::surface_from_canvas(&device, canvas)?;
        let format = surface
            .capabilities()?
            .formats
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("WebGPU canvas format"))?;
        surface
            .configure(&SurfaceConfig {
                extent,
                format,
                present_mode: PresentMode::Fifo,
                alpha_mode: CompositeAlphaMode::Opaque,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 2,
            })
            .await?;
        let mut scene =
            DescriptorIndexing::<WebGpu>::new(&device, lane, format, extent, true).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let (work, uploaded) = scene.record(&device, &image, extent, frame as f32 / 60.0)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
            if uploaded {
                scene.mark_uploaded();
            }
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Runs the fixed texture-array visual fallback through WebGL2 immediate calls.
/// This does not advertise or emulate runtime descriptor indexing.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub async fn run_webgl2_descriptor_indexing(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> Result<()> {
    use fluxel_rhi::gl::{Gl, GlInstance};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);
    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
    // SAFETY: the canvas-backed owner remains alive through the typed device
    // and surface, and this future makes all WebGL calls on the browser thread.
    let instance = unsafe { Instance::<Gl>::from_native(InstanceId::allocate()?, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or(Error::Unsupported("WebGL2 adapter"))?;
    let device = instance
        .open(DeviceRequest {
            adapter: adapter.id,
            required_features: vec![],
        })
        .await?;
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::Unsupported("WebGL2 graphics lane"))?;
    let extent = [canvas.width(), canvas.height()];
    if extent.contains(&0) {
        return Err(Error::SurfaceLost);
    }
    let format = TextureFormat::Rgba8Unorm;
    let mut surface = Gl::surface(&device)?;
    surface
        .configure(&SurfaceConfig {
            extent,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await?;
    let mut scene = DescriptorIndexing::<Gl>::new(&device, lane, format, extent, true).await?;
    for frame in 0..frames {
        let image = surface.acquire().await?;
        let ordinal = NEXT_PLAN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::OutOfMemory)?;
        // SAFETY: one immediate graphics step precedes present on this
        // device-local browser lane; no work is queued or reordered.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: device.id(),
                    ordinal,
                },
                Arc::from([
                    StepDesc {
                        lane,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                    StepDesc {
                        lane,
                        kind: StepKind::Present,
                        waits: vec![],
                    },
                ]),
            )
        };
        // SAFETY: all browser GL verbs and the following present occur in this
        // owner-thread future before its acquired image lease leaves scope.
        let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
        let mut commands = session.begin_commands().map_err(|outcome| match outcome {
            ImmediateRunOutcome::Issued(_) => Error::InvalidState,
            ImmediateRunOutcome::Failed { error, .. }
            | ImmediateRunOutcome::Indeterminate { error, .. } => error,
        })?;
        let mut encoder = commands.commands();
        let uploaded = scene.encode(&device, &image, &mut encoder, extent, frame as f32 / 60.0)?;
        drop(encoder);
        match commands.close() {
            ImmediateRunOutcome::Issued(_) => {
                if uploaded {
                    scene.mark_uploaded();
                }
            }
            ImmediateRunOutcome::Failed { error, .. }
            | ImmediateRunOutcome::Indeterminate { error, .. } => return Err(error),
        }
        match session.present(PresentBinding::new(image)) {
            ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
            ImmediatePresentResult::Run(ImmediateRunOutcome::Failed { error, .. })
            | ImmediatePresentResult::Run(ImmediateRunOutcome::Indeterminate { error, .. })
            | ImmediatePresentResult::Rejected { error, .. } => return Err(error),
        }
        match session.finish() {
            ImmediateFinishOutcome::Issued(_) => {}
            ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Failed { error, .. })
            | ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Indeterminate {
                error, ..
            }) => {
                return Err(error);
            }
            ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Issued(_))
            | ImmediateFinishOutcome::AlreadyReported => return Err(Error::InvalidState),
        }
    }
    Ok(())
}
