//! Dynamic-uniform cube grid through the RHI.
//!
//! The second group binding is a real dynamically-offset uniform binding.  One
//! aligned model matrix is selected per indexed cube draw; no instance data is
//! substituted for the operation demonstrated here.
mod common;
use fluxel_rhi::api::*;
const VS: &[u8] = include_bytes!("assets/04_dynamic_uniform_buffer/dynamic_uniform.vs.dxil");
const PS: &[u8] = include_bytes!("assets/04_dynamic_uniform_buffer/dynamic_uniform.ps.dxil");
const VS_SPV: &[u8] = include_bytes!("assets/04_dynamic_uniform_buffer/dynamic_uniform.vs.spv");
const PS_SPV: &[u8] = include_bytes!("assets/04_dynamic_uniform_buffer/dynamic_uniform.ps.spv");
const WGSL: &[u8] = include_bytes!("assets/04_dynamic_uniform_buffer/dynamic_uniform.wgsl");
pub const OBJECT_COUNT: u32 = 125;
pub const MODEL_BYTES: u64 = 64;
pub trait DynamicUniformBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> DynamicUniformBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
pub struct DynamicUniforms<B: DynamicUniformBackend> {
    vertex: Buffer<B>,
    index: Buffer<B>,
    camera: Buffer<B>,
    models: Buffer<B>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
    group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    lane: LaneId,
    alignment: u64,
}

/// Draws the dynamic uniform cube grid into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_dynamic_uniforms(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};

    let run = async {
        if frames == 0 {
            return Err(Error::InvalidRange);
        }
        let device = WebGpuInstance::new()?.open_default().await?;
        let lane = device.lanes()[0].id;
        let extent = [canvas.width(), canvas.height()];
        let mut surface = WebGpu::surface_from_canvas(&device, canvas)?;
        let format = surface.capabilities()?.formats[0];
        let config = SurfaceConfig {
            extent,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 2,
        };
        surface.configure(&config).await?;
        let workload = DynamicUniforms::<WebGpu>::new(&device, lane, format).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let work = workload.record(&device, &image, extent, frame as f32 / 60.0)?;
            common::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}
impl<B: DynamicUniformBackend> DynamicUniforms<B> {
    pub async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        let alignment = d.capabilities().limits.uniform_offset_alignment.max(256);
        let size = aligned(MODEL_BYTES, alignment) * u64::from(OBJECT_COUNT);
        let vertex = d.create_buffer(&buffer(
            "04 cube vertices",
            192,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = d.create_buffer(&buffer(
            "04 cube indices",
            144,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let camera = d.create_buffer(&buffer(
            "04 camera",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let models = d.create_buffer(&buffer(
            "04 dynamic models",
            size,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let depth = d.create_texture(&TextureDesc {
            label: Some("04 depth".into()),
            dimension: TextureDimension::D2,
            extent: [1280, 720, 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })?;
        let depth_view = d.create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("04 dynamic uniforms".into()),
            entries: vec![entry(0, false, 64), entry(1, true, 64)],
        })?;
        let es = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &camera,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: &models,
                    range: 0..MODEL_BYTES,
                },
            },
        ];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("04 group"),
            layout: &layout,
            entries: &es,
        })?;
        let pl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("04 layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let pipeline = make_pipeline(d, pl, format).await?;
        Ok(Self {
            vertex,
            index,
            camera,
            models,
            _depth: depth,
            depth_view,
            group,
            pipeline,
            lane,
            alignment,
        })
    }
    pub fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>> {
        let mut e = d.encoder(self.lane)?;
        e.write_buffer(&self.vertex, 0, &cube_vertices())?;
        e.write_buffer(&self.index, 0, &cube_indices())?;
        e.write_buffer(&self.camera, 0, &camera_matrix(extent))?;
        for n in 0..OBJECT_COUNT {
            e.write_buffer(
                &self.models,
                aligned(MODEL_BYTES, self.alignment) * u64::from(n),
                &model(n, time),
            )?
        }
        let c = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.02, 0.02, 0.03, 1.0])),
            store: StoreOp::Store,
        })];
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("04 dynamic uniforms"),
            colors: &c,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&self.depth_view),
                depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: extent[0],
                height: extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        p.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        p.set_pipeline(&self.pipeline)?;
        p.set_vertex_buffer(0, &self.vertex, 0..192)?;
        p.set_index_buffer(&self.index, 0..144, IndexFormat::Uint32)?;
        for n in 0..OBJECT_COUNT {
            let offset = aligned(MODEL_BYTES, self.alignment) * u64::from(n);
            p.set_bind_group(
                0,
                &self.group,
                &[u32::try_from(offset).map_err(|_| Error::InvalidRange)?],
            )?;
            p.draw_indexed(0..36, 0, 0..1)?
        }
        p.finish()?;
        e.finish()
    }
}
fn buffer(label: &str, size: u64, usage: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    }
}
fn entry(binding: u32, dynamic_offset: bool, min_size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Vertex],
        ty: BindingType::UniformBuffer {
            dynamic_offset,
            min_size,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
fn aligned(n: u64, a: u64) -> u64 {
    n.div_ceil(a) * a
}
async fn shader<B: DynamicUniformBackend>(
    d: &Device<B>,
    dxil: &[u8],
    spirv: &[u8],
    name: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    common::native_shader_with_wgsl(d, name, name, stage, dxil, spirv, WGSL).await
}
async fn make_pipeline<B: DynamicUniformBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
) -> Result<RasterPipeline<B>> {
    let vs = shader(d, VS, VS_SPV, "vs_main", ShaderStage::Vertex).await?;
    let ps = shader(d, PS, PS_SPV, "fs_main", ShaderStage::Fragment).await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some("04 pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "fs_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: 24,
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
                    format: VertexFormat::Float32(3),
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
fn model(n: u32, time: f32) -> [u8; 64] {
    // The C++ sample lays out a 5³ lattice at five world-unit intervals and
    // rotates every cube independently.  The previous 0.35-unit spacing made
    // full-size cubes overlap, hiding the dynamic-offset workload.
    let x = n / 25;
    let y = (n / 5) % 5;
    let z = n % 5;
    let position = [
        -10.0 + x as f32 * 5.0,
        -10.0 + y as f32 * 5.0,
        -10.0 + z as f32 * 5.0,
    ];
    // The source initializes each cube with independent rotations in
    // [-2π, 2π] and then advances it with independent angular velocities.
    // Its time seed is intentionally not reproducible, so use a local stable
    // generator while preserving the same distribution and transform order.
    let mut random = CubeRng::for_cube(n);
    let angles: [f32; 3] = std::array::from_fn(|_| random.signed() * 2.0 * std::f32::consts::PI);
    let speed: [f32; 3] = std::array::from_fn(|_| random.signed());
    let angles: [f32; 3] = std::array::from_fn(|index| angles[index] + time * speed[index]);
    encode_matrix(multiply(
        translation(position),
        multiply(
            rotation_z(angles[2]),
            multiply(rotation_y(angles[1]), rotation_axis_xy(angles[0])),
        ),
    ))
}

struct CubeRng(u32);

impl CubeRng {
    fn for_cube(index: u32) -> Self {
        // Do not seed adjacent lattice cells with adjacent LCG states: their
        // first six values are correlated, making every z layer appear to
        // share the same orientation. This avalanche mix gives each C++-style
        // independent random stream while retaining reproducible captures.
        let mut seed = index.wrapping_add(0x9e37_79b9);
        seed ^= seed >> 16;
        seed = seed.wrapping_mul(0x85eb_ca6b);
        seed ^= seed >> 13;
        seed = seed.wrapping_mul(0xc2b2_ae35);
        Self(seed ^ (seed >> 16))
    }

    fn signed(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

fn camera_matrix(extent: [u32; 2]) -> [u8; 64] {
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / 30f32.to_radians().tan();
    // The source's world/view convention is right-handed: its look-at camera
    // is at z=-30 and looks toward the origin. DX12 still retains [0, 1]
    // depth in this right-handed projection.
    let projection = [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        256. / -255.9,
        -1.,
        0.,
        0.,
        -25.6 / 255.9,
        0.,
    ];
    encode_matrix(multiply(projection, translation([0., 0., -30.])))
}

fn translation(position: [f32; 3]) -> [f32; 16] {
    [
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
        position[0],
        position[1],
        position[2],
        1.,
    ]
}

fn rotation_axis_xy(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    // Rodrigues rotation for the normalized C++ axis vec3(1, 1, 0).
    let half = 0.5 * (1.0 - c);
    let diagonal = c + half;
    let xy = half;
    let z = s * std::f32::consts::FRAC_1_SQRT_2;
    [
        diagonal, xy, -z, 0., xy, diagonal, z, 0., z, -z, c, 0., 0., 0., 0., 1.,
    ]
}

fn rotation_y(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.]
}

fn rotation_z(angle: f32) -> [f32; 16] {
    let (s, c) = angle.sin_cos();
    [c, s, 0., 0., -s, c, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
}

fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut output = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    output
}

fn encode_matrix(matrix: [f32; 16]) -> [u8; 64] {
    let mut bytes = [0; 64];
    for (index, value) in matrix.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}
fn cube_vertices() -> [u8; 192] {
    // Exact `generateCube` vertex order and colour values from the C++ demo.
    let p: [f32; 48] = [
        -1., -1., 1., 1., 0., 0., 1., -1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., -1., 1., 1., 0.,
        0., 0., -1., -1., -1., 1., 0., 0., 1., -1., -1., 0., 1., 0., 1., 1., -1., 0., 0., 1., -1.,
        1., -1., 0., 0., 0.,
    ];
    let mut b = [0; 192];
    for (i, v) in p.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
fn cube_indices() -> [u8; 144] {
    let x: [u32; 36] = [
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4,
        3, 2, 6, 6, 7, 3,
    ];
    let mut b = [0; 144];
    for (i, v) in x.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
#[cfg(windows)]
fn run<B: DynamicUniformBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut workload: Option<DynamicUniforms<B>> = None;
    let start = std::time::Instant::now();
    common::run_example(
        host,
        "Dynamic uniform buffers",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if workload.is_none() {
                workload = Some(common::block_on(DynamicUniforms::new(
                    device,
                    lane,
                    config.format,
                ))?)
            }
            let work = workload.as_ref().ok_or(Error::InvalidState)?.record(
                device,
                &image,
                extent,
                start.elapsed().as_secs_f32(),
            )?;
            common::submit_and_present(device, lane, work, image)
        },
    )
}

#[cfg(windows)]
fn main() {
    let result = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("04_dynamic_uniform_buffer: {error:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("04_dynamic_uniform_buffer requires Windows and DX12");
}
