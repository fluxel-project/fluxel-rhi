//! Push-constant sphere ring through the RHI.
//!
//! The original Vulkan workload issues sixteen indexed sphere draws.  Every
//! draw writes its own 32-byte `{ color, position }` record to the vertex stage;
//! it is deliberately not replaced with instance data or a dynamic uniform.

mod common;
#[path = "common/gltf.rs"]
mod gltf;

use fluxel_rhi::api::*;

pub const SPHERE_COUNT: usize = 16;
pub const PUSH_BYTES: u32 = 32;

/// The native DXIL source for the b0 camera / b1 root-constant ABI.
pub const VERTEX_DXIL: &[u8] = include_bytes!("assets/05_push_constants/push_constants.vs.dxil");
pub const FRAGMENT_DXIL: &[u8] = include_bytes!("assets/05_push_constants/push_constants.ps.dxil");
pub const VERTEX_SPIRV: &[u8] = include_bytes!("assets/05_push_constants/push_constants.vs.spv");
pub const FRAGMENT_SPIRV: &[u8] = include_bytes!("assets/05_push_constants/push_constants.ps.spv");
const WGSL: &[u8] = include_bytes!("assets/05_push_constants/push_constants.wgsl");
const SPHERE: &[u8] = include_bytes!("assets/models/sphere.gltf");

/// One source-equivalent `color + position` record.
#[derive(Clone, Copy)]
pub struct PushData {
    pub color: [f32; 4],
    pub position: [f32; 4],
}

impl PushData {
    pub fn bytes(self) -> [u8; PUSH_BYTES as usize] {
        let mut bytes = [0; PUSH_BYTES as usize];
        for (index, value) in self.color.into_iter().chain(self.position).enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        }
        bytes
    }
}

/// The same sixteen positions as the original sample's radius-3.5 circle.
pub fn sphere_ring() -> [PushData; SPHERE_COUNT] {
    // Fixed capture palette sampled from the source example's reference run.
    // The C++ program seeds `std::default_random_engine` from
    // `std::random_device`, so it has no reproducible colour sequence.
    const COLORS: [[f32; 3]; SPHERE_COUNT] = [
        [0.659, 0.757, 0.757],
        [0.537, 0.110, 0.929],
        [0.294, 0.800, 0.702],
        [0.345, 0.110, 0.976],
        [0.580, 0.725, 0.780],
        [0.537, 0.788, 0.125],
        [0.298, 0.933, 0.855],
        [0.867, 0.898, 0.137],
        [0.890, 0.596, 0.890],
        [0.667, 0.149, 0.667],
        [0.314, 0.533, 0.553],
        [0.973, 0.306, 0.941],
        [0.639, 0.220, 0.741],
        [0.141, 0.392, 0.518],
        [0.267, 0.388, 0.376],
        [0.459, 0.651, 0.220],
    ];
    std::array::from_fn(|index| {
        let angle = (index as f32 * 360.0 / SPHERE_COUNT as f32).to_radians();
        let (sin, cos) = angle.sin_cos();
        PushData {
            color: [COLORS[index][0], COLORS[index][1], COLORS[index][2], 1.0],
            position: [sin * 3.5, cos * 3.5, 0.0, 1.0],
        }
    })
}

/// Encodes the original sixteen push-constant indexed draws.
///
/// The caller supplies the sphere mesh, camera bind group and already compiled
/// pipeline. Graph transitions and the containing surface submission remain
/// outside the API command layer.
pub fn encode_sphere_ring<B>(
    device: &Device<B>,
    encoder: &mut CommandEncoder<B>,
    pass: &RasterPassDesc<'_, B>,
    pipeline: &RasterPipeline<B>,
    camera: &BindGroup<B>,
    vertices: &Buffer<B>,
    vertex_bytes: u64,
    indices: &Buffer<B>,
    index_bytes: u64,
    index_count: u32,
    spheres: &[PushData; SPHERE_COUNT],
) -> Result<()>
where
    B: RasterBackend + DeferredBackend,
{
    if !device.capabilities().push_constants {
        return Err(Error::Unsupported(
            "05_push_constants requires 32-byte vertex push constants",
        ));
    }
    if device.capabilities().limits.max_push_constant_bytes < PUSH_BYTES {
        return Err(Error::Unsupported(
            "05_push_constants requires a 32-byte push-constant limit",
        ));
    }
    let extras = device
        .raster_extras()
        .ok_or(Error::Unsupported("push constants"))?;
    let mut raster = encoder.begin_raster(pass)?;
    raster.viewport(Viewport {
        x: pass.render_area.x as f32,
        y: pass.render_area.y as f32,
        width: pass.render_area.width as f32,
        height: pass.render_area.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(pass.render_area)?;
    raster.set_pipeline(pipeline)?;
    raster.set_bind_group(0, camera, &[])?;
    raster.set_vertex_buffer(0, vertices, 0..vertex_bytes)?;
    raster.set_index_buffer(indices, 0..index_bytes, IndexFormat::Uint32)?;
    for sphere in spheres {
        extras.push_constants(&mut raster, StageMask::VERTEX, 0, &sphere.bytes())?;
        raster.draw_indexed(0..index_count, 0, 0..1)?;
    }
    raster.finish()
}

/// Encodes the browser equivalent of the native push-constant workload.
///
/// Standard WebGPU has no push constants.  A dynamic uniform range selects
/// the same one 32-byte `PushData` record before each indexed sphere draw.
fn encode_sphere_ring_dynamic_uniform<B>(
    encoder: &mut CommandEncoder<B>,
    pass: &RasterPassDesc<'_, B>,
    pipeline: &RasterPipeline<B>,
    camera: &BindGroup<B>,
    draw_data: &BindGroup<B>,
    alignment: u64,
    vertices: &Buffer<B>,
    vertex_bytes: u64,
    indices: &Buffer<B>,
    index_bytes: u64,
    index_count: u32,
) -> Result<()>
where
    B: RasterBackend + DeferredBackend,
{
    let mut raster = encoder.begin_raster(pass)?;
    raster.viewport(Viewport {
        x: pass.render_area.x as f32,
        y: pass.render_area.y as f32,
        width: pass.render_area.width as f32,
        height: pass.render_area.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(pass.render_area)?;
    raster.set_pipeline(pipeline)?;
    raster.set_bind_group(0, camera, &[])?;
    raster.set_vertex_buffer(0, vertices, 0..vertex_bytes)?;
    raster.set_index_buffer(indices, 0..index_bytes, IndexFormat::Uint32)?;
    for index in 0..SPHERE_COUNT {
        let offset = alignment
            .checked_mul(index as u64)
            .ok_or(Error::InvalidRange)?;
        raster.set_bind_group(
            1,
            draw_data,
            &[u32::try_from(offset).map_err(|_| Error::InvalidRange)?],
        )?;
        raster.draw_indexed(0..index_count, 0, 0..1)?;
    }
    raster.finish()
}

/// Draws the sphere ring into a host-owned browser canvas.
///
/// This path uses the WebGPU dynamic-uniform fallback; native backends retain
/// the real push-constant path below.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_push_constants(
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
        let mut workload = PushWorkload::<WebGpu>::new(&device, lane, format, extent).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let work = workload.record(&device, &image, extent)?;
            common::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn run<B: PushBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut workload: Option<PushWorkload<B>> = None;
    common::run_example(
        host,
        "Push constants",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if workload.is_none() {
                workload = Some(common::block_on(PushWorkload::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let work = workload
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent)?;
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
        eprintln!("05_push_constants: {error:?}");
        std::process::exit(1);
    }
}

pub trait PushBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> PushBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
struct PushWorkload<B: PushBackend> {
    lane: LaneId,
    vertex: Buffer<B>,
    index: Buffer<B>,
    camera: Buffer<B>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
    group: BindGroup<B>,
    draw_data: Option<Buffer<B>>,
    draw_group: Option<BindGroup<B>>,
    pipeline: RasterPipeline<B>,
    source_vertices: Vec<u8>,
    source_indices: Vec<u8>,
    uploaded: bool,
    dynamic_uniforms: bool,
    dynamic_uniform_alignment: u64,
}
impl<B: PushBackend> PushWorkload<B> {
    async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let dynamic_uniforms = !d.capabilities().push_constants
            || d.capabilities().limits.max_push_constant_bytes < PUSH_BYTES;
        let dynamic_uniform_alignment = d.capabilities().limits.uniform_offset_alignment.max(256);
        // Use the C++ sample's glTF sphere instead of a placeholder octahedron.
        let sphere = gltf::load_embedded_first_mesh(SPHERE)
            .map_err(|error| Error::Backend(error.to_string()))?;
        let vertex = buffer(
            d,
            "05 sphere vertices",
            sphere.vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        )?;
        let index = buffer(
            d,
            "05 sphere indices",
            sphere.indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        )?;
        let camera = buffer(
            d,
            "05 camera",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let depth = d.create_texture(&TextureDesc {
            label: Some("05 depth".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
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
        let group_layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("05 camera layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 128,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let entries = [BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: &camera,
                range: 0..256,
            },
        }];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("05 camera group"),
            layout: &group_layout,
            entries: &entries,
        })?;
        let draw_layout = dynamic_uniforms
            .then(|| {
                d.create_bind_group_layout(&BindGroupLayoutDesc {
                    label: Some("05 WebGPU draw-data layout".into()),
                    entries: vec![BindGroupLayoutEntry {
                        binding: 0,
                        visibility: vec![ShaderStage::Vertex],
                        ty: BindingType::UniformBuffer {
                            dynamic_offset: true,
                            min_size: PUSH_BYTES as u64,
                        },
                        count: BindingCount::One,
                        partially_bound: false,
                        nonuniform_indexing: false,
                    }],
                })
            })
            .transpose()?;
        let draw_data = if dynamic_uniforms {
            Some(buffer(
                d,
                "05 WebGPU draw data",
                dynamic_uniform_alignment
                    .checked_mul(SPHERE_COUNT as u64)
                    .ok_or(Error::InvalidRange)?,
                BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
            )?)
        } else {
            None
        };
        let draw_group = match (&draw_layout, &draw_data) {
            (Some(layout), Some(data)) => Some(d.create_bind_group(&BindGroupDesc {
                label: Some("05 WebGPU draw-data group"),
                layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: data,
                        range: 0..PUSH_BYTES as u64,
                    },
                }],
            })?),
            (None, None) => None,
            _ => return Err(Error::InvalidState),
        };
        let constants = (!dynamic_uniforms).then_some(PushConstantRange {
            stages: vec![ShaderStage::Vertex],
            range: 0..PUSH_BYTES,
        });
        let groups = match &draw_layout {
            Some(draw_layout) => vec![&group_layout, draw_layout],
            None => vec![&group_layout],
        };
        let layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("05 layout"),
            groups: &groups,
            push_constants: constants.as_slice(),
        })?;
        let pipeline = pipeline(d, layout, format, dynamic_uniforms).await?;
        Ok(Self {
            lane,
            vertex,
            index,
            camera,
            _depth: depth,
            depth_view,
            group,
            draw_data,
            draw_group,
            pipeline,
            source_vertices: sphere.vertices,
            source_indices: sphere.indices,
            uploaded: false,
            dynamic_uniforms,
            dynamic_uniform_alignment,
        })
    }
    fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<B>> {
        let mut encoder = d.encoder(self.lane)?;
        if !self.uploaded {
            encoder.write_buffer(&self.vertex, 0, &self.source_vertices)?;
            encoder.write_buffer(&self.index, 0, &self.source_indices)?;
            self.uploaded = true;
        }
        encoder.write_buffer(&self.camera, 0, &camera_bytes(extent))?;
        if let Some(draw_data) = &self.draw_data {
            for (index, sphere) in sphere_ring().into_iter().enumerate() {
                let offset = self
                    .dynamic_uniform_alignment
                    .checked_mul(index as u64)
                    .ok_or(Error::InvalidRange)?;
                encoder.write_buffer(draw_data, offset, &sphere.bytes())?;
            }
        }
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.025, 1.0])),
            store: StoreOp::Store,
        })];
        let pass = RasterPassDesc {
            label: Some("05 push constant spheres"),
            colors: &colors,
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
        };
        if self.dynamic_uniforms {
            encode_sphere_ring_dynamic_uniform(
                &mut encoder,
                &pass,
                &self.pipeline,
                &self.group,
                self.draw_group.as_ref().ok_or(Error::InvalidState)?,
                self.dynamic_uniform_alignment,
                &self.vertex,
                self.source_vertices.len() as u64,
                &self.index,
                self.source_indices.len() as u64,
                (self.source_indices.len() / 4) as u32,
            )?;
        } else {
            encode_sphere_ring(
                d,
                &mut encoder,
                &pass,
                &self.pipeline,
                &self.group,
                &self.vertex,
                self.source_vertices.len() as u64,
                &self.index,
                self.source_indices.len() as u64,
                (self.source_indices.len() / 4) as u32,
                &sphere_ring(),
            )?;
        }
        encoder.finish()
    }
}
fn buffer<B: PushBackend>(
    d: &Device<B>,
    label: &'static str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    d.create_buffer(&BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}
async fn shader<B: PushBackend>(
    d: &Device<B>,
    dxil: &[u8],
    spirv: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    common::native_shader_with_wgsl(d, "05 shader", entry, stage, dxil, spirv, WGSL).await
}
async fn pipeline<B: PushBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    dynamic_uniforms: bool,
) -> Result<RasterPipeline<B>> {
    let vs = shader(d, VERTEX_DXIL, VERTEX_SPIRV, "vs_main", ShaderStage::Vertex).await?;
    let ps = shader(
        d,
        FRAGMENT_DXIL,
        FRAGMENT_SPIRV,
        "fs_main",
        ShaderStage::Fragment,
    )
    .await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some(
            if dynamic_uniforms {
                "05 WebGPU dynamic-uniform pipeline"
            } else {
                "05 push pipeline"
            }
            .into(),
        ),
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
            stride: gltf::VKGLTF_VERTEX_STRIDE as u64,
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
                VertexAttribute {
                    location: 2,
                    offset: 32,
                    format: VertexFormat::Float32(4),
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
fn camera_bytes(extent: [u32; 2]) -> [u8; 256] {
    // The source uses a 60° camera at z=-10 and a uniform 0.5 model scale.
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / 30f32.to_radians().tan();
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
        256. / 255.9,
        1.,
        0.,
        0.,
        -25.6 / 255.9,
        0.,
    ];
    let view = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 10., 1.,
    ];
    let scale: [f32; 16] = [
        0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 1.,
    ];
    let projection_view = multiply(projection, view);
    let mut out = [0; 256];
    for (index, value) in projection_view.into_iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (index, value) in scale.into_iter().enumerate() {
        out[64 + index * 4..68 + index * 4].copy_from_slice(&value.to_le_bytes());
    }
    out
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

#[cfg(not(windows))]
fn main() {
    eprintln!("05_push_constants requires Windows and DX12");
}
