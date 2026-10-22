//! Dynamic-uniform cube grid through the RHI.
//!
//! The second group binding is a real dynamically-offset uniform binding.  One
//! aligned model matrix is selected per indexed cube draw; no instance data is
//! substituted for the operation demonstrated here.
mod scene;
#[path = "../utils/mod.rs"]
mod utils;
use fluxel_rhi::api::*;
use scene::{camera_matrix, cube_indices, cube_vertices, model};
const VS: &[u8] = include_bytes!("../assets/04_dynamic_uniform_buffer/dynamic_uniform.vs.dxil");
const PS: &[u8] = include_bytes!("../assets/04_dynamic_uniform_buffer/dynamic_uniform.ps.dxil");
const VS_SPV: &[u8] = include_bytes!("../assets/04_dynamic_uniform_buffer/dynamic_uniform.vs.spv");
const PS_SPV: &[u8] = include_bytes!("../assets/04_dynamic_uniform_buffer/dynamic_uniform.ps.spv");
const WGSL: &[u8] = include_bytes!("../assets/04_dynamic_uniform_buffer/dynamic_uniform.wgsl");
pub const OBJECT_COUNT: u32 = 125;
pub const MODEL_BYTES: u64 = 64;
pub trait DynamicUniformBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
impl<T> DynamicUniformBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
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

impl<B: DynamicUniformBackend> DynamicUniforms<B> {
    pub async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
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
    /// Issues buffer uploads and all 125 dynamic-offset draws to an open
    /// encoder. GL runs this synchronously in its immediate session; deferred
    /// backends call it before producing `RecordedWork`.
    pub fn encode(
        &self,
        encoder: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<()> {
        encoder.write_buffer(&self.vertex, 0, &cube_vertices())?;
        encoder.write_buffer(&self.index, 0, &cube_indices())?;
        encoder.write_buffer(&self.camera, 0, &camera_matrix(extent))?;
        for n in 0..OBJECT_COUNT {
            encoder.write_buffer(
                &self.models,
                aligned(MODEL_BYTES, self.alignment) * u64::from(n),
                &model(n, time),
            )?
        }
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.02, 0.02, 0.03, 1.0])),
            store: StoreOp::Store,
        })];
        encoder.transition_resources(&[utils::discarded_depth(&self._depth, 0..1)])?;
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("04 dynamic uniforms"),
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
        })?;
        pass.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        pass.set_pipeline(&self.pipeline)?;
        pass.set_vertex_buffer(0, &self.vertex, 0..192)?;
        pass.set_index_buffer(&self.index, 0..144, IndexFormat::Uint32)?;
        for n in 0..OBJECT_COUNT {
            let offset = aligned(MODEL_BYTES, self.alignment) * u64::from(n);
            pass.set_bind_group(
                0,
                &self.group,
                &[u32::try_from(offset).map_err(|_| Error::InvalidRange)?],
            )?;
            pass.draw_indexed(0..36, 0, 0..1)?
        }
        pass.finish()
    }

    pub fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>>
    where
        B: DeferredBackend,
    {
        let mut e = d.encoder(self.lane)?;
        self.encode(&mut e, image, extent, time)?;
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
    utils::native_shader_with_wgsl(d, name, name, stage, dxil, spirv, WGSL).await
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
#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
fn run<
    B: DynamicUniformBackend + DeferredBackend + ExecutionBackend<Mode = DeferredExecution>,
    H: utils::ExampleHost<B>,
>(
    host: H,
) -> Result<()> {
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
    utils::run_example(
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
                workload = Some(utils::block_on(DynamicUniforms::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?)
            }
            let work = workload.as_ref().ok_or(Error::InvalidState)?.record(
                device,
                &image,
                extent,
                start.elapsed().as_secs_f32(),
            )?;
            utils::submit_and_present(device, lane, work, image)
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    if let Err(error) = run::<fluxel_rhi::metal::Metal, _>(utils::metal_api::MetalHost) {
        eprintln!("04_dynamic_uniform_buffer: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
#[path = "immediate.rs"]
mod immediate;

#[cfg(all(target_arch = "wasm32", any(feature = "webgpu", feature = "webgl2")))]
#[path = "browser_host.rs"]
mod browser;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub use browser::run_webgl2_dynamic_uniforms;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub use browser::run_webgpu_dynamic_uniforms;

#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|args| args[0] == "--backend" && args[1] == "gl")
    {
        if let Err(error) = immediate::run() {
            eprintln!("04_dynamic_uniform_buffer: {error:?}");
            std::process::exit(1);
        }
        return;
    }
    let result: Result<()> = match utils::windows_api::selected_backend() {
        #[cfg(feature = "dx12")]
        Ok(utils::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(utils::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(utils::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(utils::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("04_dynamic_uniform_buffer: {error:?}");
        std::process::exit(1)
    }
}
#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    if let Err(error) = immediate::run() {
        eprintln!("04_dynamic_uniform_buffer: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(
    windows,
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", feature = "native-gles-egl")
)))]
fn main() {
    eprintln!("04_dynamic_uniform_buffer requires Windows and DX12");
}
