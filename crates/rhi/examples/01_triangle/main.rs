//! Indexed triangle through Fluxel's statically typed API RHI.
//!
//! The DX12 host chooses `Dx12` once outside this module. Resources, pipeline
//! creation and command recording below are all generic over the fixed backend.
//! When built with `native-gl-wgl`, the `gl` selection instead creates one WGL
//! context and issues the same frame through an immediate session.

#[path = "../utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;

// These checked-in DXIL/SPIR-V artifacts are generated from triangle.wgsl.
const TRIANGLE_VS: &[u8] = include_bytes!("../assets/01_triangle/triangle_vs.dxil");
const TRIANGLE_PS: &[u8] = include_bytes!("../assets/01_triangle/triangle_ps.dxil");
const TRIANGLE_VS_SPV: &[u8] = include_bytes!("../assets/01_triangle/triangle_vs.spv");
const TRIANGLE_PS_SPV: &[u8] = include_bytes!("../assets/01_triangle/triangle_ps.spv");
const TRIANGLE_WGSL: &[u8] = include_bytes!("../assets/01_triangle/triangle.wgsl");

/// Requirements for the raster-only baseline.
pub trait TriangleBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
impl<T> TriangleBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}

pub struct Triangle<B: TriangleBackend> {
    vertex: Buffer<B>,
    index: Buffer<B>,
    pipeline: RasterPipeline<B>,
    lane: LaneId,
}

impl<B: TriangleBackend> Triangle<B> {
    pub async fn new(device: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        let vertex = device.create_buffer(&BufferDesc {
            label: Some("triangle vertices".into()),
            size: 3 * 6 * 4,
            usage: BufferUsage(BufferUsage::VERTEX.0 | BufferUsage::COPY_DST.0),
            memory: MemoryClass::Device,
        })?;
        let index = device.create_buffer(&BufferDesc {
            label: Some("triangle indices".into()),
            size: 3 * 2,
            usage: BufferUsage(BufferUsage::INDEX.0 | BufferUsage::COPY_DST.0),
            memory: MemoryClass::Device,
        })?;
        let vertex_shader = utils::native_shader_with_wgsl(
            device,
            "triangle vertex shader",
            "vs_main",
            ShaderStage::Vertex,
            TRIANGLE_VS,
            TRIANGLE_VS_SPV,
            TRIANGLE_WGSL,
        )
        .await?;
        let fragment_shader = utils::native_shader_with_wgsl(
            device,
            "triangle fragment shader",
            "ps_main",
            ShaderStage::Fragment,
            TRIANGLE_PS,
            TRIANGLE_PS_SPV,
            TRIANGLE_WGSL,
        )
        .await?;
        let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("triangle layout"),
            groups: &[],
            push_constants: &[],
        })?;
        let pipeline = device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("triangle pipeline".into()),
                layout,
                vertex: PipelineStage {
                    module: vertex_shader,
                    entry: "vs_main".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: fragment_shader,
                    entry: "ps_main".into(),
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
                    write_mask: 0xf,
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
                    mask: u64::from(u32::MAX),
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await?;
        Ok(Self {
            vertex,
            index,
            pipeline,
            lane,
        })
    }

    /// Issues the common triangle scene to an open encoder.
    ///
    /// Deferred backends finish this encoder into `RecordedWork`; GL calls the
    /// same scene while an immediate session is open.
    pub fn encode(
        &self,
        encoder: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<()> {
        encoder.write_buffer(&self.vertex, 0, &triangle_vertices())?;
        encoder.write_buffer(&self.index, 0, &triangle_indices())?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.2, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("triangle"),
            colors: &colors,
            depth_stencil: None,
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
        pass.set_vertex_buffer(0, &self.vertex, 0..72)?;
        pass.set_index_buffer(&self.index, 0..6, IndexFormat::Uint16)?;
        pass.draw_indexed(0..3, 0, 0..1)?;
        pass.finish()
    }

    /// Records the commands for one acquired surface lease. The caller owns the
    /// execution skeleton, explicit transitions and eventual plan submission.
    pub fn record(
        &self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<B>>
    where
        B: DeferredBackend,
    {
        let mut encoder = device.encoder(self.lane)?;
        self.encode(&mut encoder, image, extent)?;
        encoder.finish()
    }
}

/// Draws the same typed example into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_triangle(
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
        let triangle = Triangle::<WebGpu>::new(&device, lane, format).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let work = triangle.record(&device, &image, extent)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

fn triangle_vertices() -> [u8; 72] {
    // C++ source vertices are `(1,1)` red, `(-1,1)` green and `(0,-1)`
    // blue, viewed from z=-2.5 with a 60° projection.  Bake that fixed
    // source-camera transform here so this minimal no-uniform example keeps
    // its indexed-triangle focus while matching the reference composition.
    // WebGPU/DX12 framebuffer coordinates put +NDC Y at the top, hence the
    // source Vulkan projection's Y inversion is represented in these values.
    let values = [
        0.3897_f32, -0.6928, 0.0, 1.0, 0.0, 0.0, -0.3897, -0.6928, 0.0, 0.0, 1.0, 0.0, 0.0, 0.6928,
        0.0, 0.0, 0.0, 1.0,
    ];
    let mut bytes = [0; 72];
    for (index, value) in values.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn triangle_indices() -> [u8; 6] {
    [0, 0, 1, 0, 2, 0]
}

#[cfg(windows)]
fn run<B: TriangleBackend + DeferredBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut triangle: Option<Triangle<B>> = None;
    utils::run_example(
        host,
        "Basic indexed triangle",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if triangle.is_none() {
                triangle = Some(utils::block_on(Triangle::new(device, lane, config.format))?);
            }
            let work = triangle
                .as_ref()
                .expect("initialized above")
                .record(device, &image, extent)?;
            utils::submit_and_present(device, lane, work, image)
        },
    )
}

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
#[path = "immediate.rs"]
mod immediate;

/// Draws the shared WGSL triangle through the immediate WebGL2 backend.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_triangle(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    immediate::run_browser(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|args| args[0] == "--backend" && args[1] == "gl")
    {
        if let Err(error) = immediate::run() {
            eprintln!("01_triangle: {error:?}");
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
        eprintln!("01_triangle: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("01_triangle requires Windows and DX12");
}
