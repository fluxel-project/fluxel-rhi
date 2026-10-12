//! Indexed triangle through Fluxel's statically typed API RHI.
//!
//! The DX12 host chooses `Dx12` once outside this module. Resources, pipeline
//! creation and command recording below are all generic over the fixed backend.

mod common;

use std::sync::Arc;

use fluxel_rhi::api::*;

// These checked-in DXIL blobs are compiled from assets/01_triangle/triangle.hlsl.
const TRIANGLE_VS: &[u8] = include_bytes!("assets/01_triangle/triangle_vs.dxil");
const TRIANGLE_PS: &[u8] = include_bytes!("assets/01_triangle/triangle_ps.dxil");

/// Requirements for the raster-only baseline.
pub trait TriangleBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> TriangleBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
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
        let vertex_shader = device
            .shader(ShaderDesc {
                label: Some("triangle vertex shader".into()),
                format: ShaderFormat::Dxil,
                bytes: Arc::<[u8]>::from(TRIANGLE_VS),
                entries: vec![ShaderEntryPoint {
                    name: "vs_main".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await?;
        let fragment_shader = device
            .shader(ShaderDesc {
                label: Some("triangle fragment shader".into()),
                format: ShaderFormat::Dxil,
                bytes: Arc::<[u8]>::from(TRIANGLE_PS),
                entries: vec![ShaderEntryPoint {
                    name: "ps_main".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
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

    /// Records the commands for one acquired surface lease. The caller owns the
    /// execution skeleton, explicit transitions and eventual plan submission.
    pub fn record(
        &self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<B>> {
        let mut encoder = device.encoder(self.lane)?;
        encoder.write_buffer(&self.vertex, 0, &triangle_vertices())?;
        encoder.write_buffer(&self.index, 0, &triangle_indices())?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.02, 0.02, 0.03, 1.0])),
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
        pass.finish()?;
        encoder.finish()
    }
}

fn triangle_vertices() -> [u8; 72] {
    let values = [
        0.0_f32, 0.5, 0.0, 1.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0,
        1.0,
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
fn main() {
    use fluxel_rhi::dx12::Dx12;

    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut triangle: Option<Triangle<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
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
                triangle = Some(common::block_on(Triangle::new(
                    device,
                    lane,
                    config.format,
                ))?);
            }
            let work = triangle
                .as_ref()
                .expect("initialized above")
                .record(device, &image, extent)?;
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("01_triangle: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("01_triangle requires Windows and DX12");
}
