//! Parallel recording of inherited, draw-only secondary raster work.
//!
//! Workers encode disjoint ranges of a shared triangle stream.  The host/main
//! thread writes that stream, opens the inherited primary pass, and executes
//! the completed secondary packets in a deterministic worker order.
mod common;

use fluxel_rhi::api::*;
use std::sync::Arc;

const VS: &[u8] = include_bytes!("assets/13_multithreaded_recording/triangles_vs.dxil");
const PS: &[u8] = include_bytes!("assets/13_multithreaded_recording/triangles_ps.dxil");
const TRIANGLES: u32 = 96;
const WORKERS: u32 = 8;
const VERTEX_BYTES: u64 = 24;

pub trait SecondaryExampleBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + CommandBackend
    + RasterBackend
    + DeferredBackend
    + SecondaryBackend
{
}
impl<T> SecondaryExampleBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + CommandBackend
        + RasterBackend
        + DeferredBackend
        + SecondaryBackend
{
}

pub fn require_parallel_secondary_recording<B: Backend>(device: &Device<B>) -> Result<()> {
    if !device.capabilities().secondary_commands {
        return Err(Error::Unsupported("secondary raster commands"));
    }
    Ok(())
}

pub struct Multithreaded<B: SecondaryExampleBackend> {
    lane: LaneId,
    vertices: Buffer<B>,
    pipeline: RasterPipeline<B>,
    inheritance: SecondaryRasterDesc,
}

impl<B: SecondaryExampleBackend> Multithreaded<B> {
    pub async fn new(device: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        require_parallel_secondary_recording(device)?;
        let vertices = device.create_buffer(&BufferDesc {
            label: Some("13 triangle stream".into()),
            size: u64::from(TRIANGLES) * 3 * VERTEX_BYTES,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("13 layout"),
            groups: &[],
            push_constants: &[],
        })?;
        let vs = device
            .shader(ShaderDesc {
                label: Some("13 VS".into()),
                format: ShaderFormat::Dxil,
                bytes: Arc::from(VS),
                entries: vec![ShaderEntryPoint {
                    name: "vs_main".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await?;
        let ps = device
            .shader(ShaderDesc {
                label: Some("13 PS".into()),
                format: ShaderFormat::Dxil,
                bytes: Arc::from(PS),
                entries: vec![ShaderEntryPoint {
                    name: "ps_main".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await?;
        let pipeline = device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("13 triangles".into()),
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
            lane,
            vertices,
            pipeline,
            inheritance: SecondaryRasterDesc {
                color_formats: vec![Some(format)],
                depth_stencil_format: None,
                samples: 1,
                multiview_mask: None,
            },
        })
    }

    pub fn record(
        &self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<B>>
    where
        B::NativeDevice: Send + Sync,
        B::NativeRasterPipeline: Send + Sync,
        B::NativeBuffer: Send + Sync,
        B::NativeSecondaryWork: Send,
    {
        let secondary = device
            .secondary()
            .ok_or(Error::Unsupported("secondary raster commands"))?;
        let chunk = TRIANGLES / WORKERS;
        let mut packets = Vec::with_capacity(WORKERS as usize);
        std::thread::scope(|scope| -> Result<()> {
            let mut jobs = Vec::with_capacity(WORKERS as usize);
            for worker in 0..WORKERS {
                let device = device.clone();
                let pipeline = self.pipeline.clone();
                let vertices = self.vertices.clone();
                let inheritance = self.inheritance.clone();
                let first = worker * chunk;
                let count = if worker + 1 == WORKERS {
                    TRIANGLES - first
                } else {
                    chunk
                };
                jobs.push(scope.spawn(move || -> Result<SecondaryRasterWork<B>> {
                    let domain = device
                        .secondary()
                        .ok_or(Error::Unsupported("secondary raster commands"))?;
                    let mut encoder = domain.begin(&inheritance)?;
                    encoder.set_pipeline(&pipeline)?;
                    encoder.viewport(Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: extent[0] as f32,
                        height: extent[1] as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    })?;
                    encoder.scissor(Rect {
                        x: 0,
                        y: 0,
                        width: extent[0],
                        height: extent[1],
                    })?;
                    encoder.set_vertex_buffer(
                        0,
                        &vertices,
                        0..u64::from(TRIANGLES) * 3 * VERTEX_BYTES,
                    )?;
                    encoder.draw(first * 3..(first + count) * 3, 0..1)?;
                    domain.finish(encoder)
                }));
            }
            for job in jobs {
                packets.push(job.join().map_err(|_| {
                    Error::Backend("13 secondary recording worker panicked".into())
                })??);
            }
            Ok(())
        })?;
        let mut encoder = device.encoder(self.lane)?;
        encoder.write_buffer(&self.vertices, 0, &triangle_stream())?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.005, 0.008, 0.02, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("13 primary execute"),
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
            secondary_inheritance: Some(self.inheritance.clone()),
        })?;
        for packet in &packets {
            secondary.execute(&mut pass, packet)?;
        }
        pass.finish()?;
        encoder.finish()
    }
}

fn triangle_stream() -> Vec<u8> {
    let mut bytes = Vec::with_capacity((TRIANGLES as usize) * 3 * VERTEX_BYTES as usize);
    for triangle in 0..TRIANGLES {
        let column = triangle % 12;
        let row = triangle / 12;
        let x = -0.94 + column as f32 * 0.17;
        let y = -0.82 + row as f32 * 0.23;
        let color = [
            0.25 + 0.75 * column as f32 / 11.0,
            0.2 + 0.8 * row as f32 / 7.0,
            0.9 - 0.6 * column as f32 / 11.0,
        ];
        for position in [[x, y, 0.0], [x + 0.12, y, 0.0], [x + 0.06, y + 0.17, 0.0]] {
            for value in position.into_iter().chain(color) {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
    }
    bytes
}

#[cfg(windows)]
fn main() {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    use fluxel_rhi::dx12::Dx12;
    let mut example: Option<Multithreaded<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
        "Parallel secondary raster recording",
        config.clone(),
        move |device, _, image, extent| {
            require_parallel_secondary_recording(device)?;
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if example.is_none() {
                example = Some(common::block_on(Multithreaded::new(
                    device,
                    lane,
                    config.format,
                ))?);
            }
            let work = example
                .as_ref()
                .expect("initialized above")
                .record(device, &image, extent)?;
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("13_multithreaded_recording: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!(
        "13_multithreaded_recording requires Windows and a backend with secondary raster commands"
    );
}
