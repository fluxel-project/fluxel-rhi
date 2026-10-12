//! Typed offscreen render, texture copy, and completed readback screenshot path.
//!
//! Surface leases are present-only. This example renders into an ordinary RGBA8
//! texture, transitions it to a copy source, and maps a readback buffer only
//! after the caller supplies completion evidence for that encoder.
mod common;

use fluxel_rhi::api::*;
use std::{
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

const VS: &[u8] = include_bytes!("assets/12_screenshot/triangle_vs.dxil");
const PS: &[u8] = include_bytes!("assets/12_screenshot/triangle_ps.dxil");
static NEXT_CAPTURE_PLAN: AtomicU64 = AtomicU64::new(1);

pub trait ScreenshotBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + CommandBackend
    + RasterBackend
    + DeferredBackend
    + ReadbackBackend
{
}
impl<T> ScreenshotBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + CommandBackend
        + RasterBackend
        + DeferredBackend
        + ReadbackBackend
{
}

/// A submitted record and its optional readback ticket. Map the ticket only
/// with a completion point which covers this exact `work` submission.
pub struct ScreenshotRecord<B: ScreenshotBackend> {
    pub work: RecordedWork<B>,
    pub ticket: Option<ReadbackTicket<B>>,
}

pub struct Screenshot<B: ScreenshotBackend> {
    lane: LaneId,
    extent: [u32; 2],
    vertices: Buffer<B>,
    target: Texture<B>,
    target_view: TextureView<B>,
    readback: Buffer<B>,
    pipeline: RasterPipeline<B>,
    layout: ImageLayout,
    row_pitch: u32,
}
impl<B: ScreenshotBackend> Screenshot<B> {
    pub async fn new(device: &Device<B>, lane: LaneId, extent: [u32; 2]) -> Result<Self> {
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let target_desc = TextureDesc {
            label: Some("12 screenshot render target".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::COPY_SRC),
            view_formats: vec![],
        };
        match device.texture_support(&target_desc) {
            Support::Supported(_) => {}
            Support::Unsupported(reason) => return Err(Error::Unsupported(reason)),
        }
        let target = device.create_texture(&target_desc)?;
        let target_view = device.create_view(
            &target,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let row_pitch = aligned_row_pitch(extent[0])?;
        let readback = device.create_buffer(&BufferDesc {
            label: Some("12 screenshot readback".into()),
            size: u64::from(row_pitch) * u64::from(extent[1]),
            usage: BufferUsage::COPY_DST.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })?;
        let vertices = device.create_buffer(&BufferDesc {
            label: Some("12 screenshot triangle".into()),
            size: 72,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("12 screenshot layout"),
            groups: &[],
            push_constants: &[],
        })?;
        let pipeline = pipeline(device, layout).await?;
        Ok(Self {
            lane,
            extent,
            vertices,
            target,
            target_view,
            readback,
            pipeline,
            layout: ImageLayout::Common,
            row_pitch,
        })
    }

    /// Render the deterministic color triangle. `capture` records texture copy
    /// plus native readback at the tail of this encoder.
    pub fn record(&mut self, device: &Device<B>, capture: bool) -> Result<ScreenshotRecord<B>> {
        let mut encoder = device.encoder(self.lane)?;
        encoder.write_buffer(&self.vertices, 0, &triangle())?;
        self.transition(
            &mut encoder,
            ImageLayout::ColorAttachment,
            TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
        )?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&self.target_view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 0.0])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("12 screenshot scene"),
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: self.extent[0],
                height: self.extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            secondary_inheritance: None,
        })?;
        pass.set_pipeline(&self.pipeline)?;
        pass.set_vertex_buffer(0, &self.vertices, 0..72)?;
        pass.draw(0..3, 0..1)?;
        pass.finish()?;
        let ticket = if capture {
            self.transition(
                &mut encoder,
                ImageLayout::CopySrc,
                TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            )?;
            Some(encode_screenshot(
                &mut encoder,
                &self.target,
                self.extent,
                &self.readback,
                aligned_row_pitch(self.extent[0])?,
            )?)
        } else {
            None
        };
        Ok(ScreenshotRecord {
            work: encoder.finish()?,
            ticket,
        })
    }

    fn transition(
        &mut self,
        encoder: &mut CommandEncoder<B>,
        next: ImageLayout,
        after: TextureUses,
    ) -> Result<()> {
        let before = match self.layout {
            ImageLayout::Common => TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            ImageLayout::ColorAttachment => TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
            ImageLayout::CopySrc => TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            _ => return Err(Error::InvalidState),
        };
        encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &self.target,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: before,
                to: after,
            },
            layout: Some((self.layout, next)),
            ownership: None,
        })])?;
        self.layout = next;
        Ok(())
    }

    /// Waits for `completion` and writes the original demo's binary PPM image.
    /// The safety precondition of `read_after` is fulfilled by the caller only
    /// when `completion` covers the record holding this ticket.
    pub async fn save_ppm_after(
        &self,
        device: &Device<B>,
        ticket: ReadbackTicket<B>,
        completion: &CompletionPoint<B>,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let data = unsafe { device.read_after(ticket, completion).await? };
        // ReadbackBackend maps a buffer; its row geometry is established by
        // the explicit TexelCopyBufferLayout recorded above.
        let ppm = ppm_rgb(&data.bytes, self.row_pitch as usize, self.extent, false)?;
        fs::write(path, ppm).map_err(|e| Error::Backend(e.to_string()))
    }
}

/// Records the explicit texture-to-buffer copy used by the screenshot demo.
pub fn encode_screenshot<B>(
    encoder: &mut CommandEncoder<B>,
    rendered_frame: &Texture<B>,
    extent: [u32; 2],
    readback: &Buffer<B>,
    row_pitch: u32,
) -> Result<ReadbackTicket<B>>
where
    B: ReadbackBackend + DeferredBackend,
{
    if extent.contains(&0) || row_pitch < extent[0].checked_mul(4).ok_or(Error::InvalidRange)? {
        return Err(Error::InvalidRange);
    }
    encoder.texture_to_buffer(
        TextureRegion {
            texture: rendered_frame,
            mip: 0,
            origin: [0, 0, 0],
            extent: [extent[0], extent[1], 1],
            aspect: TextureAspects::COLOR,
        },
        readback,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: row_pitch,
            rows_per_image: extent[1],
        },
    )?;
    encoder.readback_buffer(
        readback,
        0..u64::from(row_pitch)
            .checked_mul(u64::from(extent[1]))
            .ok_or(Error::InvalidRange)?,
    )
}

/// Writes an RGBA or BGRA readback to binary PPM, stripping D3D12 row padding.
pub fn ppm_rgb(bytes: &[u8], row_pitch: usize, extent: [u32; 2], bgra: bool) -> Result<Vec<u8>> {
    let row_bytes = (extent[0] as usize)
        .checked_mul(4)
        .ok_or(Error::InvalidRange)?;
    let height = extent[1] as usize;
    if row_pitch < row_bytes
        || bytes.len() < row_pitch.checked_mul(height).ok_or(Error::InvalidRange)?
    {
        return Err(Error::InvalidRange);
    }
    let mut ppm = format!("P6\n{}\n{}\n255\n", extent[0], extent[1]).into_bytes();
    for row in bytes.chunks_exact(row_pitch).take(height) {
        for px in row[..row_bytes].chunks_exact(4) {
            if bgra {
                ppm.extend_from_slice(&[px[2], px[1], px[0]]);
            } else {
                ppm.extend_from_slice(&px[..3]);
            }
        }
    }
    Ok(ppm)
}
fn aligned_row_pitch(width: u32) -> Result<u32> {
    width
        .checked_mul(4)
        .and_then(|n| n.checked_add(255))
        .map(|n| n & !255)
        .ok_or(Error::InvalidRange)
}
async fn pipeline<B: ScreenshotBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
) -> Result<RasterPipeline<B>> {
    let shader = |bytes: &[u8], stage: ShaderStage, name: &str| ShaderDesc {
        label: Some(format!("12 {name}")),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vs = device
        .shader(shader(VS, ShaderStage::Vertex, "vs_main"))
        .await?;
    let ps = device
        .shader(shader(PS, ShaderStage::Fragment, "ps_main"))
        .await?;
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some("12 screenshot pipeline".into()),
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
                mask: u64::from(u32::MAX),
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
}
fn triangle() -> [u8; 72] {
    let values = [
        0.0_f32, 0.65, 0.0, 1.0, 0.15, 0.1, -0.7, -0.6, 0.0, 0.1, 1.0, 0.25, 0.7, -0.6, 0.0, 0.2,
        0.35, 1.0,
    ];
    let mut bytes = [0; 72];
    for (i, v) in values.into_iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }
    bytes
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
    let requested = std::env::var_os("FLUXEL_RHI_SCREENSHOT").is_some();
    let mut screenshot: Option<Screenshot<Dx12>> = None;
    let mut saved = false;
    let result = common::run_example(
        common::windows_api::Dx12Host,
        "Screenshot readback",
        config,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if screenshot.is_none() {
                screenshot = Some(common::block_on(Screenshot::new(device, lane, extent))?);
            }
            let workload = screenshot.as_mut().expect("initialized above");
            if workload.extent != extent {
                return Err(Error::Unsupported("screenshot resize"));
            }
            let capture = requested && !saved;
            let record = workload.record(device, capture)?;
            let completion = submit_capture_and_present(device, lane, record.work, image)?;
            if let Some(ticket) = record.ticket {
                common::block_on(device.wait_completion(&completion))?;
                common::block_on(workload.save_ppm_after(
                    device,
                    ticket,
                    &completion,
                    "screenshot.ppm",
                ))?;
                saved = true;
            }
            Ok(())
        },
    );
    if let Err(error) = result {
        eprintln!("12_screenshot: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(windows)]
fn submit_capture_and_present<B: DeferredBackend>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<CompletionPoint<B>> {
    if lane.device != device.id() || image.device() != device.id() {
        return Err(Error::WrongDevice);
    }
    let id = PlanId {
        device: device.id(),
        ordinal: NEXT_CAPTURE_PLAN.fetch_add(1, Ordering::Relaxed),
    };
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![StepWait {
                        previous: fluxel_rhi::api::StepId { plan: id, index: 0 },
                        destination: AccessScope {
                            stages: StageMask::ALL_COMMANDS,
                            accesses: AccessMask::ALL,
                        },
                    }],
                },
            ]),
        )
    };
    let result = unsafe {
        device.queue().submit(SubmitPacket {
            steps: vec![
                BoundStep {
                    step: plan.step(0),
                    payload: StepPayload::Commands(vec![work]),
                },
                BoundStep {
                    step: plan.step(1),
                    payload: StepPayload::Present(PresentBinding::new(image)),
                },
            ],
            plan,
            completion: CompletionRequest::End,
        })
    };
    match result {
        SubmitOutcome::Accepted(receipt)
            if receipt.accepted.len() == 2
                && receipt.presents.len() == 1
                && receipt.presents[0].accepted =>
        {
            receipt
                .checkpoints
                .into_iter()
                .next()
                .map(|checkpoint| checkpoint.point)
                .ok_or(Error::InvalidState)
        }
        SubmitOutcome::Accepted(_) => Err(Error::InvalidState),
        SubmitOutcome::Rejected { error, .. }
        | SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => Err(error),
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("12_screenshot requires a backend implementing ReadbackBackend");
}
