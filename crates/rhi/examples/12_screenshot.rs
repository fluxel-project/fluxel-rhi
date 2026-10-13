//! Typed offscreen render, texture copy, and completed readback screenshot path.
//!
//! Surface leases are present-only. This example renders into an ordinary RGBA8
//! texture, transitions it to a copy source, and maps a readback buffer only
//! after the caller supplies completion evidence for that encoder.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
mod screenshot {
    mod scene;
    pub(crate) use scene::DragonScene;
}

use fluxel_rhi::api::*;
use std::{
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

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
    target: Texture<B>,
    target_view: TextureView<B>,
    readback: Buffer<B>,
    scene: screenshot::DragonScene<B>,
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
        let scene = screenshot::DragonScene::new(device, extent).await?;
        Ok(Self {
            lane,
            extent,
            target,
            target_view,
            readback,
            scene,
            // An ordinary Vulkan image starts undefined.  Every later state
            // change is explicit in `transition` below.
            layout: ImageLayout::Undefined,
            row_pitch,
        })
    }

    /// Render the source Chinese-dragon scene twice: once into the ordinary
    /// texture used for capture and once to the visible surface.
    pub fn record(
        &mut self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        capture: bool,
    ) -> Result<ScreenshotRecord<B>> {
        let mut encoder = device.encoder(self.lane)?;
        self.transition(
            &mut encoder,
            ImageLayout::ColorAttachment,
            TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
        )?;
        self.scene
            .record_target(&mut encoder, &self.target_view, self.extent)?;
        self.scene
            .record_surface(&mut encoder, image, self.extent)?;
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
            ImageLayout::Undefined => TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
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
        // SAFETY: callers pass the checkpoint returned by the record that
        // produced `ticket`; it is awaited by `read_after` before mapping.
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
        eprintln!("12_screenshot: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(windows)]
fn run<B, H>(host: H) -> Result<()>
where
    B: ScreenshotBackend,
    H: common::ExampleHost<B>,
{
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
    let mut screenshot: Option<Screenshot<B>> = None;
    let mut saved = false;
    common::run_example(
        host,
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
            if screenshot
                .as_ref()
                .is_some_and(|workload| workload.extent != extent)
            {
                // The old offscreen target, depth images, and readback buffer
                // may still be used by a submitted frame.  This standalone
                // example has no frame-retirement policy, so wait before
                // replacing its extent-owned resources.
                common::block_on(device.wait_idle())?;
                screenshot = Some(common::block_on(Screenshot::new(device, lane, extent))?);
            }
            let workload = screenshot.as_mut().expect("initialized above");
            let capture = requested && !saved;
            let record = workload.record(device, &image, capture)?;
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
    )
}

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
    // SAFETY: the two steps share one device-local lane and present explicitly
    // waits for commands, so this compiled order is realizable by that queue.
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
    // SAFETY: the caller retains screenshot resources and the acquired image
    // until the returned end checkpoint completes; every bound step is from
    // this plan and device.
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

/// Renders the visible dragon and verifies one native GPU-to-CPU screenshot.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_screenshot(
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
        let mut screenshot = Screenshot::<WebGpu>::new(&device, lane, extent).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let record = screenshot.record(&device, &image, frame == 0)?;
            let completion = submit_capture_and_present(&device, lane, record.work, image)?;
            device.wait_completion(&completion).await?;
            if let Some(ticket) = record.ticket {
                // SAFETY: `completion` is the end checkpoint returned for
                // `record` and was awaited above, so it covers this ticket.
                let data = unsafe { device.read_after(ticket, &completion).await? };
                let rgb = ppm_rgb(&data.bytes, screenshot.row_pitch as usize, extent, false)?;
                if !rgb.starts_with(b"P6\n") || rgb.len() <= 64 {
                    return Err(Error::InvalidState);
                }
            }
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}
#[cfg(not(windows))]
fn main() {
    eprintln!("12_screenshot requires a backend implementing ReadbackBackend");
}
