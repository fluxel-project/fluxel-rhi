//! SaschaWillems multisampled Voyager scene lowered through the RHI.
//!
//! The scene renders the original `voyager.gltf` mesh into a multisampled
//! color/depth pair and resolves the color attachment into the acquired frame.
//! The native host owns the window and surface lifecycle.
#[path = "../utils/gltf.rs"]
mod gltf;
#[path = "scene.rs"]
mod msaa;
#[path = "../utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;
pub use msaa::{Msaa, MsaaDeferredBackend};

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
#[path = "immediate.rs"]
mod immediate;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
fn main_gl() {
    if let Err(error) = immediate::run() {
        eprintln!("08_msaa: {error:?}");
        std::process::exit(1);
    }
}

pub(crate) const VOYAGER: &[u8] = include_bytes!("../assets/models/voyager.gltf");
pub(crate) const VS: &[u8] = include_bytes!("../assets/08_msaa/msaa.vs.dxil");
pub(crate) const PS: &[u8] = include_bytes!("../assets/08_msaa/msaa.ps.dxil");
pub(crate) const VS_SPIRV: &[u8] = include_bytes!("../assets/08_msaa/msaa.vs.spv");
pub(crate) const PS_SPIRV: &[u8] = include_bytes!("../assets/08_msaa/msaa.ps.spv");

/// Draws the multisampled Voyager into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_msaa(
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
        let mut demo = Msaa::<WebGpu>::new(&device, lane, format, extent).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let work = demo.record(&device, &image, extent, frame as f32 / 60.0)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Runs the Voyager scene without MSAA when the WebGL2 profile cannot expose
/// the Core multisample-texture attachment shape.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_msaa_fallback(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::gl::{Gl, GlInstance};
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);
    let run = async {
        if frames == 0 || canvas.width() == 0 || canvas.height() == 0 {
            return Err(Error::InvalidRange);
        }
        let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
        // SAFETY: the instance retains this canvas-backed context, and all
        // immediate commands below execute on the browser owner thread.
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
        let mut demo = Msaa::<Gl>::new_single_sample(&device, lane, format, extent).await?;
        for frame in 0..frames {
            let image = surface.acquire().await?;
            let ordinal = NEXT_PLAN
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| Error::OutOfMemory)?;
            // SAFETY: command and present steps target one immediate GL lane
            // and are issued in this exact order on the canvas owner thread.
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
            // SAFETY: this session and its acquired image remain on the GL
            // owner thread until the immediate present consumes the lease.
            let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
            let mut commands = session.begin_commands().map_err(|outcome| match outcome {
                ImmediateRunOutcome::Issued(_) => Error::InvalidState,
                ImmediateRunOutcome::Failed { error, .. }
                | ImmediateRunOutcome::Indeterminate { error, .. } => error,
            })?;
            let commit = demo.encode(
                commands.commands(),
                &device,
                &image,
                extent,
                frame as f32 / 60.0,
            )?;
            match commands.close() {
                ImmediateRunOutcome::Issued(_) => demo.commit_frame(commit),
                ImmediateRunOutcome::Failed { error, .. }
                | ImmediateRunOutcome::Indeterminate { error, .. } => return Err(error),
            }
            match session.present(PresentBinding::new(image)) {
                ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
                ImmediatePresentResult::Run(ImmediateRunOutcome::Failed { error, .. })
                | ImmediatePresentResult::Run(ImmediateRunOutcome::Indeterminate {
                    error, ..
                })
                | ImmediatePresentResult::Rejected { error, .. } => return Err(error),
            }
            match session.finish() {
                ImmediateFinishOutcome::Issued(_) => {}
                ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Failed { error, .. })
                | ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Indeterminate {
                    error,
                    ..
                }) => return Err(error),
                ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Issued(_))
                | ImmediateFinishOutcome::AlreadyReported => return Err(Error::InvalidState),
            }
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}
#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
fn run<B: MsaaDeferredBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut workload: Option<Msaa<B>> = None;
    let start = std::time::Instant::now();
    utils::run_example(
        host,
        "Multisample anti-aliasing",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if workload.is_none() {
                workload = Some(utils::block_on(Msaa::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let work = workload.as_mut().ok_or(Error::InvalidState)?.record(
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
        eprintln!("08_msaa: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|args| args[0] == "--backend" && args[1] == "gl")
    {
        main_gl();
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
        eprintln!("08_msaa: {error:?}");
        std::process::exit(1)
    }
}
#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    main_gl();
}

#[cfg(not(any(
    windows,
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", feature = "native-gles-egl")
)))]
fn main() {
    eprintln!("08_msaa requires a supported native or browser backend");
}
