//! Runtime RGBA8 mip-chain generation through native linear blits.
#[path = "../utils/gltf.rs"]
mod gltf;
mod scene;
#[path = "../utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;
use scene::Tunnel;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
mod immediate;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
fn main_gl() {
    if let Err(error) = immediate::run() {
        eprintln!("09_texture_mipmap: {error:?}");
        std::process::exit(1);
    }
}

pub(crate) const MODEL: &[u8] = include_bytes!("../assets/models/tunnel_cylinder.gltf");
pub(crate) const IMAGE: &[u8] = include_bytes!("../assets/textures/metalplate_nomips_rgba.ktx");
pub(crate) const VS: &[u8] = include_bytes!("../assets/09_texture_mipmap/tunnel.vs.dxil");
pub(crate) const PS: &[u8] = include_bytes!("../assets/09_texture_mipmap/tunnel.ps.dxil");
pub(crate) const VS_SPIRV: &[u8] = include_bytes!("../assets/09_texture_mipmap/tunnel.vs.spv");
pub(crate) const PS_SPIRV: &[u8] = include_bytes!("../assets/09_texture_mipmap/tunnel.ps.spv");
pub(crate) const WGSL: &[u8] = include_bytes!("../assets/09_texture_mipmap/tunnel.wgsl");

/// Draws the textured tunnel into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_texture_mipmap(
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
        let mut tunnel = Tunnel::<WebGpu>::new(&device, lane, format, extent).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let work = tunnel.record(&device, &image, extent)?;
            utils::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Runs the shared tunnel and generated mip chain on an immediate WebGL2 context.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_texture_mipmap(
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
        let mut tunnel = Tunnel::<Gl>::new(&device, lane, format, extent).await?;
        for _ in 0..frames {
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
            // SAFETY: this session and its surface lease stay on the context
            // owner thread until the present step consumes the lease.
            let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
            let mut commands = session.begin_commands().map_err(|outcome| match outcome {
                ImmediateRunOutcome::Issued(_) => Error::InvalidState,
                ImmediateRunOutcome::Failed { error, .. }
                | ImmediateRunOutcome::Indeterminate { error, .. } => error,
            })?;
            let commit = tunnel.encode(commands.commands(), &device, &image, extent)?;
            match commands.close() {
                ImmediateRunOutcome::Issued(_) => tunnel.commit_frame(commit),
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
fn run<B: scene::MipmapBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut demo: Option<Tunnel<B>> = None;
    utils::run_example(
        host,
        "GPU generated mipmaps",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if demo.is_none() {
                demo = Some(utils::block_on(Tunnel::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let work = demo
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent)?;
            utils::submit_and_present(device, lane, work, image)
        },
    )
}
#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    if let Err(error) = run::<fluxel_rhi::metal::Metal, _>(utils::metal_api::MetalHost) {
        eprintln!("09_texture_mipmap: {error:?}");
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
        eprintln!("09_texture_mipmap: {error:?}");
        std::process::exit(1);
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
    eprintln!("09_texture_mipmap requires a supported native or browser backend");
}
