//! Push-constant sphere ring through the RHI.
//!
//! The original Vulkan workload issues sixteen indexed sphere draws.  Every
//! draw writes its own 32-byte `{ color, position }` record to the vertex stage;
//! it is deliberately not replaced with instance data or a dynamic uniform.
//! The native host owns surface lifetime and submission; this module only owns
//! scene resources and command construction.

#[path = "../utils/gltf.rs"]
mod gltf;
#[path = "../utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;

pub const SPHERE_COUNT: usize = 16;
pub const PUSH_BYTES: u32 = 32;

/// The native DXIL source for the b0 camera / b1 root-constant ABI.
pub const VERTEX_DXIL: &[u8] = include_bytes!("../assets/05_push_constants/push_constants.vs.dxil");
pub const FRAGMENT_DXIL: &[u8] =
    include_bytes!("../assets/05_push_constants/push_constants.ps.dxil");
pub const VERTEX_SPIRV: &[u8] = include_bytes!("../assets/05_push_constants/push_constants.vs.spv");
pub const FRAGMENT_SPIRV: &[u8] =
    include_bytes!("../assets/05_push_constants/push_constants.ps.spv");
const WGSL: &[u8] = include_bytes!("../assets/05_push_constants/push_constants.wgsl");
const SPHERE: &[u8] = include_bytes!("../assets/models/sphere.gltf");

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
    B: RasterBackend,
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
            utils::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn run_host<B: PushBackend + DeferredBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
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
    utils::run_example(
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
                workload = Some(utils::block_on(PushWorkload::new(
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
            utils::submit_and_present(device, lane, work, image)
        },
    )
}

/// Runs the native Cargo example target.
#[cfg(windows)]
pub fn run() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .any(|args| args[0] == "--backend" && args[1] == "gl")
    {
        if let Err(error) = immediate::run() {
            eprintln!("05_push_constants: {error:?}");
            std::process::exit(1);
        }
        return;
    }
    let result: Result<()> = match utils::windows_api::selected_backend() {
        #[cfg(feature = "dx12")]
        Ok(utils::windows_api::SelectedBackend::Dx12) => {
            run_host::<fluxel_rhi::dx12::Dx12, _>(utils::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(utils::windows_api::SelectedBackend::Vulkan) => {
            run_host::<fluxel_rhi::vulkan::Vulkan, _>(utils::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("05_push_constants: {error:?}");
        std::process::exit(1);
    }
}

pub trait PushBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
impl<T> PushBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
mod immediate;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub use immediate::run_webgl2_push_constants;
mod pipeline;
mod workload;
use workload::PushWorkload;

/// Reports the unavailable native target on unsupported hosts.
#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
pub fn run() {
    if let Err(error) = immediate::run() {
        eprintln!("05_push_constants: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub fn run() {
    eprintln!("05_push_constants requires Windows and DX12");
}
