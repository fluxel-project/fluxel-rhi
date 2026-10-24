//! SaschaWillems occlusion-query scene with native query pools.
//!
//! `Workload` owns the scene, query result storage, and command recording.
//! The platform host owns its surface lease and submission; completion evidence
//! crosses back into this module only to make the query readback safe.
#[path = "../utils/gltf.rs"]
mod gltf;
#[path = "../utils/mod.rs"]
mod utils;
use fluxel_rhi::api::*;
const PLANE: &[u8] = include_bytes!("../assets/models/plane_z.gltf");
const TEAPOT: &[u8] = include_bytes!("../assets/models/teapot.gltf");
const SPHERE: &[u8] = include_bytes!("../assets/models/sphere.gltf");
const VS: &[u8] = include_bytes!("../assets/11_occlusion_query/occlusion.vs.dxil");
const SIMPLE_PS: &[u8] = include_bytes!("../assets/11_occlusion_query/simple.ps.dxil");
const MESH_PS: &[u8] = include_bytes!("../assets/11_occlusion_query/mesh.ps.dxil");
const OCCLUDER_PS: &[u8] = include_bytes!("../assets/11_occlusion_query/occluder.ps.dxil");
const VS_SPIRV: &[u8] = include_bytes!("../assets/11_occlusion_query/occlusion.vs.spv");
const SIMPLE_PS_SPIRV: &[u8] = include_bytes!("../assets/11_occlusion_query/simple.ps.spv");
const MESH_PS_SPIRV: &[u8] = include_bytes!("../assets/11_occlusion_query/mesh.ps.spv");
const OCCLUDER_PS_SPIRV: &[u8] = include_bytes!("../assets/11_occlusion_query/occluder.ps.spv");
const WGSL: &[u8] = include_bytes!("../assets/11_occlusion_query/occlusion.wgsl");
pub trait OcclusionWorkloadBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + RasterBackend
    + CommandBackend
    + QueryBackend
    + ReadbackBackend
{
}
impl<T> OcclusionWorkloadBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + CommandBackend
        + QueryBackend
        + ReadbackBackend
{
}
pub trait OcclusionBackend: OcclusionWorkloadBackend + DeferredBackend {}
impl<T> OcclusionBackend for T where T: OcclusionWorkloadBackend + DeferredBackend {}
pub struct Workload<B: OcclusionWorkloadBackend> {
    lane: LaneId,
    attachments: Option<Attachments<B>>,
    plane_vertices: Buffer<B>,
    plane_indices: Buffer<B>,
    teapot_vertices: Buffer<B>,
    teapot_indices: Buffer<B>,
    sphere_vertices: Buffer<B>,
    sphere_indices: Buffer<B>,
    plane_count: u32,
    teapot_count: u32,
    sphere_count: u32,
    plane_len: u64,
    teapot_len: u64,
    sphere_len: u64,
    plane_index_len: u64,
    teapot_index_len: u64,
    sphere_index_len: u64,
    plane_uniform: Buffer<B>,
    teapot_uniform: Buffer<B>,
    sphere_uniform: Buffer<B>,
    plane_group: BindGroup<B>,
    teapot_group: BindGroup<B>,
    sphere_group: BindGroup<B>,
    simple_pipeline: RasterPipeline<B>,
    mesh_pipeline: RasterPipeline<B>,
    occluder_pipeline: RasterPipeline<B>,
    queries: Option<QuerySet<B>>,
    results: Option<Buffer<B>>,
    pending_ticket: Option<ReadbackTicket<B>>,
    visible: [bool; 2],
}

// The host waits for each frame before recording the next one. Keep both
// texture and view alive until that completion is observed.
struct Attachments<B: OcclusionWorkloadBackend> {
    _display_depth: Texture<B>,
    display_depth_view: TextureView<B>,
    query: Option<QueryAttachments<B>>,
}

struct QueryAttachments<B: OcclusionWorkloadBackend> {
    color: Texture<B>,
    color_view: TextureView<B>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
}
pub struct Frame<B: OcclusionBackend> {
    pub work: RecordedWork<B>,
    pub ticket: Option<ReadbackTicket<B>>,
}

/// Draws the native-query workload into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_occlusion_query(
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
        let mut workload = Workload::<WebGpu>::new(&device, lane, format).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let Frame { work, ticket } = workload.record(&device, &image, extent)?;
            workload.retain_ticket(ticket)?;
            let completion =
                utils::submit_and_present_point_async(&device, lane, work, image).await?;
            let ticket = workload
                .take_completed_ticket()
                .ok_or(Error::InvalidState)?;
            workload.consume_async(&device, ticket, &completion).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

/// Displays the occlusion-query scene on WebGL2 without claiming query support.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgl2_occlusion_fallback(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    immediate::run_webgl2(canvas, frames)
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(any(
    feature = "native-gl-wgl",
    all(target_os = "linux", feature = "native-gles-egl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
mod immediate;
mod workload;

fn color<B: OcclusionWorkloadBackend>(
    d: &Device<B>,
    e: [u32; 2],
    label: &str,
) -> Result<(Texture<B>, TextureView<B>)> {
    let t = d.create_texture(&TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [e[0], e[1], 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let v = d.create_view(
        &t,
        &TextureViewDesc {
            format: TextureFormat::Rgba8Unorm,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((t, v))
}
fn depth<B: OcclusionWorkloadBackend>(
    d: &Device<B>,
    e: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let t = d.create_texture(&TextureDesc {
        label: Some("11 depth".into()),
        dimension: TextureDimension::D2,
        extent: [e[0], e[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let v = d.create_view(
        &t,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((t, v))
}
fn transform(
    extent: [u32; 2],
    translate: [f32; 3],
    scale: f32,
    tint: [f32; 4],
    visible: bool,
) -> [u8; 256] {
    // C++ camera: look-at style `translate(0, 0, -7.5) * rotateY(-123.75)`
    // and a Vulkan depth-range perspective (60°, near 1, far 256).
    let multiply = |a: [f32; 16], b: [f32; 16]| {
        let mut out = [0.0; 16];
        for column in 0..4 {
            for row in 0..4 {
                out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
            }
        }
        out
    };
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / (30.0_f32.to_radians()).tan();
    let projection = [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        256.0 / -255.0,
        -1.0,
        0.0,
        0.0,
        256.0 / -255.0,
        0.0,
    ];
    let (s, c) = (-123.75_f32.to_radians()).sin_cos();
    let rotate = [
        c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let view = multiply(
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -7.5, 1.0,
        ],
        rotate,
    );
    let model = [
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        translate[0],
        translate[1],
        translate[2],
        1.0,
    ];
    let mut bytes = [0; 256];
    for (matrix_index, matrix) in [projection, view, model].into_iter().enumerate() {
        for (index, value) in matrix.into_iter().enumerate() {
            let offset = matrix_index * 64 + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
    }
    for (index, value) in tint.into_iter().enumerate() {
        let offset = 192 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }
    // The RHI viewport mirrors the C++ Vulkan viewport in Y, so the model
    // keeps its original glTF Y and the world-space light mirrors with it.
    for (index, value) in [10.0_f32, 10.0, 10.0, 1.0].into_iter().enumerate() {
        let offset = 208 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }
    let visible = if visible { 1.0_f32 } else { 0.0_f32 };
    bytes[224..228].copy_from_slice(&visible.to_ne_bytes());
    bytes
}
#[cfg(any(windows, all(target_os = "macos", feature = "metal")))]
fn run<
    B: OcclusionBackend + ExecutionBackend<Mode = DeferredExecution>,
    H: utils::ExampleHost<B>,
>(
    host: H,
) -> Result<()> {
    let cfg = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut work: Option<Workload<B>> = None;
    utils::run_example(
        host,
        "Occlusion queries",
        cfg.clone(),
        move |d, _, image, extent| {
            let lane = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if work.is_none() {
                let workload = if d.queries(QueryKind::Occlusion).is_some() {
                    utils::block_on(Workload::new(d, lane, cfg.format))?
                } else {
                    utils::block_on(Workload::new_visual_fallback(d, lane, cfg.format))?
                };
                work = Some(workload);
            }
            let w = work.as_mut().ok_or(Error::InvalidState)?;
            let Frame {
                work: recorded,
                ticket,
            } = w.record(d, &image, extent)?;
            w.retain_ticket(ticket)?;
            // The harness waits for the frame completion.  Keep the query
            // result path explicit here so the next frame's visibility is
            // based on native query evidence.
            let point = utils::submit_and_present_point(d, lane, recorded, image)?;
            if let Some(ticket) = w.take_completed_ticket() {
                w.consume(d, ticket, &point)
            } else {
                utils::block_on(d.wait_completion(&point))
            }
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    if let Err(error) = run::<fluxel_rhi::metal::Metal, _>(utils::metal_api::MetalHost) {
        eprintln!("11_occlusion_query: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .skip_while(|arg| arg != "--backend")
        .nth(1)
        .is_some_and(|backend| backend == "gl")
    {
        if let Err(error) = immediate::run() {
            eprintln!("11_occlusion_query: {error:?}");
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
        eprintln!("11_occlusion_query: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    if let Err(error) = immediate::run() {
        eprintln!("11_occlusion_query: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(
    windows,
    all(target_os = "macos", feature = "metal"),
    all(target_os = "linux", feature = "native-gles-egl")
)))]
fn main() {
    eprintln!("11_occlusion_query requires a supported native or browser backend");
    // WASM module initialization is not GPU execution. Unsupported native
    // launches must not count as a GPU test pass.
    #[cfg(not(target_arch = "wasm32"))]
    std::process::exit(1);
}
