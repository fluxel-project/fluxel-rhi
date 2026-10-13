//! Textured indexed quad using the statically typed RHI.
//!
//! `TexturePixels` carries decoded KTX texels; the workload records a real
//! texture upload and textured indexed draw.  The native host supplies the
//! acquired surface and submits the returned work.
#[path = "../common/mod.rs"]
mod common;
use fluxel_rhi::api::*;
mod data;
use data::*;
const VS: &[u8] = include_bytes!("../assets/02_texture/texture.vs.dxil");
const PS: &[u8] = include_bytes!("../assets/02_texture/texture.ps.dxil");
const VS_SPV: &[u8] = include_bytes!("../assets/02_texture/texture.vs.spv");
const PS_SPV: &[u8] = include_bytes!("../assets/02_texture/texture.ps.spv");
const WGSL: &[u8] = include_bytes!("../assets/02_texture/texture.wgsl");
pub trait TextureBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> TextureBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
#[derive(Clone, Copy)]
pub struct TexturePixels<'a> {
    pub rgba: &'a [u8],
    pub width: u32,
    pub height: u32,
}
mod workload;
pub use workload::TextureExample;
pub(super) fn texture_transition<B: TextureBackend>(
    texture: &Texture<B>,
    from: ImageLayout,
    to: ImageLayout,
    before: TextureUses,
    after: TextureUses,
) -> ResourceTransition<'_, B> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: before,
            to: after,
        },
        layout: Some((from, to)),
        ownership: None,
    })
}
pub(super) fn buf(label: &str, size: u64, usage: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    }
}
pub(super) fn entry(
    binding: u32,
    visibility: Vec<ShaderStage>,
    ty: BindingType,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
pub(super) async fn pipeline<B: TextureBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    v: Vec<VertexBufferLayout>,
    vs: &[u8],
    ps: &[u8],
) -> Result<RasterPipeline<B>> {
    let a =
        common::native_shader_with_wgsl(d, "vs", "vs_main", ShaderStage::Vertex, vs, VS_SPV, WGSL)
            .await?;
    let b = common::native_shader_with_wgsl(
        d,
        "ps",
        "fs_main",
        ShaderStage::Fragment,
        ps,
        PS_SPV,
        WGSL,
    )
    .await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some("02 pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: a,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: b,
            entry: "fs_main".into(),
            constants: vec![],
        }),
        vertex_buffers: v,
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
            // D3D12 sample masks are 32-bit; the API pipeline lowerer rejects
            // a wider public mask instead of truncating it.
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
    .await
}

/// Draws the textured example into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_texture(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};

    let run = async {
        if frames == 0 {
            return Err(Error::InvalidRange);
        }
        let (rgba, width, height) = load_ktx()?;
        let pixels = TexturePixels {
            rgba: &rgba,
            width,
            height,
        };
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
        let example = TextureExample::<WebGpu>::new(&device, lane, format, pixels).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let work = example.record(&device, &image, extent, pixels)?;
            common::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn run_host<B: TextureBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let (rgba, width, height) = match load_ktx() {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut example: Option<TextureExample<B>> = None;
    common::run_example(
        host,
        "Textured quad",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            let pixels = TexturePixels {
                rgba: &rgba,
                width,
                height,
            };
            if example.is_none() {
                example = Some(common::block_on(TextureExample::new(
                    device,
                    lane,
                    config.format,
                    pixels,
                ))?);
            }
            let work = example
                .as_ref()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent, pixels)?;
            common::submit_and_present(device, lane, work, image)
        },
    )
}

/// Runs the native Cargo example target.
#[cfg(windows)]
pub fn run() {
    let result = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run_host::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run_host::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("02_texture: {error:?}");
        std::process::exit(1);
    }
}

/// Reports the unavailable native target on unsupported hosts.
#[cfg(not(windows))]
pub fn run() {
    eprintln!("02_texture requires Windows and DX12");
}
