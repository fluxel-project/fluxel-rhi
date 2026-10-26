//! Microsoft MeshletInstancing scene rendered with a mesh shader.

#[path = "../utils/texel_display.rs"]
mod display;
mod model;
mod record;
mod scene;
mod setup;
#[path = "../utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;

struct Demo<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
{
    lane: LaneId,
    buffers: Vec<Buffer<B>>,
    output: Texture<B>,
    view: TextureView<B>,
    depth: Texture<B>,
    depth_view: TextureView<B>,
    readback: Buffer<B>,
    row_pitch: u32,
    group: BindGroup<B>,
    pipeline: MeshPipeline<B>,
    display: display::Display<B>,
    initialized: bool,
}

#[cfg(windows)]
fn run<B, H>(host: H, shader_format: ShaderFormat) -> Result<()>
where
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + DeferredBackend
        + ExecutionBackend<Mode = DeferredExecution>
        + 'static,
    H: utils::ExampleHost<B>,
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
    let mut demo: Option<(Demo<B>, [u32; 2])> = None;
    let result = utils::run_example(
        host,
        "Meshlet Instancing",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if demo.as_ref().is_none_or(|(_, size)| *size != extent) {
                demo = Some((
                    utils::block_on(Demo::new(
                        device,
                        lane,
                        extent,
                        config.format,
                        shader_format,
                    ))?,
                    extent,
                ));
            }
            let (demo, _) = demo.as_mut().expect("initialized above");
            let (work, tickets) = demo.record(device, &image, extent)?;
            let completion = utils::submit_and_present_point(device, lane, work, image)?;
            utils::block_on(device.wait_completion(&completion))?;
            if let Some(pixels) = tickets {
                // SAFETY: this completion covers the submitted readback.
                let pixels = unsafe { utils::block_on(device.read_after(pixels, &completion))? };
                record::verify_and_save(demo, extent, &pixels.bytes)?;
            }
            demo.initialized = true;
            Ok(())
        },
    );
    result
}
#[cfg(windows)]
fn main() {
    let result = match utils::windows_api::selected_backend() {
        #[cfg(feature = "dx12")]
        Ok(utils::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(utils::windows_api::Dx12Host, ShaderFormat::Dxil)
        }
        #[cfg(feature = "vulkan")]
        Ok(utils::windows_api::SelectedBackend::Vulkan) => run::<fluxel_rhi::vulkan::Vulkan, _>(
            utils::windows_api::VulkanHost,
            ShaderFormat::SpirV,
        ),
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("18_meshlet_instancing: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("18_meshlet_instancing currently requires Windows");
    // WASM module initialization is not GPU execution. Unsupported native
    // launches must not count as a GPU test pass.
    #[cfg(not(target_arch = "wasm32"))]
    std::process::exit(1);
}
