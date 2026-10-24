//! GPU particle simulation using separate compute and graphics queues.
#[path = "../utils/texel_display.rs"]
mod display;
mod record;
mod setup;
#[path = "../utils/mod.rs"]
mod utils;
use fluxel_rhi::api::*;

struct Demo<
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + DeferredBackend,
> {
    compute: LaneId,
    graphics: LaneId,
    particles: [Buffer<B>; 2],
    compute_groups: [BindGroup<B>; 2],
    draw_groups: [BindGroup<B>; 2],
    compute_pipeline: ComputePipeline<B>,
    pipeline: RasterPipeline<B>,
    output: Texture<B>,
    view: TextureView<B>,
    readback: Buffer<B>,
    particle_readback: Buffer<B>,
    display: display::Display<B>,
    frame: u64,
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
        + MultiQueueBackend
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
    let mut demo: Option<Demo<B>> = None;
    let result = utils::run_example(
        host,
        "Multi Queue Particles",
        config.clone(),
        move |d, _, image, extent| {
            if demo.is_none() {
                demo = Some(utils::block_on(Demo::new(
                    d,
                    extent,
                    config.format,
                    shader_format,
                ))?);
            }
            let demo = demo.as_mut().unwrap();
            let (compute, graphics, ticket) = demo.record(d, &image, extent)?;
            let point = demo.submit(d, compute, graphics, image)?;
            utils::block_on(d.wait_completion(&point))?;
            if let Some((ticket, particle_ticket)) = ticket {
                // SAFETY: the graphics completion covers the readback.
                let data = unsafe { utils::block_on(d.read_after(ticket, &point))? };
                record::save(&data.bytes, extent)?;
                // SAFETY: the same completion covers the particle copy.
                let particles = unsafe { utils::block_on(d.read_after(particle_ticket, &point))? };
                if let Some(path) = std::env::var_os("FLUXEL_NBODY_PARTICLES") {
                    std::fs::write(path, particles.bytes)
                        .map_err(|e| Error::Backend(e.to_string()))?;
                }
            }
            demo.frame += 1;
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
        Err(e) => Err(e),
    };
    if let Err(e) = result {
        eprintln!("23_multi_queue: {e:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("23_multi_queue currently requires Windows and DX12");
    std::process::exit(1);
}
