//! Controlled ray-query rendering of the NVIDIA particle scene.
#[cfg(windows)]
use fluxel_rhi::api::*;
#[cfg(windows)]
#[path = "../utils/texel_display.rs"]
mod display;
mod particles;
#[cfg(windows)]
mod run;
mod scene;
#[cfg(windows)]
#[path = "../utils/mod.rs"]
mod utils;

#[cfg(windows)]
fn main() {
    use fluxel_rhi::api::ShaderFormat;
    use utils::windows_api::{SelectedBackend, selected_backend};
    let result = (|| match selected_backend()? {
        #[cfg(feature = "dx12")]
        SelectedBackend::Dx12 => {
            let device = utils::block_on(fluxel_rhi::dx12::Dx12Instance::new()?.open_default())?;
            if std::env::var_os("FLUXEL_RAY_QUERY_PRESENT").is_some() {
                present::<fluxel_rhi::dx12::Dx12, _>(
                    utils::windows_api::Dx12Host,
                    ShaderFormat::Dxil,
                )
            } else {
                run::run(&device, ShaderFormat::Dxil)
            }
        }
        #[cfg(feature = "vulkan")]
        SelectedBackend::Vulkan => {
            let device =
                utils::block_on(fluxel_rhi::vulkan::VulkanInstance::new()?.open_default())?;
            if std::env::var_os("FLUXEL_RAY_QUERY_PRESENT").is_some() {
                present::<fluxel_rhi::vulkan::Vulkan, _>(
                    utils::windows_api::VulkanHost,
                    ShaderFormat::SpirV,
                )
            } else {
                run::run(&device, ShaderFormat::SpirV)
            }
        }
    })();
    if let Err(e) = result {
        eprintln!("19_ray_query_particles: {e:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("This example currently requires Windows.");
}

#[cfg(windows)]
fn present<B, H>(host: H, shader_format: ShaderFormat) -> Result<()>
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
        extent: [run::WIDTH, run::HEIGHT],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut frames: Option<(Vec<(Texture<B>, display::Display<B>)>, std::time::Instant)> = None;
    utils::run_example(
        host,
        "Ray Query Particles",
        config.clone(),
        move |device, _, image, extent| {
            if extent != [run::WIDTH, run::HEIGHT] {
                return Err(Error::Unsupported("fixed ray query display extent"));
            }
            if frames.is_none() {
                let mut images = Vec::new();
                for (index, time) in [8.363f32, 10.212, 12.228].into_iter().enumerate() {
                    let texture = run::render(device, shader_format, index, time)?;
                    let view = device.create_view(
                        &texture,
                        &TextureViewDesc {
                            format: TextureFormat::Rgba8Unorm,
                            dimension: TextureViewDimension::D2,
                            aspects: TextureAspects::COLOR,
                            mips: 0..1,
                            layers: 0..1,
                        },
                    )?;
                    let display = utils::block_on(display::Display::new_with_format(
                        device,
                        &view,
                        config.format,
                        shader_format,
                    ))?;
                    images.push((texture, display));
                }
                frames = Some((images, std::time::Instant::now()));
            }
            let (images, start) = frames.as_ref().unwrap();
            let index = ((start.elapsed().as_secs() / 2) % 3) as usize;
            let lane = device
                .lanes()
                .iter()
                .find(|l| l.class == LaneClass::Graphics)
                .ok_or(Error::Unsupported("graphics lane"))?
                .id;
            let mut encoder = device.encoder(lane)?;
            images[index].1.encode(&mut encoder, &image, extent)?;
            let point = utils::submit_and_present_point(device, lane, encoder.finish()?, image)?;
            utils::block_on(device.wait_completion(&point))?;
            Ok(())
        },
    )
}
