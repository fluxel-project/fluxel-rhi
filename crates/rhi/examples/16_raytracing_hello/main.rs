//! Microsoft's DXR Hello World scene through the typed RHI.

#[path = "../utils/texel_display.rs"]
mod display;
mod scene;
#[path = "../utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;

struct Hello<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
{
    lane: LaneId,
    vertices: Buffer<B>,
    indices: Buffer<B>,
    blas_desc: Option<AccelerationDesc<B>>,
    tlas_desc: Option<AccelerationDesc<B>>,
    blas: Option<AccelerationStructure<B>>,
    tlas: AccelerationStructure<B>,
    blas_scratch: Buffer<B>,
    tlas_scratch: Buffer<B>,
    scratch_sizes: [u64; 2],
    output: Texture<B>,
    readback: Buffer<B>,
    row_pitch: u32,
    pipeline: RayPipeline<B>,
    group: BindGroup<B>,
    table: ShaderTable<B>,
    table_bytes: Vec<u8>,
    display: display::Display<B>,
    initialized: bool,
    captured: bool,
}

mod record;
mod setup;

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
    let mut hello: Option<(Hello<B>, [u32; 2])> = None;
    let result = utils::run_example(
        host,
        "Raytracing - Hello World",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if hello
                .as_ref()
                .is_none_or(|(_, previous)| *previous != extent)
            {
                hello = Some((
                    utils::block_on(Hello::new(
                        device,
                        lane,
                        extent,
                        config.format,
                        shader_format,
                    ))?,
                    extent,
                ));
            }
            let (hello, _) = hello.as_mut().expect("initialized above");
            let (work, ticket) = hello.record(device, &image, extent)?;
            let completion = utils::submit_and_present_point(device, lane, work, image)?;
            utils::block_on(device.wait_completion(&completion))?;
            if let Some(ticket) = ticket {
                // SAFETY: this completion point covers the submission that recorded the ticket.
                let data = unsafe { utils::block_on(device.read_after(ticket, &completion))? };
                if let Some(path) = std::env::var_os("FLUXEL_RAY_READBACK") {
                    let expected = u64::from(hello.row_pitch) * u64::from(extent[1]);
                    if data.bytes.len() as u64 != expected {
                        return Err(Error::InvalidRange);
                    }
                    let mut ppm = format!("P6\n{} {}\n255\n", extent[0], extent[1]).into_bytes();
                    for row in data
                        .bytes
                        .chunks_exact(hello.row_pitch as usize)
                        .take(extent[1] as usize)
                    {
                        for pixel in row[..extent[0] as usize * 4].chunks_exact(4) {
                            ppm.extend_from_slice(&pixel[..3]);
                        }
                    }
                    std::fs::write(path, ppm).map_err(|error| Error::Backend(error.to_string()))?;
                }
                hello.captured = true;
            }
            // The TLAS must retain the BLAS it references after the build work
            // completes. The next frame traces after all public BLAS references
            // and build descriptions have been dropped.
            hello.blas = None;
            hello.blas_desc = None;
            hello.tlas_desc = None;
            hello.initialized = true;
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
        eprintln!("16_raytracing_hello: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("16_raytracing_hello currently requires Windows");
    // WASM module initialization is not GPU execution. Unsupported native
    // launches must not count as a GPU test pass.
    #[cfg(not(target_arch = "wasm32"))]
    std::process::exit(1);
}
