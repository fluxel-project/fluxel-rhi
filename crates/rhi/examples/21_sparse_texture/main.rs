//! Nine-mip sparse texture based on Microsoft ReservedResources.

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
        + SparseBackend
        + DeferredBackend,
> {
    lane: LaneId,
    vertex: Buffer<B>,
    sparse_texture: Texture<B>,
    allocations: Vec<SparseAllocation<B>>,
    regions: Vec<SparseTile>,
    region_mips: Vec<std::ops::Range<u32>>,
    uploaded: Vec<bool>,
    mapped: Option<usize>,
    selected: u32,
    frame: u64,
    output: Texture<B>,
    view: TextureView<B>,
    readback: Buffer<B>,
    row_pitch: u32,
    groups: Vec<BindGroup<B>>,
    pipeline: RasterPipeline<B>,
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
        + SparseBackend
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
        "Sparse Texture",
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
            let selected = std::env::var("FLUXEL_SPARSE_MIP")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(((demo.frame / 120) % 9) as u32);
            if selected >= 9 {
                return Err(Error::InvalidRange);
            }
            demo.select(device, selected)
                .map_err(|e| Error::Backend(format!("select: {e:?}")))?;
            let (work, tickets) = demo
                .record(device, &image, extent)
                .map_err(|e| Error::Backend(format!("record: {e:?}")))?;
            let completion = utils::submit_and_present_point(device, lane, work, image)?;
            utils::block_on(device.wait_completion(&completion))?;
            if let Some(pixels) = tickets {
                // SAFETY: this completion covers the submitted readback.
                let pixels = unsafe { utils::block_on(device.read_after(pixels, &completion))? };
                record::verify_and_save(demo, extent, &pixels.bytes)?;
            }
            demo.uploaded[demo.mapped.expect("selected region")] = true;
            demo.initialized = true;
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
        eprintln!("21_sparse_texture: {e:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("21_sparse_texture currently requires Windows and DX12");
}
