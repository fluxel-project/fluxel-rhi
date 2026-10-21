//! Native GL immediate execution for the sequential fallback.

use super::*;

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
pub fn run_gl_multithreaded_recording() -> Result<()> {
    use std::{cell::RefCell, rc::Rc};

    use fluxel_rhi::gl::Gl;

    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let scene = Rc::new(RefCell::new(None::<Sequential<Gl>>));
    let dropped = Rc::clone(&scene);
    utils::gl_api::run_native(
        "Sequential UFO recording fallback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if scene.borrow().is_none() {
                let next = utils::block_on(Sequential::<Gl>::new(device, config.format))?;
                *scene.borrow_mut() = Some(next);
            }
            let (uploaded, outcome) = utils::execute_frame(
                device,
                lane,
                image,
                CompletionRequest::End,
                |encoder, image| {
                    scene
                        .borrow_mut()
                        .as_mut()
                        .ok_or(Error::InvalidState)?
                        .encode(device, image, encoder, extent)
                },
            )?;
            let point = utils::execution_end_checkpoint(&outcome)?;
            if uploaded {
                scene
                    .borrow_mut()
                    .as_mut()
                    .ok_or(Error::InvalidState)?
                    .mark_uploaded();
            }
            utils::block_on(device.wait_completion(&point))
        },
    )
}
