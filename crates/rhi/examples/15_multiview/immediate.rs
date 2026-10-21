//! WGL route for the sequential stereo visual fallback.
//!
//! The scene encoder calls execute on the WGL owner thread. This module only
//! drives Core's immediate session; it does not retain commands for replay.

use std::{cell::RefCell, rc::Rc};

use fluxel_rhi::{api::*, gl::Gl};

use super::{per_eye::StereoFallback, utils};

pub(super) fn run() -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let scene = Rc::new(RefCell::new(None::<StereoFallback<Gl>>));
    let dropped = Rc::clone(&scene);
    utils::gl_api::run_native(
        "Sequential stereo fallback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if scene
                .borrow()
                .as_ref()
                .is_none_or(|example| example.extent() != extent)
            {
                let fallback =
                    utils::block_on(StereoFallback::<Gl>::new(device, config.format, extent))
                        .map_err(|error| {
                            eprintln!("multiview fallback setup failed: {error:?}");
                            error
                        })?;
                *scene.borrow_mut() = Some(fallback);
            }
            let mut scene = scene.borrow_mut();
            let scene = scene.as_mut().ok_or(Error::InvalidState)?;
            let (uploaded, outcome) = utils::execute_frame(
                device,
                lane,
                image,
                CompletionRequest::End,
                |encoder, image| {
                    scene.encode(encoder, image).map_err(|error| {
                        eprintln!("multiview fallback encoding failed: {error:?}");
                        error
                    })
                },
            )?;
            let point = utils::execution_end_checkpoint(&outcome)?;
            if uploaded {
                scene.mark_uploaded();
            }
            utils::block_on(device.wait_completion(&point))
        },
    )
}
