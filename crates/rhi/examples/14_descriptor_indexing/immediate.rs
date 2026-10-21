//! WGL immediate route for the fixed texture-array fallback.
//!
//! GL does not claim runtime-sized non-uniform descriptor indexing. It uses
//! the same 32-layer WGSL texture-array visual path as WebGPU and issues each
//! RHI operation directly on the current context owner thread.

use std::{cell::RefCell, rc::Rc, time::Instant};

use fluxel_rhi::{api::*, gl::Gl};

use super::{DescriptorIndexing, utils};

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
    let start = Instant::now();
    let scene = Rc::new(RefCell::new(None::<DescriptorIndexing<Gl>>));
    let dropped = Rc::clone(&scene);
    utils::gl_api::run_native(
        "Fixed texture-array fallback",
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
                let workload = utils::block_on(DescriptorIndexing::<Gl>::new(
                    device,
                    lane,
                    config.format,
                    extent,
                    true,
                ))?;
                *scene.borrow_mut() = Some(workload);
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
                        .encode(
                            device,
                            image,
                            encoder,
                            extent,
                            start.elapsed().as_secs_f32(),
                        )
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
