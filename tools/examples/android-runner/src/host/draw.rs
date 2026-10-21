//! Per-frame execution of shared examples through deferred Vulkan or immediate GLES.
//! Submission completion covers readback tickets; no CPU command replay is introduced.

use super::*;

impl App {
    pub(super) fn draw(&mut self) -> Result<()> {
        let gpu = self.gpu.as_mut().ok_or(Error::SurfaceLost)?;
        let _scope = utils::GpuScope::new(&gpu.device);
        #[cfg(feature = "demo15")]
        if gpu.scene.extent() != gpu.extent {
            utils::block_on(gpu.device.wait_idle())?;
            #[cfg(feature = "vulkan")]
            {
                gpu.scene = utils::block_on(multiview::Multiview::new(
                    &gpu.device,
                    gpu.lane,
                    gpu.format,
                    gpu.extent,
                ))?;
            }
            #[cfg(feature = "gles")]
            {
                gpu.scene = utils::block_on(multiview::per_eye::StereoFallback::new(
                    &gpu.device,
                    gpu.format,
                    gpu.extent,
                ))?;
            }
        }
        let image = utils::block_on(gpu.surface.acquire())?;
        #[cfg(feature = "vulkan")]
        {
            #[cfg(feature = "demo15")]
            let (work, uploaded) = gpu.scene.record(&gpu.device, &image)?;
            #[cfg(feature = "demo14")]
            let (work, uploaded) = gpu.scene.record(&gpu.device, &image, gpu.extent, 0.0)?;
            #[cfg(feature = "demo13")]
            let (work, uploaded) = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo12")]
            {
                let frame = gpu.scene.record(&gpu.device, &image, self.frames == 0)?;
                gpu.scene.retain_ticket(frame.ticket)?;
                let point =
                    utils::submit_and_present_point(&gpu.device, gpu.lane, frame.work, image)?;
                gpu.scene.commit_frame(self.frames == 0);
                utils::block_on(gpu.device.wait_completion(&point))?;
                if let Some(ticket) = gpu.scene.take_completed_ticket() {
                    let path = self.capture_path.as_ref().ok_or(Error::InvalidState)?;
                    std::fs::create_dir_all(path.parent().ok_or(Error::InvalidState)?)
                        .map_err(|error| Error::Backend(error.to_string()))?;
                    utils::block_on(gpu.scene.save_ppm_after(&gpu.device, ticket, &point, path))?;
                    eprintln!("RHI_READBACK_OK={}", path.display());
                }
            }
            #[cfg(all(
                not(feature = "demo02"),
                not(feature = "demo03"),
                not(feature = "demo04"),
                not(feature = "demo05"),
                not(feature = "demo06"),
                not(feature = "demo07"),
                not(feature = "demo08"),
                not(feature = "demo09"),
                not(feature = "demo10"),
                not(feature = "demo11"),
                not(feature = "demo12"),
                not(feature = "demo13"),
                not(feature = "demo14"),
                not(feature = "demo15")
            ))]
            #[cfg(not(feature = "demo03"))]
            let work = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo06")]
            let (work, commit) = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo07")]
            let work = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo08")]
            let (work, commit) = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo09")]
            let (work, commit) = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo10")]
            let (work, commit) = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo11")]
            let frame = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo11")]
            {
                gpu.scene.retain_ticket(frame.ticket)?;
                let point =
                    utils::submit_and_present_point(&gpu.device, gpu.lane, frame.work, image)?;
                utils::block_on(gpu.device.wait_completion(&point))?;
                if let Some(ticket) = gpu.scene.take_completed_ticket() {
                    gpu.scene.consume(&gpu.device, ticket, &point)?;
                }
            }
            #[cfg(feature = "demo02")]
            let work = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                texture::TexturePixels {
                    rgba: &gpu.pixels,
                    width: gpu.pixel_extent[0],
                    height: gpu.pixel_extent[1],
                },
            )?;
            #[cfg(feature = "demo05")]
            let work = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo04")]
            let work = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo03")]
            let work = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(not(any(feature = "demo11", feature = "demo12")))]
            utils::submit_and_present(&gpu.device, gpu.lane, work, image)?;
            #[cfg(any(feature = "demo05", feature = "demo07"))]
            gpu.scene.commit_frame();
            #[cfg(any(
                feature = "demo06",
                feature = "demo08",
                feature = "demo09",
                feature = "demo10"
            ))]
            gpu.scene.commit_frame(commit);
            #[cfg(feature = "demo13")]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(any(feature = "demo14", feature = "demo15"))]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(feature = "demo03")]
            gpu.scene.mark_submitted();
        }
        #[cfg(feature = "gles")]
        {
            let completion = if cfg!(any(feature = "demo11", feature = "demo12")) {
                CompletionRequest::End
            } else {
                CompletionRequest::None
            };
            #[cfg(any(feature = "demo13", feature = "demo14", feature = "demo15"))]
            let mut uploaded = false;
            #[cfg(feature = "demo06")]
            let mut initialized = None;
            #[cfg(any(feature = "demo08", feature = "demo09", feature = "demo10"))]
            let mut commit = None;
            let (_, outcome) = utils::execute_frame(
                &gpu.device,
                gpu.lane,
                image,
                completion,
                |commands, image| {
                    #[cfg(feature = "demo15")]
                    {
                        uploaded = gpu.scene.encode(commands, &image)?;
                    }
                    #[cfg(feature = "demo14")]
                    {
                        uploaded =
                            gpu.scene
                                .encode(&gpu.device, &image, commands, gpu.extent, 0.0)?;
                    }
                    #[cfg(feature = "demo13")]
                    {
                        uploaded = gpu
                            .scene
                            .encode(&gpu.device, &image, commands, gpu.extent)?;
                    }
                    #[cfg(feature = "demo12")]
                    {
                        gpu.scene.ensure_ticket_available()?;
                        let ticket = gpu.scene.encode(commands, &image, self.frames == 0)?;
                        gpu.scene.retain_ticket(ticket)?;
                    }
                    #[cfg(all(
                        not(feature = "demo02"),
                        not(feature = "demo03"),
                        not(feature = "demo04"),
                        not(feature = "demo05"),
                        not(feature = "demo06"),
                        not(feature = "demo07"),
                        not(feature = "demo08"),
                        not(feature = "demo09"),
                        not(feature = "demo10"),
                        not(feature = "demo11"),
                        not(feature = "demo12"),
                        not(feature = "demo13"),
                        not(feature = "demo14"),
                        not(feature = "demo15")
                    ))]
                    #[cfg(not(feature = "demo03"))]
                    gpu.scene.encode(commands, &image, gpu.extent)?;
                    #[cfg(feature = "demo06")]
                    {
                        initialized =
                            Some(
                                gpu.scene
                                    .encode(commands, &gpu.device, &image, gpu.extent)?,
                            );
                    }
                    #[cfg(feature = "demo07")]
                    gpu.scene.encode(
                        commands,
                        &gpu.device,
                        &image,
                        gpu.extent,
                        gpu.time.elapsed().as_secs_f32(),
                    )?;
                    #[cfg(feature = "demo08")]
                    {
                        commit = Some(gpu.scene.encode(
                            commands,
                            &gpu.device,
                            &image,
                            gpu.extent,
                            gpu.time.elapsed().as_secs_f32(),
                        )?);
                    }
                    #[cfg(feature = "demo09")]
                    {
                        commit =
                            Some(
                                gpu.scene
                                    .encode(commands, &gpu.device, &image, gpu.extent)?,
                            );
                    }
                    #[cfg(feature = "demo10")]
                    {
                        commit = Some(gpu.scene.encode(
                            commands,
                            &gpu.device,
                            &image,
                            gpu.extent,
                            gpu.time.elapsed().as_secs_f32(),
                        )?);
                    }
                    #[cfg(feature = "demo11")]
                    {
                        gpu.scene.ensure_ticket_available()?;
                        let ticket =
                            gpu.scene
                                .encode_scene(&gpu.device, commands, &image, gpu.extent)?;
                        gpu.scene.retain_ticket(ticket)?;
                    }
                    #[cfg(feature = "demo02")]
                    gpu.scene.issue(
                        commands,
                        &image,
                        gpu.extent,
                        texture::TexturePixels {
                            rgba: &gpu.pixels,
                            width: gpu.pixel_extent[0],
                            height: gpu.pixel_extent[1],
                        },
                    )?;
                    #[cfg(feature = "demo04")]
                    gpu.scene.encode(
                        commands,
                        &image,
                        gpu.extent,
                        gpu.time.elapsed().as_secs_f32(),
                    )?;
                    #[cfg(feature = "demo05")]
                    gpu.scene.issue(commands, &image, gpu.extent)?;
                    #[cfg(feature = "demo03")]
                    gpu.scene.issue(
                        commands,
                        &image,
                        gpu.extent,
                        gpu.time.elapsed().as_secs_f32(),
                    )?;
                    Ok(())
                },
            )?;
            #[cfg(feature = "demo12")]
            gpu.scene.commit_frame(self.frames == 0);
            #[cfg(any(feature = "demo14", feature = "demo15"))]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(feature = "demo13")]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(feature = "demo06")]
            gpu.scene
                .commit_frame(initialized.ok_or(Error::InvalidState)?);
            #[cfg(any(feature = "demo05", feature = "demo07"))]
            gpu.scene.commit_frame();
            #[cfg(feature = "demo08")]
            gpu.scene.commit_frame(commit.ok_or(Error::InvalidState)?);
            #[cfg(feature = "demo09")]
            gpu.scene.commit_frame(commit.ok_or(Error::InvalidState)?);
            #[cfg(feature = "demo10")]
            gpu.scene.commit_frame(commit.ok_or(Error::InvalidState)?);
            #[cfg(feature = "demo03")]
            gpu.scene.mark_submitted();
            #[cfg(feature = "demo11")]
            {
                let point = utils::execution_end_checkpoint(&outcome)?;
                utils::block_on(gpu.device.wait_completion(&point))?;
                if let Some(ticket) = gpu.scene.take_completed_ticket() {
                    gpu.scene.consume(&gpu.device, ticket, &point)?;
                }
            }
            #[cfg(feature = "demo12")]
            {
                let point = utils::execution_end_checkpoint(&outcome)?;
                utils::block_on(gpu.device.wait_completion(&point))?;
                if let Some(ticket) = gpu.scene.take_completed_ticket() {
                    let path = self.capture_path.as_ref().ok_or(Error::InvalidState)?;
                    std::fs::create_dir_all(path.parent().ok_or(Error::InvalidState)?)
                        .map_err(|error| Error::Backend(error.to_string()))?;
                    utils::block_on(gpu.scene.save_ppm_after(&gpu.device, ticket, &point, path))?;
                    eprintln!("RHI_READBACK_OK={}", path.display());
                }
            }
        }
        self.frames = self.frames.saturating_add(1);
        if self.frames <= 2 {
            eprintln!("RHI_FRAME_OK={}", self.frames);
        }
        Ok(())
    }
}
