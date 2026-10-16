//! Per-frame execution of shared examples through deferred Vulkan or immediate GLES.
//! Submission completion covers readback tickets; no CPU command replay is introduced.

use super::*;

impl App {
    pub(super) fn draw(&mut self) -> Result<()> {
        let gpu = self.gpu.as_mut().ok_or(Error::SurfaceLost)?;
        let image = utils::block_on(gpu.surface.acquire())?;
        #[cfg(feature = "vulkan")]
        {
            #[cfg(feature = "demo14")]
            let (work, uploaded) = gpu.scene.record(&gpu.device, &image, gpu.extent, 0.0)?;
            #[cfg(feature = "demo13")]
            let (work, uploaded) = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo12")]
            {
                let frame = gpu.scene.record(&gpu.device, &image, self.frames == 0)?;
                let point = submission::submit_frame(&gpu.device, gpu.lane, frame.work, image)?;
                utils::block_on(gpu.device.wait_completion(&point))?;
                if let Some(ticket) = frame.ticket {
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
                not(feature = "demo14")
            ))]
            #[cfg(not(feature = "demo03"))]
            let work = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo06")]
            let work = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo07")]
            let work = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo08")]
            let work = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo09")]
            let work = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo10")]
            let work = gpu.scene.record(
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo11")]
            let frame = gpu.scene.record(&gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo11")]
            {
                let point = submission::submit_frame(&gpu.device, gpu.lane, frame.work, image)?;
                if let Some(ticket) = frame.ticket {
                    gpu.scene.consume(&gpu.device, ticket, &point)?;
                } else {
                    utils::block_on(gpu.device.wait_completion(&point))?;
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
            #[cfg(feature = "demo13")]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(feature = "demo14")]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(feature = "demo03")]
            gpu.scene.mark_submitted();
        }
        #[cfg(feature = "gles")]
        {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            // SAFETY: the two steps refer to this device and its one current
            // GLES lane. They are issued once in command-then-present order.
            let plan = unsafe {
                SubmissionPlan::from_compiled(
                    PlanId {
                        device: gpu.device.id(),
                        ordinal: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    },
                    Arc::from([
                        StepDesc {
                            lane: gpu.lane,
                            kind: StepKind::Commands,
                            waits: vec![],
                        },
                        StepDesc {
                            lane: gpu.lane,
                            kind: StepKind::Present,
                            waits: vec![],
                        },
                    ]),
                )
            };
            // SAFETY: immediate commands and present execute synchronously on this EGL owner thread.
            let completion = if cfg!(any(feature = "demo11", feature = "demo12")) {
                CompletionRequest::End
            } else {
                CompletionRequest::None
            };
            let mut session = unsafe { gpu.device.immediate(plan, completion)? };
            let mut commands = session.begin_commands().map_err(|x| match x {
                ImmediateRunOutcome::Failed { error, .. }
                | ImmediateRunOutcome::Indeterminate { error, .. } => error,
                _ => Error::InvalidState,
            })?;
            #[cfg(feature = "demo14")]
            let uploaded =
                gpu.scene
                    .encode(&gpu.device, &image, commands.commands(), gpu.extent, 0.0)?;
            #[cfg(feature = "demo13")]
            let uploaded =
                gpu.scene
                    .encode(&gpu.device, &image, commands.commands(), gpu.extent)?;
            #[cfg(feature = "demo12")]
            let ticket = gpu
                .scene
                .encode(commands.commands(), &image, self.frames == 0)?;
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
                not(feature = "demo14")
            ))]
            #[cfg(not(feature = "demo03"))]
            gpu.scene.encode(commands.commands(), &image, gpu.extent)?;
            #[cfg(feature = "demo06")]
            let initialized =
                gpu.scene
                    .encode(commands.commands(), &gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo07")]
            gpu.scene.encode(
                commands.commands(),
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo08")]
            let commit = gpu.scene.encode(
                commands.commands(),
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo09")]
            let commit = gpu
                .scene
                .encode(commands.commands(), &gpu.device, &image, gpu.extent)?;
            #[cfg(feature = "demo10")]
            let commit = gpu.scene.encode(
                commands.commands(),
                &gpu.device,
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo11")]
            let ticket =
                gpu.scene
                    .encode_scene(&gpu.device, commands.commands(), &image, gpu.extent)?;
            #[cfg(feature = "demo02")]
            gpu.scene.issue(
                commands.commands(),
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
                commands.commands(),
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            #[cfg(feature = "demo05")]
            gpu.scene.issue(commands.commands(), &image, gpu.extent)?;
            #[cfg(feature = "demo03")]
            gpu.scene.issue(
                commands.commands(),
                &image,
                gpu.extent,
                gpu.time.elapsed().as_secs_f32(),
            )?;
            if !matches!(commands.close(), ImmediateRunOutcome::Issued(_)) {
                return Err(Error::InvalidState);
            }
            #[cfg(feature = "demo14")]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(feature = "demo13")]
            if uploaded {
                gpu.scene.mark_uploaded();
            }
            #[cfg(feature = "demo06")]
            gpu.scene.commit_frame(initialized);
            #[cfg(feature = "demo07")]
            gpu.scene.commit_frame();
            #[cfg(feature = "demo08")]
            gpu.scene.commit_frame(commit);
            #[cfg(feature = "demo09")]
            gpu.scene.commit_frame(commit);
            #[cfg(feature = "demo10")]
            gpu.scene.commit_frame(commit);
            #[cfg(feature = "demo03")]
            gpu.scene.mark_submitted();
            if !matches!(
                session.present(PresentBinding::new(image)),
                ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_))
            ) {
                return Err(Error::InvalidState);
            }
            let _prefix = match session.finish() {
                ImmediateFinishOutcome::Issued(prefix) => prefix,
                _ => return Err(Error::InvalidState),
            };
            #[cfg(feature = "demo11")]
            {
                let point = _prefix
                    .checkpoints
                    .into_iter()
                    .next()
                    .map(|checkpoint| checkpoint.point)
                    .ok_or(Error::InvalidState)?;
                if let Some(ticket) = ticket {
                    gpu.scene.consume(&gpu.device, ticket, &point)?;
                } else {
                    utils::block_on(gpu.device.wait_completion(&point))?;
                }
            }
            #[cfg(feature = "demo12")]
            {
                let point = _prefix
                    .checkpoints
                    .into_iter()
                    .next()
                    .map(|checkpoint| checkpoint.point)
                    .ok_or(Error::InvalidState)?;
                utils::block_on(gpu.device.wait_completion(&point))?;
                if let Some(ticket) = ticket {
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
