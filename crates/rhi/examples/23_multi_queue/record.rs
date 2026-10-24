//! Record compute, draw and the explicit cross-queue dependency.
use super::*;
use std::{
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
};
const SIZE: u64 = 320000;
static NEXT: AtomicU64 = AtomicU64::new(23000000);
impl<
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + DeferredBackend,
> Demo<B>
{
    pub(super) fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<(
        RecordedWork<B>,
        RecordedWork<B>,
        Option<(ReadbackTicket<B>, ReadbackTicket<B>)>,
    )> {
        let input = self.frame as usize % 2;
        let output = 1 - input;
        let read = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_READ,
        };
        let graphics_read = BufferUses {
            stages: StageMask::VERTEX,
            accesses: AccessMask::SHADER_READ,
        };
        let write = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_WRITE,
        };
        let barrier = |i: usize, from, to, ownership| {
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &self.particles[i],
                    range: 0..SIZE,
                },
                usage: StateTransition { from, to },
                ownership,
            })
        };
        let mut c = d.encoder(self.compute)?;
        if self.frame == 0 {
            c.write_buffer(
                &self.particles[0],
                0,
                include_bytes!("../assets/23_multi_queue/initial.bin"),
            )?;
            c.transition_resources(&[
                barrier(
                    0,
                    BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    read,
                    None,
                ),
                barrier(
                    1,
                    BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    write,
                    None,
                ),
            ])?;
        } else {
            // The previous graphics submission returned this input buffer to
            // compute ownership. The host waited for that submission before
            // recording this frame, which is the inter-frame dependency.
            c.transition_resources(&[
                barrier(
                    input,
                    graphics_read,
                    read,
                    Some(OwnershipTransfer::Acquire {
                        from: self.graphics,
                        to: self.compute,
                    }),
                ),
                barrier(output, read, write, None),
            ])?;
        }
        let mut pass = d.compute().ok_or(Error::InvalidState)?.begin(&mut c)?;
        pass.set_pipeline(&self.compute_pipeline)?;
        pass.set_bind_group(0, &self.compute_groups[input], &[])?;
        pass.dispatch([79, 1, 1])?;
        pass.finish()?;
        // Vulkan exclusive resources need both sides of the compute-to-graphics
        // hand-off. The matching acquire is the first graphics command below.
        c.transition_resources(&[barrier(
            output,
            write,
            graphics_read,
            Some(OwnershipTransfer::Release {
                from: self.compute,
                to: self.graphics,
            }),
        )])?;
        let mut e = d.encoder(self.graphics)?;
        e.transition_resources(&[barrier(
            output,
            write,
            graphics_read,
            Some(OwnershipTransfer::Acquire {
                from: self.compute,
                to: self.graphics,
            }),
        )])?;
        let particle_ticket = if self.frame == 0 {
            let copy = BufferUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            };
            e.transition_resources(&[barrier(output, graphics_read, copy, None)])?;
            e.copy_buffer(&self.particles[output], 0, &self.particle_readback, 0, SIZE)?;
            let ticket = e.readback_buffer(&self.particle_readback, 0..SIZE)?;
            e.transition_resources(&[barrier(output, copy, graphics_read, None)])?;
            Some(ticket)
        } else {
            None
        };
        let range = || Subresources {
            texture: &self.output,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        };
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: if self.frame == 0 {
                    TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    }
                } else {
                    TextureUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::SHADER_READ,
                    }
                },
                to: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
            },
            layout: Some((
                if self.frame == 0 {
                    ImageLayout::Undefined
                } else {
                    ImageLayout::ShaderReadOnly
                },
                ImageLayout::ColorAttachment,
            )),
            ownership: None,
        })])?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&self.view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.1, 0.0])),
            store: StoreOp::Store,
        })];
        let mut pass = e.begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: extent[0],
                height: extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        pass.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        pass.set_pipeline(&self.pipeline)?;
        pass.set_bind_group(0, &self.draw_groups[output], &[])?;
        pass.draw(0..6, 0..10000)?;
        pass.finish()?;
        // The next time this buffer is read it is the compute input. This
        // release pairs with that next frame's acquire above; no ownership
        // tracker is involved.
        e.transition_resources(&[barrier(
            output,
            graphics_read,
            read,
            Some(OwnershipTransfer::Release {
                from: self.graphics,
                to: self.compute,
            }),
        )])?;
        let ticket = if self.frame == 0 {
            e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: range(),
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::COLOR_OUTPUT,
                        accesses: AccessMask::COLOR_WRITE,
                    },
                    to: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                layout: Some((ImageLayout::ColorAttachment, ImageLayout::CopySrc)),
                ownership: None,
            })])?;
            e.texture_to_buffer(
                TextureRegion {
                    texture: &self.output,
                    mip: 0,
                    origin: [0, 0, 0],
                    extent: [extent[0], extent[1], 1],
                    aspect: TextureAspects::COLOR,
                },
                &self.readback,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: extent[0] * 4,
                    rows_per_image: extent[1],
                },
            )?;
            Some((
                e.readback_buffer(
                    &self.readback,
                    0..u64::from(extent[0]) * u64::from(extent[1]) * 4,
                )?,
                particle_ticket.ok_or(Error::InvalidState)?,
            ))
        } else {
            None
        };
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: if ticket.is_some() {
                    TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    }
                } else {
                    TextureUses {
                        stages: StageMask::COLOR_OUTPUT,
                        accesses: AccessMask::COLOR_WRITE,
                    }
                },
                to: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((
                if ticket.is_some() {
                    ImageLayout::CopySrc
                } else {
                    ImageLayout::ColorAttachment
                },
                ImageLayout::ShaderReadOnly,
            )),
            ownership: None,
        })])?;
        self.display.encode(&mut e, image, extent)?;
        Ok((c.finish()?, e.finish()?, ticket))
    }
    pub(super) fn submit(
        &self,
        d: &Device<B>,
        compute: RecordedWork<B>,
        graphics: RecordedWork<B>,
        image: SurfaceImage<B>,
    ) -> Result<CompletionPoint<B>> {
        let id = PlanId {
            device: d.id(),
            ordinal: NEXT.fetch_add(1, Ordering::Relaxed),
        };
        let wait = |previous, stages, accesses| StepWait {
            previous: StepId {
                plan: id,
                index: previous,
            },
            destination: AccessScope { stages, accesses },
        };
        // SAFETY: queues, all barriers and both producer/consumer dependencies are explicitly specified; resources remain alive through the returned completion.
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                id,
                Arc::from([
                    StepDesc {
                        lane: self.compute,
                        kind: StepKind::Commands,
                        waits: vec![],
                    },
                    StepDesc {
                        lane: self.graphics,
                        kind: StepKind::Commands,
                        // Frame zero reads the particle buffer back before the
                        // vertex shader consumes it. Later frames only need
                        // VERTEX, but this fixed plan must cover both paths.
                        waits: vec![wait(
                            0,
                            StageMask::COPY | StageMask::VERTEX,
                            AccessMask::COPY_READ | AccessMask::SHADER_READ,
                        )],
                    },
                    StepDesc {
                        lane: self.graphics,
                        kind: StepKind::Present,
                        waits: vec![wait(1, StageMask::ALL_COMMANDS, AccessMask::ALL)],
                    },
                ]),
            )
        };
        let steps = vec![
            BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![compute]),
            },
            BoundStep {
                step: plan.step(1),
                payload: StepPayload::Commands(vec![graphics]),
            },
            BoundStep {
                step: plan.step(2),
                payload: StepPayload::Present(PresentBinding::new(image)),
            },
        ];
        // SAFETY: see plan construction; the caller waits before resource reuse.
        match unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps,
                completion: CompletionRequest::End,
            })
        } {
            SubmitOutcome::Accepted(r) => r
                .checkpoints
                .into_iter()
                .next()
                .map(|c| c.point)
                .ok_or(Error::InvalidState),
            SubmitOutcome::Rejected { error, .. } => Err(error),
            SubmitOutcome::PartiallyAccepted { error, .. }
            | SubmitOutcome::Indeterminate { error, .. } => {
                // Keep the native outcome (and its retained work) alive until
                // idle. Returning a drain error would hide the actual submit
                // failure; if drain itself fails we cannot safely drop that
                // work, so terminate without running destructors.
                if let Err(drain_error) = utils::block_on(d.wait_idle()) {
                    eprintln!(
                        "23_multi_queue: submission failed: {error:?}; GPU drain failed: {drain_error:?}"
                    );
                    std::process::abort();
                }
                Err(error)
            }
        }
    }
}
pub(super) fn save(bytes: &[u8], extent: [u32; 2]) -> Result<()> {
    let lit = bytes.chunks_exact(4).filter(|p| p[0] > 20).count();
    if bytes.len() != (extent[0] * extent[1] * 4) as usize || lit < 10_000 {
        return Err(Error::Backend(format!(
            "particle readback has only {lit} lit pixels"
        )));
    }
    if let Some(path) = std::env::var_os("FLUXEL_NBODY_READBACK") {
        use std::io::Write;
        let mut file = std::fs::File::create(path).map_err(|e| Error::Backend(e.to_string()))?;
        write!(file, "P6\n{} {}\n255\n", extent[0], extent[1])
            .map_err(|e| Error::Backend(e.to_string()))?;
        for pixel in bytes.chunks_exact(4) {
            file.write_all(&pixel[..3])
                .map_err(|e| Error::Backend(e.to_string()))?;
        }
    }
    Ok(())
}
