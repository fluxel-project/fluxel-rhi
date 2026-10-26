//! Reusable resources for continuously animated ray-query presentation.
use super::{particles, run, scene};
#[path = "live_init.rs"]
mod init;
use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;
use run::{HEIGHT, WIDTH, block_on, bytes};
use std::sync::Arc;
pub struct Live<B: ResourceBackend + PipelineBackend> {
    device: Device<B>,
    lane: LaneId,
    output: Texture<B>,
    particle: Buffer<B>,
    constants: Buffer<B>,
    ui_buffer: Buffer<B>,
    pipeline: ComputePipeline<B>,
    group: BindGroup<B>,
    blas: AccelerationStructure<B>,
    particle_blas: AccelerationStructure<B>,
    quad_blas: AccelerationStructure<B>,
    quad_buffer: Buffer<B>,
    quad_desc: AccelerationDesc<B>,
    qscratch: Buffer<B>,
    qscratch_size: u64,
    tlas: AccelerationStructure<B>,
    tscratch: Buffer<B>,
    tscratch_size: u64,
    first: bool,
    last_beam: bool,
    last_view: Option<super::camera::View>,
    frame: u64,
}
impl<B> Live<B>
where
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + ReadbackBackend
        + DeferredBackend
        + ExecutionBackend<Mode = DeferredExecution>,
{
    pub fn output(&self) -> &Texture<B> {
        &self.output
    }
    pub fn render(
        &mut self,
        sim: &particles::Simulation,
        panel: &super::ui::Panel,
        view: super::camera::View,
    ) -> Result<()> {
        let mut values = Vec::new();
        for p in sim.particles.iter().filter(|p| p.active) {
            values.extend_from_slice(&[
                p.position[0],
                p.position[1],
                p.position[2],
                p.radius,
                p.color[0],
                p.color[1],
                p.color[2],
                p.opacity,
                p.rotation,
                0.,
                0.,
                0.,
            ]);
        }
        if panel.settings.animate || panel.settings.update_orientation_when_paused {
            self.last_beam = panel.settings.orientation == super::ui::Orientation::Beam;
            self.last_view = Some(view);
        }
        let geometry_view = self.last_view.unwrap_or(view);
        let geometry_beam = self.last_beam;
        let count = values.len() as u32 / 12;
        if values.is_empty() {
            values.resize(12, 0.);
        }
        let data = bytes(&values);
        let mut camera_values = Vec::with_capacity(16);
        for vector in [view.position, view.forward, view.right, view.up] {
            camera_values.extend_from_slice(&[vector[0], vector[1], vector[2], 0.]);
        }
        let mut cb = bytes(&camera_values);
        for n in [
            WIDTH,
            HEIGHT,
            count,
            panel.flags() | (u32::from(geometry_beam) << 8),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ] {
            cb.extend_from_slice(&n.to_ne_bytes());
        }
        cb.resize(256, 0);
        let read = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_READ,
        };
        let uniform = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::UNIFORM_READ,
        };
        // write_buffer performs its upload and returns to the idle state.
        let idle = BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::EMPTY,
        };
        let barrier = |buffer, size, from, to| {
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer,
                    range: 0..size,
                },
                usage: StateTransition { from, to },
                ownership: None,
            })
        };
        let mut enc = self.device.encoder(self.lane)?;
        let quad_data = bytes(&super::particle_acceleration::quad_vertices_view(
            sim,
            geometry_beam,
            geometry_view.right,
            geometry_view.up,
        ));
        enc.transition_resources(&[barrier(
            &self.quad_buffer,
            quad_data.len() as u64,
            BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            },
            idle,
        )])?;
        enc.write_buffer(&self.quad_buffer, 0, &quad_data)?;
        enc.transition_resources(&[barrier(
            &self.quad_buffer,
            quad_data.len() as u64,
            idle,
            BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            },
        )])?;
        if !self.first {
            enc.transition_resources(&[
                barrier(
                    &self.particle,
                    (particles::MAX_PARTICLES * 48) as u64,
                    read,
                    idle,
                ),
                barrier(&self.constants, 256, uniform, idle),
                barrier(
                    &self.ui_buffer,
                    u64::from(super::ui::WIDTH * super::ui::HEIGHT * 4),
                    read,
                    idle,
                ),
            ])?;
        }
        enc.write_buffer(&self.particle, 0, &data)?;
        enc.write_buffer(&self.constants, 0, &cb)?;
        enc.write_buffer(&self.ui_buffer, 0, &panel.pixels())?;
        enc.transition_resources(&[
            barrier(
                &self.particle,
                (particles::MAX_PARTICLES * 48) as u64,
                idle,
                read,
            ),
            barrier(&self.constants, 256, idle, uniform),
            barrier(
                &self.ui_buffer,
                u64::from(super::ui::WIDTH * super::ui::HEIGHT * 4),
                idle,
                read,
            ),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &self.output,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: if self.first {
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
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                },
                layout: Some((
                    if self.first {
                        ImageLayout::Undefined
                    } else {
                        ImageLayout::ShaderReadOnly
                    },
                    ImageLayout::General,
                )),
                ownership: None,
            }),
        ])?;
        let ray = self
            .device
            .ray()
            .ok_or(Error::Unsupported("acceleration structures"))?;
        let tlas_desc = AccelerationDesc {
            input: AccelerationInput::Tlas(super::particle_acceleration::instances(
                &self.blas,
                &self.particle_blas,
                &self.quad_blas,
                sim,
            )),
            fast_trace: true,
            allow_update: false,
        };
        // Each render waits for the preceding submission to finish, so no earlier
        // traversal can still read the TLAS when this build begins.
        enc.transition_resources(&[barrier(
            &self.tscratch,
            self.tscratch_size,
            BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
        )])?;
        enc.transition_resources(&[barrier(
            &self.qscratch,
            self.qscratch_size,
            BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
        )])?;
        ray.build(&mut enc, &self.quad_blas, &self.qscratch, &self.quad_desc)?;
        enc.transition_resources(&[ResourceTransition::Acceleration {
            target: &self.quad_blas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            },
        }])?;
        ray.build(&mut enc, &self.tlas, &self.tscratch, &tlas_desc)?;
        enc.transition_resources(&[ResourceTransition::Acceleration {
            target: &self.tlas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::ACCEL_READ,
            },
        }])?;
        let compute = self.device.compute().ok_or(Error::InvalidState)?;
        let mut pass = compute.begin(&mut enc)?;
        pass.set_pipeline(&self.pipeline)?;
        pass.set_bind_group(0, &self.group, &[])?;
        pass.dispatch([WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1])?;
        pass.finish()?;
        enc.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &self.output,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::General, ImageLayout::ShaderReadOnly)),
            ownership: None,
        })])?;
        submit(
            &self.device,
            self.lane,
            enc.finish()?,
            19000001 + self.frame,
        )?;
        self.first = false;
        self.frame += 1;
        Ok(())
    }
}
fn submit<B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    ordinal: u64,
) -> Result<()> {
    // SAFETY: the work uses the same device and lane as this one-step plan.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: resources referenced by work remain alive through the wait below.
    let result = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(receipt) => {
            receipt
                .checkpoints
                .into_iter()
                .next()
                .ok_or(Error::InvalidState)?
                .point
        }
        SubmitOutcome::Rejected { error, .. } => return Err(error),
        outcome => {
            let error = match &outcome {
                SubmitOutcome::PartiallyAccepted { error, .. }
                | SubmitOutcome::Indeterminate { error, .. } => error.clone(),
                _ => unreachable!("accepted and rejected outcomes handled above"),
            };
            // Keep the native outcome (and its retained work) alive through
            // the drain. A drain failure cannot replace the submit failure:
            // terminate before the retained outcome is dropped instead.
            if let Err(drain_error) = block_on(device.wait_idle()) {
                eprintln!(
                    "19_ray_query_particles live submission failed: {error:?}; GPU drain failed: {drain_error:?}"
                );
                std::process::abort();
            }
            drop(outcome);
            return Err(error);
        }
    };
    block_on(device.wait_completion(&point))
}
