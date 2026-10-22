//! Explicit transitions, acceleration builds and ray tracing commands.

use super::*;

impl<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
    Hello<B>
{
    pub(super) fn record(
        &self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<(RecordedWork<B>, Option<ReadbackTicket<B>>)>
    where
        B: DeferredBackend,
    {
        let ray = device.ray().ok_or(Error::Unsupported("ray tracing"))?;
        let mut encoder = device.encoder(self.lane)?;
        if !self.initialized {
            let vertices: Vec<u8> = scene::VERTICES
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            let indices: Vec<u8> = scene::INDICES
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            encoder.write_buffer(&self.vertices, 0, &vertices)?;
            encoder.write_buffer(&self.indices, 0, &indices)?;
            encoder.write_buffer(&self.table.raygen.buffer, 0, &self.table_bytes)?;
            let common = BufferUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            };
            let accel_input = BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            };
            let accel_scratch = BufferUses {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            };
            let shader_table = BufferUses {
                stages: StageMask::RAY,
                accesses: AccessMask::RAY_TRACING_SHADER_DATA_READ,
            };
            let barrier = |buffer, size, to| {
                ResourceTransition::Buffer(BufferBarrier {
                    target: BufferSlice {
                        buffer,
                        range: 0..size,
                    },
                    usage: StateTransition { from: common, to },
                    ownership: None,
                })
            };
            encoder.transition_resources(&[
                barrier(&self.vertices, 36, accel_input),
                barrier(&self.indices, 6, accel_input),
                barrier(&self.blas_scratch, self.scratch_sizes[0], accel_scratch),
                barrier(&self.tlas_scratch, self.scratch_sizes[1], accel_scratch),
                barrier(
                    &self.table.raygen.buffer,
                    self.table_bytes.len() as u64,
                    shader_table,
                ),
            ])?;
            let blas = self.blas.as_ref().ok_or(Error::InvalidState)?;
            ray.build(
                &mut encoder,
                blas,
                &self.blas_scratch,
                self.blas_desc.as_ref().ok_or(Error::InvalidState)?,
            )?;
            encoder.transition_resources(&[ResourceTransition::Acceleration {
                target: blas,
                before: AccessScope {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
                after: AccessScope {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_READ,
                },
            }])?;
            ray.build(
                &mut encoder,
                &self.tlas,
                &self.tlas_scratch,
                self.tlas_desc.as_ref().ok_or(Error::InvalidState)?,
            )?;
            encoder.transition_resources(&[ResourceTransition::Acceleration {
                target: &self.tlas,
                before: AccessScope {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
                after: AccessScope {
                    stages: StageMask::RAY,
                    accesses: AccessMask::ACCEL_READ,
                },
            }])?;
        }
        let range = || Subresources {
            texture: &self.output,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        };
        encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: if self.initialized {
                    TextureUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::SHADER_READ,
                    }
                } else {
                    TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    }
                },
                to: TextureUses {
                    stages: StageMask::RAY,
                    accesses: AccessMask::SHADER_WRITE,
                },
            },
            layout: Some((
                if self.initialized {
                    ImageLayout::ShaderReadOnly
                } else {
                    ImageLayout::Undefined
                },
                ImageLayout::General,
            )),
            ownership: None,
        })])?;
        ray.set_pipeline(&mut encoder, &self.pipeline)?;
        ray.set_bind_group(&mut encoder, 0, &self.group, &[])?;
        ray.trace(
            &mut encoder,
            &self.pipeline,
            &self.table,
            [extent[0], extent[1], 1],
        )?;
        let ticket = if self.initialized && !self.captured {
            encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: range(),
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::RAY,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                    to: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                layout: Some((ImageLayout::General, ImageLayout::CopySrc)),
                ownership: None,
            })])?;
            encoder.texture_to_buffer(
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
                    bytes_per_row: self.row_pitch,
                    rows_per_image: extent[1],
                },
            )?;
            Some(encoder.readback_buffer(
                &self.readback,
                0..u64::from(self.row_pitch) * u64::from(extent[1]),
            )?)
        } else {
            None
        };
        encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: if ticket.is_some() {
                    TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    }
                } else {
                    TextureUses {
                        stages: StageMask::RAY,
                        accesses: AccessMask::SHADER_WRITE,
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
                    ImageLayout::General
                },
                ImageLayout::ShaderReadOnly,
            )),
            ownership: None,
        })])?;
        self.display.encode(&mut encoder, image, extent)?;
        Ok((encoder.finish()?, ticket))
    }
}
