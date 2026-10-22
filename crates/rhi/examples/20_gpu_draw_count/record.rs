//! Record the GPU count dispatch and read back the first frame.

use super::*;

impl<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
    Demo<B>
{
    pub(super) fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<(
        RecordedWork<B>,
        Option<(ReadbackTicket<B>, ReadbackTicket<B>, ReadbackTicket<B>)>,
    )>
    where
        B: DeferredBackend,
    {
        let mut e = d.encoder(self.lane)?;
        if !self.initialized {
            let vertices = [0.0f32, 0.05, 1.0, 0.05, -0.05, 1.0, -0.05, -0.05, 1.0];
            let vertices: Vec<u8> = vertices.into_iter().flat_map(f32::to_ne_bytes).collect();
            e.write_buffer(&self.vertex, 0, &vertices)?;
            e.write_buffer(
                &self.scene,
                0,
                include_bytes!("../assets/20_gpu_draw_count/frame180.bin"),
            )?;
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
            e.transition_resources(&[
                barrier(
                    &self.vertex,
                    36,
                    BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    BufferUses {
                        stages: StageMask::VERTEX_INPUT,
                        accesses: AccessMask::VERTEX_READ,
                    },
                ),
                barrier(
                    &self.scene,
                    262144,
                    BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                    BufferUses {
                        stages: StageMask::COMPUTE | StageMask::VERTEX_INPUT,
                        accesses: AccessMask::SHADER_READ | AccessMask::VERTEX_READ,
                    },
                ),
                barrier(
                    &self.draws,
                    16384,
                    BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    BufferUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                ),
                barrier(
                    &self.count,
                    4,
                    BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    BufferUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                ),
            ])?;
            let mut compute = d.compute().ok_or(Error::InvalidState)?.begin(&mut e)?;
            compute.set_pipeline(&self.compute_pipeline)?;
            compute.set_bind_group(0, &self.compute_group, &[])?;
            compute.dispatch([1, 1, 1])?;
            compute.finish()?;
            e.transition_resources(&[
                barrier(
                    &self.draws,
                    16384,
                    BufferUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                    BufferUses {
                        stages: StageMask::INDIRECT,
                        accesses: AccessMask::INDIRECT_READ,
                    },
                ),
                barrier(
                    &self.count,
                    4,
                    BufferUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                    BufferUses {
                        stages: StageMask::INDIRECT,
                        accesses: AccessMask::INDIRECT_READ,
                    },
                ),
            ])?;
            e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &self.depth,
                    aspects: TextureAspects::DEPTH,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: TextureUses {
                        stages: StageMask::FRAGMENT,
                        accesses: AccessMask::DEPTH_WRITE,
                    },
                },
                layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
                ownership: None,
            })])?;
        }
        let range = || Subresources {
            texture: &self.output,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        };
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
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
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
            },
            layout: Some((
                if self.initialized {
                    ImageLayout::ShaderReadOnly
                } else {
                    ImageLayout::Undefined
                },
                ImageLayout::ColorAttachment,
            )),
            ownership: None,
        })])?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&self.view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.2, 0.4, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = e.begin_raster(&RasterPassDesc {
            label: Some("official indirect triangles"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&self.depth_view),
                depth: Some((LoadOp::Clear(1.0), StoreOp::Store)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
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
            x: 320,
            y: 0,
            width: 640,
            height: extent[1],
        })?;
        pass.set_pipeline(&self.pipeline)?;
        pass.set_vertex_buffer(0, &self.vertex, 0..36)?;
        pass.set_vertex_buffer(1, &self.scene, 0..262144)?;
        d.raster_extras()
            .ok_or(Error::InvalidState)?
            .multi_draw_count(&mut pass, &self.draws, 0, &self.count, 0, 1024, 16, false)?;
        pass.finish()?;
        let tickets = if !self.initialized {
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
                    bytes_per_row: self.row_pitch,
                    rows_per_image: extent[1],
                },
            )?;
            let pixels = e.readback_buffer(
                &self.readback,
                0..u64::from(self.row_pitch) * u64::from(extent[1]),
            )?;
            e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &self.count,
                    range: 0..4,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::INDIRECT,
                        accesses: AccessMask::INDIRECT_READ,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            })])?;
            e.copy_buffer(&self.count, 0, &self.count_readback, 0, 4)?;
            let count = e.readback_buffer(&self.count_readback, 0..4)?;
            e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &self.count,
                    range: 0..4,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                    to: BufferUses {
                        stages: StageMask::INDIRECT,
                        accesses: AccessMask::INDIRECT_READ,
                    },
                },
                ownership: None,
            })])?;
            e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &self.draws,
                    range: 0..16384,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::INDIRECT,
                        accesses: AccessMask::INDIRECT_READ,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            })])?;
            e.copy_buffer(&self.draws, 0, &self.draws_readback, 0, 16384)?;
            let draws = e.readback_buffer(&self.draws_readback, 0..16384)?;
            e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &self.draws,
                    range: 0..16384,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                    to: BufferUses {
                        stages: StageMask::INDIRECT,
                        accesses: AccessMask::INDIRECT_READ,
                    },
                },
                ownership: None,
            })])?;
            Some((pixels, count, draws))
        } else {
            None
        };
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: if tickets.is_some() {
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
                if tickets.is_some() {
                    ImageLayout::CopySrc
                } else {
                    ImageLayout::ColorAttachment
                },
                ImageLayout::ShaderReadOnly,
            )),
            ownership: None,
        })])?;
        self.display.encode(&mut e, image, extent)?;
        Ok((e.finish()?, tickets))
    }
}

pub(super) fn verify_and_save<
    B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend,
>(
    demo: &Demo<B>,
    extent: [u32; 2],
    pixels: &[u8],
    count: &[u8],
    draws: &[u8],
) -> Result<()> {
    let count = u32::from_ne_bytes(count.try_into().map_err(|_| Error::InvalidRange)?);
    if count != 452 || draws.len() != 16384 {
        return Err(Error::Backend(format!("unexpected draw count {count}")));
    }
    let mut previous = None;
    for command in draws.chunks_exact(16).take(count as usize) {
        let word = |offset| {
            u32::from_ne_bytes(
                command[offset..offset + 4]
                    .try_into()
                    .expect("command word"),
            )
        };
        let instance = word(12);
        if word(0) != 3
            || word(4) != 1
            || word(8) != 0
            || instance >= 1024
            || previous.is_some_and(|last| last >= instance)
        {
            return Err(Error::Backend("invalid GPU-generated draw command".into()));
        }
        previous = Some(instance);
    }
    println!("GPU_DRAW_COUNT={count}");
    if pixels.len() as u64 != u64::from(demo.row_pitch) * u64::from(extent[1]) {
        return Err(Error::InvalidRange);
    }
    if let Some(path) = std::env::var_os("FLUXEL_DRAW_COUNT_READBACK") {
        let mut ppm = format!("P6\n{} {}\n255\n", extent[0], extent[1]).into_bytes();
        for row in pixels
            .chunks_exact(demo.row_pitch as usize)
            .take(extent[1] as usize)
        {
            for pixel in row[..extent[0] as usize * 4].chunks_exact(4) {
                ppm.extend_from_slice(&pixel[..3]);
            }
        }
        std::fs::write(path, ppm).map_err(|error| Error::Backend(error.to_string()))?;
    }
    Ok(())
}
