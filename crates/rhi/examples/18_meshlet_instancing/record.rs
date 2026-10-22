//! Record the mesh dispatch and read back the first frame.

use super::*;

impl<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
    Demo<B>
{
    pub(super) fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<(RecordedWork<B>, Option<ReadbackTicket<B>>)>
    where
        B: DeferredBackend,
    {
        let mut e = d.encoder(self.lane)?;
        if !self.initialized {
            let mesh = model::load(include_bytes!(
                "../assets/18_meshlet_instancing/ToyRobot.bin"
            ))
            .ok_or(Error::InvalidState)?;
            let mut info = vec![0u8; 256];
            info[0..4].copy_from_slice(&mesh.index_size.to_le_bytes());
            info[4..8].copy_from_slice(&mesh.meshlet_count.to_le_bytes());
            let globals = scene::globals();
            let instance = scene::identity_instance();
            let data = [
                globals.as_slice(),
                info.as_slice(),
                mesh.vertices,
                mesh.meshlets,
                mesh.unique_indices,
                mesh.primitives,
                instance.as_slice(),
            ];
            for (i, bytes) in data.iter().enumerate() {
                e.write_buffer(&self.buffers[i], 0, bytes)?;
                e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                    target: BufferSlice {
                        buffer: &self.buffers[i],
                        range: 0..bytes.len() as u64,
                    },
                    usage: StateTransition {
                        from: BufferUses {
                            stages: StageMask::COPY,
                            accesses: AccessMask::COPY_WRITE,
                        },
                        to: BufferUses {
                            stages: StageMask::MESH
                                | if i == 0 {
                                    StageMask::FRAGMENT
                                } else {
                                    StageMask::EMPTY
                                },
                            accesses: if i < 2 {
                                AccessMask::UNIFORM_READ
                            } else {
                                AccessMask::SHADER_READ
                            },
                        },
                    },
                    ownership: None,
                })])?;
            }
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
            label: Some("official meshlets"),
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
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        d.mesh()
            .ok_or(Error::InvalidState)?
            .bind_pipeline(&mut pass, &self.pipeline)?;
        pass.set_bind_group(0, &self.group, &[])?;
        d.raster_extras()
            .ok_or(Error::InvalidState)?
            .push_constants(&mut pass, StageMask::MESH, 0, &[1, 0, 0, 0, 0, 0, 0, 0])?;
        d.mesh()
            .ok_or(Error::InvalidState)?
            .dispatch(&mut pass, [311, 1, 1])?;
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
            Some(pixels)
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
) -> Result<()> {
    if pixels.len() as u64 != u64::from(demo.row_pitch) * u64::from(extent[1]) {
        return Err(Error::InvalidRange);
    }
    if let Some(path) = std::env::var_os("FLUXEL_MESH_READBACK") {
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
