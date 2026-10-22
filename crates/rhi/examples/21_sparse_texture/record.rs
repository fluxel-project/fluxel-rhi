//! Record the sparse checkerboard and read back the first frame.

use super::*;

impl<
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + SparseBackend
        + DeferredBackend,
> Demo<B>
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
            let h = 0.25f32 * 1280.0 / 720.0;
            let quad: [f32; 30] = [
                -0.25, -h, 0.0, 0.0, 1.0, -0.25, h, 0.0, 0.0, 0.0, 0.25, -h, 0.0, 1.0, 1.0, -0.25,
                h, 0.0, 0.0, 0.0, 0.25, h, 0.0, 1.0, 0.0, 0.25, -h, 0.0, 1.0, 1.0,
            ];
            let vertices: Vec<u8> = quad.into_iter().flat_map(f32::to_ne_bytes).collect();
            e.write_buffer(&self.vertex, 0, &vertices)?;
        }
        let region = self.mapped.ok_or(Error::InvalidState)?;
        if !self.uploaded[region] {
            for mip in self.region_mips[region].clone() {
                let size = (256u32 >> mip).max(1);
                let cell = (size / 8).max(1);
                let mut texels = vec![0u8; (size * size * 4) as usize];
                for y in 0..size {
                    for x in 0..size {
                        let white = (x / cell) % 2 != (y / cell) % 2;
                        let i = ((y * size + x) * 4) as usize;
                        texels[i..i + 4].copy_from_slice(&[
                            255,
                            if white { 255 } else { 0 },
                            if white { 255 } else { 0 },
                            255,
                        ]);
                    }
                }
                e.write_texture(
                    TextureRegion {
                        texture: &self.sparse_texture,
                        mip,
                        origin: [0, 0, 0],
                        extent: [size, size, 1],
                        aspect: TextureAspects::COLOR,
                    },
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: size * 4,
                        rows_per_image: size,
                    },
                    &texels,
                )
                .map_err(|e| Error::Backend(format!("write texture mip {mip}: {e:?}")))?;
                e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                    target: Subresources {
                        texture: &self.sparse_texture,
                        aspects: TextureAspects::COLOR,
                        mips: mip..mip + 1,
                        layers: 0..1,
                    },
                    usage: StateTransition {
                        from: TextureUses {
                            stages: StageMask::EMPTY,
                            accesses: AccessMask::EMPTY,
                        },
                        to: TextureUses {
                            stages: StageMask::FRAGMENT,
                            accesses: AccessMask::SHADER_READ,
                        },
                    },
                    layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
                    ownership: None,
                })])
                .map_err(|e| Error::Backend(format!("transition sparse mip {mip}: {e:?}")))?;
            }
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
            label: Some("reserved checkerboard"),
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
        pass.set_bind_group(0, &self.groups[self.selected as usize], &[])?;
        pass.set_vertex_buffer(0, &self.vertex, 0..120)?;
        pass.draw(0..6, 0..1)?;
        pass.finish()?;
        let tickets = if !self.initialized || std::env::var_os("FLUXEL_SPARSE_MIP").is_some() {
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
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + SparseBackend
        + DeferredBackend,
>(
    demo: &Demo<B>,
    extent: [u32; 2],
    pixels: &[u8],
) -> Result<()> {
    if pixels.len() as u64 != u64::from(demo.row_pitch) * u64::from(extent[1]) {
        return Err(Error::InvalidRange);
    }
    if let Some(path) = std::env::var_os("FLUXEL_SPARSE_READBACK") {
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
