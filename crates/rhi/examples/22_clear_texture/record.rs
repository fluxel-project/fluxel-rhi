//! Clear the texture, draw the triangle and verify the completed readback.

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
            let y = 0.25 * extent[0] as f32 / extent[1] as f32;
            let vertices: [f32; 21] = [
                0.0, y, 0.0, 1.0, 0.0, 0.0, 1.0, 0.25, -y, 0.0, 0.0, 1.0, 0.0, 1.0, -0.25, -y, 0.0,
                0.0, 0.0, 1.0, 1.0,
            ];
            let bytes: Vec<u8> = vertices.into_iter().flat_map(f32::to_ne_bytes).collect();
            e.write_buffer(&self.vertex, 0, &bytes)?;
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
        e.clear_texture(
            &self.output,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Float([0.0, 0.2, 0.4, 1.0])),
        )?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Texture(&self.view),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let mut pass = e.begin_raster(&RasterPassDesc {
            label: Some("official triangle"),
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
        pass.set_vertex_buffer(0, &self.vertex, 0..84)?;
        pass.draw(0..3, 0..1)?;
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
    if pixels.get(0..4) != Some(&[0, 51, 102, 255]) {
        return Err(Error::Backend(
            "standalone texture clear produced an unexpected background".into(),
        ));
    }
    println!("CLEAR_TEXTURE_OK");
    if let Some(path) = std::env::var_os("FLUXEL_CLEAR_READBACK") {
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
