//! Record the triangle, query its execution and verify completed readbacks.

use super::*;

type Tickets<B> = (ReadbackTicket<B>, ReadbackTicket<B>);
impl<
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + QueryBackend,
> Demo<B>
{
    pub(super) fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<(RecordedWork<B>, Option<Tickets<B>>)>
    where
        B: DeferredBackend,
    {
        let mut e = d.encoder(self.lane)?;
        let ts = d.queries(QueryKind::Timestamp).ok_or(Error::InvalidState)?;
        let stats = d
            .queries(QueryKind::PipelineStatistics)
            .ok_or(Error::InvalidState)?;
        if !self.initialized {
            let y = 0.25 * extent[0] as f32 / extent[1] as f32;
            let vertices: [f32; 21] = [
                0.0, y, 0.0, 1.0, 0.0, 0.0, 1.0, 0.25, -y, 0.0, 0.0, 1.0, 0.0, 1.0, -0.25, -y, 0.0,
                0.0, 0.0, 1.0, 1.0,
            ];
            let bytes: Vec<u8> = vertices.into_iter().flat_map(f32::to_ne_bytes).collect();
            e.write_buffer(&self.vertex, 0, &bytes)?;
            ts.timestamp(&mut e, &self.timestamps, 0)?;
            stats.begin_statistics(&mut e, &self.statistics, 0)?;
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
            stats.end_statistics(&mut e)?;
            ts.timestamp(&mut e, &self.timestamps, 1)?;
            ts.resolve(&mut e, &self.timestamps, 0..2, &self.query_buffer, 0)?;
            stats.resolve(&mut e, &self.statistics, 0..1, &self.query_buffer, 16)?;
            e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &self.query_buffer,
                    range: 0..104,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::ALL_COMMANDS,
                        accesses: AccessMask::EMPTY,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            })])?;
            let queries = e.readback_buffer(&self.query_buffer, 0..104)?;
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
            Some((pixels, queries))
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
        + QueryBackend,
>(
    demo: &Demo<B>,
    extent: [u32; 2],
    pixels: &[u8],
    queries: &[u8],
) -> Result<()> {
    if queries.len() != 104
        || pixels.len() as u64 != u64::from(demo.row_pitch) * u64::from(extent[1])
    {
        return Err(Error::InvalidRange);
    }
    let value = |offset| {
        u64::from_ne_bytes(
            queries[offset..offset + 8]
                .try_into()
                .expect("validated query length"),
        )
    };
    let (start, end, vertices, primitives) = (value(0), value(8), value(16), value(24));
    if start == 0 || end < start || vertices != 3 || primitives != 1 {
        return Err(Error::Backend(format!(
            "unexpected GPU query data: {start}, {end}, {vertices}, {primitives}"
        )));
    }
    println!(
        "GPU_QUERY_OK timestamp_start={start} timestamp_end={end} frequency={} ia_vertices={vertices} ia_primitives={primitives}",
        demo.frequency
    );
    if let Some(path) = std::env::var_os("FLUXEL_QUERY_READBACK") {
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
