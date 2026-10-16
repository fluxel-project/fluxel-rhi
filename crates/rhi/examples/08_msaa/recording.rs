//! Resize handling and per-frame MSAA pass recording for the Voyager scene.
//!
//! This module owns pass-local command order; it does not submit work or own
//! the acquired surface lease after returning recorded work.

use super::resources::{color_target, decode_material_image, depth_target};
use super::{Msaa, MsaaBackend, MsaaDeferredBackend};
use fluxel_rhi::api::*;

#[derive(Clone, Copy)]
pub struct FrameCommit {
    uploaded: bool,
    color_state_initialized: bool,
}

impl<B: MsaaBackend> Msaa<B> {
    pub(super) fn resize(
        &mut self,
        device: &Device<B>,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<()> {
        if extent == self.target_extent {
            return Ok(());
        }
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let (color, color_view) = if self.samples > 1 {
            let (color, view) = color_target(device, extent, format, self.samples)?;
            (Some(color), Some(view))
        } else {
            (None, None)
        };
        let (depth, depth_view) = depth_target(device, extent, self.samples)?;
        self.color = color;
        self.color_view = color_view;
        self.depth = depth;
        self.depth_view = depth_view;
        self.target_extent = extent;
        self.color_state_initialized = false;
        Ok(())
    }

    pub fn record(
        &mut self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>>
    where
        B: MsaaDeferredBackend,
    {
        let mut encoder = device.encoder(self.lane)?;
        let commit = self.encode(&mut encoder, device, image, extent, time)?;
        let work = encoder.finish()?;
        self.commit_frame(commit);
        Ok(work)
    }

    /// Records the shared workload into a caller-owned encoder. The immediate
    /// GL adapter uses this path so it can issue each verb synchronously.
    pub fn encode(
        &mut self,
        encoder: &mut CommandEncoder<B>,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<FrameCommit> {
        self.resize(device, self.surface_format, extent)?;
        let first_upload = !self.uploaded;
        let initialize_color = !self.color_state_initialized;
        if first_upload {
            let model = super::super::gltf::load_embedded_model(
                super::super::VOYAGER,
                super::super::gltf::LoadOptions::PRETRANSFORM_FLIP_Y,
            )
            .map_err(|error| Error::Backend(error.to_string()))?;
            encoder.write_buffer(&self.vertex, 0, &model.vertices)?;
            encoder.write_buffer(&self.index, 0, &model.indices)?;
            if model.images.len() < self.material_textures.len() {
                return Err(Error::InvalidRange);
            }
            for (texture, image) in self.material_textures.iter().zip(&model.images) {
                let pixels = decode_material_image(image)?;
                encoder.write_texture(
                    TextureRegion {
                        texture,
                        mip: 0,
                        origin: [0; 3],
                        extent: [1024, 1024, 1],
                        aspect: TextureAspects::COLOR,
                    },
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: 4096,
                        rows_per_image: 1024,
                    },
                    &pixels,
                )?;
            }
            let reads = self
                .material_textures
                .iter()
                .map(|texture| {
                    ResourceTransition::Texture(TextureBarrier {
                        target: Subresources {
                            texture,
                            aspects: TextureAspects::COLOR,
                            mips: 0..1,
                            layers: 0..1,
                        },
                        usage: StateTransition {
                            from: TextureUses {
                                stages: StageMask::COPY,
                                accesses: AccessMask::COPY_WRITE,
                            },
                            to: TextureUses {
                                stages: StageMask::FRAGMENT,
                                accesses: AccessMask::SHADER_READ,
                            },
                        },
                        layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
                        ownership: None,
                    })
                })
                .collect::<Vec<_>>();
            encoder.transition_resources(&reads)?;
        }
        encoder.write_buffer(&self.uniform, 0, &transform(extent, time))?;
        if self.samples > 1 && initialize_color {
            let color = self.color.as_ref().ok_or(Error::InvalidState)?;
            encoder.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: color,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: TextureUses {
                        stages: StageMask::COLOR_OUTPUT,
                        accesses: AccessMask::COLOR_WRITE,
                    },
                },
                // The multisampled target has never been read. Its first
                // clear may therefore discard the allocation's undefined
                // contents; subsequent frames retain COLOR_ATTACHMENT_OPTIMAL
                // even though their StoreOp discards the pixels.
                layout: Some((ImageLayout::Undefined, ImageLayout::ColorAttachment)),
                ownership: None,
            })])?;
        }
        let (view, resolve_target, store) = if self.samples > 1 {
            (
                AttachmentView::Texture(self.color_view.as_ref().ok_or(Error::InvalidState)?),
                Some(AttachmentView::Surface(image)),
                StoreOp::Discard,
            )
        } else {
            (AttachmentView::Surface(image), None, StoreOp::Store)
        };
        let colors = [Some(ColorAttachment {
            view,
            resolve_target,
            load: LoadOp::Clear(ClearColor::Float([0.98, 0.98, 1.0, 1.0])),
            store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.depth_view),
            depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("08 MSAA crate::VOYAGER resolve"),
            colors: &colors,
            depth_stencil: Some(depth),
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
        pass.set_vertex_buffer(0, &self.vertex, 0..self.vertex_len)?;
        pass.set_index_buffer(&self.index, 0..self.index_len, IndexFormat::Uint32)?;
        for draw in &self.draws {
            pass.set_bind_group(0, &self.groups[draw.material], &[])?;
            pass.draw_indexed(draw.first..draw.first + draw.count, 0, 0..1)?;
        }
        pass.finish()?;
        Ok(FrameCommit {
            uploaded: first_upload,
            color_state_initialized: initialize_color,
        })
    }

    /// Publishes initialization facts only after the caller reports that the
    /// command step was accepted. For immediate GL, commands are already live
    /// at that point; deferred backends publish after encoder finish.
    pub fn commit_frame(&mut self, commit: FrameCommit) {
        self.uploaded |= commit.uploaded;
        self.color_state_initialized |= commit.color_state_initialized;
    }
}
fn transform(extent: [u32; 2], _time: f32) -> [u8; 256] {
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    // Matches the reference camera's 60 degree perspective, with the public
    // RHI depth interval [0, 1].
    let f = 1.0 / (60f32.to_radians() * 0.5).tan();
    let near = 0.1;
    let far = 256.0;
    let (s, c) = (-90f32.to_radians()).sin_cos();
    let projection = [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        -f,
        0.,
        0.,
        0.,
        0.,
        far / (near - far),
        -1.,
        0.,
        0.,
        far * near / (near - far),
        0.,
    ];
    // Column-major layout expected by this HLSL cbuffer. Keep this as exactly
    // sixteen floats: the old seventeen-float literal shifted the light data
    // into the matrix and gave model.w the camera's z translation.
    let model = [
        c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 2.5, 2.5, -7.5, 1.,
    ];
    let mut bytes = [0; 256];
    for (index, value) in projection
        .into_iter()
        .chain(model)
        .chain([5., -5., 5., 1.])
        .enumerate()
    {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
