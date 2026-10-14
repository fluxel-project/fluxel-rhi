//! Native and visual-fallback pass encoding for the occlusion-query scene.

use super::*;

impl<B: OcclusionWorkloadBackend> Workload<B> {
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<Frame<B>>
    where
        B: OcclusionBackend,
    {
        let mut e = d.encoder(self.lane)?;
        let ticket = self.encode(d, &mut e, image, extent)?;
        Ok(Frame {
            work: e.finish()?,
            ticket,
        })
    }

    /// Issues the same query/display workload into a caller-owned encoder.
    /// Immediate GL calls execute here; the shared scene and query order stay
    /// identical to the Deferred DX12/Vulkan/WebGPU path.
    pub fn encode(
        &mut self,
        d: &Device<B>,
        e: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<ReadbackTicket<B>> {
        self.encode_scene(d, e, image, extent)?
            .ok_or(Error::InvalidState)
    }

    pub fn encode_scene(
        &mut self,
        d: &Device<B>,
        e: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<Option<ReadbackTicket<B>>> {
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        if self.visible == [true, true] {
            let options = gltf::LoadOptions::CPP_PORT.with_flip_y(false);
            let plane = gltf::load_embedded_model(PLANE, options)
                .map_err(|x| Error::Backend(x.to_string()))?;
            let teapot = gltf::load_embedded_model(TEAPOT, options)
                .map_err(|x| Error::Backend(x.to_string()))?;
            let sphere = gltf::load_embedded_model(SPHERE, options)
                .map_err(|x| Error::Backend(x.to_string()))?;
            e.write_buffer(&self.plane_vertices, 0, &plane.vertices)?;
            e.write_buffer(&self.plane_indices, 0, &plane.indices)?;
            e.write_buffer(&self.teapot_vertices, 0, &teapot.vertices)?;
            e.write_buffer(&self.teapot_indices, 0, &teapot.indices)?;
            e.write_buffer(&self.sphere_vertices, 0, &sphere.vertices)?;
            e.write_buffer(&self.sphere_indices, 0, &sphere.indices)?;
        }
        e.write_buffer(
            &self.plane_uniform,
            0,
            &transform(extent, [0.0, 0.0, 0.0], 6.0, [0.0, 0.0, 1.0, 0.5], true),
        )?;
        e.write_buffer(
            &self.teapot_uniform,
            0,
            &transform(
                extent,
                [0.0, 0.0, -3.0],
                1.0,
                [1.0, 0.0, 0.0, 1.0],
                self.visible[0],
            ),
        )?;
        e.write_buffer(
            &self.sphere_uniform,
            0,
            &transform(
                extent,
                [0.0, 0.0, 3.0],
                1.0,
                [0.0, 1.0, 0.0, 1.0],
                self.visible[1],
            ),
        )?;
        let (_display_depth, display_depth_view) = depth(d, extent)?;
        if let (Some(query_set), Some(_results)) = (&self.queries, &self.results) {
            // The C++ sample queries into a pass that it clears before display.
            // Core has no intra-pass attachment clear yet, so use a disposable
            // color/depth pair for the exact same query visibility work.
            // The scratch color exists only to reproduce query-pass depth/order;
            // RGBA8 is the GL backend's proven attachment format, while the
            // caller-owned surface retains the requested display format.
            let (query_color, query_color_view) = color(d, extent, "11 query color")?;
            let (_query_depth, query_depth_view) = depth(d, extent)?;
            e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &query_color,
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
                layout: Some((ImageLayout::Undefined, ImageLayout::ColorAttachment)),
                ownership: None,
            })])?;
            let colors = [Some(ColorAttachment {
                view: AttachmentView::Texture(&query_color_view),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 0.])),
                store: StoreOp::Store,
            })];
            let da = DepthStencilAttachment {
                view: AttachmentView::Texture(&query_depth_view),
                depth: Some((LoadOp::Clear(1.), StoreOp::Discard)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            };
            let queries = d
                .queries(QueryKind::Occlusion)
                .ok_or(Error::Unsupported("occlusion queries"))?;
            queries.reset(e, query_set, 0..2)?;
            let mut p = e.begin_raster(&RasterPassDesc {
                label: Some("11 occlusion query pass"),
                colors: &colors,
                depth_stencil: Some(da),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: extent[0],
                    height: extent[1],
                },
                multiview_mask: None,
                timestamp_writes: None,
                // WebGPU declares the query set when beginning this render pass;
                // begin_occlusion/end_occlusion below select its individual slots.
                occlusion_query_set: Some(query_set),
                secondary_inheritance: None,
            })?;
            p.viewport(Viewport {
                x: 0.,
                y: 0.,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.,
                max_depth: 1.,
            })?;
            p.scissor(Rect {
                x: 0,
                y: 0,
                width: extent[0],
                height: extent[1],
            })?;
            p.set_pipeline(&self.simple_pipeline)?;
            p.set_bind_group(0, &self.plane_group, &[])?;
            p.set_vertex_buffer(0, &self.plane_vertices, 0..self.plane_len)?;
            p.set_index_buffer(
                &self.plane_indices,
                0..self.plane_index_len,
                IndexFormat::Uint32,
            )?;
            p.draw_indexed(0..self.plane_count, 0, 0..1)?;
            queries.begin_occlusion(&mut p, query_set, 0)?;
            p.set_bind_group(0, &self.teapot_group, &[])?;
            p.set_vertex_buffer(0, &self.teapot_vertices, 0..self.teapot_len)?;
            p.set_index_buffer(
                &self.teapot_indices,
                0..self.teapot_index_len,
                IndexFormat::Uint32,
            )?;
            p.draw_indexed(0..self.teapot_count, 0, 0..1)?;
            queries.end_occlusion(&mut p)?;
            queries.begin_occlusion(&mut p, query_set, 1)?;
            p.set_bind_group(0, &self.sphere_group, &[])?;
            p.set_vertex_buffer(0, &self.sphere_vertices, 0..self.sphere_len)?;
            p.set_index_buffer(
                &self.sphere_indices,
                0..self.sphere_index_len,
                IndexFormat::Uint32,
            )?;
            p.draw_indexed(0..self.sphere_count, 0, 0..1)?;
            queries.end_occlusion(&mut p)?;
            p.finish()?;
        }
        // Visible pass: clear independently and draw all models. The mesh
        // shader uses the preceding query result only to dim an occluded mesh,
        // matching the C++ sample's `visible` uniform rather than skipping it.
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 0.])),
            store: StoreOp::Store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&display_depth_view),
            depth: Some((LoadOp::Clear(1.), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("11 visible pass"),
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
        p.viewport(Viewport {
            x: 0.,
            y: 0.,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.,
            max_depth: 1.,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        p.set_pipeline(&self.mesh_pipeline)?;
        p.set_bind_group(0, &self.teapot_group, &[])?;
        p.set_vertex_buffer(0, &self.teapot_vertices, 0..self.teapot_len)?;
        p.set_index_buffer(
            &self.teapot_indices,
            0..self.teapot_index_len,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..self.teapot_count, 0, 0..1)?;
        p.set_bind_group(0, &self.sphere_group, &[])?;
        p.set_vertex_buffer(0, &self.sphere_vertices, 0..self.sphere_len)?;
        p.set_index_buffer(
            &self.sphere_indices,
            0..self.sphere_index_len,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..self.sphere_count, 0, 0..1)?;
        p.set_pipeline(&self.occluder_pipeline)?;
        p.set_bind_group(0, &self.plane_group, &[])?;
        p.set_vertex_buffer(0, &self.plane_vertices, 0..self.plane_len)?;
        p.set_index_buffer(
            &self.plane_indices,
            0..self.plane_index_len,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..self.plane_count, 0, 0..1)?;
        p.finish()?;
        if let (Some(query_set), Some(results)) = (&self.queries, &self.results) {
            d.queries(QueryKind::Occlusion)
                .ok_or(Error::Unsupported("occlusion queries"))?
                .resolve(e, query_set, 0..2, results, 0)?;
            e.readback_buffer(results, 0..16).map(Some)
        } else {
            Ok(None)
        }
    }
}
