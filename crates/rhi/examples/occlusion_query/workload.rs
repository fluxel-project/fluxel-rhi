//! Occlusion-query workload creation, recording, and completion consumption.

use super::*;

impl<B: OcclusionBackend> Workload<B> {
    pub async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        let q = d
            .queries(QueryKind::Occlusion)
            .ok_or(Error::Unsupported("occlusion queries"))?
            .create(&QueryDesc {
                kind: QueryKind::Occlusion,
                count: 2,
            })?;
        let make = |name: &str, n: usize, usage: BufferUsage| {
            d.create_buffer(&BufferDesc {
                label: Some(name.into()),
                size: n as u64,
                usage: usage.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
        };
        // C++ uses FlipY with a positive-height Vulkan viewport. Fluxel's
        // WebGPU-compatible viewport has negative native height, so retain
        // the same final orientation by cancelling that loader-space flip.
        let options = gltf::LoadOptions::CPP_PORT.with_flip_y(false);
        let plane =
            gltf::load_embedded_model(PLANE, options).map_err(|x| Error::Backend(x.to_string()))?;
        let teapot = gltf::load_embedded_model(TEAPOT, options)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let sphere = gltf::load_embedded_model(SPHERE, options)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let plane_vertices = make(
            "11 occluder vertices",
            plane.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let plane_indices = make(
            "11 occluder indices",
            plane.indices.len(),
            BufferUsage::INDEX,
        )?;
        let teapot_vertices = make(
            "11 teapot vertices",
            teapot.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let teapot_indices = make(
            "11 teapot indices",
            teapot.indices.len(),
            BufferUsage::INDEX,
        )?;
        let sphere_vertices = make(
            "11 sphere vertices",
            sphere.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let sphere_indices = make(
            "11 sphere indices",
            sphere.indices.len(),
            BufferUsage::INDEX,
        )?;
        let plane_uniform = make("11 occluder transform", 256, BufferUsage::UNIFORM)?;
        let teapot_uniform = make("11 teapot transform", 256, BufferUsage::UNIFORM)?;
        let sphere_uniform = make("11 sphere transform", 256, BufferUsage::UNIFORM)?;
        let vs = common::native_shader_with_wgsl(
            d,
            "11 VS",
            "vs_main",
            ShaderStage::Vertex,
            VS,
            VS_SPIRV,
            WGSL,
        )
        .await?;
        let simple_ps = common::native_shader_with_wgsl(
            d,
            "11 simple PS",
            "simple_ps",
            ShaderStage::Fragment,
            SIMPLE_PS,
            SIMPLE_PS_SPIRV,
            WGSL,
        )
        .await?;
        let mesh_ps = common::native_shader_with_wgsl(
            d,
            "11 mesh PS",
            "mesh_ps",
            ShaderStage::Fragment,
            MESH_PS,
            MESH_PS_SPIRV,
            WGSL,
        )
        .await?;
        let occluder_ps = common::native_shader_with_wgsl(
            d,
            "11 occluder PS",
            "occluder_ps",
            ShaderStage::Fragment,
            OCCLUDER_PS,
            OCCLUDER_PS_SPIRV,
            WGSL,
        )
        .await?;
        let layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("11 transform layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    // WGSL uniform structures have a 16-byte final alignment;
                    // the 240 bytes of fields therefore occupy a 256-byte binding.
                    min_size: 256,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let bind = |buffer: &Buffer<B>, name: &str| {
            d.create_bind_group(&BindGroupDesc {
                label: Some(name.into()),
                layout: &layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer,
                        range: 0..256,
                    },
                }],
            })
        };
        let plane_group = bind(&plane_uniform, "11 occluder group")?;
        let teapot_group = bind(&teapot_uniform, "11 teapot group")?;
        let sphere_group = bind(&sphere_uniform, "11 sphere group")?;
        let l = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("11 layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let face = StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        };
        let pipeline = |label: &str,
                        fragment: Shader<B>,
                        entry: &str,
                        blend: Option<BlendState>| RasterPipelineDesc {
            label: Some(label.into()),
            layout: l.clone(),
            vertex: PipelineStage {
                module: vs.clone(),
                entry: "vs_main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: entry.into(),
                constants: vec![],
            }),
            vertex_buffers: vec![VertexBufferLayout {
                stride: 96,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    VertexAttribute {
                        location: 1,
                        offset: 12,
                        format: VertexFormat::Float32(3),
                    },
                    VertexAttribute {
                        location: 2,
                        offset: 24,
                        format: VertexFormat::Float32(2),
                    },
                    VertexAttribute {
                        location: 3,
                        offset: 32,
                        format: VertexFormat::Float32(3),
                    },
                ],
            }],
            color_targets: vec![Some(ColorTargetState {
                format,
                blend,
                write_mask: 15,
            })],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil_front: face,
                stencil_back: face,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.,
                depth_bias_clamp: 0.,
            }),
            multisample: MultisampleState {
                count: 1,
                mask: u64::from(u32::MAX),
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        };
        let simple_pipeline = d
            .raster_pipeline(pipeline("11 simple pipeline", simple_ps, "simple_ps", None))
            .await?;
        let mesh_pipeline = d
            .raster_pipeline(pipeline("11 mesh pipeline", mesh_ps, "mesh_ps", None))
            .await?;
        let occluder_pipeline = d
            .raster_pipeline(pipeline(
                "11 occluder pipeline",
                occluder_ps,
                "occluder_ps",
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::SrcColor,
                        dst_factor: BlendFactor::OneMinusSrcColor,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                }),
            ))
            .await?;
        let results = d.create_buffer(&BufferDesc {
            label: Some("11 resolved occlusion results".into()),
            size: 16,
            // WebGPU cannot map a QUERY_RESOLVE buffer directly: MAP_READ
            // buffers may only also carry COPY_DST.  ReadbackBackend copies
            // this resolve target to its MAP_READ staging allocation.
            usage: BufferUsage::QUERY_RESOLVE.union(BufferUsage::COPY_SRC),
            memory: MemoryClass::Device,
        })?;
        Ok(Self {
            lane,
            plane_vertices,
            plane_indices,
            teapot_vertices,
            teapot_indices,
            sphere_vertices,
            sphere_indices,
            plane_count: (plane.indices.len() / 4) as u32,
            teapot_count: (teapot.indices.len() / 4) as u32,
            sphere_count: (sphere.indices.len() / 4) as u32,
            plane_len: plane.vertices.len() as u64,
            teapot_len: teapot.vertices.len() as u64,
            sphere_len: sphere.vertices.len() as u64,
            plane_index_len: plane.indices.len() as u64,
            teapot_index_len: teapot.indices.len() as u64,
            sphere_index_len: sphere.indices.len() as u64,
            plane_uniform,
            teapot_uniform,
            sphere_uniform,
            plane_group,
            teapot_group,
            sphere_group,
            simple_pipeline,
            mesh_pipeline,
            occluder_pipeline,
            queries: q,
            results,
            visible: [true; 2],
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<Frame<B>> {
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let mut e = d.encoder(self.lane)?;
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
        // The C++ sample queries into a pass that it clears before display.
        // Core has no intra-pass attachment clear yet, so use a disposable
        // color/depth pair for the exact same query visibility work.
        let (query_color, query_color_view) = color(d, extent, "11 query color")?;
        let (_query_depth, query_depth_view) = depth(d, extent)?;
        let (_display_depth, display_depth_view) = depth(d, extent)?;
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
        queries.reset(&mut e, &self.queries, 0..2)?;
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
            occlusion_query_set: Some(&self.queries),
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
        queries.begin_occlusion(&mut p, &self.queries, 0)?;
        p.set_bind_group(0, &self.teapot_group, &[])?;
        p.set_vertex_buffer(0, &self.teapot_vertices, 0..self.teapot_len)?;
        p.set_index_buffer(
            &self.teapot_indices,
            0..self.teapot_index_len,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..self.teapot_count, 0, 0..1)?;
        queries.end_occlusion(&mut p)?;
        queries.begin_occlusion(&mut p, &self.queries, 1)?;
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
        queries.resolve(&mut e, &self.queries, 0..2, &self.results, 0)?;
        let ticket = e.readback_buffer(&self.results, 0..16)?;
        Ok(Frame {
            work: e.finish()?,
            ticket,
        })
    }
    pub(super) fn consume(
        &mut self,
        d: &Device<B>,
        ticket: ReadbackTicket<B>,
        c: &CompletionPoint<B>,
    ) -> Result<()> {
        common::block_on(d.wait_completion(c))?;
        // SAFETY: `submit` returned this end checkpoint for the command step
        // that created `ticket`, and the wait above observed that exact point.
        let v = common::block_on(unsafe { d.read_after(ticket, c) })?.bytes;
        self.consume_bytes(&v)
    }
    pub(super) async fn consume_async(
        &mut self,
        d: &Device<B>,
        ticket: ReadbackTicket<B>,
        c: &CompletionPoint<B>,
    ) -> Result<()> {
        d.wait_completion(c).await?;
        // SAFETY: this ticket and checkpoint come from one `Frame`; the
        // awaited end checkpoint covers that frame's query resolve.
        let readback = unsafe { d.read_after(ticket, c) }.await?;
        self.consume_bytes(&readback.bytes)
    }
    fn consume_bytes(&mut self, v: &[u8]) -> Result<()> {
        if v.len() != 16 {
            return Err(Error::Backend("occlusion result size".into()));
        }
        let passed = [
            u64::from_le_bytes(v[..8].try_into().map_err(|_| Error::InvalidState)?),
            u64::from_le_bytes(v[8..].try_into().map_err(|_| Error::InvalidState)?),
        ];
        if std::env::var_os("FLUXEL_QUERY_TRACE").is_some() {
            println!(
                "occlusion samples: teapot={} sphere={}",
                passed[0], passed[1]
            );
        }
        self.visible = [passed[0] != 0, passed[1] != 0];
        Ok(())
    }
}
