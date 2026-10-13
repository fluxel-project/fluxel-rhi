use super::{
    ktx::{Ktx, frame_uniform},
    pipeline::*,
    scene::{
        GROUND_PS, GROUND_PS_SPIRV, GROUND_TEX, GROUND_VS, GROUND_VS_SPIRV,
        INDEXED_INDIRECT_STRIDE, IndirectScene, PLANT_TEX, PLANTS_PS, PLANTS_PS_SPIRV, PLANTS_VS,
        PLANTS_VS_SPIRV, SKY_PS, SKY_PS_SPIRV, SKY_VS, SKY_VS_SPIRV, instance_bytes,
    },
};
use fluxel_rhi::api::*;

pub(crate) trait IndirectBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> IndirectBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}

/// Complete plant/ground/sky workload.  Plant texture layers are uploaded as
/// individual array subresources, then sampled by the instance layer attribute.
pub(crate) struct Workload<B: IndirectBackend> {
    lane: LaneId,
    scene: IndirectScene<B>,
    uniform: Buffer<B>,
    plant_texture: Texture<B>,
    ground_texture: Texture<B>,
    _depth: Texture<B>,
    plant_group: BindGroup<B>,
    ground_group: BindGroup<B>,
    sky_group: BindGroup<B>,
    plants: RasterPipeline<B>,
    ground: RasterPipeline<B>,
    sky: RasterPipeline<B>,
    depth: TextureView<B>,
    depth_extent: [u32; 2],
    plant_ktx: Ktx,
    ground_ktx: Ktx,
    uploaded: bool,
}
impl<B: IndirectBackend> Workload<B> {
    pub(crate) async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        if !d.capabilities().indirect_draw {
            return Err(Error::Unsupported("indexed indirect draw"));
        }
        let plant_ktx = Ktx::parse(PLANT_TEX)?;
        let ground_ktx = Ktx::parse(GROUND_TEX)?;
        if plant_ktx.layers < 1 || ground_ktx.layers != 1 {
            return Err(Error::Unsupported(
                "10 expects plant array and one ground layer",
            ));
        }
        let mut scene = IndirectScene::new(d)?;
        // Regenerate with the actual KTX layer count, so shader layer values stay in range.
        scene.instance_data = instance_bytes(
            scene.argument_data.len() as u32 / INDEXED_INDIRECT_STRIDE,
            plant_ktx.layers,
        );
        let uniform = buffer(
            d,
            "10 frame uniform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let plant_texture = texture(d, "10 plant texture array", &plant_ktx)?;
        let ground_texture = texture(d, "10 ground texture", &ground_ktx)?;
        let plant_view = d.create_view(
            &plant_texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2Array,
                aspects: TextureAspects::COLOR,
                mips: 0..plant_ktx.mips,
                layers: 0..plant_ktx.layers,
            },
        )?;
        let ground_view = d.create_view(
            &ground_texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..ground_ktx.mips,
                layers: 0..1,
            },
        )?;
        let sampler = d.create_sampler(&SamplerDesc {
            label: Some("10 terrain sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..plant_ktx.mips.max(ground_ktx.mips) as f32,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let plant_layout = layout(d, TextureViewDimension::D2Array, "10 plant layout")?;
        let ground_layout = layout(d, TextureViewDimension::D2, "10 ground layout")?;
        let sky_layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("10 sky layout".into()),
            entries: vec![uniform_entry(0, vec![ShaderStage::Vertex])],
        })?;
        let plant_group = textured_group(
            d,
            &plant_layout,
            &uniform,
            &plant_view,
            &sampler,
            "10 plants",
        )?;
        let ground_group = textured_group(
            d,
            &ground_layout,
            &uniform,
            &ground_view,
            &sampler,
            "10 ground",
        )?;
        let sky_group = uniform_group(d, &sky_layout, &uniform, "10 sky")?;
        let plant_pipeline_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("10 plant pipeline layout".into()),
            groups: &[&plant_layout],
            push_constants: &[],
        })?;
        let ground_pipeline_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("10 ground pipeline layout".into()),
            groups: &[&ground_layout],
            push_constants: &[],
        })?;
        let sky_pipeline_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("10 sky pipeline layout".into()),
            groups: &[&sky_layout],
            push_constants: &[],
        })?;
        let plants = pipeline(
            d,
            "10 plants",
            plant_pipeline_layout,
            format,
            PLANTS_VS,
            PLANTS_VS_SPIRV,
            "plants_vs",
            PLANTS_PS,
            PLANTS_PS_SPIRV,
            "plants_ps",
            true,
            true,
            CullMode::None,
        )
        .await?;
        let ground = pipeline(
            d,
            "10 ground",
            ground_pipeline_layout,
            format,
            GROUND_VS,
            GROUND_VS_SPIRV,
            "ground_vs",
            GROUND_PS,
            GROUND_PS_SPIRV,
            "ground_ps",
            true,
            false,
            CullMode::Back,
        )
        .await?;
        let sky = pipeline(
            d,
            "10 sky",
            sky_pipeline_layout,
            format,
            SKY_VS,
            SKY_VS_SPIRV,
            "sky_vs",
            SKY_PS,
            SKY_PS_SPIRV,
            "sky_ps",
            false,
            false,
            CullMode::Front,
        )
        .await?;
        let (depth_texture, depth) = depth(d, extent)?;
        Ok(Self {
            lane,
            scene,
            uniform,
            plant_texture,
            ground_texture,
            _depth: depth_texture,
            plant_group,
            ground_group,
            sky_group,
            plants,
            ground,
            sky,
            depth,
            depth_extent: extent,
            plant_ktx,
            ground_ktx,
            uploaded: false,
        })
    }
    pub(crate) fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>> {
        let mut e = d.encoder(self.lane)?;
        if !self.uploaded {
            self.scene.upload(&mut e)?;
            upload_ktx(&mut e, &self.plant_texture, &self.plant_ktx)?;
            upload_ktx(&mut e, &self.ground_texture, &self.ground_ktx)?;
            let reads = [
                (
                    &self.plant_texture,
                    self.plant_ktx.mips,
                    self.plant_ktx.layers,
                ),
                (&self.ground_texture, self.ground_ktx.mips, 1),
            ]
            .map(|(texture, mips, layers)| {
                ResourceTransition::Texture(TextureBarrier {
                    target: Subresources {
                        texture,
                        aspects: TextureAspects::COLOR,
                        mips: 0..mips,
                        layers: 0..layers,
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
            });
            e.transition_resources(&reads)?;
        }
        self.resize_depth(d, extent)?;
        e.write_buffer(&self.uniform, 0, &frame_uniform(extent, time))?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.18, 0.27, 0.5, 0.0])),
            store: StoreOp::Store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.depth),
            // This is one native rendering scope.  Vulkan may therefore use
            // its clear/discard attachment contract without an implicit
            // cross-pass depth-state cache.
            depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let pass = RasterPassDesc {
            label: Some("10 indirect draw scene"),
            colors: &colors,
            depth_stencil: Some(depth),
            render_area: area(extent),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        };
        self.encode_scene(&mut e, d, &pass, extent)?;
        self.uploaded = true;
        e.finish()
    }
    fn encode_scene(
        &self,
        e: &mut CommandEncoder<B>,
        d: &Device<B>,
        pass: &RasterPassDesc<'_, B>,
        extent: [u32; 2],
    ) -> Result<()> {
        let mut p = e.begin_raster(pass)?;
        p.viewport(viewport(extent))?;
        p.scissor(area(extent))?;
        for (pipeline, group, vertices, indices, vertex_bytes, index_bytes) in [
            (
                &self.sky,
                &self.sky_group,
                &self.scene.sky_vertices,
                &self.scene.sky_indices,
                self.scene.sky_data.len() as u64,
                self.scene.sky_indices_data.len() as u64,
            ),
            (
                &self.ground,
                &self.ground_group,
                &self.scene.ground_vertices,
                &self.scene.ground_indices,
                self.scene.ground_data.len() as u64,
                self.scene.ground_indices_data.len() as u64,
            ),
        ] {
            p.set_pipeline(pipeline)?;
            p.set_bind_group(0, group, &[])?;
            p.set_vertex_buffer(0, vertices, 0..vertex_bytes)?;
            p.set_index_buffer(indices, 0..index_bytes, IndexFormat::Uint32)?;
            p.draw_indexed(0..(index_bytes / 4) as u32, 0, 0..1)?;
        }
        p.set_pipeline(&self.plants)?;
        p.set_bind_group(0, &self.plant_group, &[])?;
        p.set_vertex_buffer(
            0,
            &self.scene.plant_vertices,
            0..self.scene.plant_data.len() as u64,
        )?;
        p.set_vertex_buffer(
            1,
            &self.scene.instances,
            0..self.scene.instance_data.len() as u64,
        )?;
        p.set_index_buffer(
            &self.scene.plant_indices,
            0..self.scene.plant_indices_data.len() as u64,
            IndexFormat::Uint32,
        )?;
        let draw_count = (self.scene.argument_data.len() / INDEXED_INDIRECT_STRIDE as usize) as u32;
        let extras = d
            .raster_extras()
            .ok_or(Error::Unsupported("indexed indirect draw"))?;
        if d.capabilities().multi_draw {
            extras.multi_draw(
                &mut p,
                &self.scene.arguments,
                0,
                draw_count,
                INDEXED_INDIRECT_STRIDE,
                true,
            )?;
        } else {
            for draw in 0..draw_count {
                extras.draw_indirect(
                    &mut p,
                    &self.scene.arguments,
                    u64::from(draw) * u64::from(INDEXED_INDIRECT_STRIDE),
                    true,
                )?;
            }
        }
        p.finish()
    }

    fn resize_depth(&mut self, d: &Device<B>, extent: [u32; 2]) -> Result<()> {
        if self.depth_extent == extent {
            return Ok(());
        }
        let (texture, view) = depth(d, extent)?;
        self._depth = texture;
        self.depth = view;
        self.depth_extent = extent;
        Ok(())
    }
}
