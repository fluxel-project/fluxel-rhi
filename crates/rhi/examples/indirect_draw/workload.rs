use super::{
    ktx::{Ktx, frame_uniform},
    pipeline::*,
    scene::{
        GROUND_PS, GROUND_TEX, GROUND_VS, INDEXED_INDIRECT_STRIDE, IndirectScene, PLANT_TEX,
        PLANTS_PS, PLANTS_VS, SKY_PS, SKY_VS, encode_plants, instance_bytes,
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
            return Err(Error::Unsupported("DX12 indexed indirect draw"));
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
            "plants_vs",
            PLANTS_PS,
            "plants_ps",
            true,
            true,
        )
        .await?;
        let ground = pipeline(
            d,
            "10 ground",
            ground_pipeline_layout,
            format,
            GROUND_VS,
            "ground_vs",
            GROUND_PS,
            "ground_ps",
            true,
            false,
        )
        .await?;
        let sky = pipeline(
            d,
            "10 sky",
            sky_pipeline_layout,
            format,
            SKY_VS,
            "sky_vs",
            SKY_PS,
            "sky_ps",
            false,
            false,
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
        }
        e.write_buffer(&self.uniform, 0, &frame_uniform(extent, time))?;
        self.direct_pass(&mut e, image, extent, true)?;
        self.direct_pass(&mut e, image, extent, false)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.depth),
            depth: Some((LoadOp::Load, StoreOp::Store)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let pass = RasterPassDesc {
            label: Some("10 indexed indirect plants"),
            colors: &colors,
            depth_stencil: Some(depth),
            render_area: area(extent),
            multiview_mask: None,
            timestamp_writes: None,
            secondary_inheritance: None,
        };
        encode_plants(
            d,
            &mut e,
            &pass,
            &self.plants,
            &self.plant_group,
            &self.scene.plant_vertices,
            self.scene.plant_data.len() as u64,
            &self.scene.plant_indices,
            self.scene.plant_indices_data.len() as u64,
            &self.scene.instances,
            self.scene.instance_data.len() as u64,
            &self.scene.arguments,
            (self.scene.argument_data.len() / INDEXED_INDIRECT_STRIDE as usize) as u32,
        )?;
        self.uploaded = true;
        e.finish()
    }
    fn direct_pass(
        &self,
        e: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        sky: bool,
    ) -> Result<()> {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: if sky {
                LoadOp::Clear(ClearColor::Float([0.18, 0.27, 0.5, 1.0]))
            } else {
                LoadOp::Load
            },
            store: StoreOp::Store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.depth),
            depth: Some((
                if sky {
                    LoadOp::Clear(1.0)
                } else {
                    LoadOp::Load
                },
                StoreOp::Store,
            )),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some(if sky { "10 sky" } else { "10 ground" }),
            colors: &colors,
            depth_stencil: Some(depth),
            render_area: area(extent),
            multiview_mask: None,
            timestamp_writes: None,
            secondary_inheritance: None,
        })?;
        p.viewport(viewport(extent))?;
        p.scissor(area(extent))?;
        let (pipeline, group, vertices, indices, vc, ic) = if sky {
            (
                &self.sky,
                &self.sky_group,
                &self.scene.sky_vertices,
                &self.scene.sky_indices,
                self.scene.sky_data.len() as u64,
                self.scene.sky_indices_data.len() as u64,
            )
        } else {
            (
                &self.ground,
                &self.ground_group,
                &self.scene.ground_vertices,
                &self.scene.ground_indices,
                self.scene.ground_data.len() as u64,
                self.scene.ground_indices_data.len() as u64,
            )
        };
        p.set_pipeline(pipeline)?;
        p.set_bind_group(0, group, &[])?;
        p.set_vertex_buffer(0, vertices, 0..vc)?;
        p.set_index_buffer(indices, 0..ic, IndexFormat::Uint32)?;
        p.draw_indexed(0..(ic / 4) as u32, 0, 0..1)?;
        p.finish()
    }
}
