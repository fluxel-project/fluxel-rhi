//! GPU resources and command recording for the textured quad.
//!
//! The workload owns persistent GPU objects. It accepts decoded pixels from the
//! data module and leaves host surface acquisition and submission to its caller.
use super::*;

pub struct TextureExample<B: TextureBackend> {
    vertex: Buffer<B>,
    index: Buffer<B>,
    uniform: Buffer<B>,
    texture: Texture<B>,
    group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    lane: LaneId,
}
impl<B: TextureBackend> TextureExample<B> {
    pub async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        p: TexturePixels<'_>,
    ) -> Result<Self> {
        if p.width == 0 || p.height == 0 || p.rgba.len() != p.width as usize * p.height as usize * 4
        {
            return Err(Error::InvalidRange);
        }
        let td = TextureDesc {
            label: Some("02 KTX texture".into()),
            dimension: TextureDimension::D2,
            extent: [p.width, p.height, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING.union(TextureUsage::COPY_DST),
            view_formats: vec![],
        };
        match d.texture_support(&td) {
            Support::Supported(s) if s.filterable => (),
            Support::Supported(_) => return Err(Error::Unsupported("filterable rgba8 texture")),
            Support::Unsupported(x) => return Err(Error::Unsupported(x)),
        };
        let vertex = d.create_buffer(&buf(
            "02 vertices",
            128,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = d.create_buffer(&buf(
            "02 indices",
            12,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let uniform = d.create_buffer(&buf(
            "02 transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let texture = d.create_texture(&td)?;
        let view = d.create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let sampler = d.create_sampler(&SamplerDesc {
            label: Some("02 sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let l = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("02 bindings".into()),
            entries: vec![
                entry(
                    0,
                    vec![ShaderStage::Vertex],
                    BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 256,
                    },
                ),
                entry(
                    1,
                    vec![ShaderStage::Fragment],
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    vec![ShaderStage::Fragment],
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                ),
            ],
        })?;
        let es = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &uniform,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("02 group"),
            layout: &l,
            entries: &es,
        })?;
        let pl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("02 layout"),
            groups: &[&l],
            push_constants: &[],
        })?;
        let pipeline = pipeline(
            d,
            pl,
            format,
            vec![VertexBufferLayout {
                stride: 32,
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
                        format: VertexFormat::Float32(2),
                    },
                    VertexAttribute {
                        location: 2,
                        offset: 20,
                        format: VertexFormat::Float32(3),
                    },
                ],
            }],
            VS,
            PS,
        )
        .await?;
        Ok(Self {
            vertex,
            index,
            uniform,
            texture,
            group,
            pipeline,
            lane,
        })
    }
    pub fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        p: TexturePixels<'_>,
    ) -> Result<RecordedWork<B>> {
        let mut e = d.encoder(self.lane)?;
        e.write_buffer(&self.vertex, 0, &quad())?;
        e.write_buffer(&self.index, 0, &[0, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 0])?;
        e.write_buffer(&self.uniform, 0, &texture_transform(extent))?;
        e.write_texture(
            TextureRegion {
                texture: &self.texture,
                mip: 0,
                origin: [0; 3],
                extent: [p.width, p.height, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: p.width * 4,
                rows_per_image: p.height,
            },
            p.rgba,
        )?;
        e.transition_resources(&[texture_transition(
            &self.texture,
            ImageLayout::Common,
            ImageLayout::ShaderReadOnly,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
        )])?;
        let c = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.02, 0.02, 0.02, 1.0])),
            store: StoreOp::Store,
        })];
        let mut r = e.begin_raster(&RasterPassDesc {
            label: Some("02 texture"),
            colors: &c,
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
        r.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        r.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        r.set_pipeline(&self.pipeline)?;
        r.set_bind_group(0, &self.group, &[])?;
        r.set_vertex_buffer(0, &self.vertex, 0..128)?;
        r.set_index_buffer(&self.index, 0..12, IndexFormat::Uint16)?;
        r.draw_indexed(0..6, 0, 0..1)?;
        r.finish()?;
        e.transition_resources(&[texture_transition(
            &self.texture,
            ImageLayout::ShaderReadOnly,
            ImageLayout::Common,
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
            TextureUses {
                stages: StageMask::ALL_COMMANDS,
                accesses: AccessMask::EMPTY,
            },
        )])?;
        e.finish()
    }
}
