//! Recording and resource ownership for the storage-texture filter workload.
use super::{ComputeBackend, binding, buffer, ktx, pipelines};
use fluxel_rhi::api::*;

pub struct Compute<B: ComputeBackend> {
    lane: LaneId,
    pipelines: [ComputePipeline<B>; 3],
    compute_group: BindGroup<B>,
    display_groups: [BindGroup<B>; 2],
    display_pipeline: RasterPipeline<B>,
    vertex: Buffer<B>,
    index: Buffer<B>,
    source: Texture<B>,
    result: Texture<B>,
    extent: [u32; 2],
    pixels: Option<Vec<u8>>,
    source_initialized: bool,
    result_initialized: bool,
    display_extent: Option<[u32; 2]>,
    filter: usize,
}

impl<B: ComputeBackend> Compute<B> {
    pub async fn new(
        device: &Device<B>,
        lane: LaneId,
        surface_format: TextureFormat,
    ) -> Result<Self> {
        if device.compute().is_none() || !device.capabilities().storage_textures {
            return Err(Error::Unsupported(
                "06_compute requires compute and storage textures",
            ));
        }
        let image = ktx::load_rgba8()?;
        let texture = TextureDesc {
            label: Some("06 source".into()),
            dimension: TextureDimension::D2,
            extent: [image.width, image.height, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST.union(TextureUsage::TEXTURE_BINDING),
            view_formats: vec![],
        };
        let source = device.create_texture(&texture)?;
        let result = device.create_texture(&TextureDesc {
            label: Some("06 filtered result".into()),
            usage: TextureUsage::STORAGE_BINDING.union(TextureUsage::TEXTURE_BINDING),
            ..texture
        })?;
        let view_desc = TextureViewDesc {
            format: TextureFormat::Rgba8Unorm,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        };
        let source_view = device.create_view(&source, &view_desc)?;
        let result_view = device.create_view(&result, &view_desc)?;
        let compute_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("06 compute bindings".into()),
            entries: vec![
                binding(
                    0,
                    vec![ShaderStage::Compute],
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                binding(
                    1,
                    vec![ShaderStage::Compute],
                    BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2,
                        access: StorageAccess::Write,
                    },
                ),
            ],
        })?;
        let compute_entries = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&source_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&result_view),
            },
        ];
        let compute_group = device.create_bind_group(&BindGroupDesc {
            label: Some("06 compute group"),
            layout: &compute_layout,
            entries: &compute_entries,
        })?;
        let compute_layout_pipeline = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("06 compute layout"),
            groups: &[&compute_layout],
            push_constants: &[],
        })?;
        let pipelines = pipelines::filters(device, compute_layout_pipeline).await?;
        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("06 display sampler".into()),
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let display_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("06 display bindings".into()),
            entries: vec![
                binding(
                    1,
                    vec![ShaderStage::Fragment],
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                binding(
                    2,
                    vec![ShaderStage::Fragment],
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                ),
            ],
        })?;
        let source_entries = [
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&source_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let result_entries = [
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&result_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let display_groups = [
            device.create_bind_group(&BindGroupDesc {
                label: Some("06 source display group"),
                layout: &display_layout,
                entries: &source_entries,
            })?,
            device.create_bind_group(&BindGroupDesc {
                label: Some("06 result display group"),
                layout: &display_layout,
                entries: &result_entries,
            })?,
        ];
        let display_layout_pipeline = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("06 display layout"),
            groups: &[&display_layout],
            push_constants: &[],
        })?;
        let display_pipeline =
            pipelines::display(device, display_layout_pipeline, surface_format).await?;
        let vertex = device.create_buffer(&buffer(
            "06 display vertices",
            80,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = device.create_buffer(&buffer(
            "06 display indices",
            12,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        Ok(Self {
            lane,
            pipelines,
            compute_group,
            display_groups,
            display_pipeline,
            vertex,
            index,
            source,
            result,
            extent: [image.width, image.height],
            pixels: Some(image.pixels),
            source_initialized: false,
            result_initialized: false,
            display_extent: None,
            filter: 0,
        })
    }

    pub fn record(
        &mut self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        surface_extent: [u32; 2],
    ) -> Result<RecordedWork<B>> {
        let mut encoder = device.encoder(self.lane)?;
        let uploading_source = self.pixels.is_some();
        if let Some(pixels) = self.pixels.as_deref() {
            encoder.write_texture(
                TextureRegion {
                    texture: &self.source,
                    mip: 0,
                    origin: [0; 3],
                    extent: [self.extent[0], self.extent[1], 1],
                    aspect: TextureAspects::COLOR,
                },
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: self.extent[0] * 4,
                    rows_per_image: self.extent[1],
                },
                pixels,
            )?;
            encoder.write_buffer(&self.index, 0, &[0, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 0])?;
            encoder.transition_resources(&[transition(
                &self.source,
                ImageLayout::Common,
                ImageLayout::ShaderReadOnly,
                TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                TextureUses {
                    stages: StageMask::COMPUTE.union(StageMask::FRAGMENT),
                    accesses: AccessMask::SHADER_READ,
                },
            )])?;
        }
        if !uploading_source && !self.source_initialized {
            return Err(Error::InvalidState);
        }
        let display_changed = self.display_extent != Some(surface_extent);
        if display_changed {
            encoder.write_buffer(&self.vertex, 0, &quad(surface_extent))?;
        }
        let (old_layout, old_use) = if self.result_initialized {
            (
                ImageLayout::ShaderReadOnly,
                TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            )
        } else {
            (
                // A newly allocated texture has no defined contents.  This is
                // deliberately `Undefined`, rather than the reusable Common
                // upload boundary, so Vulkan can lower the first transition
                // from its real native layout without state guessing.
                ImageLayout::Undefined,
                TextureUses {
                    stages: StageMask::HOST,
                    accesses: AccessMask::EMPTY,
                },
            )
        };
        encoder.transition_resources(&[transition(
            &self.result,
            old_layout,
            ImageLayout::General,
            old_use,
            TextureUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            },
        )])?;
        let mut compute = device
            .compute()
            .ok_or(Error::Unsupported("compute"))?
            .begin(&mut encoder)?;
        compute.set_pipeline(&self.pipelines[self.filter])?;
        compute.set_bind_group(0, &self.compute_group, &[])?;
        compute.dispatch([self.extent[0].div_ceil(16), self.extent[1].div_ceil(16), 1])?;
        compute.finish()?;
        encoder.transition_resources(&[transition(
            &self.result,
            ImageLayout::General,
            ImageLayout::ShaderReadOnly,
            TextureUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            },
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
        )])?;
        self.draw(&mut encoder, image, surface_extent)?;
        let work = encoder.finish()?;
        if uploading_source {
            self.pixels = None;
            self.source_initialized = true;
        }
        self.result_initialized = true;
        if display_changed {
            self.display_extent = Some(surface_extent);
        }
        Ok(work)
    }

    fn draw(
        &self,
        encoder: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<()> {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.015, 0.015, 0.02, 1.0])),
            store: StoreOp::Store,
        })];
        let mut raster = encoder.begin_raster(&RasterPassDesc {
            label: Some("06 source and filtered result"),
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
        raster.set_pipeline(&self.display_pipeline)?;
        raster.set_vertex_buffer(0, &self.vertex, 0..80)?;
        raster.set_index_buffer(&self.index, 0..12, IndexFormat::Uint16)?;
        let split = extent[0] / 2;
        draw_half(&mut raster, &self.display_groups[0], 0, split, extent)?;
        draw_half(
            &mut raster,
            &self.display_groups[1],
            split,
            extent[0] - split,
            extent,
        )?;
        raster.finish()
    }
}

fn draw_half<B: ComputeBackend>(
    r: &mut RasterPass<'_, B>,
    group: &BindGroup<B>,
    x: u32,
    width: u32,
    extent: [u32; 2],
) -> Result<()> {
    if width == 0 || extent[1] == 0 {
        return Ok(());
    }
    r.set_bind_group(0, group, &[])?;
    r.viewport(Viewport {
        x: x as f32,
        y: 0.0,
        width: width as f32,
        height: extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    r.scissor(Rect {
        x,
        y: 0,
        width,
        height: extent[1],
    })?;
    r.draw_indexed(0..6, 0, 0..1)
}

fn transition<'a, B: ComputeBackend>(
    texture: &'a Texture<B>,
    from_layout: ImageLayout,
    to_layout: ImageLayout,
    from: TextureUses,
    to: TextureUses,
) -> ResourceTransition<'a, B> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition { from, to },
        layout: Some((from_layout, to_layout)),
        ownership: None,
    })
}

fn quad(extent: [u32; 2]) -> [u8; 80] {
    // The C++ sample projects a 2x2 quad from camera z=-2 with a 60-degree
    // vertical field of view, then uses a viewport covering half the window.
    // Apply the same projection directly to this quad's clip-space vertices.
    let y = 0.866_025_4_f32;
    let x = y * (extent[1] as f32 * 2.0 / extent[0] as f32);
    let vertices: [f32; 20] = [
        -x, -y, 0., 0., 1., x, -y, 0., 1., 1., x, y, 0., 1., 0., -x, y, 0., 0., 0.,
    ];
    let mut bytes = [0; 80];
    for (index, value) in vertices.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
