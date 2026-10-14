//! MSAA scene state, separated from resource, pipeline, and recording lowering.
mod pipeline;
mod recording;
mod resources;
pub(crate) use recording::FrameCommit;

use fluxel_rhi::api::*;
use resources::*;

pub trait MsaaBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
impl<T> MsaaBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend
{
}
pub trait MsaaDeferredBackend: MsaaBackend + DeferredBackend {}
impl<T> MsaaDeferredBackend for T where T: MsaaBackend + DeferredBackend {}
pub struct Msaa<B: MsaaBackend> {
    lane: LaneId,
    vertex: Buffer<B>,
    index: Buffer<B>,
    uniform: Buffer<B>,
    groups: Vec<BindGroup<B>>,
    material_textures: [Texture<B>; 2],
    pipeline: RasterPipeline<B>,
    color: Option<Texture<B>>,
    color_view: Option<TextureView<B>>,
    depth: Texture<B>,
    depth_view: TextureView<B>,
    vertex_len: u64,
    index_len: u64,
    draws: Vec<Draw>,
    samples: u32,
    surface_format: TextureFormat,
    target_extent: [u32; 2],
    uploaded: bool,
    color_state_initialized: bool,
}

struct Draw {
    first: u32,
    count: u32,
    material: usize,
}

impl<B: MsaaBackend> Msaa<B> {
    pub async fn new(
        device: &Device<B>,
        lane: LaneId,
        surface_format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let samples = supported_samples(device, surface_format)?;
        Self::new_with_samples(device, lane, surface_format, extent, samples).await
    }

    /// Builds the same Voyager workload with an explicitly selected sample count.
    ///
    /// Browser WebGL2 uses one sample because its native API has multisample
    /// renderbuffers but no multisample texture storage, which is the current
    /// Core attachment representation. That path preserves the scene while
    /// leaving the backend's MSAA capability unclaimed.
    pub async fn new_single_sample(
        device: &Device<B>,
        lane: LaneId,
        surface_format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        Self::new_with_samples(device, lane, surface_format, extent, 1).await
    }

    async fn new_with_samples(
        device: &Device<B>,
        lane: LaneId,
        surface_format: TextureFormat,
        extent: [u32; 2],
        samples: u32,
    ) -> Result<Self> {
        if extent.contains(&0) || samples == 0 {
            return Err(Error::InvalidRange);
        }
        let model = super::gltf::load_embedded_model(
            super::VOYAGER,
            super::gltf::LoadOptions::PRETRANSFORM_FLIP_Y,
        )
        .map_err(|error| Error::Backend(error.to_string()))?;
        let vertex_len = model.vertices.len() as u64;
        let index_len = model.indices.len() as u64;
        let vertex = buffer(
            device,
            "08 voyager vertices",
            vertex_len,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        )?;
        let index = buffer(
            device,
            "08 voyager indices",
            index_len,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        )?;
        let uniform = buffer(
            device,
            "08 voyager transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("08 voyager layout".into()),
            entries: vec![
                layout_entry(
                    0,
                    vec![ShaderStage::Vertex, ShaderStage::Fragment],
                    BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 144,
                    },
                ),
                layout_entry(
                    1,
                    vec![ShaderStage::Fragment],
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                layout_entry(
                    2,
                    vec![ShaderStage::Fragment],
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                ),
            ],
        })?;
        let material_textures = [
            material_texture(device, "08 voyager material 0")?,
            material_texture(device, "08 voyager material 1")?,
        ];
        let material_views = [
            texture_view(device, &material_textures[0])?,
            texture_view(device, &material_textures[1])?,
        ];
        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("08 voyager sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let groups = material_views
            .iter()
            .enumerate()
            .map(|(_index, view)| {
                let entries = [
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: &uniform,
                            range: 0..256,
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(&sampler),
                    },
                ];
                device.create_bind_group(&BindGroupDesc {
                    label: Some("08 voyager material group"),
                    layout: &layout,
                    entries: &entries,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("08 voyager pipeline layout".into()),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let pipeline = pipeline::pipeline(device, pipeline_layout, surface_format, samples).await?;
        let (color, color_view) = if samples > 1 {
            let (color, view) = color_target(device, extent, surface_format, samples)?;
            (Some(color), Some(view))
        } else {
            (None, None)
        };
        let (depth, depth_view) = depth_target(device, extent, samples)?;
        Ok(Self {
            lane,
            vertex,
            index,
            uniform,
            groups,
            material_textures,
            pipeline,
            color,
            color_view,
            depth,
            depth_view,
            vertex_len,
            index_len,
            draws: model
                .primitives
                .iter()
                .map(|primitive| Draw {
                    first: primitive.first_index,
                    count: primitive.index_count,
                    // Voyager's first two materials both reference texture 0;
                    // only material 2 selects texture 1.  Bind groups follow
                    // image identity rather than raw glTF material ordinal.
                    material: match primitive.material_index {
                        Some(2) => 1,
                        _ => 0,
                    },
                })
                .collect(),
            samples,
            surface_format,
            target_extent: extent,
            uploaded: false,
            color_state_initialized: false,
        })
    }
}
