//! GPU resources, pipeline construction, and recording for the sphere ring.
//!
//! This module selects the native push-constant route or the WebGPU dynamic
//! uniform fallback. Host presentation and example entry points stay in the
//! parent scene module.
use super::*;

use super::pipeline::{buffer, camera_bytes, create_pipeline};
/// Reusable sphere-ring workload shared by deferred and immediate hosts.
pub struct PushWorkload<B: PushBackend> {
    lane: LaneId,
    vertex: Buffer<B>,
    index: Buffer<B>,
    camera: Buffer<B>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
    group: BindGroup<B>,
    draw_data: Option<Buffer<B>>,
    draw_group: Option<BindGroup<B>>,
    pipeline: RasterPipeline<B>,
    source_vertices: Vec<u8>,
    source_indices: Vec<u8>,
    uploaded: bool,
    dynamic_uniforms: bool,
    dynamic_uniform_alignment: u64,
}
impl<B: PushBackend> PushWorkload<B> {
    pub async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let dynamic_uniforms = !d.capabilities().push_constants
            || d.capabilities().limits.max_push_constant_bytes < PUSH_BYTES;
        let dynamic_uniform_alignment = d.capabilities().limits.uniform_offset_alignment.max(256);
        // Use the C++ sample's glTF sphere instead of a placeholder octahedron.
        let sphere = gltf::load_embedded_first_mesh(SPHERE)
            .map_err(|error| Error::Backend(error.to_string()))?;
        let vertex = buffer(
            d,
            "05 sphere vertices",
            sphere.vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        )?;
        let index = buffer(
            d,
            "05 sphere indices",
            sphere.indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        )?;
        let camera = buffer(
            d,
            "05 camera",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let depth = d.create_texture(&TextureDesc {
            label: Some("05 depth".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })?;
        let depth_view = d.create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let group_layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("05 camera layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 128,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let entries = [BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: &camera,
                range: 0..256,
            },
        }];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("05 camera group"),
            layout: &group_layout,
            entries: &entries,
        })?;
        let draw_layout = dynamic_uniforms
            .then(|| {
                d.create_bind_group_layout(&BindGroupLayoutDesc {
                    label: Some("05 WebGPU draw-data layout".into()),
                    entries: vec![BindGroupLayoutEntry {
                        binding: 0,
                        visibility: vec![ShaderStage::Vertex],
                        ty: BindingType::UniformBuffer {
                            dynamic_offset: true,
                            min_size: PUSH_BYTES as u64,
                        },
                        count: BindingCount::One,
                        partially_bound: false,
                        nonuniform_indexing: false,
                    }],
                })
            })
            .transpose()?;
        let draw_data = if dynamic_uniforms {
            Some(buffer(
                d,
                "05 WebGPU draw data",
                dynamic_uniform_alignment
                    .checked_mul(SPHERE_COUNT as u64)
                    .ok_or(Error::InvalidRange)?,
                BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
            )?)
        } else {
            None
        };
        let draw_group = match (&draw_layout, &draw_data) {
            (Some(layout), Some(data)) => Some(d.create_bind_group(&BindGroupDesc {
                label: Some("05 WebGPU draw-data group"),
                layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: data,
                        range: 0..PUSH_BYTES as u64,
                    },
                }],
            })?),
            (None, None) => None,
            _ => return Err(Error::InvalidState),
        };
        let constants = (!dynamic_uniforms).then_some(PushConstantRange {
            stages: vec![ShaderStage::Vertex],
            range: 0..PUSH_BYTES,
        });
        let groups = match &draw_layout {
            Some(draw_layout) => vec![&group_layout, draw_layout],
            None => vec![&group_layout],
        };
        let layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("05 layout"),
            groups: &groups,
            push_constants: constants.as_slice(),
        })?;
        let pipeline = create_pipeline(d, layout, format, dynamic_uniforms).await?;
        Ok(Self {
            lane,
            vertex,
            index,
            camera,
            _depth: depth,
            depth_view,
            group,
            draw_data,
            draw_group,
            pipeline,
            source_vertices: sphere.vertices,
            source_indices: sphere.indices,
            uploaded: false,
            dynamic_uniforms,
            dynamic_uniform_alignment,
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<B>>
    where
        B: DeferredBackend,
    {
        let mut encoder = d.encoder(self.lane)?;
        if !self.uploaded {
            encoder.write_buffer(&self.vertex, 0, &self.source_vertices)?;
            encoder.write_buffer(&self.index, 0, &self.source_indices)?;
        }
        encoder.write_buffer(&self.camera, 0, &camera_bytes(extent))?;
        if let Some(draw_data) = &self.draw_data {
            for (index, sphere) in sphere_ring().into_iter().enumerate() {
                let offset = self
                    .dynamic_uniform_alignment
                    .checked_mul(index as u64)
                    .ok_or(Error::InvalidRange)?;
                encoder.write_buffer(draw_data, offset, &sphere.bytes())?;
            }
        }
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.025, 1.0])),
            store: StoreOp::Store,
        })];
        let pass = RasterPassDesc {
            label: Some("05 push constant spheres"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&self.depth_view),
                depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
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
        };
        if self.dynamic_uniforms {
            encode_sphere_ring_dynamic_uniform(
                &mut encoder,
                &pass,
                &self.pipeline,
                &self.group,
                self.draw_group.as_ref().ok_or(Error::InvalidState)?,
                self.dynamic_uniform_alignment,
                &self.vertex,
                self.source_vertices.len() as u64,
                &self.index,
                self.source_indices.len() as u64,
                (self.source_indices.len() / 4) as u32,
            )?;
        } else {
            encode_sphere_ring(
                d,
                &mut encoder,
                &pass,
                &self.pipeline,
                &self.group,
                &self.vertex,
                self.source_vertices.len() as u64,
                &self.index,
                self.source_indices.len() as u64,
                (self.source_indices.len() / 4) as u32,
                &sphere_ring(),
            )?;
        }
        encoder.finish()
    }

    /// Publish the upload only after the command step has been accepted.
    pub fn commit_frame(&mut self) {
        self.uploaded = true;
    }
}

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl"),
    all(target_os = "android", feature = "native-gles-egl"),
    all(target_arch = "wasm32", feature = "webgl2")
))]
impl PushWorkload<fluxel_rhi::gl::Gl> {
    /// Issues the existing dynamic-uniform fallback on GL's owner thread.
    /// Every encoder call reaches GL before returning; no recorded work is
    /// produced or kept for a later queue submission.
    pub fn issue(
        &mut self,
        encoder: &mut CommandEncoder<fluxel_rhi::gl::Gl>,
        image: &SurfaceImage<fluxel_rhi::gl::Gl>,
        extent: [u32; 2],
    ) -> Result<()> {
        if !self.dynamic_uniforms {
            return Err(Error::InvalidState);
        }
        if !self.uploaded {
            encoder.write_buffer(&self.vertex, 0, &self.source_vertices)?;
            encoder.write_buffer(&self.index, 0, &self.source_indices)?;
        }
        encoder.write_buffer(&self.camera, 0, &camera_bytes(extent))?;
        let draw_data = self.draw_data.as_ref().ok_or(Error::InvalidState)?;
        for (index, sphere) in sphere_ring().into_iter().enumerate() {
            let offset = self
                .dynamic_uniform_alignment
                .checked_mul(index as u64)
                .ok_or(Error::InvalidRange)?;
            encoder.write_buffer(draw_data, offset, &sphere.bytes())?;
        }
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.025, 1.0])),
            store: StoreOp::Store,
        })];
        let pass = RasterPassDesc {
            label: Some("05 GL dynamic-uniform spheres"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&self.depth_view),
                depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
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
        };
        encode_sphere_ring_dynamic_uniform(
            encoder,
            &pass,
            &self.pipeline,
            &self.group,
            self.draw_group.as_ref().ok_or(Error::InvalidState)?,
            self.dynamic_uniform_alignment,
            &self.vertex,
            self.source_vertices.len() as u64,
            &self.index,
            self.source_indices.len() as u64,
            (self.source_indices.len() / 4) as u32,
        )
    }
}
