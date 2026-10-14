//! Persistent GPU resources and frame recording for the instanced rock-field scene.
use super::*;
use super::{pipeline, resources};

pub struct Instancing<B: InstancingBackend> {
    rock_vertices: Buffer<B>,
    rock_indices: Buffer<B>,
    planet_vertices: Buffer<B>,
    planet_indices: Buffer<B>,
    instances: Buffer<B>,
    uniform: Buffer<B>,
    rocks: Texture<B>,
    planet: Texture<B>,
    rock_group: BindGroup<B>,
    planet_group: BindGroup<B>,
    rocks_pipeline: RasterPipeline<B>,
    planet_pipeline: RasterPipeline<B>,
    stars_pipeline: RasterPipeline<B>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
    rock_vertex_bytes: u64,
    rock_index_bytes: u64,
    planet_vertex_bytes: u64,
    planet_index_bytes: u64,
    rock_pixels: resources::Ktx,
    planet_pixels: Vec<u8>,
    planet_extent: [u32; 2],
    instance_pixels: Vec<u8>,
    uploaded: bool,
    lane: LaneId,
    source_rock_vertices: Vec<u8>,
    source_rock_indices: Vec<u8>,
    source_planet_vertices: Vec<u8>,
    source_planet_indices: Vec<u8>,
}
impl<B: InstancingBackend> Instancing<B> {
    pub async fn new(
        device: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let scene = resources::load_scene()?;
        let rock_vertices = device.create_buffer(&buffer(
            "03 rock vertices",
            scene.rock_vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let rock_indices = device.create_buffer(&buffer(
            "03 rock indices",
            scene.rock_indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let planet_vertices = device.create_buffer(&buffer(
            "03 planet vertices",
            scene.planet_vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let planet_indices = device.create_buffer(&buffer(
            "03 planet indices",
            scene.planet_indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let uniform = device.create_buffer(&buffer(
            "03 scene uniform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let instances = device.create_buffer(&buffer(
            "03 rock instances",
            scene.instances.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let (depth, depth_view) = resources::depth(device, extent)?;
        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("03 base-mip sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..scene.rock_pixels.mips as f32,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let (rocks_pipeline, planet_pipeline, stars_pipeline, rock_layout, planet_layout) =
            pipeline::scene_pipelines(device, format).await?;
        let rocks = resources::array_texture(device, &scene.rock_pixels)?;
        let rock_view = device.create_view(&rocks, &resources::array_view(&scene.rock_pixels))?;
        let rock_group = group(
            device,
            "03 rock array group",
            &rock_layout,
            &uniform,
            &rock_view,
            &sampler,
        )?;
        let planet = resources::texture(
            device,
            "03 planet texture",
            scene.planet_extent[0],
            scene.planet_extent[1],
        )?;
        let planet_view = device.create_view(&planet, &resources::color_view())?;
        let planet_group = group(
            device,
            "03 planet group",
            &planet_layout,
            &uniform,
            &planet_view,
            &sampler,
        )?;
        Ok(Self {
            rock_vertices,
            rock_indices,
            planet_vertices,
            planet_indices,
            instances,
            uniform,
            rocks,
            planet,
            rock_group,
            planet_group,
            rocks_pipeline,
            planet_pipeline,
            stars_pipeline,
            _depth: depth,
            depth_view,
            rock_vertex_bytes: scene.rock_vertices.len() as u64,
            rock_index_bytes: scene.rock_indices.len() as u64,
            planet_vertex_bytes: scene.planet_vertices.len() as u64,
            planet_index_bytes: scene.planet_indices.len() as u64,
            rock_pixels: scene.rock_pixels,
            planet_pixels: scene.planet_pixels,
            planet_extent: scene.planet_extent,
            instance_pixels: scene.instances,
            uploaded: false,
            lane,
            source_rock_vertices: scene.rock_vertices,
            source_rock_indices: scene.rock_indices,
            source_planet_vertices: scene.planet_vertices,
            source_planet_indices: scene.planet_indices,
        })
    }
    /// Issues the scene into an already-open encoder.
    ///
    /// The immediate GL route calls this while its context-owning session is
    /// open; deferred callers use [`Self::record`] to create and finish a
    /// native command list around the same operation sequence.
    pub fn issue(
        &mut self,
        encoder: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<()> {
        if !self.uploaded {
            encoder.write_buffer(&self.rock_vertices, 0, &self.source_rock_vertices)?;
            encoder.write_buffer(&self.rock_indices, 0, &self.source_rock_indices)?;
            encoder.write_buffer(&self.planet_vertices, 0, &self.source_planet_vertices)?;
            encoder.write_buffer(&self.planet_indices, 0, &self.source_planet_indices)?;
            encoder.write_buffer(&self.instances, 0, &self.instance_pixels)?;
            resources::upload_ktx(encoder, &self.rocks, &self.rock_pixels)?;
            resources::upload_base(
                encoder,
                &self.planet,
                &self.planet_pixels,
                self.planet_extent,
            )?;
            encoder.transition_resources(&[
                sampled_transition(
                    &self.rocks,
                    0..self.rock_pixels.mips,
                    0..self.rock_pixels.layers,
                ),
                sampled_transition(&self.planet, 0..1, 0..1),
            ])?;
        }
        encoder.write_buffer(&self.uniform, 0, &uniform(extent, time))?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0.2, 1.])),
            store: StoreOp::Store,
        })];
        let depth = DepthStencilAttachment {
            view: AttachmentView::Texture(&self.depth_view),
            depth: Some((LoadOp::Clear(1.), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("03 rock/planet/star scene"),
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
            x: 0.,
            y: 0.,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.,
            max_depth: 1.,
        })?;
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        pass.set_pipeline(&self.stars_pipeline)?;
        pass.draw(0..3, 0..1)?;
        pass.set_pipeline(&self.planet_pipeline)?;
        pass.set_bind_group(0, &self.planet_group, &[])?;
        pass.set_vertex_buffer(0, &self.planet_vertices, 0..self.planet_vertex_bytes)?;
        pass.set_index_buffer(
            &self.planet_indices,
            0..self.planet_index_bytes,
            IndexFormat::Uint32,
        )?;
        pass.draw_indexed(0..(self.planet_index_bytes / 4) as u32, 0, 0..1)?;
        pass.set_pipeline(&self.rocks_pipeline)?;
        pass.set_vertex_buffer(0, &self.rock_vertices, 0..self.rock_vertex_bytes)?;
        pass.set_index_buffer(
            &self.rock_indices,
            0..self.rock_index_bytes,
            IndexFormat::Uint32,
        )?;
        pass.set_bind_group(0, &self.rock_group, &[])?;
        pass.set_vertex_buffer(1, &self.instances, 0..self.instance_pixels.len() as u64)?;
        pass.draw_indexed(
            0..(self.rock_index_bytes / 4) as u32,
            0,
            0..(self.instance_pixels.len() / 32) as u32,
        )?;
        pass.finish()?;
        Ok(())
    }

    /// Records the shared scene for a deferred backend.
    pub fn record(
        &mut self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>>
    where
        B: DeferredBackend,
    {
        let mut encoder = device.encoder(self.lane)?;
        self.issue(&mut encoder, image, extent, time)?;
        encoder.finish()
    }

    pub fn mark_submitted(&mut self) {
        self.uploaded = true;
    }
}
fn sampled_transition<B: InstancingBackend>(
    texture: &Texture<B>,
    mips: std::ops::Range<u32>,
    layers: std::ops::Range<u32>,
) -> ResourceTransition<'_, B> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips,
            layers,
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
}
fn group<B: InstancingBackend>(
    device: &Device<B>,
    label: &str,
    layout: &BindGroupLayout<B>,
    uniform: &Buffer<B>,
    view: &TextureView<B>,
    sampler: &Sampler<B>,
) -> Result<BindGroup<B>> {
    let entries = [
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: uniform,
                range: 0..256,
            },
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::TextureView(view),
        },
        BindGroupEntry {
            binding: 2,
            resource: BindingResource::Sampler(sampler),
        },
    ];
    device.create_bind_group(&BindGroupDesc {
        label: Some(label.into()),
        layout,
        entries: &entries,
    })
}
fn uniform(extent: [u32; 2], time: f32) -> [u8; 256] {
    let f = 1.0 / 30f32.to_radians().tan();
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let projection = [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        // Sascha Willems' Vulkan example uses its positive-height viewport,
        // whose framebuffer Y mapping is opposite DX12/WebGPU. Keep that
        // adaptation in this scene matrix rather than changing RHI semantics.
        -f,
        0.,
        0.,
        0.,
        0.,
        256. / -255.,
        -1.,
        0.,
        0.,
        256. / -255.,
        0.,
    ];
    let (sin_x, cos_x) = (-17.2f32).to_radians().sin_cos();
    let (sin_y, cos_y) = (-4.7f32).to_radians().sin_cos();
    let view = multiply(
        [
            1., 0., 0., 0., 0., cos_x, sin_x, 0., 0., -sin_x, cos_x, 0., 5.5, -1.85, -18.5, 1.,
        ],
        [
            cos_y, 0., -sin_y, 0., 0., 1., 0., 0., sin_y, 0., cos_y, 0., 0., 0., 0., 1.,
        ],
    );
    let mut output = [0; 256];
    for (index, value) in projection
        .into_iter()
        .chain(view)
        .chain([0., -5., 0., 1., time * 0.35, time * 0.01, 0., 0.])
        .enumerate()
    {
        output[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    output
}
fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut output = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    output
}
