//! Runtime-sized, non-uniformly indexed 32-texture cube scene using RHI.
mod common;
use fluxel_rhi::api::*;
use std::sync::Arc;
const TEXTURES: usize = 32;
const CUBES: usize = 5;
const VERTEX_BYTES: u64 = 24;
const VERTICES_PER_CUBE: u32 = 24;
const INDICES_PER_CUBE: u32 = 36;
const VS: &[u8] = include_bytes!("assets/14_descriptor_indexing/descriptor_indexing.vs.dxil");
const PS: &[u8] = include_bytes!("assets/14_descriptor_indexing/descriptor_indexing.ps.dxil");
pub trait DescriptorIndexingBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> DescriptorIndexingBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
pub fn require_descriptor_indexing<B: Backend>(d: &Device<B>) -> Result<()> {
    let c = d.capabilities();
    if !c.binding_arrays {
        return Err(Error::Unsupported("binding arrays"));
    }
    if !c.runtime_sized_binding_arrays {
        return Err(Error::Unsupported("runtime-sized binding arrays"));
    }
    if !c.nonuniform_sampled_texture_and_storage_buffer_indexing {
        return Err(Error::Unsupported("non-uniform sampled-texture indexing"));
    }
    Ok(())
}
pub fn descriptor_indexing_layout<B: Backend + ResourceBackend>(
    d: &Device<B>,
) -> Result<BindGroupLayout<B>> {
    require_descriptor_indexing(d)?;
    d.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("14 runtime texture array".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 192,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::RuntimeSized,
                partially_bound: false,
                nonuniform_indexing: true,
            },
        ],
    })
}
pub struct DescriptorIndexing<B: DescriptorIndexingBackend> {
    lane: LaneId,
    vertex: Buffer<B>,
    index: Buffer<B>,
    uniform: Buffer<B>,
    textures: Vec<Texture<B>>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
    group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    uploaded: bool,
}
impl<B: DescriptorIndexingBackend> DescriptorIndexing<B> {
    pub async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        require_descriptor_indexing(d)?;
        let vertex = d.create_buffer(&buffer(
            "14 cube vertices",
            VERTEX_BYTES * u64::from(VERTICES_PER_CUBE) * CUBES as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = d.create_buffer(&buffer(
            "14 cube indices",
            u64::from(INDICES_PER_CUBE) * CUBES as u64 * 4,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let uniform = d.create_buffer(&buffer(
            "14 matrices",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let layout = descriptor_indexing_layout(d)?;
        let sampler = d.create_sampler(&SamplerDesc {
            label: Some("14 nearest sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let (mut textures, mut views) =
            (Vec::with_capacity(TEXTURES), Vec::with_capacity(TEXTURES));
        for i in 0..TEXTURES {
            let t = d.create_texture(&TextureDesc {
                label: Some(format!("14 image {i}")),
                dimension: TextureDimension::D2,
                extent: [3, 3, 1],
                array_layers: 1,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::TEXTURE_BINDING.union(TextureUsage::COPY_DST),
                view_formats: vec![],
            })?;
            views.push(d.create_view(
                &t,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
            )?);
            textures.push(t)
        }
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
                resource: BindingResource::Sampler(&sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Array(
                    views
                        .iter()
                        .map(|v| Some(BindingResource::TextureView(v)))
                        .collect(),
                ),
            },
        ];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("14 textures"),
            layout: &layout,
            entries: &entries,
        })?;
        let pl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("14 layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let pipeline = pipeline(d, pl, format).await?;
        let (depth, depth_view) = depth_target(d, extent)?;
        Ok(Self {
            lane,
            vertex,
            index,
            uniform,
            textures,
            _depth: depth,
            depth_view,
            group,
            pipeline,
            uploaded: false,
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>> {
        let mut e = d.encoder(self.lane)?;
        if !self.uploaded {
            e.write_buffer(&self.vertex, 0, &cube_vertices())?;
            e.write_buffer(&self.index, 0, &cube_indices())?;
            for (i, t) in self.textures.iter().enumerate() {
                let p = texture_pixels(i as u8);
                e.write_texture(
                    TextureRegion {
                        texture: t,
                        mip: 0,
                        origin: [0; 3],
                        extent: [3, 3, 1],
                        aspect: TextureAspects::COLOR,
                    },
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: 12,
                        rows_per_image: 3,
                    },
                    &p,
                )?
            }
            self.uploaded = true
        }
        e.write_buffer(&self.uniform, 0, &matrices(extent, time))?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.04, 1.])),
            store: StoreOp::Store,
        })];
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("14 descriptor indexed cubes"),
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
            secondary_inheritance: None,
        })?;
        p.set_pipeline(&self.pipeline)?;
        p.set_bind_group(0, &self.group, &[])?;
        p.set_vertex_buffer(
            0,
            &self.vertex,
            0..VERTEX_BYTES * u64::from(VERTICES_PER_CUBE) * CUBES as u64,
        )?;
        p.set_index_buffer(
            &self.index,
            0..u64::from(INDICES_PER_CUBE) * CUBES as u64 * 4,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..INDICES_PER_CUBE * CUBES as u32, 0, 0..1)?;
        p.finish()?;
        e.finish()
    }
}
fn buffer(label: &str, size: u64, usage: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    }
}
fn depth_target<B: DescriptorIndexingBackend>(
    d: &Device<B>,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = d.create_texture(&TextureDesc {
        label: Some("14 depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = d.create_view(
        &texture,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((texture, view))
}
async fn pipeline<B: DescriptorIndexingBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
) -> Result<RasterPipeline<B>> {
    let desc = |label: &str, bytes: &[u8], entry: &str, stage| ShaderDesc {
        label: Some(label.into()),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: entry.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vs = d
        .shader(desc(
            "14 descriptor vertex",
            VS,
            "vs_main",
            ShaderStage::Vertex,
        ))
        .await?;
    let ps = d
        .shader(desc(
            "14 descriptor fragment",
            PS,
            "ps_main",
            ShaderStage::Fragment,
        ))
        .await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some("14 descriptor-indexing pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "ps_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: VERTEX_BYTES,
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
                    format: VertexFormat::Sint32(1),
                },
            ],
        }],
        color_targets: vec![Some(ColorTargetState {
            format,
            blend: None,
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            // The generated vertex order follows the C++ sample's Vulkan
            // coordinate convention.  Keep both windings while this demo
            // intentionally uses a hand-expanded, non-indexed cube stream.
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
            stencil_front: StencilFaceState {
                compare: CompareFunction::Always,
                fail: StencilOperation::Keep,
                depth_fail: StencilOperation::Keep,
                pass: StencilOperation::Keep,
            },
            stencil_back: StencilFaceState {
                compare: CompareFunction::Always,
                fail: StencilOperation::Keep,
                depth_fail: StencilOperation::Keep,
                pass: StencilOperation::Keep,
            },
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
    .await
}
fn texture_pixels(n: u8) -> [u8; 36] {
    let mut p = [0; 36];
    for y in 0..3 {
        for x in 0..3 {
            let i = (y * 3 + x) * 4;
            p[i..i + 4].copy_from_slice(&[
                n.wrapping_mul(37).wrapping_add(x as u8 * 45),
                n.wrapping_mul(17).wrapping_add(y as u8 * 65),
                n.wrapping_mul(73),
                255,
            ])
        }
    }
    p
}
fn cube_vertices() -> Vec<u8> {
    // Same 24 per-face vertices as the C++ sample.  The indexed stream keeps
    // each face's texture index and winding intact.
    let c = [
        ([-1., -1., 1.], [0., 0.]),
        ([1., -1., 1.], [1., 0.]),
        ([1., 1., 1.], [1., 1.]),
        ([-1., 1., 1.], [0., 1.]),
        ([1., 1., 1.], [0., 0.]),
        ([1., 1., -1.], [1., 0.]),
        ([1., -1., -1.], [1., 1.]),
        ([1., -1., 1.], [0., 1.]),
        ([-1., -1., -1.], [0., 0.]),
        ([1., -1., -1.], [1., 0.]),
        ([1., 1., -1.], [1., 1.]),
        ([-1., 1., -1.], [0., 1.]),
        ([-1., -1., -1.], [0., 0.]),
        ([-1., -1., 1.], [1., 0.]),
        ([-1., 1., 1.], [1., 1.]),
        ([-1., 1., -1.], [0., 1.]),
        ([1., 1., 1.], [0., 0.]),
        ([-1., 1., 1.], [1., 0.]),
        ([-1., 1., -1.], [1., 1.]),
        ([1., 1., -1.], [0., 1.]),
        ([-1., -1., -1.], [0., 0.]),
        ([1., -1., -1.], [1., 0.]),
        ([1., -1., 1.], [1., 1.]),
        ([-1., -1., 1.], [0., 1.]),
    ];
    let mut o = Vec::new();
    for cube in 0..CUBES {
        for (vertex, (mut position, uv)) in c.into_iter().enumerate() {
            position[0] += 2.5 * cube as f32 - (CUBES as f32 * 2.5 / 2.0) + 1.25;
            for value in position.into_iter().chain(uv) {
                o.extend_from_slice(&value.to_ne_bytes());
            }
            o.extend_from_slice(&(((cube * 6 + vertex / 4) % TEXTURES) as u32).to_ne_bytes());
        }
    }
    o
}
fn cube_indices() -> Vec<u8> {
    let local = [
        0u32, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    ];
    (0..CUBES)
        .flat_map(|cube| {
            local
                .into_iter()
                .map(move |index| index + cube as u32 * VERTICES_PER_CUBE)
        })
        .flat_map(u32::to_ne_bytes)
        .collect()
}
fn matrices(extent: [u32; 2], _time: f32) -> [u8; 256] {
    let a = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1. / (45f32.to_radians() * 0.5).tan();
    // C++ `Camera::lookat`: position (0, 0, -10), rotation (-35, 0, 0).
    let (s, c) = (-35.0_f32.to_radians()).sin_cos();
    let v = [
        f / a,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        -1.001,
        -1.,
        0.,
        0.,
        -0.1001,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        c,
        s,
        0.,
        0.,
        -s,
        c,
        0.,
        0.,
        0.,
        -10.,
        1.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
    ];
    let mut b = [0; 256];
    for (i, x) in v.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&x.to_ne_bytes())
    }
    b
}
#[cfg(windows)]
fn main() {
    use fluxel_rhi::dx12::Dx12;
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let start = std::time::Instant::now();
    let mut scene: Option<DescriptorIndexing<Dx12>> = None;
    let r = common::run_example(
        common::windows_api::Dx12Host,
        "Descriptor indexing",
        config.clone(),
        move |d, _, i, e| {
            let lane = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene.is_none() {
                scene = Some(common::block_on(DescriptorIndexing::new(
                    d,
                    lane,
                    config.format,
                    e,
                ))?)
            }
            let w = scene.as_mut().ok_or(Error::InvalidState)?.record(
                d,
                &i,
                e,
                start.elapsed().as_secs_f32(),
            )?;
            common::submit_and_present(d, lane, w, i)
        },
    );
    if let Err(e) = r {
        eprintln!("14_descriptor_indexing: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("14_descriptor_indexing requires Windows and runtime descriptor indexing")
}
