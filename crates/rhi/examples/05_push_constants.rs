//! Push-constant sphere ring through the RHI.
//!
//! The original Vulkan workload issues sixteen indexed sphere draws.  Every
//! draw writes its own 32-byte `{ color, position }` record to the vertex stage;
//! it is deliberately not replaced with instance data or a dynamic uniform.

mod common;

use fluxel_rhi::api::*;
use std::sync::Arc;

pub const SPHERE_COUNT: usize = 16;
pub const PUSH_BYTES: u32 = 32;

/// The native DXIL source for the b0 camera / b1 root-constant ABI.
pub const VERTEX_DXIL: &[u8] = include_bytes!("assets/05_push_constants/push_constants.vs.dxil");
pub const FRAGMENT_DXIL: &[u8] = include_bytes!("assets/05_push_constants/push_constants.ps.dxil");

/// One source-equivalent `color + position` record.
#[derive(Clone, Copy)]
pub struct PushData {
    pub color: [f32; 4],
    pub position: [f32; 4],
}

impl PushData {
    pub fn bytes(self) -> [u8; PUSH_BYTES as usize] {
        let mut bytes = [0; PUSH_BYTES as usize];
        for (index, value) in self.color.into_iter().chain(self.position).enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        }
        bytes
    }
}

/// The same sixteen positions as the original sample's radius-3.5 circle.
pub fn sphere_ring() -> [PushData; SPHERE_COUNT] {
    std::array::from_fn(|index| {
        let angle = (index as f32 * 360.0 / SPHERE_COUNT as f32).to_radians();
        let (sin, cos) = angle.sin_cos();
        // Deterministic colors keep the original 0.1..1.0 range without
        // making the API command workload depend on a host RNG.
        let seed = index as f32;
        PushData {
            color: [
                0.1 + (seed * 0.37).fract() * 0.9,
                0.1 + (seed * 0.61).fract() * 0.9,
                0.1 + (seed * 0.83).fract() * 0.9,
                1.0,
            ],
            position: [sin * 3.5, cos * 3.5, 0.0, 1.0],
        }
    })
}

/// Encodes the original sixteen push-constant indexed draws.
///
/// The caller supplies the sphere mesh, camera bind group and already compiled
/// pipeline. Graph transitions and the containing surface submission remain
/// outside the API command layer.
pub fn encode_sphere_ring<B>(
    device: &Device<B>,
    encoder: &mut CommandEncoder<B>,
    pass: &RasterPassDesc<'_, B>,
    pipeline: &RasterPipeline<B>,
    camera: &BindGroup<B>,
    vertices: &Buffer<B>,
    vertex_bytes: u64,
    indices: &Buffer<B>,
    index_bytes: u64,
    index_count: u32,
    spheres: &[PushData; SPHERE_COUNT],
) -> Result<()>
where
    B: RasterBackend + DeferredBackend,
{
    if !device.capabilities().push_constants {
        return Err(Error::Unsupported(
            "05_push_constants requires 32-byte vertex push constants",
        ));
    }
    if device.capabilities().limits.max_push_constant_bytes < PUSH_BYTES {
        return Err(Error::Unsupported(
            "05_push_constants requires a 32-byte push-constant limit",
        ));
    }
    let extras = device
        .raster_extras()
        .ok_or(Error::Unsupported("push constants"))?;
    let mut raster = encoder.begin_raster(pass)?;
    raster.viewport(Viewport {
        x: pass.render_area.x as f32,
        y: pass.render_area.y as f32,
        width: pass.render_area.width as f32,
        height: pass.render_area.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    raster.scissor(pass.render_area)?;
    raster.set_pipeline(pipeline)?;
    raster.set_bind_group(0, camera, &[])?;
    raster.set_vertex_buffer(0, vertices, 0..vertex_bytes)?;
    raster.set_index_buffer(indices, 0..index_bytes, IndexFormat::Uint32)?;
    for sphere in spheres {
        extras.push_constants(&mut raster, StageMask::VERTEX, 0, &sphere.bytes())?;
        raster.draw_indexed(0..index_count, 0, 0..1)?;
    }
    raster.finish()
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
    let mut workload: Option<PushWorkload<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
        "Push constants",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if workload.is_none() {
                workload = Some(common::block_on(PushWorkload::new(
                    device,
                    lane,
                    config.format,
                ))?);
            }
            let work = workload
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent)?;
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("05_push_constants: {error:?}");
        std::process::exit(1);
    }
}

pub trait PushBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> PushBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
struct PushWorkload<B: PushBackend> {
    lane: LaneId,
    vertex: Buffer<B>,
    index: Buffer<B>,
    camera: Buffer<B>,
    group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    uploaded: bool,
}
impl<B: PushBackend> PushWorkload<B> {
    async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        if !d.capabilities().push_constants
            || d.capabilities().limits.max_push_constant_bytes < PUSH_BYTES
        {
            return Err(Error::Unsupported("05 push constants"));
        }
        let vertex = buffer(
            d,
            "05 sphere vertices",
            240,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        )?;
        let index = buffer(
            d,
            "05 sphere indices",
            96,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        )?;
        let camera = buffer(
            d,
            "05 camera",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let group_layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("05 camera layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 64,
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
        let constants = [PushConstantRange {
            stages: vec![ShaderStage::Vertex],
            range: 0..PUSH_BYTES,
        }];
        let layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("05 layout"),
            groups: &[&group_layout],
            push_constants: &constants,
        })?;
        let pipeline = pipeline(d, layout, format).await?;
        Ok(Self {
            lane,
            vertex,
            index,
            camera,
            group,
            pipeline,
            uploaded: false,
        })
    }
    fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<B>> {
        let mut encoder = d.encoder(self.lane)?;
        if !self.uploaded {
            encoder.write_buffer(&self.vertex, 0, &sphere_vertices())?;
            encoder.write_buffer(&self.index, 0, &sphere_indices())?;
            self.uploaded = true;
        }
        encoder.write_buffer(&self.camera, 0, &camera_bytes())?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.01, 0.01, 0.02, 1.0])),
            store: StoreOp::Store,
        })];
        let pass = RasterPassDesc {
            label: Some("05 push constant spheres"),
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
            secondary_inheritance: None,
        };
        encode_sphere_ring(
            d,
            &mut encoder,
            &pass,
            &self.pipeline,
            &self.group,
            &self.vertex,
            240,
            &self.index,
            96,
            24,
            &sphere_ring(),
        )?;
        encoder.finish()
    }
}
fn buffer<B: PushBackend>(
    d: &Device<B>,
    label: &'static str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    d.create_buffer(&BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}
async fn shader<B: PushBackend>(
    d: &Device<B>,
    bytes: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    d.shader(ShaderDesc {
        label: Some("05 shader".into()),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: entry.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    })
    .await
}
async fn pipeline<B: PushBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
) -> Result<RasterPipeline<B>> {
    let vs = shader(d, VERTEX_DXIL, "vs_main", ShaderStage::Vertex).await?;
    let ps = shader(d, FRAGMENT_DXIL, "fs_main", ShaderStage::Fragment).await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some("05 push pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "fs_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: 40,
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
                    format: VertexFormat::Float32(4),
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
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
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
fn sphere_vertices() -> [u8; 240] {
    let p: [[f32; 10]; 6] = [
        [0., 1., 0., 0., 1., 0., 1., 1., 1., 1.],
        [0., -1., 0., 0., -1., 0., 1., 1., 1., 1.],
        [-1., 0., 0., -1., 0., 0., 1., 1., 1., 1.],
        [1., 0., 0., 1., 0., 0., 1., 1., 1., 1.],
        [0., 0., -1., 0., 0., -1., 1., 1., 1., 1.],
        [0., 0., 1., 0., 0., 1., 1., 1., 1., 1.],
    ];
    let mut out = [0; 240];
    for (i, x) in p.into_iter().flatten().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_ne_bytes())
    }
    out
}
fn sphere_indices() -> [u8; 96] {
    let x: [u32; 24] = [
        0, 3, 5, 0, 5, 2, 0, 2, 4, 0, 4, 3, 1, 5, 3, 1, 2, 5, 1, 4, 2, 1, 3, 4,
    ];
    let mut out = [0; 96];
    for (i, v) in x.into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    out
}
fn camera_bytes() -> [u8; 256] {
    let mut out = [0; 256];
    for i in 0..4 {
        out[(i * 4 + i) * 4..(i * 4 + i + 1) * 4]
            .copy_from_slice(&(if i < 2 { 0.2f32 } else { 1.0 }).to_ne_bytes())
    }
    out
}

#[cfg(not(windows))]
fn main() {
    eprintln!("05_push_constants requires Windows and DX12");
}
