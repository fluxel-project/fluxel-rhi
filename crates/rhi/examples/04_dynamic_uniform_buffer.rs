//! Dynamic-uniform cube grid through the RHI.
//!
//! The second group binding is a real dynamically-offset uniform binding.  One
//! aligned model matrix is selected per indexed cube draw; no instance data is
//! substituted for the operation demonstrated here.
mod common;
use fluxel_rhi::api::*;
use std::sync::Arc;
const VS: &[u8] = include_bytes!("assets/04_dynamic_uniform_buffer/dynamic_uniform.vs.dxil");
const PS: &[u8] = include_bytes!("assets/04_dynamic_uniform_buffer/dynamic_uniform.ps.dxil");
pub const OBJECT_COUNT: u32 = 125;
pub const MODEL_BYTES: u64 = 64;
pub trait DynamicUniformBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> DynamicUniformBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
pub struct DynamicUniforms<B: DynamicUniformBackend> {
    vertex: Buffer<B>,
    index: Buffer<B>,
    camera: Buffer<B>,
    models: Buffer<B>,
    group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    lane: LaneId,
    alignment: u64,
}
impl<B: DynamicUniformBackend> DynamicUniforms<B> {
    pub async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        let alignment = d.capabilities().limits.uniform_offset_alignment.max(256);
        let size = aligned(MODEL_BYTES, alignment) * u64::from(OBJECT_COUNT);
        let vertex = d.create_buffer(&buffer(
            "04 cube vertices",
            192,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = d.create_buffer(&buffer(
            "04 cube indices",
            72,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let camera = d.create_buffer(&buffer(
            "04 camera",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let models = d.create_buffer(&buffer(
            "04 dynamic models",
            size,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("04 dynamic uniforms".into()),
            entries: vec![entry(0, false, 64), entry(1, true, 64)],
        })?;
        let es = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &camera,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: &models,
                    range: 0..MODEL_BYTES,
                },
            },
        ];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("04 group"),
            layout: &layout,
            entries: &es,
        })?;
        let pl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("04 layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let pipeline = make_pipeline(d, pl, format).await?;
        Ok(Self {
            vertex,
            index,
            camera,
            models,
            group,
            pipeline,
            lane,
            alignment,
        })
    }
    pub fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<RecordedWork<B>> {
        let mut e = d.encoder(self.lane)?;
        e.write_buffer(&self.vertex, 0, &cube_vertices())?;
        e.write_buffer(&self.index, 0, &cube_indices())?;
        e.write_buffer(&self.camera, 0, &identity())?;
        for n in 0..OBJECT_COUNT {
            e.write_buffer(
                &self.models,
                aligned(MODEL_BYTES, self.alignment) * u64::from(n),
                &model(n, time),
            )?
        }
        let c = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.02, 0.02, 0.03, 1.0])),
            store: StoreOp::Store,
        })];
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("04 dynamic uniforms"),
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
            secondary_inheritance: None,
        })?;
        p.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        p.set_pipeline(&self.pipeline)?;
        p.set_vertex_buffer(0, &self.vertex, 0..192)?;
        p.set_index_buffer(&self.index, 0..72, IndexFormat::Uint16)?;
        for n in 0..OBJECT_COUNT {
            let offset = aligned(MODEL_BYTES, self.alignment) * u64::from(n);
            p.set_bind_group(
                0,
                &self.group,
                &[u32::try_from(offset).map_err(|_| Error::InvalidRange)?],
            )?;
            p.draw_indexed(0..36, 0, 0..1)?
        }
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
fn entry(binding: u32, dynamic_offset: bool, min_size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Vertex],
        ty: BindingType::UniformBuffer {
            dynamic_offset,
            min_size,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
fn aligned(n: u64, a: u64) -> u64 {
    n.div_ceil(a) * a
}
async fn shader<B: DynamicUniformBackend>(
    d: &Device<B>,
    bytes: &[u8],
    name: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    d.shader(ShaderDesc {
        label: Some(name.into()),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: name.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    })
    .await
}
async fn make_pipeline<B: DynamicUniformBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
) -> Result<RasterPipeline<B>> {
    let vs = shader(d, VS, "vs_main", ShaderStage::Vertex).await?;
    let ps = shader(d, PS, "fs_main", ShaderStage::Fragment).await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some("04 pipeline".into()),
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
            stride: 24,
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
fn identity() -> [u8; 64] {
    let mut b = [0; 64];
    for i in 0..4 {
        b[i * 20..i * 20 + 4].copy_from_slice(&1f32.to_ne_bytes())
    }
    b
}
fn model(n: u32, t: f32) -> [u8; 64] {
    let mut b = identity();
    let x = (n % 5) as f32 - 2.0;
    let y = ((n / 5) % 5) as f32 - 2.0;
    let z = (n / 25) as f32 - 2.0;
    b[48..52].copy_from_slice(&(x * 0.35).to_ne_bytes());
    b[52..56].copy_from_slice(&(y * 0.35).to_ne_bytes());
    b[56..60].copy_from_slice(&(z * 0.35 + t.sin() * 0.05).to_ne_bytes());
    b
}
fn cube_vertices() -> [u8; 192] {
    let p: [f32; 48] = [
        -1., -1., -1., 1., 0., 0., 1., -1., -1., 0., 1., 0., 1., 1., -1., 0., 0., 1., -1., 1., -1.,
        1., 1., 0., -1., -1., 1., 1., 0., 1., 1., -1., 1., 0., 1., 1., 1., 1., 1., 1., 1., -1., 1.,
        1., 0., 0., 0., 0.,
    ];
    let mut b = [0; 192];
    for (i, v) in p.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
fn cube_indices() -> [u8; 72] {
    let x: [u16; 36] = [
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 4, 5, 5, 1, 0, 3, 2, 6, 6, 7, 3, 1, 5, 6, 6, 2, 1,
        0, 3, 7, 7, 4, 0,
    ];
    let mut b = [0; 72];
    for (i, v) in x.into_iter().enumerate() {
        b[i * 2..i * 2 + 2].copy_from_slice(&v.to_ne_bytes())
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
    let mut workload: Option<DynamicUniforms<Dx12>> = None;
    let start = std::time::Instant::now();
    let result = common::run_example(
        common::windows_api::Dx12Host,
        "Dynamic uniform buffers",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if workload.is_none() {
                workload = Some(common::block_on(DynamicUniforms::new(
                    device,
                    lane,
                    config.format,
                ))?)
            }
            let work = workload.as_ref().ok_or(Error::InvalidState)?.record(
                device,
                &image,
                extent,
                start.elapsed().as_secs_f32(),
            )?;
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("04_dynamic_uniform_buffer: {error:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("04_dynamic_uniform_buffer requires Windows and DX12");
}
