//! Occlusion queries with native DX12 BeginQuery/EndQuery/ResolveQueryData.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
use fluxel_rhi::api::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
const PLANE: &[u8] = include_bytes!("assets/models/plane_z.gltf");
const TEAPOT: &[u8] = include_bytes!("assets/models/teapot.gltf");
const SPHERE: &[u8] = include_bytes!("assets/models/sphere.gltf");
const VS: &[u8] = include_bytes!("assets/11_occlusion_query/occlusion.vs.dxil");
const PS: &[u8] = include_bytes!("assets/11_occlusion_query/occlusion.ps.dxil");
static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);
pub trait OcclusionBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + RasterBackend
    + DeferredBackend
    + QueryBackend
    + ReadbackBackend
{
}
impl<T> OcclusionBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
        + QueryBackend
        + ReadbackBackend
{
}
pub struct Workload<B: OcclusionBackend> {
    lane: LaneId,
    plane_vertices: Buffer<B>,
    plane_indices: Buffer<B>,
    teapot_vertices: Buffer<B>,
    teapot_indices: Buffer<B>,
    sphere_vertices: Buffer<B>,
    sphere_indices: Buffer<B>,
    plane_count: u32,
    teapot_count: u32,
    sphere_count: u32,
    plane_len: u64,
    teapot_len: u64,
    sphere_len: u64,
    plane_index_len: u64,
    teapot_index_len: u64,
    sphere_index_len: u64,
    plane_uniform: Buffer<B>,
    teapot_uniform: Buffer<B>,
    sphere_uniform: Buffer<B>,
    plane_group: BindGroup<B>,
    teapot_group: BindGroup<B>,
    sphere_group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    queries: QuerySet<B>,
    results: Buffer<B>,
    visible: [bool; 2],
}
pub struct Frame<B: OcclusionBackend> {
    work: RecordedWork<B>,
    ticket: ReadbackTicket<B>,
}
impl<B: OcclusionBackend> Workload<B> {
    pub async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        let q = d
            .queries(QueryKind::Occlusion)
            .ok_or(Error::Unsupported("occlusion queries"))?
            .create(&QueryDesc {
                kind: QueryKind::Occlusion,
                count: 2,
            })?;
        let make = |name: &str, n: usize, usage: BufferUsage| {
            d.create_buffer(&BufferDesc {
                label: Some(name.into()),
                size: n as u64,
                usage: usage.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
        };
        let plane =
            gltf::load_embedded_model(PLANE, gltf::LoadOptions::CPP_PORT.with_flip_y(false))
                .map_err(|x| Error::Backend(x.to_string()))?;
        let teapot =
            gltf::load_embedded_model(TEAPOT, gltf::LoadOptions::CPP_PORT.with_flip_y(false))
                .map_err(|x| Error::Backend(x.to_string()))?;
        let sphere =
            gltf::load_embedded_model(SPHERE, gltf::LoadOptions::CPP_PORT.with_flip_y(false))
                .map_err(|x| Error::Backend(x.to_string()))?;
        let plane_vertices = make(
            "11 occluder vertices",
            plane.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let plane_indices = make(
            "11 occluder indices",
            plane.indices.len(),
            BufferUsage::INDEX,
        )?;
        let teapot_vertices = make(
            "11 teapot vertices",
            teapot.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let teapot_indices = make(
            "11 teapot indices",
            teapot.indices.len(),
            BufferUsage::INDEX,
        )?;
        let sphere_vertices = make(
            "11 sphere vertices",
            sphere.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let sphere_indices = make(
            "11 sphere indices",
            sphere.indices.len(),
            BufferUsage::INDEX,
        )?;
        let plane_uniform = make("11 occluder transform", 256, BufferUsage::UNIFORM)?;
        let teapot_uniform = make("11 teapot transform", 256, BufferUsage::UNIFORM)?;
        let sphere_uniform = make("11 sphere transform", 256, BufferUsage::UNIFORM)?;
        let vs = shader(d, "11 VS", VS, "vs_main", ShaderStage::Vertex).await?;
        let ps = shader(d, "11 PS", PS, "ps_main", ShaderStage::Fragment).await?;
        let layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("11 transform layout".into()),
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
        let bind = |buffer: &Buffer<B>, name: &str| {
            d.create_bind_group(&BindGroupDesc {
                label: Some(name.into()),
                layout: &layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer,
                        range: 0..256,
                    },
                }],
            })
        };
        let plane_group = bind(&plane_uniform, "11 occluder group")?;
        let teapot_group = bind(&teapot_uniform, "11 teapot group")?;
        let sphere_group = bind(&sphere_uniform, "11 sphere group")?;
        let l = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("11 layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let face = StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        };
        let p = d
            .raster_pipeline(RasterPipelineDesc {
                label: Some("11 query pipeline".into()),
                layout: l,
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
                    stride: 96,
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
                            format: VertexFormat::Float32(2),
                        },
                        VertexAttribute {
                            location: 3,
                            offset: 32,
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
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::LessEqual,
                    stencil_front: face,
                    stencil_back: face,
                    stencil_read_mask: 0,
                    stencil_write_mask: 0,
                    depth_bias_constant: 0,
                    depth_bias_slope_scale: 0.,
                    depth_bias_clamp: 0.,
                }),
                multisample: MultisampleState {
                    count: 1,
                    mask: u64::from(u32::MAX),
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await?;
        let results = d.create_buffer(&BufferDesc {
            label: Some("11 resolved occlusion results".into()),
            size: 16,
            usage: BufferUsage::QUERY_RESOLVE.union(BufferUsage::MAP_READ),
            memory: MemoryClass::Readback,
        })?;
        Ok(Self {
            lane,
            plane_vertices,
            plane_indices,
            teapot_vertices,
            teapot_indices,
            sphere_vertices,
            sphere_indices,
            plane_count: (plane.indices.len() / 4) as u32,
            teapot_count: (teapot.indices.len() / 4) as u32,
            sphere_count: (sphere.indices.len() / 4) as u32,
            plane_len: plane.vertices.len() as u64,
            teapot_len: teapot.vertices.len() as u64,
            sphere_len: sphere.vertices.len() as u64,
            plane_index_len: plane.indices.len() as u64,
            teapot_index_len: teapot.indices.len() as u64,
            sphere_index_len: sphere.indices.len() as u64,
            plane_uniform,
            teapot_uniform,
            sphere_uniform,
            plane_group,
            teapot_group,
            sphere_group,
            pipeline: p,
            queries: q,
            results,
            visible: [true; 2],
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<Frame<B>> {
        if extent.contains(&0) {
            return Err(Error::InvalidRange);
        }
        let mut e = d.encoder(self.lane)?;
        if self.visible == [true, true] {
            let plane =
                gltf::load_embedded_model(PLANE, gltf::LoadOptions::CPP_PORT.with_flip_y(false))
                    .map_err(|x| Error::Backend(x.to_string()))?;
            let teapot =
                gltf::load_embedded_model(TEAPOT, gltf::LoadOptions::CPP_PORT.with_flip_y(false))
                    .map_err(|x| Error::Backend(x.to_string()))?;
            let sphere =
                gltf::load_embedded_model(SPHERE, gltf::LoadOptions::CPP_PORT.with_flip_y(false))
                    .map_err(|x| Error::Backend(x.to_string()))?;
            e.write_buffer(&self.plane_vertices, 0, &plane.vertices)?;
            e.write_buffer(&self.plane_indices, 0, &plane.indices)?;
            e.write_buffer(&self.teapot_vertices, 0, &teapot.vertices)?;
            e.write_buffer(&self.teapot_indices, 0, &teapot.indices)?;
            e.write_buffer(&self.sphere_vertices, 0, &sphere.vertices)?;
            e.write_buffer(&self.sphere_indices, 0, &sphere.indices)?;
        }
        e.write_buffer(
            &self.plane_uniform,
            0,
            &transform(extent, [0.0, 0.0, 0.0], 6.0, [0.0, 0.0, 1.0, 0.5]),
        )?;
        e.write_buffer(
            &self.teapot_uniform,
            0,
            &transform(extent, [0.0, 0.0, -3.0], 1.0, [1.0, 0.0, 0.0, 1.0]),
        )?;
        e.write_buffer(
            &self.sphere_uniform,
            0,
            &transform(extent, [0.0, 0.0, 3.0], 1.0, [0.0, 1.0, 0.0, 1.0]),
        )?;
        let (_depth, view) = depth(d, extent)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.025, 1.])),
            store: StoreOp::Store,
        })];
        let da = DepthStencilAttachment {
            view: AttachmentView::Texture(&view),
            depth: Some((LoadOp::Clear(1.), StoreOp::Discard)),
            stencil: None,
            depth_read_only: false,
            stencil_read_only: false,
        };
        let queries = d
            .queries(QueryKind::Occlusion)
            .ok_or(Error::Unsupported("occlusion queries"))?;
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("11 occlusion query pass"),
            colors: &colors,
            depth_stencil: Some(da),
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
            x: 0.,
            y: 0.,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.,
            max_depth: 1.,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        p.set_pipeline(&self.pipeline)?;
        p.set_bind_group(0, &self.plane_group, &[])?;
        p.set_vertex_buffer(0, &self.plane_vertices, 0..self.plane_len)?;
        p.set_index_buffer(
            &self.plane_indices,
            0..self.plane_index_len,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..self.plane_count, 0, 0..1)?;
        queries.begin_occlusion(&mut p, &self.queries, 0)?;
        p.set_bind_group(0, &self.teapot_group, &[])?;
        p.set_vertex_buffer(0, &self.teapot_vertices, 0..self.teapot_len)?;
        p.set_index_buffer(
            &self.teapot_indices,
            0..self.teapot_index_len,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..self.teapot_count, 0, 0..1)?;
        queries.end_occlusion(&mut p)?;
        queries.begin_occlusion(&mut p, &self.queries, 1)?;
        p.set_bind_group(0, &self.sphere_group, &[])?;
        p.set_vertex_buffer(0, &self.sphere_vertices, 0..self.sphere_len)?;
        p.set_index_buffer(
            &self.sphere_indices,
            0..self.sphere_index_len,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..self.sphere_count, 0, 0..1)?;
        queries.end_occlusion(&mut p)?;
        if self.visible[0] {
            p.set_bind_group(0, &self.teapot_group, &[])?;
            p.set_vertex_buffer(0, &self.teapot_vertices, 0..self.teapot_len)?;
            p.set_index_buffer(
                &self.teapot_indices,
                0..self.teapot_index_len,
                IndexFormat::Uint32,
            )?;
            p.draw_indexed(0..self.teapot_count, 0, 0..1)?
        }
        if self.visible[1] {
            p.set_bind_group(0, &self.sphere_group, &[])?;
            p.set_vertex_buffer(0, &self.sphere_vertices, 0..self.sphere_len)?;
            p.set_index_buffer(
                &self.sphere_indices,
                0..self.sphere_index_len,
                IndexFormat::Uint32,
            )?;
            p.draw_indexed(0..self.sphere_count, 0, 0..1)?
        }
        p.finish()?;
        queries.resolve(&mut e, &self.queries, 0..2, &self.results, 0)?;
        let ticket = e.readback_buffer(&self.results, 0..16)?;
        Ok(Frame {
            work: e.finish()?,
            ticket,
        })
    }
    fn consume(
        &mut self,
        d: &Device<B>,
        ticket: ReadbackTicket<B>,
        c: &CompletionPoint<B>,
    ) -> Result<()> {
        common::block_on(d.wait_completion(c))?;
        let v = common::block_on(unsafe { d.read_after(ticket, c) })?.bytes;
        if v.len() != 16 {
            return Err(Error::Backend("occlusion result size".into()));
        }
        let passed = [
            u64::from_le_bytes(v[..8].try_into().map_err(|_| Error::InvalidState)?),
            u64::from_le_bytes(v[8..].try_into().map_err(|_| Error::InvalidState)?),
        ];
        self.visible = [passed[0] != 0, passed[1] != 0];
        Ok(())
    }
}
async fn shader<B: OcclusionBackend>(
    d: &Device<B>,
    n: &str,
    b: &[u8],
    e: &str,
    s: ShaderStage,
) -> Result<Shader<B>> {
    d.shader(ShaderDesc {
        label: Some(n.into()),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(b),
        entries: vec![ShaderEntryPoint {
            name: e.into(),
            stage: s,
            workgroup_size: None,
        }],
        required_features: vec![],
    })
    .await
}
fn depth<B: OcclusionBackend>(d: &Device<B>, e: [u32; 2]) -> Result<(Texture<B>, TextureView<B>)> {
    let t = d.create_texture(&TextureDesc {
        label: Some("11 depth".into()),
        dimension: TextureDimension::D2,
        extent: [e[0], e[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let v = d.create_view(
        &t,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((t, v))
}
fn tri(a: [f32; 3], b: [f32; 3], c: [f32; 3], k: [f32; 3]) -> Vec<u8> {
    [a, b, c]
        .into_iter()
        .flat_map(|p| p.into_iter().chain(k).flat_map(f32::to_le_bytes))
        .collect()
}
fn transform(extent: [u32; 2], translate: [f32; 3], scale: f32, tint: [f32; 4]) -> [u8; 256] {
    // C++ camera: look-at style `translate(0, 0, -7.5) * rotateY(-123.75)`
    // and a Vulkan depth-range perspective (60°, near 1, far 256).
    let multiply = |a: [f32; 16], b: [f32; 16]| {
        let mut out = [0.0; 16];
        for column in 0..4 {
            for row in 0..4 {
                out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
            }
        }
        out
    };
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / (30.0_f32.to_radians()).tan();
    let projection = [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        256.0 / -255.0,
        -1.0,
        0.0,
        0.0,
        256.0 / -255.0,
        0.0,
    ];
    let (s, c) = (-123.75_f32.to_radians()).sin_cos();
    let rotate = [
        c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let view = multiply(
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -7.5, 1.0,
        ],
        rotate,
    );
    let model = [
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        translate[0],
        translate[1],
        translate[2],
        1.0,
    ];
    let matrix = multiply(projection, multiply(view, model));
    let mut bytes = [0; 256];
    for (index, value) in matrix.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    for (index, value) in tint.into_iter().enumerate() {
        let offset = 64 + index * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
fn quad(a: [f32; 3], b: [f32; 3], k: [f32; 3]) -> Vec<u8> {
    tri(a, [b[0], a[1], a[2]], b, k)
        .into_iter()
        .chain(tri(a, b, [a[0], b[1], a[2]], k))
        .collect()
}
#[cfg(windows)]
fn main() {
    use fluxel_rhi::dx12::Dx12;
    let cfg = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut work: Option<Workload<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
        "Occlusion queries",
        cfg.clone(),
        move |d, _, image, extent| {
            let lane = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if work.is_none() {
                work = Some(common::block_on(Workload::new(d, lane, cfg.format))?)
            }
            let w = work.as_mut().ok_or(Error::InvalidState)?;
            let Frame {
                work: recorded,
                ticket,
            } = w.record(d, &image, extent)?;
            let c = submit(d, lane, recorded, image)?;
            w.consume(d, ticket, &c)
        },
    );
    if let Err(e) = result {
        eprintln!("11_occlusion_query: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("11_occlusion_query requires Windows and DX12")
}
fn submit<B: DeferredBackend>(
    d: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<CompletionPoint<B>> {
    let id = PlanId {
        device: d.id(),
        ordinal: NEXT_PLAN.fetch_add(1, Ordering::Relaxed),
    };
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![StepWait {
                        previous: StepId { plan: id, index: 0 },
                        destination: AccessScope {
                            stages: StageMask::ALL_COMMANDS,
                            accesses: AccessMask::ALL,
                        },
                    }],
                },
            ]),
        )
    };
    match unsafe {
        d.queue().submit(SubmitPacket {
            steps: vec![
                BoundStep {
                    step: plan.step(0),
                    payload: StepPayload::Commands(vec![work]),
                },
                BoundStep {
                    step: plan.step(1),
                    payload: StepPayload::Present(PresentBinding::new(image)),
                },
            ],
            plan,
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(r)
            if r.accepted.len() == 2 && r.presents.len() == 1 && r.presents[0].accepted =>
        {
            r.checkpoints
                .into_iter()
                .next()
                .map(|x| x.point)
                .ok_or(Error::InvalidState)
        }
        SubmitOutcome::Accepted(_) => Err(Error::InvalidState),
        SubmitOutcome::Rejected { error, .. }
        | SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => Err(error),
    }
}
