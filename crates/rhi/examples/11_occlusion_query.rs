//! Occlusion queries with native DX12 BeginQuery/EndQuery/ResolveQueryData.
mod common;
use fluxel_rhi::api::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
const VS: &[u8] = include_bytes!("assets/01_triangle/triangle_vs.dxil");
const PS: &[u8] = include_bytes!("assets/01_triangle/triangle_ps.dxil");
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
    plane: Buffer<B>,
    red: Buffer<B>,
    green: Buffer<B>,
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
        let make = |name: &str, n: usize| {
            d.create_buffer(&BufferDesc {
                label: Some(name.into()),
                size: n as u64,
                usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
        };
        let plane = make("11 blue occluder", 144)?;
        let red = make("11 queried teapot", 72)?;
        let green = make("11 queried sphere", 72)?;
        let vs = shader(d, "11 VS", VS, "vs_main", ShaderStage::Vertex).await?;
        let ps = shader(d, "11 PS", PS, "ps_main", ShaderStage::Fragment).await?;
        let l = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("11 layout"),
            groups: &[],
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
            plane,
            red,
            green,
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
        e.write_buffer(
            &self.plane,
            0,
            &quad([-0.85, -0.85, 0.2], [0.85, 0.85, 0.2], [0.08, 0.20, 0.95]),
        )?;
        e.write_buffer(
            &self.red,
            0,
            &tri(
                [-0.55, -0.45, 0.10],
                [-0.05, -0.45, 0.10],
                [-0.30, 0.42, 0.10],
                [0.95, 0.12, 0.08],
            ),
        )?;
        e.write_buffer(
            &self.green,
            0,
            &tri(
                [0.05, -0.45, 0.75],
                [0.55, -0.45, 0.75],
                [0.30, 0.42, 0.75],
                [0.10, 0.90, 0.20],
            ),
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
        p.set_vertex_buffer(0, &self.plane, 0..144)?;
        p.draw(0..6, 0..1)?;
        queries.begin_occlusion(&mut p, &self.queries, 0)?;
        p.set_vertex_buffer(0, &self.red, 0..72)?;
        p.draw(0..3, 0..1)?;
        queries.end_occlusion(&mut p)?;
        queries.begin_occlusion(&mut p, &self.queries, 1)?;
        p.set_vertex_buffer(0, &self.green, 0..72)?;
        p.draw(0..3, 0..1)?;
        queries.end_occlusion(&mut p)?;
        if self.visible[0] {
            p.set_vertex_buffer(0, &self.red, 0..72)?;
            p.draw(0..3, 0..1)?
        }
        if self.visible[1] {
            p.set_vertex_buffer(0, &self.green, 0..72)?;
            p.draw(0..3, 0..1)?
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
