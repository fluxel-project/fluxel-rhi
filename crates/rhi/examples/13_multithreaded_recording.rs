//! Port of SaschaWillems/Vulkan `examples/multithreading`.
//!
//! Every visible UFO is recorded in an inherited, draw-only raster packet on
//! one of the C++ example's worker threads.  The primary pass records the star
//! sphere first, then executes those packets in thread/object order.

pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use std::{sync::Arc, time::Instant};

use fluxel_rhi::api::{
    command::{
        ColorAttachment, ColorAttachmentView, ColorClearValue, DepthAttachmentMode,
        DepthStencilAttachment, IndexFormat, LoadOp, RasterScopeDescriptor, RecorderDescriptor,
        Rect, StoreOp, Viewport,
    },
    error::{RhiError, RhiErrorKind, RhiResult},
    format::TextureFormat,
    pipeline::{
        ColorTargetState, CullMode, DepthState, DepthStencilState, ImmediateRange,
        PipelineInterfaceDescriptor, PrimitiveState, PrimitiveTopology, RasterPipeline,
        RasterPipelineDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat,
        VertexInputState, VertexStepMode,
    },
    platform::Device,
    presentation::AcquiredFrame,
    resource::{
        Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor, BufferUsage,
        CompareFunction, Extent3d, TextureDescriptor, TextureUsage, TextureView,
        TextureViewDescriptor, TextureViewDimension,
    },
    shader::{
        ArtifactHash, ArtifactProducerVersion, InterpolationMode, InterpolationSampling,
        ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderImmediateRequirement, ShaderInterface,
        ShaderInterpolation, ShaderLocation, ShaderLocationInterface, ShaderNumericType,
        ShaderRequirements, ShaderStage, ShaderStages,
    },
    submission::{CompletionPoint, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder},
};

const FRAMES: usize = 2;
const OBJECTS: usize = 512;
const PUSH_BYTES: u32 = 80; // mat4 MVP + vec3 colour, matching std140 padding.
const UFO: &[u8] = include_bytes!("assets/models/retroufo_red_lowpoly.gltf");
const STAR_SPHERE: &[u8] = include_bytes!("assets/models/sphere.gltf");

// Direct translations of shaders/glsl/multithreading/{phong,starsphere}.*.
const PHONG: &str = r#"
struct Push { mvp:mat4x4<f32>, color:vec3<f32>, _pad:f32, };
var<push_constant> pc:Push;
struct I { @location(0) p:vec3<f32>, @location(1) n:vec3<f32>, @location(2) c:vec3<f32>, };
struct O { @builtin(position) p:vec4<f32>, @location(0) n:vec3<f32>, @location(1) c:vec3<f32>, @location(3) v:vec3<f32>, @location(4) l:vec3<f32>, };
@vertex fn vs_main(i:I)->O { var o:O; o.c=select(i.c,pc.color,all(i.c==vec3<f32>(1.0,0.0,0.0))); o.p=pc.mvp*vec4<f32>(i.p,1.0); let q=pc.mvp*vec4<f32>(i.p,1.0); o.n=mat3x3<f32>(pc.mvp[0].xyz,pc.mvp[1].xyz,pc.mvp[2].xyz)*i.n; o.l=-q.xyz; o.v=-q.xyz; return o; }
@fragment fn fs_main(i:O)->@location(0) vec4<f32> { let n=normalize(i.n); let l=normalize(i.l); let v=normalize(i.v); let r=reflect(-l,n); return vec4<f32>(max(dot(n,l),0.0)*i.c+pow(max(dot(r,v),0.0),8.0)*vec3<f32>(0.75),1.0); }
"#;
const STARS: &str = r#"
struct Push { mvp:mat4x4<f32>, color:vec3<f32>, _pad:f32, }; var<push_constant> pc:Push;
struct I { @location(0) p:vec3<f32>, }; struct O { @builtin(position) p:vec4<f32>, @location(0) uvw:vec3<f32>, };
@vertex fn vs_main(i:I)->O { var o:O;o.uvw=i.p;o.p=pc.mvp*vec4<f32>(i.p,1.0);return o; }
fn hash33(p:vec3<f32>)->f32 { var q=fract(p*vec3<f32>(443.897,441.423,437.195));q+=dot(q,q.yxz+vec3<f32>(19.19));return fract((q.x+q.y)*q.z+(q.x+q.z)*q.y+(q.y+q.z)*q.x); }
@fragment fn fs_main(i:O)->@location(0) vec4<f32> { let r=hash33(i.uvw);let star=select(0.0,pow((r-0.99)/0.01,16.0),r>=0.99);let atmosphere=clamp(vec3<f32>(0.1,0.15,0.4)*(i.uvw.y+0.25),vec3<f32>(0.0),vec3<f32>(1.0));return vec4<f32>(vec3<f32>(star)+atmosphere,1.0); }
"#;

fn main() {
    if let Err(error) = common::run_example("Multi threaded command buffer", create_example()) {
        eprintln!("13_multithreaded_recording: {error}");
        std::process::exit(1)
    }
}
pub fn create_example() -> Example {
    Example {
        work: None,
        lane: None,
        last: None,
    }
}
pub struct Example {
    work: Option<Workload>,
    lane: Option<SubmissionLaneId>,
    last: Option<Instant>,
}
impl common::Example for Example {
    fn init(
        &mut self,
        c: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let d = c.device().clone();
        let l = d
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|x| {
                x.domains()
                    .contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY))
            })
            .map(|x| x.id())
            .ok_or_else(|| std::io::Error::other("13 needs COPY|RASTER lane"))?;
        let e = c.extent();
        self.work = Some(common::block_on(Workload::new(
            &d,
            c.presentation_mut().configuration().format(),
            Extent3d::d2(e.width, e.height),
            l,
        ))?);
        self.lane = Some(l);
        Ok(())
    }
    fn update(
        &mut self,
        _: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = Instant::now();
        if let (Some(last), Some(w)) = (self.last, self.work.as_mut()) {
            w.delta = (now - last).as_secs_f32()
        }
        self.last = Some(now);
        Ok(())
    }
    fn render(
        &mut self,
        c: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let f = common::block_on(c.presentation_mut().acquire())?;
        common::block_on(
            self.work
                .as_mut()
                .ok_or_else(|| std::io::Error::other("not initialized"))?
                .render(
                    c.device(),
                    self.lane.ok_or_else(|| std::io::Error::other("no lane"))?,
                    f,
                ),
        )?;
        Ok(())
    }
    fn resize(
        &mut self,
        c: &mut common::ExampleContext<'_>,
        w: u32,
        h: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(x) = &mut self.work {
            x.resize(c.device(), Extent3d::d2(w, h))?
        }
        Ok(())
    }
    fn device_lost(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.work = None;
        self.lane = None;
        Ok(())
    }
    fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.device_lost()
    }
}

struct Object {
    pos: [f32; 3],
    rotation: f32,
    direction: f32,
    speed: f32,
    scale: f32,
    delta: f32,
    color: [f32; 3],
}
struct Workload {
    phong: RasterPipeline,
    stars: RasterPipeline,
    ufo_v: BufferBinding,
    ufo_i: BufferBinding,
    star_v: BufferBinding,
    star_i: BufferBinding,
    ufo_draws: Vec<(u32, u32)>,
    ufo_cull_radius: f32,
    star_draws: Vec<(u32, u32)>,
    depth: TextureView,
    extent: Extent3d,
    objects: Vec<Vec<Object>>,
    delta: f32,
    next: usize,
    done: [Option<CompletionPoint>; FRAMES],
}
impl Workload {
    async fn new(
        d: &Device,
        fmt: TextureFormat,
        e: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let ufo = gltf::load_embedded_model(UFO, gltf::LoadOptions::CPP_PORT).map_err(asset)?;
        let star =
            gltf::load_embedded_model(STAR_SPHERE, gltf::LoadOptions::CPP_PORT).map_err(asset)?;
        let interface =
            d.create_pipeline_interface(
                &PipelineInterfaceDescriptor::new(vec![])
                    .with_immediate_range(ImmediateRange::new(0, PUSH_BYTES, ShaderStages::VERTEX)),
            )?;
        let vi = input();
        let phong = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &artifact(ShaderStage::Vertex, "vs_main", PHONG, [0x81; 32], false),
                    )
                    .await?,
                    interface.clone(),
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &artifact(ShaderStage::Fragment, "fs_main", PHONG, [0x82; 32], false),
                    )
                    .await?,
                )
                .with_vertex_input(vi.clone())
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Back),
                )
                .with_depth_stencil(
                    DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                        DepthState::new(CompareFunction::LessEqual).with_write_enabled(true),
                    ),
                )
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let stars = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &artifact(ShaderStage::Vertex, "vs_main", STARS, [0x83; 32], true),
                    )
                    .await?,
                    interface,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &artifact(ShaderStage::Fragment, "fs_main", STARS, [0x84; 32], true),
                    )
                    .await?,
                )
                .with_vertex_input(vi)
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Front),
                )
                .with_depth_stencil(
                    DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                        DepthState::new(CompareFunction::LessEqual).with_write_enabled(false),
                    ),
                )
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let ufo_cull_radius = model_radius(&ufo.vertices) * 0.5;
        let (uv, ui) = buffers(d, "ufo", ufo.vertices.len(), ufo.indices.len())?;
        let (sv, si) = buffers(d, "star sphere", star.vertices.len(), star.indices.len())?;
        let uvl = ufo.vertices.len() as u64;
        let uil = ufo.indices.len() as u64;
        let svl = star.vertices.len() as u64;
        let sil = star.indices.len() as u64;
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        for (b, x) in [
            (&uv, ufo.vertices),
            (&ui, ufo.indices),
            (&sv, star.vertices),
            (&si, star.indices),
        ] {
            r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
                b.clone(),
                0,
                x,
            ))?)?
        }
        let mut p = SubmissionPlanBuilder::new(d);
        p.add_batch(lane, vec![r.finish()?])?;
        let receipt = d.submit(p.build()?)?;
        let _ = d.wait_completion(receipt.completion()).await?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let each = OBJECTS / threads;
        let mut seed = 0x5eed_1234u32;
        let mut objects = Vec::with_capacity(threads);
        for _ in 0..threads {
            let mut v = Vec::with_capacity(each);
            for _ in 0..each {
                let theta = std::f32::consts::TAU * rnd(&mut seed);
                let phi = (1. - 2. * rnd(&mut seed)).acos();
                let rotation = rnd(&mut seed) * 360.;
                let delta = rnd(&mut seed);
                let dir = if rnd(&mut seed) < 0.5 { 1. } else { -1. };
                let speed = (2. + rnd(&mut seed) * 4.) * dir;
                let scale = 0.75 + rnd(&mut seed) * 0.5;
                v.push(Object {
                    pos: [phi.sin() * theta.cos() * 35., 0., phi.cos() * 35.],
                    rotation,
                    direction: dir,
                    speed,
                    scale,
                    delta,
                    color: [rnd(&mut seed), rnd(&mut seed), rnd(&mut seed)],
                })
            }
            objects.push(v)
        }
        Ok(Self {
            phong,
            stars,
            ufo_v: bind(uv, uvl),
            ufo_i: bind(ui, uil),
            star_v: bind(sv, svl),
            star_i: bind(si, sil),
            ufo_draws: ufo
                .primitives
                .iter()
                .map(|x| (x.first_index, x.index_count))
                .collect(),
            ufo_cull_radius,
            star_draws: star
                .primitives
                .iter()
                .map(|x| (x.first_index, x.index_count))
                .collect(),
            depth: depth(d, e)?,
            extent: e,
            objects,
            delta: 0.,
            next: 0,
            done: [None, None],
        })
    }
    fn resize(&mut self, d: &Device, e: Extent3d) -> RhiResult<()> {
        self.depth = depth(d, e)?;
        self.extent = e;
        Ok(())
    }
    async fn render(
        &mut self,
        d: &Device,
        lane: SubmissionLaneId,
        frame: AcquiredFrame,
    ) -> RhiResult<()> {
        let slot = self.next;
        if let Some(x) = self.done[slot].take() {
            let _ = d.wait_completion(x).await?;
        }
        let pv = projection(self.extent);
        let view = translate(0., 0., -32.5);
        let vp = mul(pv, view);
        let frame_attachment = frame.attachment();
        let desc = RasterScopeDescriptor::new()
            .with_color(
                ShaderLocation::new(0),
                ColorAttachment {
                    view: ColorAttachmentView::Frame(frame_attachment.clone()),
                    load: LoadOp::Clear(ColorClearValue::Float([0., 0., 0., 1.])),
                    store: StoreOp::Store,
                    resolve: None,
                    depth_slice: None,
                },
            )
            .with_depth_stencil(DepthStencilAttachment {
                view: self.depth.clone(),
                depth: Some(DepthAttachmentMode::ReadWrite {
                    load: LoadOp::Clear(1.),
                    store: StoreOp::Discard,
                }),
                stencil: None,
            });
        // `finish_secondary_raster` deliberately rejects pass effects: its
        // attachment declaration only establishes the inheritance signature.
        let inherited = RasterScopeDescriptor::new()
            .with_color(
                ShaderLocation::new(0),
                ColorAttachment {
                    view: ColorAttachmentView::Frame(frame_attachment),
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                    resolve: None,
                    depth_slice: None,
                },
            )
            .with_depth_stencil(DepthStencilAttachment {
                view: self.depth.clone(),
                depth: Some(DepthAttachmentMode::ReadWrite {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil: None,
            });
        let extent = self.extent;
        let mut packets = Vec::new();
        std::thread::scope(|scope| -> RhiResult<()> {
            let mut jobs = Vec::new();
            for row in &mut self.objects {
                let d = d.clone();
                let desc = inherited.clone();
                let pipeline = self.phong.clone();
                let vb = self.ufo_v.clone();
                let ib = self.ufo_i.clone();
                let draws = self.ufo_draws.clone();
                let radius = self.ufo_cull_radius;
                let dt = self.delta;
                jobs.push(scope.spawn(move || -> RhiResult<Vec<_>> {
                    let mut out = Vec::new();
                    for o in row {
                        let visible = visible(vp, o.pos, radius);
                        if !visible {
                            continue;
                        }
                        o.rotation = (o.rotation + 2.5 * o.speed * dt) % 360.;
                        o.delta = (o.delta + 0.15 * dt) % 1.;
                        o.pos[1] = (o.delta * std::f32::consts::TAU).sin() * 2.5;
                        let m = model(o);
                        let push = push(mul(vp, m), o.color);
                        let mut r =
                            d.create_secondary_raster_recorder(&RecorderDescriptor::new())?;
                        let mut s = r.begin_raster(&desc)?;
                        s.set_viewport(Viewport::new(
                            0.,
                            0.,
                            extent.width as f32,
                            extent.height as f32,
                            0.,
                            1.,
                        ))?;
                        s.set_scissor(Rect::new(0, 0, extent.width, extent.height))?;
                        s.set_pipeline(&pipeline)?;
                        s.set_immediates(0, &push)?;
                        s.set_vertex_buffer(0, &vb)?;
                        s.set_index_buffer(&ib, IndexFormat::Uint32)?;
                        for (first, count) in &draws {
                            s.draw_indexed(*first..*first + *count, 0, 0..1)?
                        }
                        s.end()?;
                        out.push(r.finish_secondary_raster()?)
                    }
                    Ok(out)
                }));
            }
            for j in jobs {
                packets.extend(j.join().map_err(|_| {
                    RhiError::new(RhiErrorKind::BackendFailure, "13 worker panicked")
                })??)
            }
            Ok(())
        })?;
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        let mut s = r.begin_raster(&desc)?;
        s.set_viewport(Viewport::new(
            0.,
            0.,
            self.extent.width as f32,
            self.extent.height as f32,
            0.,
            1.,
        ))?;
        s.set_scissor(Rect::new(0, 0, self.extent.width, self.extent.height))?;
        s.set_pipeline(&self.stars)?;
        s.set_immediates(0, &push(star_mvp(mul(pv, view)), [0., 0., 0.]))?;
        s.set_vertex_buffer(0, &self.star_v)?;
        s.set_index_buffer(&self.star_i, IndexFormat::Uint32)?;
        for (first, count) in &self.star_draws {
            s.draw_indexed(*first..*first + *count, 0, 0..1)?
        }
        for packet in packets {
            s.execute_secondary(packet)?
        }
        s.end()?;
        let mut p = SubmissionPlanBuilder::new(d);
        let point = p.add_batch(lane, vec![r.finish()?])?;
        p.present_after(frame, point)?;
        let receipt = d.submit(p.build()?)?;
        self.done[slot] = Some(receipt.completion());
        let _ = d.wait_present(receipt.presents()[0].id()).await?;
        self.next = (slot + 1) % FRAMES;
        Ok(())
    }
}
fn asset(e: impl std::fmt::Display) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, e.to_string())
}
fn rnd(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    (*s >> 8) as f32 / (1u32 << 24) as f32
}
fn buffers(d: &Device, n: &str, v: usize, i: usize) -> RhiResult<(Buffer, Buffer)> {
    Ok((
        d.create_buffer(
            &BufferDescriptor::new(v as u64, BufferUsage::VERTEX.union(BufferUsage::COPY_DST))
                .with_label(format!("13 {n} vertices")),
        )?,
        d.create_buffer(
            &BufferDescriptor::new(i as u64, BufferUsage::INDEX.union(BufferUsage::COPY_DST))
                .with_label(format!("13 {n} indices")),
        )?,
    ))
}
fn bind(b: Buffer, n: u64) -> BufferBinding {
    BufferBinding::new(b, BufferRange::new(0, n))
}
/// Mirrors `vkglTF::Model::dimensions.radius`: half the diagonal from the
/// model-wide minimum position to its maximum position.
fn model_radius(vertices: &[u8]) -> f32 {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for vertex in vertices.chunks_exact(gltf::VKGLTF_VERTEX_STRIDE) {
        for axis in 0..3 {
            let start = axis * 4;
            let value = f32::from_le_bytes(vertex[start..start + 4].try_into().expect("position"));
            lo[axis] = lo[axis].min(value);
            hi[axis] = hi[axis].max(value);
        }
    }
    ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt() * 0.5
}
fn depth(d: &Device, e: Extent3d) -> RhiResult<TextureView> {
    let t = d.create_texture(&TextureDescriptor::new_2d(
        e.width,
        e.height,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ))?;
    d.create_texture_view(
        &t,
        &TextureViewDescriptor::whole(&t, TextureViewDimension::D2)?,
    )
}
fn input() -> VertexInputState {
    VertexInputState::new().with_buffer(
        VertexBufferLayout::new(gltf::VKGLTF_VERTEX_STRIDE as u64, VertexStepMode::Vertex)
            .with_attribute(VertexAttribute::new(
                ShaderLocation::new(0),
                VertexFormat::Float32x3,
                0,
            ))
            .with_attribute(VertexAttribute::new(
                ShaderLocation::new(1),
                VertexFormat::Float32x3,
                12,
            ))
            .with_attribute(VertexAttribute::new(
                ShaderLocation::new(2),
                VertexFormat::Float32x3,
                32,
            )),
    )
}
fn artifact(
    stage: ShaderStage,
    entry: &'static str,
    source: &'static str,
    hash: [u8; 32],
    star: bool,
) -> ShaderArtifact {
    let i = ShaderInterpolation {
        mode: InterpolationMode::Perspective,
        sampling: InterpolationSampling::Center,
    };
    let io = |n, c, x| ShaderLocationInterface {
        location: ShaderLocation::new(n),
        numeric_type: ShaderNumericType::Float32,
        components: c,
        interpolation: x,
    };
    let interface = if stage == ShaderStage::Vertex {
        let x = ShaderInterface::new()
            .with_immediate_requirement(ShaderImmediateRequirement {
                offset: 0,
                size: PUSH_BYTES,
            })
            .with_input(io(0, 3, None));
        let x = if star {
            x
        } else {
            x.with_input(io(1, 3, None)).with_input(io(2, 3, None))
        };
        if star {
            x.with_output(io(0, 3, Some(i))).with_writes_position(true)
        } else {
            x.with_output(io(0, 3, Some(i)))
                .with_output(io(1, 3, Some(i)))
                .with_output(io(3, 3, Some(i)))
                .with_output(io(4, 3, Some(i)))
                .with_writes_position(true)
        }
    } else if star {
        ShaderInterface::new()
            .with_input(io(0, 3, Some(i)))
            .with_output(io(0, 4, None))
    } else {
        ShaderInterface::new()
            .with_input(io(0, 3, Some(i)))
            .with_input(io(1, 3, Some(i)))
            .with_input(io(3, 3, Some(i)))
            .with_input(io(4, 3, Some(i)))
            .with_output(io(0, 4, None))
    };
    ShaderArtifact::new(
        stage,
        entry,
        ShaderCode::Wgsl(Arc::from(source)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        ArtifactHash(hash),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
fn projection(e: Extent3d) -> [f32; 16] {
    let f = 1. / 30f32.to_radians().tan();
    let a = e.width.max(1) as f32 / e.height.max(1) as f32;
    let (n, z) = (0.1, 256.);
    [
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
        z / (n - z),
        -1.,
        0.,
        0.,
        z * n / (n - z),
        0.,
    ]
}
fn translate(x: f32, y: f32, z: f32) -> [f32; 16] {
    [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., x, y, z, 1.]
}
fn model(o: &Object) -> [f32; 16] {
    let a = o.delta * std::f32::consts::TAU;
    // glm applies these in this exact order in the C++ renderer.
    mul(
        translate(o.pos[0], o.pos[1], o.pos[2]),
        mul(
            rotate_x(-a.sin() * 0.25 * o.direction),
            mul(
                rotate_y(o.rotation.to_radians() * o.direction),
                mul(rotate_y(a * o.direction), uniform_scale(o.scale)),
            ),
        ),
    )
}
fn uniform_scale(s: f32) -> [f32; 16] {
    [s, 0., 0., 0., 0., s, 0., 0., 0., 0., s, 0., 0., 0., 0., 1.]
}
fn rotate_x(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [1., 0., 0., 0., 0., c, s, 0., 0., -s, c, 0., 0., 0., 0., 1.]
}
fn rotate_y(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.]
}
fn star_mvp(mut m: [f32; 16]) -> [f32; 16] {
    // C++ clears the view translation, then glm::scale(mvp, vec3(2)).
    m[12] = 0.;
    m[13] = 0.;
    m[14] = 0.;
    m[15] = 1.;
    for column in 0..3 {
        for row in 0..4 {
            m[column * 4 + row] *= 2.;
        }
    }
    m
}
fn mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut o = [0.; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum()
        }
    }
    o
}
fn push(m: [f32; 16], c: [f32; 3]) -> Vec<u8> {
    m.into_iter()
        .chain(c)
        .chain([0.])
        .flat_map(f32::to_le_bytes)
        .collect()
}
fn visible(vp: [f32; 16], p: [f32; 3], radius: f32) -> bool {
    let q = [
        vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12],
        vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13],
        vp[2] * p[0] + vp[6] * p[1] + vp[10] * p[2] + vp[14],
        vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15],
    ];
    q[3] > 0.
        && q[0].abs() <= q[3] + radius
        && q[1].abs() <= q[3] + radius
        && q[2] >= -radius
        && q[2] <= q[3] + radius
}
