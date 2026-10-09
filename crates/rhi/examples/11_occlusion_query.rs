//! Faithful port of SaschaWillems/Vulkan `examples/occlusionquery`.
//!
//! It draws the scaled blue plane and brackets the hidden teapot and sphere in
//! two occlusion queries.  It then clears the same pass's color/depth targets,
//! draws the objects using the preceding query results, and overlays the plane.

pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use std::sync::Arc;

use fluxel_rhi::api::{
    binding::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayout,
        BindGroupLayoutDescriptor, BindingCount, BindingKind, BindingResource, BindingSlot,
        BindingSlotId,
    },
    command::{
        ColorAttachment, ColorAttachmentView, ColorClearValue, DepthAttachmentMode,
        DepthStencilAttachment, IndexFormat, LoadOp, RasterAttachmentClear, RasterScopeDescriptor,
        RecorderDescriptor, Rect, StoreOp, Viewport,
    },
    error::{RhiError, RhiErrorKind, RhiResult},
    format::TextureFormat,
    identity::Label,
    pipeline::{
        BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, CullMode,
        DepthState, DepthStencilState, PipelineInterfaceDescriptor, PrimitiveState,
        PrimitiveTopology, RasterPipeline, RasterPipelineDescriptor, VertexAttribute,
        VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
    },
    platform::{Device, OptionalFeature},
    presentation::AcquiredFrame,
    query::{QuerySet, QuerySetDescriptor, QueryType},
    resource::{
        Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor, BufferUsage,
        CompareFunction, Extent3d, ReadbackRequest, ReadbackViewData, TextureDescriptor,
        TextureUsage, TextureView, TextureViewDescriptor, TextureViewDimension,
    },
    shader::{
        ArtifactHash, ArtifactProducerVersion, InterpolationMode, InterpolationSampling,
        ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderInterface, ShaderInterpolation,
        ShaderLocation, ShaderLocationInterface, ShaderNumericType, ShaderRequirements,
        ShaderResourceRequirement, ShaderStage, ShaderStages,
    },
    submission::{CompletionPoint, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder},
};

const FRAMES: usize = 2;
const UBO_BYTES: u64 = 240; // three mat4s, color, light position, visible + padding
const PLANE: &[u8] = include_bytes!("assets/models/plane_z.gltf");
const TEAPOT: &[u8] = include_bytes!("assets/models/teapot.gltf");
const SPHERE: &[u8] = include_bytes!("assets/models/sphere.gltf");

// Direct WGSL translations of shaders/glsl/occlusionquery/{mesh,simple,occluder}.*.
const MESH: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, model:mat4x4<f32>, color:vec4<f32>, light:vec4<f32>, visible:vec4<f32>, };
@group(0) @binding(0) var<uniform> u:U;
struct I { @location(0) p:vec3<f32>, @location(1) n:vec3<f32>, @location(2) c:vec3<f32>, };
struct O { @builtin(position) p:vec4<f32>, @location(0) n:vec3<f32>, @location(1) c:vec3<f32>, @location(2) visible:f32, @location(3) view:vec3<f32>, @location(4) light:vec3<f32>, };
@vertex fn vs_main(i:I)->O { var o:O; o.n=(mat3x3<f32>(u.model[0].xyz,u.model[1].xyz,u.model[2].xyz)*i.n);o.c=i.c*u.color.rgb;o.visible=u.visible.x;o.p=u.projection*u.view*u.model*vec4<f32>(i.p,1.0);let pos=(u.model*vec4<f32>(i.p,1.0)).xyz;o.light=u.light.xyz-pos;o.view=-pos;return o; }
@fragment fn fs_main(i:O)->@location(0) vec4<f32> { if(i.visible>0.0){let n=normalize(i.n);let l=normalize(i.light);let v=normalize(i.view);let r=reflect(-l,n);return vec4<f32>(max(dot(n,l),0.25)*i.c+pow(max(dot(r,v),0.0),8.0)*vec3<f32>(0.75),1.0);}return vec4<f32>(vec3<f32>(0.1),1.0); }
"#;
const SIMPLE: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, model:mat4x4<f32>, color:vec4<f32>, light:vec4<f32>, visible:vec4<f32>, };
@group(0) @binding(0) var<uniform> u:U;
@vertex fn vs_main(@location(0) p:vec3<f32>)->@builtin(position) vec4<f32>{return u.projection*u.view*u.model*vec4<f32>(p,1.0);}
@fragment fn fs_main()->@location(0) vec4<f32>{return vec4<f32>(1.0);}
"#;
const OCCLUDER: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, model:mat4x4<f32>, color:vec4<f32>, light:vec4<f32>, visible:vec4<f32>, };
@group(0) @binding(0) var<uniform> u:U;
struct O { @builtin(position) p:vec4<f32>, @location(0) c:vec3<f32>, };
@vertex fn vs_main(@location(0) p:vec3<f32>,@location(1) _n:vec3<f32>,@location(2) c:vec3<f32>)->O { var o:O;o.c=c*u.color.rgb;o.p=u.projection*u.view*u.model*vec4<f32>(p,1.0);return o; }
@fragment fn fs_main(i:O)->@location(0) vec4<f32>{return vec4<f32>(i.c,0.5);}
"#;

fn main() {
    if let Err(error) = common::run_example("Occlusion queries", create_example()) {
        eprintln!("11_occlusion_query: {error}");
        std::process::exit(1);
    }
}

pub fn create_example() -> Example {
    Example {
        work: None,
        lane: None,
    }
}
pub struct Example {
    work: Option<Workload>,
    lane: Option<SubmissionLaneId>,
}

impl common::Example for Example {
    fn init(
        &mut self,
        c: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let d = c.device().clone();
        if !d
            .capabilities()
            .supports_feature(OptionalFeature::OcclusionQuery)
            || !d
                .capabilities()
                .supports_feature(OptionalFeature::QueryResolve)
        {
            return Err(std::io::Error::other(
                "11_occlusion_query requires OcclusionQuery and QueryResolve",
            )
            .into());
        }
        let lane = d
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|x| {
                x.domains()
                    .contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY))
            })
            .map(|x| x.id())
            .ok_or_else(|| std::io::Error::other("11_occlusion_query requires COPY|RASTER lane"))?;
        let e = c.extent();
        self.work = Some(common::block_on(Workload::new(
            &d,
            c.presentation_mut().configuration().format(),
            Extent3d::d2(e.width, e.height),
            lane,
        ))?);
        self.lane = Some(lane);
        Ok(())
    }
    fn update(
        &mut self,
        _: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

struct Mesh {
    vertex: BufferBinding,
    index: BufferBinding,
    draws: Vec<(u32, u32)>,
}
struct Workload {
    solid: RasterPipeline,
    simple: RasterPipeline,
    occluder: RasterPipeline,
    groups: [[BindGroup; 3]; FRAMES],
    uniforms: [[Buffer; 3]; FRAMES],
    plane: Mesh,
    teapot: Mesh,
    sphere: Mesh,
    query: QuerySet,
    resolved: Buffer,
    depth: TextureView,
    extent: Extent3d,
    passed: [u64; 2],
    next: usize,
    done: [Option<CompletionPoint>; FRAMES],
}
impl Workload {
    async fn new(
        d: &Device,
        fmt: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let plane = gltf::load_embedded_model(PLANE, gltf::LoadOptions::CPP_PORT).map_err(asset)?;
        let teapot =
            gltf::load_embedded_model(TEAPOT, gltf::LoadOptions::CPP_PORT).map_err(asset)?;
        let sphere =
            gltf::load_embedded_model(SPHERE, gltf::LoadOptions::CPP_PORT).map_err(asset)?;
        let layout =
            d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::VERTEX,
                BindingKind::UniformBuffer {
                    min_size: UBO_BYTES,
                },
            )]))?;
        let interface =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        let ds = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(true));
        let solid = pipeline(d, interface.clone(), MESH, fmt, ds.clone(), None, [11; 32]).await?;
        let simple = pipeline(
            d,
            interface.clone(),
            SIMPLE,
            fmt,
            ds.clone(),
            None,
            [12; 32],
        )
        .await?;
        let blend = BlendState::new(
            BlendComponent::new(
                BlendFactor::SrcColor,
                BlendFactor::OneMinusSrcColor,
                BlendOperation::Add,
            ),
            BlendComponent::new(BlendFactor::One, BlendFactor::Zero, BlendOperation::Add),
        );
        let occluder = pipeline(d, interface, OCCLUDER, fmt, ds, Some(blend), [13; 32]).await?;
        let plane_buffers = mesh_buffers(d, "plane", &plane)?;
        let teapot_buffers = mesh_buffers(d, "teapot", &teapot)?;
        let sphere_buffers = mesh_buffers(d, "sphere", &sphere)?;
        let plane_mesh = finish_mesh((plane_buffers.0.clone(), plane_buffers.1.clone()), &plane);
        let teapot_mesh = finish_mesh(
            (teapot_buffers.0.clone(), teapot_buffers.1.clone()),
            &teapot,
        );
        let sphere_mesh = finish_mesh(
            (sphere_buffers.0.clone(), sphere_buffers.1.clone()),
            &sphere,
        );
        let uniforms = [
            [
                ubo(d, "11 plane 0")?,
                ubo(d, "11 teapot 0")?,
                ubo(d, "11 sphere 0")?,
            ],
            [
                ubo(d, "11 plane 1")?,
                ubo(d, "11 teapot 1")?,
                ubo(d, "11 sphere 1")?,
            ],
        ];
        let groups = [
            [
                group(d, &layout, &uniforms[0][0])?,
                group(d, &layout, &uniforms[0][1])?,
                group(d, &layout, &uniforms[0][2])?,
            ],
            [
                group(d, &layout, &uniforms[1][0])?,
                group(d, &layout, &uniforms[1][1])?,
                group(d, &layout, &uniforms[1][2])?,
            ],
        ];
        let query = d.create_query_set(
            &QuerySetDescriptor::new(QueryType::Occlusion, 2).with_label("11 occlusion queries"),
        )?;
        let resolved = d.create_buffer(
            &BufferDescriptor::new(16, BufferUsage::QUERY_RESOLVE.union(BufferUsage::COPY_SRC))
                .with_label("11 query results"),
        )?;
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        for (b, data) in [
            (&plane_buffers.0, plane.vertices),
            (&plane_buffers.1, plane.indices),
            (&teapot_buffers.0, teapot.vertices),
            (&teapot_buffers.1, teapot.indices),
            (&sphere_buffers.0, sphere.vertices),
            (&sphere_buffers.1, sphere.indices),
        ] {
            r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
                b.clone(),
                0,
                data,
            ))?)?;
        }
        let mut p = SubmissionPlanBuilder::new(d);
        p.add_batch(lane, vec![r.finish()?])?;
        let receipt = d.submit(p.build()?)?;
        let _ = d.wait_completion(receipt.completion()).await?;
        Ok(Self {
            solid,
            simple,
            occluder,
            groups,
            uniforms,
            plane: plane_mesh,
            teapot: teapot_mesh,
            sphere: sphere_mesh,
            query,
            resolved,
            depth: depth(d, extent)?,
            extent,
            passed: [1, 1],
            next: 0,
            done: [None, None],
        })
    }
    fn resize(&mut self, d: &Device, e: Extent3d) -> RhiResult<()> {
        self.extent = e;
        self.depth = depth(d, e)?;
        Ok(())
    }
    async fn render(
        &mut self,
        d: &Device,
        lane: SubmissionLaneId,
        f: AcquiredFrame,
    ) -> RhiResult<()> {
        let n = self.next;
        if let Some(x) = self.done[n].take() {
            let _ = d.wait_completion(x).await?;
        }
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        for (b, bytes) in [
            (
                &self.uniforms[n][0],
                uniform(self.extent, 6., [0., 0., 1., 0.5], 1.),
            ),
            (
                &self.uniforms[n][1],
                uniform(
                    self.extent,
                    1.,
                    [1., 0., 0., 1.],
                    if self.passed[0] > 0 { 1. } else { 0. },
                ),
            ),
            (
                &self.uniforms[n][2],
                uniform(
                    self.extent,
                    1.,
                    [0., 1., 0., 1.],
                    if self.passed[1] > 0 { 1. } else { 0. },
                ),
            ),
        ] {
            r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
                b.clone(),
                0,
                bytes,
            ))?)?;
        }
        {
            let mut q = r.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_label("11 occlusion and visible passes")
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(f.attachment()),
                            load: LoadOp::Clear(ColorClearValue::Float([0.025, 0.025, 0.025, 1.])),
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
                    }),
            )?;
            view(&mut q, self.extent)?;
            q.set_pipeline(&self.simple)?;
            draw(&mut q, &self.plane, &self.groups[n][0])?;
            q.begin_query(&self.query, 0)?;
            draw(&mut q, &self.teapot, &self.groups[n][1])?;
            q.end_query(&self.query, 0)?;
            q.begin_query(&self.query, 1)?;
            draw(&mut q, &self.sphere, &self.groups[n][2])?;
            q.end_query(&self.query, 1)?;
            let mut clear = RasterAttachmentClear::new(
                Rect::new(0, 0, self.extent.width, self.extent.height),
                0,
                1,
            );
            clear
                .colors
                .push((0, ColorClearValue::Float([0.025, 0.025, 0.025, 1.])));
            clear.depth = Some(1.);
            q.clear_attachments(&clear)?;
            q.set_pipeline(&self.solid)?;
            draw(&mut q, &self.teapot, &self.groups[n][1])?;
            draw(&mut q, &self.sphere, &self.groups[n][2])?;
            q.set_pipeline(&self.occluder)?;
            draw(&mut q, &self.plane, &self.groups[n][0])?;
            q.end()?;
        }
        r.resolve_query_set(&self.query, 0, 2, &self.resolved, 0)?;
        let ticket = r.encode_readback(ReadbackRequest::Buffer {
            label: Label(Some("11 occlusion query results".into())),
            src: self.resolved.clone(),
            range: BufferRange::new(0, 16),
        })?;
        let mut p = SubmissionPlanBuilder::new(d);
        let point = p.add_batch(lane, vec![r.finish()?])?;
        p.present_after(f, point)?;
        let receipt = d.submit(p.build()?)?;
        self.done[n] = Some(receipt.completion());
        let _ = d.wait_completion(receipt.completion()).await?;
        let read = ticket.try_read()?.ok_or_else(|| {
            RhiError::new(
                RhiErrorKind::BackendFailure,
                "occlusion-query readback did not complete",
            )
        })?;
        let ReadbackViewData::Buffer { bytes } = read.data() else {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "occlusion-query resolve returned texture data",
            ));
        };
        if bytes.len() != 16 {
            return Err(RhiError::new(
                RhiErrorKind::BackendFailure,
                "occlusion-query result length is not two u64 values",
            ));
        };
        self.passed = [
            u64::from_le_bytes(bytes[0..8].try_into().expect("u64")),
            u64::from_le_bytes(bytes[8..16].try_into().expect("u64")),
        ];
        let _ = d.wait_present(receipt.presents()[0].id()).await?;
        self.next = (n + 1) % FRAMES;
        Ok(())
    }
}

fn mesh_buffers(d: &Device, n: &str, m: &gltf::EmbeddedModel) -> RhiResult<(Buffer, Buffer)> {
    Ok((
        d.create_buffer(
            &BufferDescriptor::new(
                m.vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label(format!("11 {n} vertices")),
        )?,
        d.create_buffer(
            &BufferDescriptor::new(
                m.indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label(format!("11 {n} indices")),
        )?,
    ))
}
fn finish_mesh((v, i): (Buffer, Buffer), m: &gltf::EmbeddedModel) -> Mesh {
    Mesh {
        vertex: BufferBinding::new(v, BufferRange::new(0, m.vertices.len() as u64)),
        index: BufferBinding::new(i, BufferRange::new(0, m.indices.len() as u64)),
        draws: m
            .primitives
            .iter()
            .map(|p| (p.first_index, p.index_count))
            .collect(),
    }
}
fn ubo(d: &Device, label: &str) -> RhiResult<Buffer> {
    d.create_buffer(
        &BufferDescriptor::new(UBO_BYTES, BufferUsage::UNIFORM.union(BufferUsage::COPY_DST))
            .with_label(label),
    )
}
fn group(d: &Device, l: &BindGroupLayout, b: &Buffer) -> RhiResult<BindGroup> {
    d.create_bind_group(
        &BindGroupDescriptor::new(l.clone()).with_entry(BindGroupEntry::new(
            BindingSlotId::new(0),
            BindingResource::Buffer(BufferBinding::new(
                b.clone(),
                BufferRange::new(0, UBO_BYTES),
            )),
        )),
    )
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
fn view(q: &mut fluxel_rhi::api::command::RasterScope<'_>, e: Extent3d) -> RhiResult<()> {
    q.set_viewport(Viewport::new(
        0.,
        0.,
        e.width as f32,
        e.height as f32,
        0.,
        1.,
    ))?;
    q.set_scissor(Rect::new(0, 0, e.width, e.height))
}
fn draw(
    q: &mut fluxel_rhi::api::command::RasterScope<'_>,
    m: &Mesh,
    g: &BindGroup,
) -> RhiResult<()> {
    q.set_bind_group(BindGroupIndex::new(0), g, &[])?;
    q.set_vertex_buffer(0, &m.vertex)?;
    q.set_index_buffer(&m.index, IndexFormat::Uint32)?;
    for &(first, count) in &m.draws {
        q.draw_indexed(first..first + count, 0, 0..1)?;
    }
    Ok(())
}

async fn pipeline(
    d: &Device,
    i: fluxel_rhi::api::pipeline::PipelineInterface,
    code: &'static str,
    fmt: TextureFormat,
    ds: DepthStencilState,
    blend: Option<BlendState>,
    hash: [u8; 32],
) -> RhiResult<RasterPipeline> {
    let color = if let Some(x) = blend {
        ColorTargetState::new(fmt).with_blend(x)
    } else {
        ColorTargetState::new(fmt)
    };
    d.create_raster_pipeline(
        &RasterPipelineDescriptor::new(
            common::shader::create_shader(d, &artifact(ShaderStage::Vertex, code, hash)).await?,
            i,
        )
        .with_fragment(
            common::shader::create_shader(
                d,
                &artifact(ShaderStage::Fragment, code, hash.map(|x| x + 10)),
            )
            .await?,
        )
        .with_vertex_input(input())
        .with_primitive(
            PrimitiveState::new(PrimitiveTopology::TriangleList).with_cull_mode(CullMode::None),
        )
        .with_depth_stencil(ds)
        .with_color_target(ShaderLocation::new(0), color),
    )
    .await
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

fn uniform(e: Extent3d, scale: f32, color: [f32; 4], visible: f32) -> Vec<u8> {
    let f = 1. / 30f32.to_radians().tan();
    let a = e.width.max(1) as f32 / e.height.max(1) as f32;
    let near = 1.;
    let far = 256.;
    let p = [
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
        far / (near - far),
        -1.,
        0.,
        0.,
        far * near / (near - far),
        0.,
    ];
    let (sx, cx) = (0f32).to_radians().sin_cos();
    let (sy, cy) = (-123.75f32).to_radians().sin_cos();
    let rx = [
        1., 0., 0., 0., 0., cx, sx, 0., 0., -sx, cx, 0., 0., 0., 0., 1.,
    ];
    let ry = [
        cy, 0., -sy, 0., 0., 1., 0., 0., sy, 0., cy, 0., 0., 0., 0., 1.,
    ];
    let v = mm(mm(tr([0., 0., -7.5]), rx), ry);
    let m = if scale == 6. {
        sc([6., 6., 6.])
    } else if color[0] > 0. {
        tr([0., 0., -3.])
    } else {
        tr([0., 0., 3.])
    };
    p.into_iter()
        .chain(v)
        .chain(m)
        .chain(color)
        .chain([10., -10., 10., 1.])
        .chain([visible, 0., 0., 0.])
        .flat_map(f32::to_le_bytes)
        .collect()
}
fn tr(x: [f32; 3]) -> [f32; 16] {
    [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., x[0], x[1], x[2], 1.,
    ]
}
fn sc(x: [f32; 3]) -> [f32; 16] {
    [
        x[0], 0., 0., 0., 0., x[1], 0., 0., 0., 0., x[2], 0., 0., 0., 0., 1.,
    ]
}
fn mm(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut o = [0.; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    o
}
fn asset(e: impl ToString) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, e.to_string())
}
fn artifact(stage: ShaderStage, code: &'static str, hash: [u8; 32]) -> ShaderArtifact {
    let interp = Some(ShaderInterpolation {
        mode: InterpolationMode::Perspective,
        sampling: InterpolationSampling::Center,
    });
    let io = |n, c, i| ShaderLocationInterface {
        location: ShaderLocation::new(n),
        numeric_type: ShaderNumericType::Float32,
        components: c,
        interpolation: i,
    };
    let mut iface = ShaderInterface::new();
    if stage == ShaderStage::Vertex {
        iface = iface
            .with_resource(ShaderResourceRequirement {
                group: BindGroupIndex::new(0),
                slot: BindingSlotId::new(0),
                kind: BindingKind::UniformBuffer {
                    min_size: UBO_BYTES,
                },
                count: BindingCount::One,
            })
            .with_input(io(0, 3, None))
            .with_writes_position(true);
        if code != SIMPLE {
            iface = iface.with_input(io(1, 3, None)).with_input(io(2, 3, None));
            if code == MESH {
                iface = iface
                    .with_output(io(0, 3, interp))
                    .with_output(io(1, 3, interp))
                    .with_output(io(2, 1, interp))
                    .with_output(io(3, 3, interp))
                    .with_output(io(4, 3, interp));
            } else {
                iface = iface.with_output(io(0, 3, interp));
            }
        }
    } else if code == MESH {
        iface = iface
            .with_input(io(0, 3, interp))
            .with_input(io(1, 3, interp))
            .with_input(io(2, 1, interp))
            .with_input(io(3, 3, interp))
            .with_input(io(4, 3, interp))
            .with_output(io(0, 4, None));
    } else if code == OCCLUDER {
        iface = iface
            .with_input(io(0, 3, interp))
            .with_output(io(0, 4, None));
    } else {
        iface = iface.with_output(io(0, 4, None));
    }
    ShaderArtifact::new(
        stage,
        if stage == ShaderStage::Vertex {
            "vs_main"
        } else {
            "fs_main"
        },
        ShaderCode::Wgsl(Arc::from(code)),
        ShaderAbiVersion { major: 1, minor: 0 },
        iface,
        ShaderRequirements::new(),
        ArtifactHash(hash),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
