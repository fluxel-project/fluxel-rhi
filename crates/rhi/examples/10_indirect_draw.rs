//! Port of SaschaWillems/Vulkan `examples/indirectdraw/indirectdraw.cpp`.
//! The original five assets are retained.  On desktop it builds 2,048 records
//! per plant mesh and submits the C++ `VkDrawIndexedIndirectCommand` layout.

pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use fluxel_rhi::api::{
    binding::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayout,
        BindGroupLayoutDescriptor, BindingCount, BindingKind, BindingResource, BindingSlot,
        BindingSlotId, SamplerKind, TextureSampleType,
    },
    command::{
        ColorAttachment, ColorAttachmentView, ColorClearValue, DepthAttachmentMode,
        DepthStencilAttachment, IndexFormat, LoadOp, RasterScopeDescriptor, RecorderDescriptor,
        Rect, StoreOp, Viewport,
    },
    error::{RhiError, RhiErrorKind, RhiResult},
    format::TextureFormat,
    pipeline::{
        ColorTargetState, CullMode, DepthState, DepthStencilState, PipelineInterfaceDescriptor,
        PrimitiveState, PrimitiveTopology, RasterPipeline, RasterPipelineDescriptor,
        VertexAttribute, VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
    },
    platform::{Device, LimitKey, OptionalFeature},
    presentation::AcquiredFrame,
    resource::{
        AddressMode, Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor,
        BufferUsage, CompareFunction, Extent3d, FilterMode, HostTexelLayout, Origin3d, Sampler,
        SamplerDescriptor, Texture, TextureAspect, TextureDescriptor, TextureSubresourceLayers,
        TextureUploadDescriptor, TextureUsage, TextureView, TextureViewDescriptor,
        TextureViewDimension,
    },
    shader::{
        ArtifactHash, ArtifactProducerVersion, InterpolationMode, InterpolationSampling,
        ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderInterface, ShaderInterpolation,
        ShaderLocation, ShaderLocationInterface, ShaderNumericType, ShaderRequirements,
        ShaderResourceRequirement, ShaderStage, ShaderStages,
    },
    submission::{CompletionPoint, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder},
};
use std::{
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const FRAMES: usize = 2;
const INSTANCES: u32 = 2048;
const RADIUS: f32 = 25.;
const UBO: u64 = 128;
const PLANTS: &[u8] = include_bytes!("assets/models/plants.gltf");
const GROUND: &[u8] = include_bytes!("assets/models/plane_circle.gltf");
const SKY: &[u8] = include_bytes!("assets/models/sphere.gltf");
const PLANT_TEX: &[u8] = include_bytes!("assets/textures/texturearray_plants_rgba.ktx");
const GROUND_TEX: &[u8] = include_bytes!("assets/textures/ground_dry_rgba.ktx");

// These are direct WGSL translations of indirectdraw.{vert,frag}, ground and
// skysphere.  Texture/sampler are separate RHI bindings for the Vulkan combined sampler.
const PLANT_SHADER: &str = r#"struct U{p:mat4x4<f32>,v:mat4x4<f32>,};@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(1)var t:texture_2d_array<f32>;@group(0)@binding(2)var s:sampler;struct I{@location(0)p:vec3<f32>,@location(1)n:vec3<f32>,@location(2)uv:vec2<f32>,@location(3)c:vec3<f32>,@location(4)ip:vec3<f32>,@location(5)rot:vec3<f32>,@location(6)scale:f32,@location(7)layer:i32,};struct O{@builtin(position)p:vec4<f32>,@location(0)n:vec3<f32>,@location(1)c:vec3<f32>,@location(2)uv:vec3<f32>,@location(3)v:vec3<f32>,@location(4)l:vec3<f32>,};@vertex fn vs_main(i:I)->O{var o:O;let sx=sin(i.rot.x);let cx=cos(i.rot.x);let sy=sin(i.rot.y);let cy=cos(i.rot.y);let sz=sin(i.rot.z);let cz=cos(i.rot.z);let mx=mat4x4<f32>(vec4<f32>(cx,sx,0.,0.),vec4<f32>(-sx,cx,0.,0.),vec4<f32>(0.,0.,1.,0.),vec4<f32>(0.,0.,0.,1.));let my=mat4x4<f32>(vec4<f32>(cy,0.,sy,0.),vec4<f32>(0.,1.,0.,0.),vec4<f32>(-sy,0.,cy,0.),vec4<f32>(0.,0.,0.,1.));let mz=mat4x4<f32>(vec4<f32>(1.,0.,0.,0.),vec4<f32>(0.,cz,sz,0.),vec4<f32>(0.,-sz,cz,0.),vec4<f32>(0.,0.,0.,1.));let r=mz*my*mx;let q=transpose(r)*vec4<f32>(i.p*i.scale+i.ip,1.);o.c=i.c;o.uv=vec3<f32>(i.uv,f32(i.layer));o.n=transpose(mat3x3<f32>(r[0].xyz,r[1].xyz,r[2].xyz))*i.n;o.p=u.p*u.v*q;o.l=vec3<f32>(0.,-5.,0.)-q.xyz;o.v=-q.xyz;return o;}@fragment fn fs_main(i:O)->@location(0)vec4<f32>{let x=textureSample(t,s,i.uv.xy,i32(i.uv.z));if(x.a<.5){discard;}let diffuse=max(dot(normalize(i.n),normalize(i.l)),0.)*i.c;return vec4<f32>((vec3<f32>(.65)+diffuse)*x.rgb,1.);}"#;
const SURFACE_SHADER: &str = r#"struct U{p:mat4x4<f32>,v:mat4x4<f32>,};@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(1)var t:texture_2d<f32>;@group(0)@binding(2)var s:sampler;struct I{@location(0)p:vec3<f32>,@location(2)uv:vec2<f32>,};struct O{@builtin(position)p:vec4<f32>,@location(0)uv:vec2<f32>,};@vertex fn vs_main(i:I)->O{var o:O;o.uv=i.uv*32.;o.p=u.p*u.v*vec4<f32>(i.p,1.);return o;}@fragment fn fs_main(i:O)->@location(0)vec4<f32>{return textureSample(t,s,i.uv);}"#;
const SKY_SHADER: &str = r#"struct U{p:mat4x4<f32>,v:mat4x4<f32>,};@group(0)@binding(0)var<uniform>u:U;struct I{@location(0)p:vec3<f32>,@location(2)uv:vec2<f32>,};struct O{@builtin(position)p:vec4<f32>,@location(0)uv:vec2<f32>,};@vertex fn vs_main(i:I)->O{var o:O;o.uv=i.uv;let rotation=mat4x4<f32>(vec4<f32>(u.v[0].xyz,0.),vec4<f32>(u.v[1].xyz,0.),vec4<f32>(u.v[2].xyz,0.),vec4<f32>(0.,0.,0.,1.));o.p=u.p*rotation*vec4<f32>(i.p,1.);return o;}@fragment fn fs_main(i:O)->@location(0)vec4<f32>{return mix(vec4<f32>(.93,.9,.81,1.),vec4<f32>(.35,.5,1.,1.),min(.5-(i.uv.y+.05),.5)/.15+.5);}"#;

fn main() {
    if let Err(e) = common::run_example("Indirect draw", create_example()) {
        eprintln!("10_indirect_draw: {e}");
        std::process::exit(1)
    }
}
pub fn create_example() -> Example {
    Example {
        w: None,
        lane: None,
        time: 0.,
        last: None,
    }
}
pub struct Example {
    w: Option<Workload>,
    lane: Option<SubmissionLaneId>,
    time: f32,
    last: Option<Instant>,
}
impl common::Example for Example {
    fn init(
        &mut self,
        c: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let d = c.device().clone();
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
            .ok_or_else(|| std::io::Error::other("no COPY|RASTER lane"))?;
        let e = c.extent();
        self.w = Some(common::block_on(Workload::new(
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
        let n = Instant::now();
        if let Some(x) = self.last {
            self.time += (n - x).as_secs_f32()
        }
        self.last = Some(n);
        Ok(())
    }
    fn render(
        &mut self,
        c: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let f = common::block_on(c.presentation_mut().acquire())?;
        common::block_on(
            self.w
                .as_mut()
                .ok_or_else(|| std::io::Error::other("not initialized"))?
                .render(
                    c.device(),
                    self.lane.ok_or_else(|| std::io::Error::other("no lane"))?,
                    f,
                    self.time,
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
        if let Some(x) = &mut self.w {
            x.e = Extent3d::d2(w, h);
            x.depth = depth(c.device(), x.e)?
        }
        Ok(())
    }
    fn device_lost(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.w = None;
        self.lane = None;
        Ok(())
    }
    fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.device_lost()
    }
}

// The upload and draw sequence follows the C++ order: sky, ground, then every
// indexed-indirect plant mesh.  The public multi-draw capability is optional.
struct Workload {
    plants: RasterPipeline,
    ground: RasterPipeline,
    sky: RasterPipeline,
    pg: [BindGroup; FRAMES],
    gg: [BindGroup; FRAMES],
    sg: [BindGroup; FRAMES],
    ubo: [Buffer; FRAMES],
    pv: BufferBinding,
    pi: BufferBinding,
    gv: BufferBinding,
    gi: BufferBinding,
    sv: BufferBinding,
    si: BufferBinding,
    instance: BufferBinding,
    indirect: Buffer,
    draws: u32,
    depth: TextureView,
    e: Extent3d,
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
        let pm = gltf::load_embedded_model(PLANTS, gltf::LoadOptions::CPP_PORT)
            .map_err(|x| asset(x.to_string()))?;
        let gm = gltf::load_embedded_model(GROUND, gltf::LoadOptions::CPP_PORT)
            .map_err(|x| asset(x.to_string()))?;
        let sm = gltf::load_embedded_model(SKY, gltf::LoadOptions::CPP_PORT)
            .map_err(|x| asset(x.to_string()))?;
        let pk = Ktx::parse(PLANT_TEX).map_err(asset)?;
        let gk = Ktx::parse(GROUND_TEX).map_err(asset)?;
        let pl = layout(d, TextureViewDimension::D2Array)?;
        let gl = layout(d, TextureViewDimension::D2)?;
        let sl = sky_layout(d)?;
        let pif =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![pl.clone()]))?;
        let gif =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![gl.clone()]))?;
        let sif =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![sl.clone()]))?;
        let write = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(true));
        let read = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(false));
        let plants = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Vertex,
                            "vs_main",
                            PLANT_SHADER,
                            plant_vs(),
                            [41; 32],
                        ),
                    )
                    .await?,
                    pif,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Fragment,
                            "fs_main",
                            PLANT_SHADER,
                            plant_fs(),
                            [42; 32],
                        ),
                    )
                    .await?,
                )
                .with_vertex_input(mesh_input().with_buffer(instance_input()))
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::None),
                )
                .with_depth_stencil(write)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let ground = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Vertex,
                            "vs_main",
                            SURFACE_SHADER,
                            surface_vs(),
                            [43; 32],
                        ),
                    )
                    .await?,
                    gif,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Fragment,
                            "fs_main",
                            SURFACE_SHADER,
                            surface_fs(),
                            [44; 32],
                        ),
                    )
                    .await?,
                )
                .with_vertex_input(mesh_input())
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Back),
                )
                .with_depth_stencil(write)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let sky = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Vertex,
                            "vs_main",
                            SKY_SHADER,
                            sky_vs(),
                            [45; 32],
                        ),
                    )
                    .await?,
                    sif,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Fragment,
                            "fs_main",
                            SKY_SHADER,
                            sky_fs(),
                            [46; 32],
                        ),
                    )
                    .await?,
                )
                .with_vertex_input(mesh_input())
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Front),
                )
                .with_depth_stencil(read)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let pt = texture(d, &pk, "10 plants")?;
        let gt = texture(d, &gk, "10 ground")?;
        let ptv = d.create_texture_view(
            &pt,
            &TextureViewDescriptor::whole(&pt, TextureViewDimension::D2Array)?,
        )?;
        let gtv = d.create_texture_view(
            &gt,
            &TextureViewDescriptor::whole(&gt, TextureViewDimension::D2)?,
        )?;
        let samp = sampler(d, pk.mips.max(gk.mips))?;
        let ubo = [ubo(d, 0)?, ubo(d, 1)?];
        let pg = [
            group(d, &pl, &ubo[0], &ptv, &samp)?,
            group(d, &pl, &ubo[1], &ptv, &samp)?,
        ];
        let gg = [
            group(d, &gl, &ubo[0], &gtv, &samp)?,
            group(d, &gl, &ubo[1], &gtv, &samp)?,
        ];
        let sg = [sky_group(d, &sl, &ubo[0])?, sky_group(d, &sl, &ubo[1])?];
        let (pvb, pib) = mesh_buffers(d, "plants", &pm)?;
        let (gvb, gib) = mesh_buffers(d, "ground", &gm)?;
        let (svb, sib) = mesh_buffers(d, "sky", &sm)?;
        let (pvl, pil) = (pm.vertices.len(), pm.indices.len());
        let (gvl, gil) = (gm.vertices.len(), gm.indices.len());
        let (svl, sil) = (sm.vertices.len(), sm.indices.len());
        let draws = pm.primitives.len() as u32;
        let records = instances(draws);
        let mut args = Vec::new();
        for (i, p) in pm.primitives.iter().enumerate() {
            for v in [
                p.index_count,
                INSTANCES,
                p.first_index,
                0,
                i as u32 * INSTANCES,
            ] {
                args.extend_from_slice(&v.to_le_bytes())
            }
        }
        let ib = d.create_buffer(
            &BufferDescriptor::new(
                records.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("10 instances"),
        )?;
        let indirect = d.create_buffer(
            &BufferDescriptor::new(
                args.len() as u64,
                BufferUsage::INDIRECT.union(BufferUsage::COPY_DST),
            )
            .with_label("10 indexed indirect commands"),
        )?;
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        for (b, x) in [
            (&pvb, pm.vertices),
            (&pib, pm.indices),
            (&gvb, gm.vertices),
            (&gib, gm.indices),
            (&svb, sm.vertices),
            (&sib, sm.indices),
            (&ib, records),
            (&indirect, args),
        ] {
            r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
                b.clone(),
                0,
                x,
            ))?)?
        }
        upload(d, &mut r, &pt, &pk)?;
        upload(d, &mut r, &gt, &gk)?;
        let mut plan = SubmissionPlanBuilder::new(d);
        plan.add_batch(lane, vec![r.finish()?])?;
        let receipt = d.submit(plan.build()?)?;
        let _ = d.wait_completion(receipt.completion()).await?;
        Ok(Self {
            plants,
            ground,
            sky,
            pg,
            gg,
            sg,
            ubo,
            pv: binding(pvb, pvl),
            pi: binding(pib, pil),
            gv: binding(gvb, gvl),
            gi: binding(gib, gil),
            sv: binding(svb, svl),
            si: binding(sib, sil),
            instance: binding(ib, draws as usize * INSTANCES as usize * 32),
            indirect,
            draws,
            depth: depth(d, e)?,
            e,
            next: 0,
            done: [None, None],
        })
    }
    async fn render(
        &mut self,
        d: &Device,
        lane: SubmissionLaneId,
        f: AcquiredFrame,
        time: f32,
    ) -> RhiResult<()> {
        let n = self.next;
        if let Some(x) = self.done[n].take() {
            let _ = d.wait_completion(x).await?;
        }
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            self.ubo[n].clone(),
            0,
            Uniform::new(self.e, time).bytes(),
        ))?)?;
        let mut s = r.begin_raster(
            &RasterScopeDescriptor::new()
                .with_color(
                    ShaderLocation::new(0),
                    ColorAttachment {
                        view: ColorAttachmentView::Frame(f.attachment()),
                        load: LoadOp::Clear(ColorClearValue::Float([0.18, 0.27, 0.5, 0.])),
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
        s.set_viewport(Viewport::new(
            0.,
            0.,
            self.e.width as f32,
            self.e.height as f32,
            0.,
            1.,
        ))?;
        s.set_scissor(Rect::new(0, 0, self.e.width, self.e.height))?;
        s.set_bind_group(BindGroupIndex::new(0), &self.sg[n], &[])?;
        s.set_pipeline(&self.sky)?;
        s.set_vertex_buffer(0, &self.sv)?;
        s.set_index_buffer(&self.si, IndexFormat::Uint32)?;
        s.draw_indexed(0..(self.si.range().size() / 4) as u32, 0, 0..1)?;
        s.set_bind_group(BindGroupIndex::new(0), &self.gg[n], &[])?;
        s.set_pipeline(&self.ground)?;
        s.set_vertex_buffer(0, &self.gv)?;
        s.set_index_buffer(&self.gi, IndexFormat::Uint32)?;
        s.draw_indexed(0..(self.gi.range().size() / 4) as u32, 0, 0..1)?;
        s.set_bind_group(BindGroupIndex::new(0), &self.pg[n], &[])?;
        s.set_pipeline(&self.plants)?;
        s.set_vertex_buffer(0, &self.pv)?;
        s.set_vertex_buffer(1, &self.instance)?;
        s.set_index_buffer(&self.pi, IndexFormat::Uint32)?;
        if d.capabilities()
            .supports_feature(OptionalFeature::MultiDrawIndirect)
        {
            s.multi_draw_indexed_indirect(&self.indirect, 0, self.draws, 20)?
        } else {
            for i in 0..self.draws {
                s.draw_indexed_indirect(&self.indirect, u64::from(i) * 20)?
            }
        }
        s.end()?;
        let mut plan = SubmissionPlanBuilder::new(d);
        let p = plan.add_batch(lane, vec![r.finish()?])?;
        plan.present_after(f, p)?;
        let receipt = d.submit(plan.build()?)?;
        self.done[n] = Some(receipt.completion());
        let _ = d.wait_present(receipt.presents()[0].id()).await?;
        self.next = (n + 1) % FRAMES;
        Ok(())
    }
}

fn asset(x: String) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, x)
}
fn binding(b: Buffer, n: usize) -> BufferBinding {
    BufferBinding::new(b, BufferRange::new(0, n as u64))
}
fn mesh_input() -> VertexInputState {
    VertexInputState::new().with_buffer(
        VertexBufferLayout::new(96, VertexStepMode::Vertex)
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
                VertexFormat::Float32x2,
                24,
            ))
            .with_attribute(VertexAttribute::new(
                ShaderLocation::new(3),
                VertexFormat::Float32x3,
                32,
            )),
    )
}
fn instance_input() -> VertexBufferLayout {
    VertexBufferLayout::new(32, VertexStepMode::Instance)
        .with_attribute(VertexAttribute::new(
            ShaderLocation::new(4),
            VertexFormat::Float32x3,
            0,
        ))
        .with_attribute(VertexAttribute::new(
            ShaderLocation::new(5),
            VertexFormat::Float32x3,
            12,
        ))
        .with_attribute(VertexAttribute::new(
            ShaderLocation::new(6),
            VertexFormat::Float32,
            24,
        ))
        .with_attribute(VertexAttribute::new(
            ShaderLocation::new(7),
            VertexFormat::Sint32,
            28,
        ))
}
fn layout(d: &Device, dim: TextureViewDimension) -> RhiResult<BindGroupLayout> {
    d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
        BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer { min_size: UBO },
        ),
        BindingSlot::new(
            BindingSlotId::new(1),
            ShaderStages::FRAGMENT,
            BindingKind::SampledTexture {
                dimension: dim,
                sample_type: TextureSampleType::Float,
                multisampled: false,
            },
        ),
        BindingSlot::new(
            BindingSlotId::new(2),
            ShaderStages::FRAGMENT,
            BindingKind::Sampler {
                kind: SamplerKind::Filtering,
            },
        ),
    ]))
}
fn sky_layout(d: &Device) -> RhiResult<BindGroupLayout> {
    d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
        BindingSlotId::new(0),
        ShaderStages::VERTEX,
        BindingKind::UniformBuffer { min_size: UBO },
    )]))
}
fn ubo(d: &Device, n: usize) -> RhiResult<Buffer> {
    d.create_buffer(
        &BufferDescriptor::new(UBO, BufferUsage::UNIFORM.union(BufferUsage::COPY_DST))
            .with_label(format!("10 ubo {n}")),
    )
}
fn group(
    d: &Device,
    l: &BindGroupLayout,
    b: &Buffer,
    t: &TextureView,
    s: &Sampler,
) -> RhiResult<BindGroup> {
    d.create_bind_group(
        &BindGroupDescriptor::new(l.clone())
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(0),
                BindingResource::Buffer(BufferBinding::new(b.clone(), BufferRange::new(0, UBO))),
            ))
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(1),
                BindingResource::Texture(t.clone()),
            ))
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(2),
                BindingResource::Sampler(s.clone()),
            )),
    )
}
fn sky_group(d: &Device, l: &BindGroupLayout, b: &Buffer) -> RhiResult<BindGroup> {
    d.create_bind_group(
        &BindGroupDescriptor::new(l.clone()).with_entry(BindGroupEntry::new(
            BindingSlotId::new(0),
            BindingResource::Buffer(BufferBinding::new(b.clone(), BufferRange::new(0, UBO))),
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
fn sampler(d: &Device, m: u32) -> RhiResult<Sampler> {
    let a = if d
        .capabilities()
        .supports_feature(OptionalFeature::SamplerAnisotropy)
    {
        d.capabilities()
            .limit(LimitKey::MaxSamplerAnisotropy)
            .unwrap_or(1)
            .clamp(1, u16::MAX as u64) as u16
    } else {
        1
    };
    d.create_sampler(
        &SamplerDescriptor::new()
            .with_address_modes(
                AddressMode::Repeat,
                AddressMode::Repeat,
                AddressMode::Repeat,
            )
            .with_filters(FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
            .with_lod_clamp(0., m as f32)
            .with_max_anisotropy(a),
    )
}
fn mesh_buffers(d: &Device, n: &str, m: &gltf::EmbeddedModel) -> RhiResult<(Buffer, Buffer)> {
    Ok((
        d.create_buffer(
            &BufferDescriptor::new(
                m.vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label(format!("10 {n} vertices")),
        )?,
        d.create_buffer(
            &BufferDescriptor::new(
                m.indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label(format!("10 {n} indices")),
        )?,
    ))
}

struct Ktx {
    width: u32,
    height: u32,
    layers: u32,
    mips: u32,
    data: Vec<Vec<Vec<u8>>>,
}
impl Ktx {
    fn parse(b: &[u8]) -> Result<Self, String> {
        const ID: [u8; 12] = [
            0xAB, b'K', b'T', b'X', b' ', b'1', b'1', 0xBB, 13, 10, 26, 10,
        ];
        if b.len() < 64 || b[..12] != ID {
            return Err("expected KTX1".into());
        }
        let w = |o| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        if w(12) != 0x04030201 || w(16) != 0x1401 || w(24) != 0x1908 || w(28) != 0x8058 {
            return Err("expected RGBA8 KTX1".into());
        }
        let (width, height, raw, mips) = (w(36), w(40), w(48), w(56));
        let layers = raw.max(1);
        let mut at = 64 + w(60) as usize;
        let mut data = Vec::new();
        for mip in 0..mips {
            if at + 4 > b.len() {
                return Err("truncated KTX".into());
            }
            let size = w(at) as usize;
            at += 4;
            let expected = (width >> mip).max(1) as usize * (height >> mip).max(1) as usize * 4;
            if size != expected {
                return Err("unexpected KTX image size".into());
            }
            let mut level = Vec::new();
            for _ in 0..layers {
                let end = at + size;
                if end > b.len() {
                    return Err("truncated KTX image".into());
                }
                level.push(b[at..end].to_vec());
                at = (end + 3) & !3
            }
            data.push(level)
        }
        Ok(Self {
            width,
            height,
            layers,
            mips,
            data,
        })
    }
}
fn texture(d: &Device, k: &Ktx, label: &str) -> RhiResult<Texture> {
    d.create_texture(
        &TextureDescriptor::new_2d(
            k.width,
            k.height,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
        )
        .with_mip_levels(k.mips)
        .with_array_layers(k.layers)
        .with_label(label),
    )
}
fn upload(
    d: &Device,
    r: &mut fluxel_rhi::api::command::CommandRecorder,
    t: &Texture,
    k: &Ktx,
) -> RhiResult<()> {
    for (mip, layers) in k.data.iter().enumerate() {
        let (w, h) = ((k.width >> mip).max(1), (k.height >> mip).max(1));
        for (layer, bytes) in layers.iter().enumerate() {
            r.encode_upload(&d.create_texture_upload(TextureUploadDescriptor::new(
                t.clone(),
                TextureSubresourceLayers {
                    aspect: TextureAspect::Color,
                    mip_level: mip as u32,
                    base_layer: layer as u32,
                    layer_count: 1,
                },
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d::d2(w, h),
                HostTexelLayout {
                    bytes_per_row: w * 4,
                    rows_per_image: h,
                },
                bytes.clone(),
            ))?)?
        }
    }
    Ok(())
}
fn instances(meshes: u32) -> Vec<u8> {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|x| x.as_secs() as u32)
        .unwrap_or(0);
    let mut r = Rng(seed);
    let mut out = vec![0; meshes as usize * INSTANCES as usize * 32];
    for i in 0..meshes as usize * INSTANCES as usize {
        let theta = 2. * std::f32::consts::PI * r.f();
        let phi = (1. - 2. * r.f()).acos();
        let p = [phi.sin() * theta.cos() * RADIUS, 0., phi.cos() * RADIUS];
        let rot = [0., std::f32::consts::PI * r.f(), 0.];
        let scale = 1. + 2. * r.f();
        let base = i * 32;
        for (j, x) in p.into_iter().chain(rot).chain([scale]).enumerate() {
            out[base + j * 4..base + j * 4 + 4].copy_from_slice(&x.to_le_bytes())
        }
        out[base + 28..base + 32].copy_from_slice(&((i as u32) / INSTANCES).to_le_bytes())
    }
    out
}
struct Rng(u32);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / 16777216.
    }
}
struct Uniform {
    p: [f32; 16],
    v: [f32; 16],
}
impl Uniform {
    fn new(e: Extent3d, _: f32) -> Self {
        let f = 1. / 30f32.to_radians().tan();
        let a = e.width.max(1) as f32 / e.height.max(1) as f32;
        Self {
            p: [
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
                512. / (0.1 - 512.),
                -1.,
                0.,
                0.,
                0.1 * 512. / (0.1 - 512.),
                0.,
            ],
            v: mm(
                mm(rx((-12f32).to_radians()), ry(159f32.to_radians())),
                tr([0.4, 1.25, 0.]),
            ),
        }
    }
    fn bytes(self) -> Vec<u8> {
        self.p
            .into_iter()
            .chain(self.v)
            .flat_map(f32::to_le_bytes)
            .collect()
    }
}
fn tr(x: [f32; 3]) -> [f32; 16] {
    [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., x[0], x[1], x[2], 1.,
    ]
}
fn rx(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [1., 0., 0., 0., 0., c, s, 0., 0., -s, c, 0., 0., 0., 0., 1.]
}
fn ry(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.]
}
fn mm(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut o = [0.; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum()
        }
    }
    o
}
fn art(
    s: ShaderStage,
    e: &'static str,
    c: &'static str,
    i: ShaderInterface,
    h: [u8; 32],
) -> ShaderArtifact {
    ShaderArtifact::new(
        s,
        e,
        ShaderCode::Wgsl(Arc::from(c)),
        ShaderAbiVersion { major: 1, minor: 0 },
        i,
        ShaderRequirements::new(),
        ArtifactHash(h),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
fn rr(n: u32, k: BindingKind) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(n),
        kind: k,
        count: BindingCount::One,
    }
}
fn io(n: u32, c: u8) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(n),
        numeric_type: ShaderNumericType::Float32,
        components: c,
        interpolation: Some(ShaderInterpolation {
            mode: InterpolationMode::Perspective,
            sampling: InterpolationSampling::Center,
        }),
    }
}
fn ii(n: u32) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(n),
        numeric_type: ShaderNumericType::Sint32,
        components: 1,
        interpolation: None,
    }
}
fn plant_vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(0, BindingKind::UniformBuffer { min_size: UBO }))
        .with_input(io(0, 3))
        .with_input(io(1, 3))
        .with_input(io(2, 2))
        .with_input(io(3, 3))
        .with_input(io(4, 3))
        .with_input(io(5, 3))
        .with_input(io(6, 1))
        .with_input(ii(7))
        .with_output(io(0, 3))
        .with_output(io(1, 3))
        .with_output(io(2, 3))
        .with_output(io(3, 3))
        .with_output(io(4, 3))
        .with_writes_position(true)
}
fn plant_fs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(
            1,
            BindingKind::SampledTexture {
                dimension: TextureViewDimension::D2Array,
                sample_type: TextureSampleType::Float,
                multisampled: false,
            },
        ))
        .with_resource(rr(
            2,
            BindingKind::Sampler {
                kind: SamplerKind::Filtering,
            },
        ))
        .with_input(io(0, 3))
        .with_input(io(1, 3))
        .with_input(io(2, 3))
        .with_input(io(3, 3))
        .with_input(io(4, 3))
        .with_output(ShaderLocationInterface {
            interpolation: None,
            ..io(0, 4)
        })
}
fn surface_vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(0, BindingKind::UniformBuffer { min_size: UBO }))
        .with_input(io(0, 3))
        .with_input(io(2, 2))
        .with_output(io(0, 2))
        .with_writes_position(true)
}
fn surface_fs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(
            1,
            BindingKind::SampledTexture {
                dimension: TextureViewDimension::D2,
                sample_type: TextureSampleType::Float,
                multisampled: false,
            },
        ))
        .with_resource(rr(
            2,
            BindingKind::Sampler {
                kind: SamplerKind::Filtering,
            },
        ))
        .with_input(io(0, 2))
        .with_output(ShaderLocationInterface {
            interpolation: None,
            ..io(0, 4)
        })
}
fn sky_vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(0, BindingKind::UniformBuffer { min_size: UBO }))
        .with_input(io(0, 3))
        .with_input(io(2, 2))
        .with_output(io(0, 2))
        .with_writes_position(true)
}
fn sky_fs() -> ShaderInterface {
    ShaderInterface::new()
        .with_input(io(0, 2))
        .with_output(ShaderLocationInterface {
            interpolation: None,
            ..io(0, 4)
        })
}
