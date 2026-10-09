//! Port of SaschaWillems/Vulkan `examples/instancing/instancing.cpp`.
//! It retains the original meshes, KTX texture array, camera, 8,192 instances,
//! starfield/planet/rocks ordering, and per-frame animation rates.

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
        SamplerDescriptor, TextureAspect, TextureDescriptor, TextureSubresourceLayers,
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
const INSTANCES: u32 = 8192;
const UBO_SIZE: u64 = 160; // 2 mat4 + vec4 + 2 f32 + WGSL's 8-byte tail padding.
const ROCK: &[u8] = include_bytes!("assets/models/rock01.gltf");
const PLANET: &[u8] = include_bytes!("assets/models/lavaplanet.gltf");
const ROCK_TEX: &[u8] = include_bytes!("assets/textures/texturearray_rocks_rgba.ktx");
const PLANET_TEX: &[u8] = include_bytes!("assets/textures/lavaplanet_rgba.ktx");

const ROCK_SHADER: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, light:vec4<f32>, local_speed:f32, global_speed:f32, pad:vec2<f32>, };
@group(0) @binding(0) var<uniform> u:U; @group(0) @binding(1) var tex:texture_2d_array<f32>; @group(0) @binding(2) var samp:sampler;
struct I { @location(0) p:vec3<f32>, @location(1) n:vec3<f32>, @location(2) uv:vec2<f32>, @location(3) c:vec3<f32>, @location(4) ip:vec3<f32>, @location(5) ir:vec3<f32>, @location(6) scale:f32, @location(7) layer:i32, };
struct O { @builtin(position) p:vec4<f32>, @location(0) n:vec3<f32>, @location(1) c:vec3<f32>, @location(2) uv:vec3<f32>, @location(3) vv:vec3<f32>, @location(4) lv:vec3<f32>, };
@vertex fn vs_main(i:I)->O { var o:O; o.c=i.c;o.uv=vec3<f32>(i.uv,f32(i.layer)); let x=i.ir.x+u.local_speed;let y=i.ir.y+u.local_speed;let z=i.ir.z+u.local_speed;let mx=mat3x3<f32>(vec3<f32>(cos(x),sin(x),0),vec3<f32>(-sin(x),cos(x),0),vec3<f32>(0,0,1));let my=mat3x3<f32>(vec3<f32>(cos(y),0,sin(y)),vec3<f32>(0,1,0),vec3<f32>(-sin(y),0,cos(y)));let mz=mat3x3<f32>(vec3<f32>(1,0,0),vec3<f32>(0,cos(z),sin(z)),vec3<f32>(0,-sin(z),cos(z)));let r=mz*my*mx;let gy=i.ir.y+u.global_speed;let g=mat4x4<f32>(vec4<f32>(cos(gy),0,sin(gy),0),vec4<f32>(0,1,0,0),vec4<f32>(-sin(gy),0,cos(gy),0),vec4<f32>(0,0,0,1));let lp=vec4<f32>(i.p*r,1);let p=vec4<f32>(lp.xyz*i.scale+i.ip,1);o.p=u.projection*u.view*g*p;let vg=u.view*g;o.n=mat3x3<f32>(vg[0].xyz,vg[1].xyz,vg[2].xyz)*inverse(r)*i.n;let vp=u.view*vec4<f32>(i.p+i.ip,1);o.lv=mat3x3<f32>(u.view[0].xyz,u.view[1].xyz,u.view[2].xyz)*u.light.xyz-vp.xyz;o.vv=-vp.xyz;return o; }
@fragment fn fs_main(i:O)->@location(0) vec4<f32>{let c=textureSample(tex,samp,i.uv.xy,i32(i.uv.z))*vec4<f32>(i.c,1);let n=normalize(i.n);let l=normalize(i.lv);let v=normalize(i.vv);let d=max(dot(n,l),0.1)*i.c;let s=select(vec3<f32>(0),pow(max(dot(reflect(-l,n),v),0),16)*vec3<f32>(0.75)*c.r,dot(n,l)>0);return vec4<f32>(d*c.rgb+s,1);}"#;
const PLANET_SHADER: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, light:vec4<f32>, local_speed:f32, global_speed:f32, pad:vec2<f32>, }; @group(0) @binding(0) var<uniform> u:U; @group(0) @binding(1) var tex:texture_2d<f32>; @group(0) @binding(2) var samp:sampler;
struct I {@location(0)p:vec3<f32>,@location(1)n:vec3<f32>,@location(2)uv:vec2<f32>,@location(3)c:vec3<f32>,};struct O{@builtin(position)p:vec4<f32>,@location(0)n:vec3<f32>,@location(1)c:vec3<f32>,@location(2)uv:vec2<f32>,@location(3)vv:vec3<f32>,@location(4)lv:vec3<f32>,};
@vertex fn vs_main(i:I)->O{var o:O;o.c=i.c;o.uv=i.uv;o.p=u.projection*u.view*vec4<f32>(i.p,1);let p=u.view*vec4<f32>(i.p,1);let m=mat3x3<f32>(u.view[0].xyz,u.view[1].xyz,u.view[2].xyz);o.n=m*i.n;o.lv=m*u.light.xyz-p.xyz;o.vv=-p.xyz;return o;}@fragment fn fs_main(i:O)->@location(0)vec4<f32>{let c=textureSample(tex,samp,i.uv)*vec4<f32>(i.c,1)*1.5;let n=normalize(i.n);let l=normalize(i.lv);let r=reflect(-l,n);let d=max(dot(n,l),0)*i.c;let s=pow(max(dot(r,normalize(i.vv)),0),4)*vec3<f32>(0.5)*c.r;return vec4<f32>(d*c.rgb+s,1);}"#;
const STAR_SHADER: &str = r#"struct O{@builtin(position)p:vec4<f32>,@location(0)uvw:vec3<f32>,};@vertex fn vs_main(@builtin(vertex_index)i:u32)->O{var o:O;o.uvw=vec3<f32>(f32((i<<1u)&2u),f32(i&2u),f32(i&2u));o.p=vec4<f32>(o.uvw.xy*2-1,0,1);return o;}fn h(p:vec3<f32>)->f32{var q=fract(p*vec3<f32>(443.897,441.423,437.195));q+=dot(q,q.yxz+vec3<f32>(19.19));return fract((q.x+q.y)*q.z+(q.x+q.z)*q.y+(q.y+q.z)*q.x);}@fragment fn fs_main(i:O)->@location(0)vec4<f32>{let r=h(i.uvw);let s=select(0.,pow((r-.99)/.01,16.),r>=.99);return vec4<f32>(vec3<f32>(s),1);}"#;

fn main() {
    if let Err(e) = common::run_example("Instanced mesh rendering", create_example()) {
        eprintln!("03_instancing: {e}");
        std::process::exit(1)
    }
}
pub fn create_example() -> Example {
    Example {
        workload: None,
        lane: None,
        time: 0.0,
        last: None,
    }
}
pub struct Example {
    workload: Option<Workload>,
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
        let s = c.extent();
        self.workload = Some(common::block_on(Workload::new(
            &d,
            c.presentation_mut().configuration().format(),
            Extent3d::d2(s.width, s.height),
            lane,
        ))?);
        self.lane = Some(lane);
        Ok(())
    }
    fn update(
        &mut self,
        _: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = Instant::now();
        if let Some(last) = self.last {
            self.time += (now - last).as_secs_f32()
        }
        self.last = Some(now);
        Ok(())
    }
    fn render(
        &mut self,
        c: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let f = common::block_on(c.presentation_mut().acquire())?;
        let lane = self.lane.ok_or_else(|| std::io::Error::other("no lane"))?;
        common::block_on(
            self.workload
                .as_mut()
                .ok_or_else(|| std::io::Error::other("not initialized"))?
                .render(c.device(), lane, f, self.time),
        )?;
        Ok(())
    }
    fn resize(
        &mut self,
        c: &mut common::ExampleContext<'_>,
        w: u32,
        h: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(x) = &mut self.workload {
            x.extent = Extent3d::d2(w, h);
            x.depth = depth(c.device(), x.extent)?
        }
        Ok(())
    }
    fn device_lost(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.workload = None;
        self.lane = None;
        Ok(())
    }
    fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.device_lost()
    }
}

struct Workload {
    rocks: RasterPipeline,
    planet: RasterPipeline,
    stars: RasterPipeline,
    rock_groups: [BindGroup; FRAMES],
    static_groups: [BindGroup; FRAMES],
    ubos: [Buffer; FRAMES],
    rv: BufferBinding,
    ri: BufferBinding,
    pv: BufferBinding,
    pi: BufferBinding,
    instances: BufferBinding,
    depth: TextureView,
    extent: Extent3d,
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
        let rock_mesh = gltf::load_embedded_model(ROCK, gltf::LoadOptions::CPP_PORT)
            .map_err(|e| asset(e.to_string()))?;
        let planet_mesh = gltf::load_embedded_model(PLANET, gltf::LoadOptions::CPP_PORT)
            .map_err(|e| asset(e.to_string()))?;
        let rock_ktx = Ktx::parse(ROCK_TEX).map_err(asset)?;
        let planet_ktx = Ktx::parse(PLANET_TEX).map_err(asset)?;
        let rl = layout(d, TextureViewDimension::D2Array)?;
        let sl = layout(d, TextureViewDimension::D2)?;
        let ri =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![rl.clone()]))?;
        let si =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![sl.clone()]))?;
        let ds = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(true));
        let ro = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Vertex,
                            "vs_main",
                            ROCK_SHADER,
                            rock_vs(),
                            [31; 32],
                        ),
                    )
                    .await?,
                    ri,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Fragment,
                            "fs_main",
                            ROCK_SHADER,
                            rock_fs(),
                            [32; 32],
                        ),
                    )
                    .await?,
                )
                .with_vertex_input(mesh_input().with_buffer(instance_input()))
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Back),
                )
                .with_depth_stencil(ds)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let po = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Vertex,
                            "vs_main",
                            PLANET_SHADER,
                            planet_vs(),
                            [33; 32],
                        ),
                    )
                    .await?,
                    si.clone(),
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Fragment,
                            "fs_main",
                            PLANET_SHADER,
                            planet_fs(),
                            [34; 32],
                        ),
                    )
                    .await?,
                )
                .with_vertex_input(mesh_input())
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Back),
                )
                .with_depth_stencil(ds)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let sd = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(false));
        let st = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Vertex,
                            "vs_main",
                            STAR_SHADER,
                            star_vs(),
                            [35; 32],
                        ),
                    )
                    .await?,
                    si,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &art(
                            ShaderStage::Fragment,
                            "fs_main",
                            STAR_SHADER,
                            star_fs(),
                            [36; 32],
                        ),
                    )
                    .await?,
                )
                .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleList))
                .with_depth_stencil(sd)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let rt = texture(d, &rock_ktx, "03_instancing rocks")?;
        let pt = texture(d, &planet_ktx, "03_instancing planet")?;
        let rvw = d.create_texture_view(
            &rt,
            &TextureViewDescriptor::whole(&rt, TextureViewDimension::D2Array)?,
        )?;
        let pvw = d.create_texture_view(
            &pt,
            &TextureViewDescriptor::whole(&pt, TextureViewDimension::D2)?,
        )?;
        let sampler = sampler(d, rock_ktx.mips)?;
        let ubos = [ubo(d, 0)?, ubo(d, 1)?];
        let rock_groups = [
            group(d, &rl, &ubos[0], &rvw, &sampler)?,
            group(d, &rl, &ubos[1], &rvw, &sampler)?,
        ];
        let static_groups = [
            group(d, &sl, &ubos[0], &pvw, &sampler)?,
            group(d, &sl, &ubos[1], &pvw, &sampler)?,
        ];
        let (rvb, rib) = mesh_buffers(d, "rock01", &rock_mesh)?;
        let (pvb, pib) = mesh_buffers(d, "lavaplanet", &planet_mesh)?;
        let ib = instances(rock_ktx.layers);
        let ins = d.create_buffer(
            &BufferDescriptor::new(
                ib.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("03 instances"),
        )?;
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        for (b, x) in [
            (&rvb, rock_mesh.vertices),
            (&rib, rock_mesh.indices),
            (&pvb, planet_mesh.vertices),
            (&pib, planet_mesh.indices),
            (&ins, ib),
        ] {
            r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
                b.clone(),
                0,
                x,
            ))?)?;
        }
        upload(d, &mut r, &rt, &rock_ktx)?;
        upload(d, &mut r, &pt, &planet_ktx)?;
        let mut p = SubmissionPlanBuilder::new(d);
        p.add_batch(lane, vec![r.finish()?])?;
        let receipt = d.submit(p.build()?)?;
        let _ = d.wait_completion(receipt.completion()).await?;
        Ok(Self {
            rocks: ro,
            planet: po,
            stars: st,
            rock_groups,
            static_groups,
            ubos,
            rv: BufferBinding::new(rvb, BufferRange::new(0, rock_mesh.vertices.len() as u64)),
            ri: BufferBinding::new(rib, BufferRange::new(0, rock_mesh.indices.len() as u64)),
            pv: BufferBinding::new(pvb, BufferRange::new(0, planet_mesh.vertices.len() as u64)),
            pi: BufferBinding::new(pib, BufferRange::new(0, planet_mesh.indices.len() as u64)),
            instances: BufferBinding::new(ins, BufferRange::new(0, u64::from(INSTANCES) * 32)),
            depth: depth(d, extent)?,
            extent,
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
            self.ubos[n].clone(),
            0,
            Uniform::new(self.extent, time).bytes(),
        ))?)?;
        {
            let mut s = r.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(f.attachment()),
                            load: LoadOp::Clear(ColorClearValue::Float([0., 0., 0.2, 0.])),
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
                self.extent.width as f32,
                self.extent.height as f32,
                0.,
                1.,
            ))?;
            s.set_scissor(Rect::new(0, 0, self.extent.width, self.extent.height))?;
            s.set_bind_group(BindGroupIndex::new(0), &self.static_groups[n], &[])?;
            s.set_pipeline(&self.stars)?;
            s.draw(0..3, 0..1)?;
            s.set_pipeline(&self.planet)?;
            s.set_vertex_buffer(0, &self.pv)?;
            s.set_index_buffer(&self.pi, IndexFormat::Uint32)?;
            s.draw_indexed(0..(self.pi.range().size() / 4) as u32, 0, 0..1)?;
            s.set_bind_group(BindGroupIndex::new(0), &self.rock_groups[n], &[])?;
            s.set_pipeline(&self.rocks)?;
            s.set_vertex_buffer(0, &self.rv)?;
            s.set_vertex_buffer(1, &self.instances)?;
            s.set_index_buffer(&self.ri, IndexFormat::Uint32)?;
            s.draw_indexed(0..(self.ri.range().size() / 4) as u32, 0, 0..INSTANCES)?;
            s.end()?;
        }
        let mut p = SubmissionPlanBuilder::new(d);
        let point = p.add_batch(lane, vec![r.finish()?])?;
        p.present_after(f, point)?;
        let receipt = d.submit(p.build()?)?;
        self.done[n] = Some(receipt.completion());
        let _ = d.wait_present(receipt.presents()[0].id()).await?;
        self.next = (n + 1) % FRAMES;
        Ok(())
    }
}

fn asset(x: String) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, x)
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
            BindingKind::UniformBuffer { min_size: UBO_SIZE },
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
fn ubo(d: &Device, n: usize) -> RhiResult<Buffer> {
    d.create_buffer(
        &BufferDescriptor::new(UBO_SIZE, BufferUsage::UNIFORM.union(BufferUsage::COPY_DST))
            .with_label(format!("03 ubo {n}")),
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
                BindingResource::Buffer(BufferBinding::new(
                    b.clone(),
                    BufferRange::new(0, UBO_SIZE),
                )),
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
fn sampler(d: &Device, mips: u32) -> RhiResult<Sampler> {
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
            .with_lod_clamp(0., mips as f32)
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
            .with_label(format!("03 {n} vertices")),
        )?,
        d.create_buffer(
            &BufferDescriptor::new(
                m.indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label(format!("03 {n} indices")),
        )?,
    ))
}

// KTX1 has one `imageSize` followed by every array layer at each mip.
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
        let (width, height, raw_layers, mips) = (w(36), w(40), w(48), w(56));
        let layers = raw_layers.max(1);
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
                at = (end + 3) & !3;
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
fn texture(d: &Device, k: &Ktx, label: &str) -> RhiResult<fluxel_rhi::api::resource::Texture> {
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
    t: &fluxel_rhi::api::resource::Texture,
    k: &Ktx,
) -> RhiResult<()> {
    for (mip, ls) in k.data.iter().enumerate() {
        let (w, h) = ((k.width >> mip).max(1), (k.height >> mip).max(1));
        for (layer, x) in ls.iter().enumerate() {
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
                x.clone(),
            ))?)?
        }
    }
    Ok(())
}

fn instances(layers: u32) -> Vec<u8> {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|x| x.as_secs() as u32)
        .unwrap_or(0);
    let mut r = Rng(seed);
    let mut out = vec![0; INSTANCES as usize * 32];
    for index in 0..INSTANCES as usize / 2 {
        // The reference creates both records in one iteration, then stores the
        // second at `i + INSTANCE_COUNT / 2`. Preserve that random draw order.
        let inner = instance_record(&mut r, [7.0, 11.0], layers);
        let outer = instance_record(&mut r, [14.0, 18.0], layers);
        out[index * 32..(index + 1) * 32].copy_from_slice(&inner);
        let outer_index = index + INSTANCES as usize / 2;
        out[outer_index * 32..(outer_index + 1) * 32].copy_from_slice(&outer);
    }
    out
}

fn instance_record(r: &mut Rng, ring: [f32; 2], layers: u32) -> [u8; 32] {
    let rho = ((ring[1] * ring[1] - ring[0] * ring[0]) * r.f() + ring[0] * ring[0]).sqrt();
    let theta = 2.0 * std::f32::consts::PI * r.f();
    let position = [rho * theta.cos(), r.f() * 0.5 - 0.25, rho * theta.sin()];
    let rotation = [
        std::f32::consts::PI * r.f(),
        std::f32::consts::PI * r.f(),
        std::f32::consts::PI * r.f(),
    ];
    let scale = (1.5 + r.f() - r.f()) * 0.75;
    let layer = (r.f() * layers as f32).floor() as u32 % layers;
    let mut record = [0; 32];
    for (index, value) in position
        .into_iter()
        .chain(rotation)
        .chain([scale])
        .enumerate()
    {
        record[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    record[28..32].copy_from_slice(&layer.to_le_bytes());
    record
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
    l: [f32; 4],
    a: f32,
    b: f32,
}
impl Uniform {
    fn new(e: Extent3d, t: f32) -> Self {
        let f = 1. / 30f32.to_radians().tan();
        let aspect = e.width.max(1) as f32 / e.height.max(1) as f32;
        let p = [
            f / aspect,
            0.,
            0.,
            0.,
            0.,
            f,
            0.,
            0.,
            0.,
            0.,
            256. / (1. - 256.),
            -1.,
            0.,
            0.,
            256. / (1. - 256.),
            0.,
        ];
        let v = mm(
            tr([5.5, -1.85, -18.5]),
            mm(rx((-17.2f32).to_radians()), ry((-4.7f32).to_radians())),
        );
        Self {
            p,
            v,
            l: [0., -5., 0., 1.],
            a: t * 0.35,
            b: t * 0.01,
        }
    }
    fn bytes(self) -> Vec<u8> {
        self.p
            .into_iter()
            .chain(self.v)
            .chain(self.l)
            .chain([self.a, self.b, 0., 0.])
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
fn rock_vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(0, BindingKind::UniformBuffer { min_size: UBO_SIZE }))
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
fn rock_fs() -> ShaderInterface {
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
fn planet_vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(0, BindingKind::UniformBuffer { min_size: UBO_SIZE }))
        .with_input(io(0, 3))
        .with_input(io(1, 3))
        .with_input(io(2, 2))
        .with_input(io(3, 3))
        .with_output(io(0, 3))
        .with_output(io(1, 3))
        .with_output(io(2, 2))
        .with_output(io(3, 3))
        .with_output(io(4, 3))
        .with_writes_position(true)
}
fn planet_fs() -> ShaderInterface {
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
        .with_input(io(0, 3))
        .with_input(io(1, 3))
        .with_input(io(2, 2))
        .with_input(io(3, 3))
        .with_input(io(4, 3))
        .with_output(ShaderLocationInterface {
            interpolation: None,
            ..io(0, 4)
        })
}
fn star_vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_output(io(0, 3))
        .with_writes_position(true)
}
fn star_fs() -> ShaderInterface {
    ShaderInterface::new()
        .with_input(io(0, 3))
        .with_output(ShaderLocationInterface {
            interpolation: None,
            ..io(0, 4)
        })
}
