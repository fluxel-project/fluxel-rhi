//! Faithful port of SaschaWillems/Vulkan `examples/offscreen/offscreen.cpp`.
//!
//! The first raster scope renders the Y-mirrored Chinese dragon into a 512x512
//! RGBA8 colour target.  The second draws the original plane with the source
//! shader's 7x7 (49 tap) reflection blur, then the unmirrored dragon.

pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use std::{sync::Arc, time::Instant};

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
    platform::Device,
    presentation::AcquiredFrame,
    resource::{
        AddressMode, Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor,
        BufferUsage, CompareFunction, Extent3d, FilterMode, Sampler, SamplerDescriptor,
        TextureDescriptor, TextureUsage, TextureView, TextureViewDescriptor, TextureViewDimension,
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
const OFFSCREEN_DIM: u32 = 512;
const UBO_BYTES: u64 = 208; // projection, view, model, light position
const DRAGON: &[u8] = include_bytes!("assets/models/chinesedragon.gltf");
const PLANE: &[u8] = include_bytes!("assets/models/plane.gltf");

// Direct WGSL translation of shaders/glsl/offscreen/{phong,mirror}.{vert,frag}.
const PHONG: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, model:mat4x4<f32>, light:vec4<f32>, };
@group(0) @binding(0) var<uniform> u:U;
struct I { @location(0) p:vec3<f32>, @location(1) n:vec3<f32>, @location(2) uv:vec2<f32>, @location(3) c:vec3<f32>, };
struct O { @builtin(position) p:vec4<f32>, @location(0) n:vec3<f32>, @location(1) c:vec3<f32>, @location(2) eye:vec3<f32>, @location(3) light:vec3<f32>, };
@vertex fn vs_main(i:I)->O { var o:O; o.n=i.n;o.c=i.c;o.p=u.projection*u.view*u.model*vec4<f32>(i.p,1.0);let e=(u.view*u.model*vec4<f32>(i.p,1.0)).xyz;o.eye=e;o.light=normalize(u.light.xyz-e);return o; }
@fragment fn fs_main(i:O)->@location(0) vec4<f32> { let eye=normalize(-i.eye);let r=normalize(reflect(-i.light,i.n));let ambient=vec4<f32>(0.1,0.1,0.1,1.0);let diffuse=vec4<f32>(max(dot(i.n,i.light),0.0));let spec=select(vec4<f32>(0.0),vec4<f32>(0.5,0.5,0.5,1.0)*pow(max(dot(r,eye),0.0),16.0)*0.75,dot(i.eye,i.n)<0.0);return (ambient+diffuse)*vec4<f32>(i.c,1.0)+spec; }
"#;
const MIRROR: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, model:mat4x4<f32>, light:vec4<f32>, };
@group(0) @binding(0) var<uniform> u:U; @group(0) @binding(1) var image:texture_2d<f32>; @group(0) @binding(2) var samp:sampler;
struct I { @location(0) p:vec3<f32>, }; struct VOut { @builtin(position) p:vec4<f32>, @location(0) clip:vec4<f32>, }; struct FIn { @location(0) clip:vec4<f32>, @builtin(front_facing) front:bool, };
@vertex fn vs_main(i:I)->VOut { var o:VOut; o.clip=u.projection*u.view*u.model*vec4<f32>(i.p,1.0);o.p=o.clip;return o; }
@fragment fn fs_main(i:FIn)->@location(0) vec4<f32> { if(!i.front){return vec4<f32>(0.0,0.0,0.0,1.0);}let q=(i.clip/i.clip.w+vec4<f32>(1.0))*0.5;var reflection=vec4<f32>(0.0);for(var x:i32=-3;x<=3;x=x+1){for(var y:i32=-3;y<=3;y=y+1){reflection+=textureSample(image,samp,q.xy+vec2<f32>(f32(x),f32(y))/512.0)/49.0;}}return vec4<f32>(0.0,0.0,0.0,1.0)+reflection; }
"#;
// C++ toggles this with its UI overlay. The host has no input event API, so
// preserve the branch through a startup flag: FLUXEL_RHI_OFFSCREEN_DEBUG=1.
const DEBUG_QUAD: &str = r#"
@group(0) @binding(1) var image:texture_2d<f32>; @group(0) @binding(2) var samp:sampler;
struct V { @builtin(position) p:vec4<f32>, @location(0) uv:vec2<f32>, };
@vertex fn vs_main(@builtin(vertex_index) i:u32)->V { var o:V;o.uv=vec2<f32>(f32((i<<1u)&2u),f32(i&2u));o.p=vec4<f32>(o.uv*2.0-1.0,0.0,1.0);return o; }
@fragment fn fs_main(i:V)->@location(0) vec4<f32>{return textureSample(image,samp,i.uv);}
"#;

fn main() {
    if let Err(e) = common::run_example("Offscreen mirror rendering", create_example()) {
        eprintln!("07_offscreen: {e}");
        std::process::exit(1)
    }
}
pub fn create_example() -> Example {
    Example {
        work: None,
        lane: None,
        rotation: 0.,
        last: None,
    }
}
pub struct Example {
    work: Option<Workload>,
    lane: Option<SubmissionLaneId>,
    rotation: f32,
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
            .ok_or_else(|| std::io::Error::other("07_offscreen requires COPY|RASTER lane"))?;
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
        let now = Instant::now();
        if let Some(last) = self.last {
            self.rotation += (now - last).as_secs_f32() * 10.;
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
                    self.rotation,
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

struct Workload {
    shaded: RasterPipeline,
    shaded_offscreen: RasterPipeline,
    mirror: RasterPipeline,
    debug_quad: RasterPipeline,
    debug_display: bool,
    model_groups: [BindGroup; FRAMES],
    off_groups: [BindGroup; FRAMES],
    mirror_groups: [BindGroup; FRAMES],
    ubos: [[Buffer; 3]; FRAMES],
    dragon_v: BufferBinding,
    dragon_i: BufferBinding,
    plane_v: BufferBinding,
    plane_i: BufferBinding,
    off_color: TextureView,
    off_depth: TextureView,
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
        let dragon =
            gltf::load_embedded_model(DRAGON, gltf::LoadOptions::CPP_PORT).map_err(asset)?;
        let plane = gltf::load_embedded_model(PLANE, gltf::LoadOptions::CPP_PORT).map_err(asset)?;
        let sl = shade_layout(d)?;
        let ml = mirror_layout(d)?;
        let si =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![sl.clone()]))?;
        let mi =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![ml.clone()]))?;
        let ds = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(true));
        let shaded = pipeline(
            d,
            si.clone(),
            PHONG,
            fmt,
            ds.clone(),
            CullMode::Back,
            [1; 32],
        )
        .await?;
        let shaded_offscreen = pipeline(
            d,
            si,
            PHONG,
            TextureFormat::Rgba8Unorm,
            ds.clone(),
            CullMode::Front,
            [2; 32],
        )
        .await?;
        let debug_quad = debug_pipeline(d, mi.clone(), fmt, ds.clone()).await?;
        let mirror = pipeline(d, mi, MIRROR, fmt, ds, CullMode::None, [3; 32]).await?;
        let (dvb, dib) = buffers(d, "dragon", &dragon)?;
        let (pvb, pib) = buffers(d, "plane", &plane)?;
        let off_texture = d.create_texture(&TextureDescriptor::new_2d(
            OFFSCREEN_DIM,
            OFFSCREEN_DIM,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
        ))?;
        let off_color = d.create_texture_view(
            &off_texture,
            &TextureViewDescriptor::whole(&off_texture, TextureViewDimension::D2)?,
        )?;
        let off_depth = depth(d, Extent3d::d2(OFFSCREEN_DIM, OFFSCREEN_DIM))?;
        let sampler = d.create_sampler(
            &SamplerDescriptor::new()
                .with_address_modes(
                    AddressMode::ClampToEdge,
                    AddressMode::ClampToEdge,
                    AddressMode::ClampToEdge,
                )
                .with_filters(FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
                .with_lod_clamp(0., 1.),
        )?;
        let ubos = [
            [
                ubo(d, "model 0".into())?,
                ubo(d, "mirror 0".into())?,
                ubo(d, "offscreen 0".into())?,
            ],
            [
                ubo(d, "model 1".into())?,
                ubo(d, "mirror 1".into())?,
                ubo(d, "offscreen 1".into())?,
            ],
        ];
        let model_groups = [
            shade_group(d, &sl, &ubos[0][0])?,
            shade_group(d, &sl, &ubos[1][0])?,
        ];
        let off_groups = [
            shade_group(d, &sl, &ubos[0][2])?,
            shade_group(d, &sl, &ubos[1][2])?,
        ];
        let mirror_groups = [
            mirror_group(d, &ml, &ubos[0][1], &off_color, &sampler)?,
            mirror_group(d, &ml, &ubos[1][1], &off_color, &sampler)?,
        ];
        let dragon_vertex_len = dragon.vertices.len() as u64;
        let dragon_index_len = dragon.indices.len() as u64;
        let plane_vertex_len = plane.vertices.len() as u64;
        let plane_index_len = plane.indices.len() as u64;
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        for (b, bytes) in [
            (&dvb, dragon.vertices),
            (&dib, dragon.indices),
            (&pvb, plane.vertices),
            (&pib, plane.indices),
        ] {
            r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
                b.clone(),
                0,
                bytes,
            ))?)?;
        }
        let mut p = SubmissionPlanBuilder::new(d);
        p.add_batch(lane, vec![r.finish()?])?;
        let receipt = d.submit(p.build()?)?;
        let _ = d.wait_completion(receipt.completion()).await?;
        Ok(Self {
            shaded,
            shaded_offscreen,
            mirror,
            debug_quad,
            debug_display: std::env::var("FLUXEL_RHI_OFFSCREEN_DEBUG").as_deref() == Ok("1"),
            model_groups,
            off_groups,
            mirror_groups,
            ubos,
            dragon_v: BufferBinding::new(dvb, BufferRange::new(0, dragon_vertex_len)),
            dragon_i: BufferBinding::new(dib, BufferRange::new(0, dragon_index_len)),
            plane_v: BufferBinding::new(pvb, BufferRange::new(0, plane_vertex_len)),
            plane_i: BufferBinding::new(pib, BufferRange::new(0, plane_index_len)),
            off_color,
            off_depth,
            depth: depth(d, extent)?,
            extent,
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
        rot: f32,
    ) -> RhiResult<()> {
        let n = self.next;
        if let Some(x) = self.done[n].take() {
            let _ = d.wait_completion(x).await?;
        }
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        let [model, mirror, off] = uniforms(self.extent, rot);
        for (b, x) in [
            (&self.ubos[n][0], model),
            (&self.ubos[n][1], mirror),
            (&self.ubos[n][2], off),
        ] {
            r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
                b.clone(),
                0,
                x,
            ))?)?;
        }
        {
            let mut q = r.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_label("07 offscreen mirrored dragon")
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Texture(self.off_color.clone()),
                            load: LoadOp::Clear(ColorClearValue::Float([0., 0., 0., 0.])),
                            store: StoreOp::Store,
                            resolve: None,
                            depth_slice: None,
                        },
                    )
                    .with_depth_stencil(DepthStencilAttachment {
                        view: self.off_depth.clone(),
                        depth: Some(DepthAttachmentMode::ReadWrite {
                            load: LoadOp::Clear(1.),
                            store: StoreOp::Discard,
                        }),
                        stencil: None,
                    }),
            )?;
            set_view(&mut q, Extent3d::d2(OFFSCREEN_DIM, OFFSCREEN_DIM))?;
            q.set_pipeline(&self.shaded_offscreen)?;
            q.set_bind_group(BindGroupIndex::new(0), &self.off_groups[n], &[])?;
            q.set_vertex_buffer(0, &self.dragon_v)?;
            q.set_index_buffer(&self.dragon_i, IndexFormat::Uint32)?;
            q.draw_indexed(0..(self.dragon_i.range().size() / 4) as u32, 0, 0..1)?;
            q.end()?;
        }
        {
            let mut q = r.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_label("07 mirror and dragon")
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(f.attachment()),
                            load: LoadOp::Clear(ColorClearValue::Float([0.025, 0.025, 0.025, 1.0])),
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
            set_view(&mut q, self.extent)?;
            q.set_bind_group(BindGroupIndex::new(0), &self.mirror_groups[n], &[])?;
            if self.debug_display {
                q.set_pipeline(&self.debug_quad)?;
                q.draw(0..3, 0..1)?;
            } else {
                q.set_pipeline(&self.mirror)?;
                q.set_vertex_buffer(0, &self.plane_v)?;
                q.set_index_buffer(&self.plane_i, IndexFormat::Uint32)?;
                q.draw_indexed(0..(self.plane_i.range().size() / 4) as u32, 0, 0..1)?;
                q.set_pipeline(&self.shaded)?;
                q.set_bind_group(BindGroupIndex::new(0), &self.model_groups[n], &[])?;
                q.set_vertex_buffer(0, &self.dragon_v)?;
                q.set_index_buffer(&self.dragon_i, IndexFormat::Uint32)?;
                q.draw_indexed(0..(self.dragon_i.range().size() / 4) as u32, 0, 0..1)?;
            }
            q.end()?;
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

fn set_view(q: &mut fluxel_rhi::api::command::RasterScope<'_>, e: Extent3d) -> RhiResult<()> {
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
fn asset(e: impl ToString) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, e.to_string())
}
fn ubo(d: &Device, label: String) -> RhiResult<Buffer> {
    d.create_buffer(
        &BufferDescriptor::new(UBO_BYTES, BufferUsage::UNIFORM.union(BufferUsage::COPY_DST))
            .with_label(label),
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
fn buffers(d: &Device, n: &str, m: &gltf::EmbeddedModel) -> RhiResult<(Buffer, Buffer)> {
    Ok((
        d.create_buffer(
            &BufferDescriptor::new(
                m.vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label(format!("07 {n} vertices")),
        )?,
        d.create_buffer(
            &BufferDescriptor::new(
                m.indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label(format!("07 {n} indices")),
        )?,
    ))
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
async fn pipeline(
    d: &Device,
    i: fluxel_rhi::api::pipeline::PipelineInterface,
    code: &'static str,
    fmt: TextureFormat,
    ds: DepthStencilState,
    cull: CullMode,
    h: [u8; 32],
) -> RhiResult<RasterPipeline> {
    d.create_raster_pipeline(
        &RasterPipelineDescriptor::new(
            common::shader::create_shader(
                d,
                &art(ShaderStage::Vertex, "vs_main", code, vs(code), h),
            )
            .await?,
            i,
        )
        .with_fragment(
            common::shader::create_shader(
                d,
                &art(
                    ShaderStage::Fragment,
                    "fs_main",
                    code,
                    fs(code),
                    h.map(|x| x + 10),
                ),
            )
            .await?,
        )
        .with_vertex_input(mesh_input())
        .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleList).with_cull_mode(cull))
        .with_depth_stencil(ds)
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
    )
    .await
}
async fn debug_pipeline(
    d: &Device,
    i: fluxel_rhi::api::pipeline::PipelineInterface,
    fmt: TextureFormat,
    ds: DepthStencilState,
) -> RhiResult<RasterPipeline> {
    d.create_raster_pipeline(
        &RasterPipelineDescriptor::new(
            common::shader::create_shader(
                d,
                &art(
                    ShaderStage::Vertex,
                    "vs_main",
                    DEBUG_QUAD,
                    debug_vs(),
                    [4; 32],
                ),
            )
            .await?,
            i,
        )
        .with_fragment(
            common::shader::create_shader(
                d,
                &art(
                    ShaderStage::Fragment,
                    "fs_main",
                    DEBUG_QUAD,
                    debug_fs(),
                    [5; 32],
                ),
            )
            .await?,
        )
        .with_primitive(
            PrimitiveState::new(PrimitiveTopology::TriangleList).with_cull_mode(CullMode::None),
        )
        .with_depth_stencil(ds)
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
    )
    .await
}
fn shade_layout(d: &Device) -> RhiResult<BindGroupLayout> {
    d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![BindingSlot::new(
        BindingSlotId::new(0),
        ShaderStages::VERTEX,
        BindingKind::UniformBuffer {
            min_size: UBO_BYTES,
        },
    )]))
}
fn mirror_layout(d: &Device) -> RhiResult<BindGroupLayout> {
    d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
        BindingSlot::new(
            BindingSlotId::new(0),
            ShaderStages::VERTEX,
            BindingKind::UniformBuffer {
                min_size: UBO_BYTES,
            },
        ),
        BindingSlot::new(
            BindingSlotId::new(1),
            ShaderStages::FRAGMENT,
            BindingKind::SampledTexture {
                dimension: TextureViewDimension::D2,
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
fn shade_group(d: &Device, l: &BindGroupLayout, b: &Buffer) -> RhiResult<BindGroup> {
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
fn mirror_group(
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
                    BufferRange::new(0, UBO_BYTES),
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
fn uniforms(e: Extent3d, r: f32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let f = 1. / 30f32.to_radians().tan();
    let a = e.width.max(1) as f32 / e.height.max(1) as f32;
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
        256. / (0.1 - 256.),
        -1.,
        0.,
        0.,
        256. * 0.1 / (0.1 - 256.),
        0.,
    ];
    let v = mm(tr([0., 1., -6.]), rx((-2.5f32).to_radians()));
    let m = mm(ry(r.to_radians()), tr([0., -1., 0.]));
    let off = mm(mm(ry(r.to_radians()), sc([1., -1., 1.])), tr([0., -1., 0.]));
    (
        bytes(p, v, m),
        bytes(
            p,
            v,
            [
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
        ),
        bytes(p, v, off),
    )
}
fn bytes(p: [f32; 16], v: [f32; 16], m: [f32; 16]) -> Vec<u8> {
    p.into_iter()
        .chain(v)
        .chain(m)
        .chain([0., 0., 0., 1.])
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
            o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
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
fn vertex_input(n: u32, c: u8) -> ShaderLocationInterface {
    ShaderLocationInterface {
        interpolation: None,
        ..io(n, c)
    }
}
fn vs(code: &str) -> ShaderInterface {
    let mut x = ShaderInterface::new()
        .with_resource(rr(
            0,
            BindingKind::UniformBuffer {
                min_size: UBO_BYTES,
            },
        ))
        .with_input(vertex_input(0, 3))
        .with_writes_position(true);
    if code == PHONG {
        x = x
            .with_input(vertex_input(1, 3))
            .with_input(vertex_input(2, 2))
            .with_input(vertex_input(3, 3))
            .with_output(io(0, 3))
            .with_output(io(1, 3))
            .with_output(io(2, 3))
            .with_output(io(3, 3));
    } else {
        x = x.with_output(io(0, 4));
    }
    x
}
fn fs(code: &str) -> ShaderInterface {
    let mut x = ShaderInterface::new().with_output(ShaderLocationInterface {
        interpolation: None,
        ..io(0, 4)
    });
    if code == PHONG {
        x = x
            .with_input(io(0, 3))
            .with_input(io(1, 3))
            .with_input(io(2, 3))
            .with_input(io(3, 3));
    } else {
        x = x
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
            .with_input(io(0, 4));
    }
    x
}
fn debug_vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_output(io(0, 2))
        .with_writes_position(true)
}
fn debug_fs() -> ShaderInterface {
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
