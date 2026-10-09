//! Port of SaschaWillems/Vulkan `texturemipmapgen`.
//! The original top KTX level is uploaded once; every following mip is made by
//! a linear blit from its predecessor.  The three source samplers are retained.

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
        BlitFilter, ColorAttachment, ColorAttachmentView, ColorClearValue, DepthAttachmentMode,
        DepthStencilAttachment, IndexFormat, LoadOp, RasterScopeDescriptor, RecorderDescriptor,
        Rect, StoreOp, TextureBlit, Viewport,
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
use std::sync::Arc;

const FRAMES: usize = 2;
const UBO: u64 = 224;
const KTX_ID: [u8; 12] = [
    0xAB, b'K', b'T', b'X', b' ', b'1', b'1', 0xBB, 13, 10, 26, 10,
];
const MODEL: &[u8] = include_bytes!("assets/models/tunnel_cylinder.gltf");
const IMAGE: &[u8] = include_bytes!("assets/textures/metalplate_nomips_rgba.ktx");
const WGSL: &str = r#"struct U{p:mat4x4<f32>,v:mat4x4<f32>,m:mat4x4<f32>,eye:vec4<f32>,bias:f32,};@group(0)@binding(0)var<uniform>u:U;@group(0)@binding(1)var tex:texture_2d<f32>;@group(0)@binding(2)var smp:sampler;struct I{@location(0)p:vec3<f32>,@location(1)uv:vec2<f32>,@location(2)n:vec3<f32>,};struct O{@builtin(position)p:vec4<f32>,@location(0)uv:vec2<f32>,@location(1)bias:f32,@location(2)n:vec3<f32>,@location(3)view:vec3<f32>,@location(4)light:vec3<f32>,};@vertex fn vs_main(i:I)->O{var o:O;o.uv=i.uv*vec2<f32>(2,1);o.bias=u.bias;let w=(u.m*vec4<f32>(i.p,1)).xyz;o.p=u.p*u.v*vec4<f32>(w,1);o.n=mat3x3<f32>(u.m[0].xyz,u.m[1].xyz,u.m[2].xyz)*i.n;o.light=w-vec3<f32>(-30,0,0);o.view=u.eye.xyz-w;return o;}@fragment fn fs_main(i:O)->@location(0)vec4<f32>{let c=textureSampleBias(tex,smp,i.uv,i.bias);let n=normalize(i.n);let l=normalize(i.light);let v=normalize(i.view);let r=reflect(l,n);let d=max(dot(n,l),.65);let s=pow(max(dot(r,v),0),16)*c.a;return vec4<f32>(d*c.rgb+s,1);}"#;
fn main() {
    if let Err(e) = common::run_example("Runtime mip map generation", create_example()) {
        eprintln!("09_texture_mipmap: {e}");
        std::process::exit(1)
    }
}
pub fn create_example() -> Example {
    Example {
        w: None,
        l: None,
        time: 0.,
        last: None,
    }
}
pub struct Example {
    w: Option<Work>,
    l: Option<SubmissionLaneId>,
    time: f32,
    last: Option<std::time::Instant>,
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
            .ok_or_else(|| std::io::Error::other("09 needs COPY|RASTER"))?;
        let e = c.extent();
        self.w = Some(common::block_on(Work::new(
            &d,
            c.presentation_mut().configuration().format(),
            Extent3d::d2(e.width, e.height),
            l,
        ))?);
        self.l = Some(l);
        Ok(())
    }
    fn update(
        &mut self,
        _: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::Instant::now();
        if let Some(last) = self.last {
            self.time += (now - last).as_secs_f32() * 0.05
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
            self.w
                .as_mut()
                .ok_or_else(|| std::io::Error::other("not initialized"))?
                .render(
                    c.device(),
                    self.l.ok_or_else(|| std::io::Error::other("no lane"))?,
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
            x.resize(c.device(), Extent3d::d2(w, h))?
        };
        Ok(())
    }
    fn device_lost(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.w = None;
        self.l = None;
        Ok(())
    }
    fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.device_lost()
    }
}
struct Work {
    p: RasterPipeline,
    g: [[BindGroup; 3]; FRAMES],
    u: [Buffer; FRAMES],
    v: BufferBinding,
    i: BufferBinding,
    draws: Vec<(u32, u32)>,
    z: TextureView,
    e: Extent3d,
    next: usize,
    done: [Option<CompletionPoint>; FRAMES],
    sampler: usize,
    mips: u32,
}
impl Work {
    async fn new(
        d: &Device,
        fmt: TextureFormat,
        e: Extent3d,
        l: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let m = gltf::load_embedded_model(MODEL, gltf::LoadOptions::PRETRANSFORM_FLIP_Y)
            .map_err(asset)?;
        let k = ktx(IMAGE).map_err(asset)?;
        let layout = d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
                BindingKind::UniformBuffer { min_size: UBO },
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
        ]))?;
        let pi =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        let p = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &art(ShaderStage::Vertex, "vs_main", vs(), [9; 32]),
                    )
                    .await?,
                    pi,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &art(ShaderStage::Fragment, "fs_main", fs(), [10; 32]),
                    )
                    .await?,
                )
                .with_vertex_input(input())
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
        let vb = d.create_buffer(&BufferDescriptor::new(
            m.vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let ib = d.create_buffer(&BufferDescriptor::new(
            m.indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let vl = m.vertices.len() as u64;
        let il = m.indices.len() as u64;
        let draws = m
            .primitives
            .iter()
            .map(|p| (p.first_index, p.index_count))
            .collect();
        let u = [ub(d, 0)?, ub(d, 1)?];
        let t = d.create_texture(
            &TextureDescriptor::new_2d(
                k.w,
                k.h,
                TextureFormat::Rgba8Unorm,
                TextureUsage::SAMPLED
                    .union(TextureUsage::COPY_DST)
                    .union(TextureUsage::COPY_SRC),
            )
            .with_mip_levels(k.mips),
        )?;
        let view = d.create_texture_view(
            &t,
            &TextureViewDescriptor::whole(&t, TextureViewDimension::D2)?,
        )?;
        let ss = samplers(d, k.mips)?;
        let g = [
            group3(d, &layout, &u[0], &view, &ss)?,
            group3(d, &layout, &u[1], &view, &ss)?,
        ];
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            vb.clone(),
            0,
            m.vertices,
        ))?)?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            ib.clone(),
            0,
            m.indices,
        ))?)?;
        r.encode_upload(&d.create_texture_upload(TextureUploadDescriptor::new(
            t.clone(),
            sub(0),
            Origin3d { x: 0, y: 0, z: 0 },
            Extent3d::d2(k.w, k.h),
            HostTexelLayout {
                bytes_per_row: k.w * 4,
                rows_per_image: k.h,
            },
            k.base,
        ))?)?;
        for x in 1..k.mips {
            r.blit_texture(&TextureBlit {
                src: t.clone(),
                src_subresource: sub(x - 1),
                src_origin: Origin3d { x: 0, y: 0, z: 0 },
                src_extent: Extent3d::d2((k.w >> (x - 1)).max(1), (k.h >> (x - 1)).max(1)),
                dst: t.clone(),
                dst_subresource: sub(x),
                dst_origin: Origin3d { x: 0, y: 0, z: 0 },
                dst_extent: Extent3d::d2((k.w >> x).max(1), (k.h >> x).max(1)),
                filter: BlitFilter::Linear,
            })?
        }
        let mut plan = SubmissionPlanBuilder::new(d);
        plan.add_batch(l, vec![r.finish()?])?;
        let rec = d.submit(plan.build()?)?;
        let _ = d.wait_completion(rec.completion()).await?;
        Ok(Self {
            p,
            g,
            u,
            v: BufferBinding::new(vb, BufferRange::new(0, vl)),
            i: BufferBinding::new(ib, BufferRange::new(0, il)),
            draws,
            z: depth(d, e)?,
            e,
            next: 0,
            done: [None, None],
            sampler: std::env::var("FLUXEL_RHI_MIP_SAMPLER")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(2)
                .min(2),
            mips: k.mips,
        })
    }
    fn resize(&mut self, d: &Device, e: Extent3d) -> RhiResult<()> {
        self.e = e;
        self.z = depth(d, e)?;
        Ok(())
    }
    async fn render(
        &mut self,
        d: &Device,
        l: SubmissionLaneId,
        f: AcquiredFrame,
        t: f32,
    ) -> RhiResult<()> {
        let n = self.next;
        if let Some(x) = self.done[n].take() {
            let _ = d.wait_completion(x).await?;
        }
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            self.u[n].clone(),
            0,
            Uni::new(self.e, t, self.mips).bytes(),
        ))?)?;
        {
            let mut q = r.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(f.attachment()),
                            load: LoadOp::Clear(ColorClearValue::Float([0., 0., 0., 1.])),
                            store: StoreOp::Store,
                            resolve: None,
                            depth_slice: None,
                        },
                    )
                    .with_depth_stencil(DepthStencilAttachment {
                        view: self.z.clone(),
                        depth: Some(DepthAttachmentMode::ReadWrite {
                            load: LoadOp::Clear(1.),
                            store: StoreOp::Discard,
                        }),
                        stencil: None,
                    }),
            )?;
            q.set_pipeline(&self.p)?;
            q.set_viewport(Viewport::new(
                0.,
                0.,
                self.e.width as f32,
                self.e.height as f32,
                0.,
                1.,
            ))?;
            q.set_scissor(Rect::new(0, 0, self.e.width, self.e.height))?;
            q.set_bind_group(BindGroupIndex::new(0), &self.g[n][self.sampler], &[])?;
            q.set_vertex_buffer(0, &self.v)?;
            q.set_index_buffer(&self.i, IndexFormat::Uint32)?;
            for (a, b) in &self.draws {
                q.draw_indexed(*a..*a + *b, 0, 0..1)?
            }
            q.end()?
        };
        let mut p = SubmissionPlanBuilder::new(d);
        let point = p.add_batch(l, vec![r.finish()?])?;
        p.present_after(f, point)?;
        let rec = d.submit(p.build()?)?;
        self.done[n] = Some(rec.completion());
        let _ = d.wait_present(rec.presents()[0].id()).await?;
        self.next = (n + 1) % FRAMES;
        Ok(())
    }
}
fn sub(mip: u32) -> TextureSubresourceLayers {
    TextureSubresourceLayers {
        aspect: TextureAspect::Color,
        mip_level: mip,
        base_layer: 0,
        layer_count: 1,
    }
}
fn ub(d: &Device, n: usize) -> RhiResult<Buffer> {
    d.create_buffer(
        &BufferDescriptor::new(UBO, BufferUsage::UNIFORM.union(BufferUsage::COPY_DST))
            .with_label(format!("09 uniform {n}")),
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
fn samplers(d: &Device, m: u32) -> RhiResult<[Sampler; 3]> {
    let base = SamplerDescriptor::new()
        .with_address_modes(
            AddressMode::MirrorRepeat,
            AddressMode::MirrorRepeat,
            AddressMode::MirrorRepeat,
        )
        .with_filters(FilterMode::Linear, FilterMode::Linear, FilterMode::Linear);
    let a = d.create_sampler(&base.clone().with_lod_clamp(0., 0.))?;
    let b = d.create_sampler(&base.clone().with_lod_clamp(0., m as f32))?;
    let c = if d
        .capabilities()
        .supports_feature(OptionalFeature::SamplerAnisotropy)
    {
        d.create_sampler(
            &base.with_lod_clamp(0., m as f32).with_max_anisotropy(
                d.capabilities()
                    .limit(LimitKey::MaxSamplerAnisotropy)
                    .unwrap_or(1)
                    .min(u16::MAX as u64) as u16,
            ),
        )?
    } else {
        d.create_sampler(&base.with_lod_clamp(0., m as f32))?
    };
    Ok([a, b, c])
}
fn one_group(
    d: &Device,
    l: &BindGroupLayout,
    u: &Buffer,
    v: &TextureView,
    s: &Sampler,
) -> RhiResult<BindGroup> {
    d.create_bind_group(
        &BindGroupDescriptor::new(l.clone())
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(0),
                BindingResource::Buffer(BufferBinding::new(u.clone(), BufferRange::new(0, UBO))),
            ))
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(1),
                BindingResource::Texture(v.clone()),
            ))
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(2),
                BindingResource::Sampler(s.clone()),
            )),
    )
}
fn group3(
    d: &Device,
    l: &BindGroupLayout,
    u: &Buffer,
    v: &TextureView,
    s: &[Sampler; 3],
) -> RhiResult<[BindGroup; 3]> {
    Ok([
        one_group(d, l, u, v, &s[0])?,
        one_group(d, l, u, v, &s[1])?,
        one_group(d, l, u, v, &s[2])?,
    ])
}
struct K {
    w: u32,
    h: u32,
    mips: u32,
    base: Vec<u8>,
}
fn ktx(b: &[u8]) -> Result<K, String> {
    if b.len() < 68 || b[..12] != KTX_ID {
        return Err("expected little-endian RGBA8 KTX1".into());
    }
    let w = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("ktx word"));
    if w(12) != 0x04030201
        || w(16) != 0x1401
        || w(20) != 1
        || w(24) != 0x1908
        || w(28) != 0x8058
        || w(32) != 0x1908
    {
        return Err("KTX is not RGBA8".into());
    }
    let (width, height, depth, array, faces, levels, key) =
        (w(36), w(40), w(44), w(48), w(52), w(56), w(60));
    if width == 0 || height == 0 || depth != 0 || array != 0 || faces != 1 || levels != 1 {
        return Err("source KTX must contain exactly one 2D mip".into());
    }
    let p = 64usize + key as usize;
    let bytes = w(p) as usize;
    let data = b
        .get(p + 4..p + 4 + bytes)
        .ok_or("truncated KTX base image")?;
    if bytes != width as usize * height as usize * 4 {
        return Err("KTX base image has wrong byte length".into());
    }
    let mips = 1 + (width.max(height)).ilog2();
    Ok(K {
        w: width,
        h: height,
        mips,
        base: data.to_vec(),
    })
}
struct Uni {
    p: [f32; 16],
    v: [f32; 16],
    m: [f32; 16],
    eye: [f32; 4],
    bias: f32,
}
impl Uni {
    fn new(e: Extent3d, t: f32, mips: u32) -> Self {
        let a = e.width.max(1) as f32 / e.height.max(1) as f32;
        let f = 1.0 / 30f32.to_radians().tan();
        let (n, z) = (0.1, 1024.);
        let r = (t * 360.).to_radians();
        let (s, c) = r.sin_cos();
        let model = [1., 0., 0., 0., 0., c, s, 0., 0., -s, c, 0., 0., 0., 0., 1.];
        let y = 90f32.to_radians();
        let (sy, cy) = y.sin_cos();
        let rotation = [
            cy, 0., -sy, 0., 0., 1., 0., 0., sy, 0., cy, 0., 0., 0., 0., 1.,
        ];
        let translation = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 40.75, 0., 0., 1.,
        ];
        let view = mul(rotation, translation);
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
                z / (n - z),
                -1.,
                0.,
                0.,
                z * n / (n - z),
                0.,
            ],
            v: view,
            m: model,
            eye: [-40.75, 0., 0., 0.],
            bias: std::env::var("FLUXEL_RHI_MIP_LOD_BIAS")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0.)
                .clamp(0., mips as f32),
        }
    }
    fn bytes(self) -> Vec<u8> {
        self.p
            .into_iter()
            .chain(self.v)
            .chain(self.m)
            .chain(self.eye)
            .chain([self.bias, 0., 0., 0.])
            .flat_map(f32::to_le_bytes)
            .collect()
    }
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
                VertexFormat::Float32x2,
                24,
            ))
            .with_attribute(VertexAttribute::new(
                ShaderLocation::new(2),
                VertexFormat::Float32x3,
                12,
            )),
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
fn io(n: u32, c: u8, i: Option<ShaderInterpolation>) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(n),
        numeric_type: ShaderNumericType::Float32,
        components: c,
        interpolation: i,
    }
}
fn interp() -> ShaderInterpolation {
    ShaderInterpolation {
        mode: InterpolationMode::Perspective,
        sampling: InterpolationSampling::Center,
    }
}
fn vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(0, BindingKind::UniformBuffer { min_size: UBO }))
        .with_input(io(0, 3, None))
        .with_input(io(1, 2, None))
        .with_input(io(2, 3, None))
        .with_output(io(0, 2, Some(interp())))
        .with_output(io(1, 1, Some(interp())))
        .with_output(io(2, 3, Some(interp())))
        .with_output(io(3, 3, Some(interp())))
        .with_output(io(4, 3, Some(interp())))
        .with_writes_position(true)
}
fn fs() -> ShaderInterface {
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
        .with_input(io(0, 2, Some(interp())))
        .with_input(io(1, 1, Some(interp())))
        .with_input(io(2, 3, Some(interp())))
        .with_input(io(3, 3, Some(interp())))
        .with_input(io(4, 3, Some(interp())))
        .with_output(io(0, 4, None))
}
fn art(s: ShaderStage, e: &'static str, i: ShaderInterface, h: [u8; 32]) -> ShaderArtifact {
    ShaderArtifact::new(
        s,
        e,
        ShaderCode::Wgsl(Arc::from(WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        i,
        ShaderRequirements::new(),
        ArtifactHash(h),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
fn asset(x: impl Into<String>) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, x.into())
}
