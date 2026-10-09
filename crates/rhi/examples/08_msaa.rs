//! Faithful port of SaschaWillems/Vulkan `examples/multisampling`.
//!
//! It loads the original `voyager.gltf`, decodes every embedded PNG exactly as
//! `vkglTF::Texture::fromglTfImage` does, builds its mip chain by linear GPU
//! blits, and renders each primitive through its glTF material texture into a
//! multisampled colour/depth pair resolved directly to the acquired frame.

pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use std::{io::Cursor, sync::Arc};

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
    format::{TextureFormat, TextureSupportQuery},
    pipeline::{
        ColorTargetState, CullMode, DepthState, DepthStencilState, MultisampleState,
        PipelineInterfaceDescriptor, PrimitiveState, PrimitiveTopology, RasterPipeline,
        RasterPipelineDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat,
        VertexInputState, VertexStepMode,
    },
    platform::{Device, OptionalFeature},
    presentation::AcquiredFrame,
    resource::{
        AddressMode, Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor,
        BufferUsage, CompareFunction, Extent3d, FilterMode, HostTexelLayout, Origin3d, Sampler,
        SamplerDescriptor, Texture, TextureAspect, TextureDescriptor, TextureDimension,
        TextureSubresourceLayers, TextureUploadDescriptor, TextureUsage, TextureView,
        TextureViewDescriptor, TextureViewDimension,
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
const UBO_BYTES: u64 = 144; // projection + model + light position
const VOYAGER: &[u8] = include_bytes!("assets/models/voyager.gltf");

// Direct WGSL transcription of shaders/glsl/multisampling/mesh.{vert,frag}.
const SHADER: &str = r#"
struct U { projection:mat4x4<f32>, model:mat4x4<f32>, light:vec4<f32>, };
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var image:texture_2d<f32>;
@group(0) @binding(2) var samp:sampler;
struct I { @location(0) pos:vec3<f32>, @location(1) normal:vec3<f32>, @location(2) uv:vec2<f32>, @location(3) color:vec3<f32>, };
struct O { @builtin(position) pos:vec4<f32>, @location(0) normal:vec3<f32>, @location(1) color:vec3<f32>, @location(2) uv:vec2<f32>, @location(3) view_vec:vec3<f32>, @location(4) light_vec:vec3<f32>, };
@vertex fn vs_main(i:I)->O { var o:O; o.color=i.color;o.uv=i.uv;o.pos=u.projection*u.model*vec4<f32>(i.pos,1);let p=u.model*vec4<f32>(i.pos,1);o.normal=mat3x3<f32>(u.model[0].xyz,u.model[1].xyz,u.model[2].xyz)*i.normal;let l=mat3x3<f32>(u.model[0].xyz,u.model[1].xyz,u.model[2].xyz)*u.light.xyz;o.light_vec=l-p.xyz;o.view_vec=-p.xyz;return o; }
@fragment fn fs_main(i:O)->@location(0) vec4<f32> { let c=textureSample(image,samp,i.uv)*vec4<f32>(i.color,1);let n=normalize(i.normal);let l=normalize(i.light_vec);let v=normalize(i.view_vec);let r=reflect(-l,n);let d=max(dot(n,l),0.15)*i.color;let s=pow(max(dot(r,v),0),16)*vec3<f32>(0.75);return vec4<f32>(d*c.rgb+s,1); }
"#;

fn main() {
    if let Err(e) = common::run_example("Multisampling", create_example()) {
        eprintln!("08_msaa: {e}");
        std::process::exit(1)
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
            .ok_or_else(|| std::io::Error::other("08_msaa requires COPY|RASTER lane"))?;
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
        };
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

struct Draw {
    first: u32,
    count: u32,
    group: usize,
}
struct Workload {
    pipe: RasterPipeline,
    groups: [Vec<BindGroup>; FRAMES],
    ubo: [Buffer; FRAMES],
    vertex: BufferBinding,
    index: BufferBinding,
    draws: Vec<Draw>,
    color: TextureView,
    depth: TextureView,
    extent: Extent3d,
    samples: u32,
    next: usize,
    done: [Option<CompletionPoint>; FRAMES],
}
impl Workload {
    async fn new(
        d: &Device,
        frame_format: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let model = gltf::load_embedded_model(VOYAGER, gltf::LoadOptions::PRETRANSFORM_FLIP_Y)
            .map_err(asset)?;
        let samples = highest_samples(d, frame_format)?;
        let layout = d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
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
        ]))?;
        let interface =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        // This is the C++ UI checkbox. `fluxel-host` has no input callback yet, so
        // retain the opt-in branch as a startup setting and only form the pipeline
        // state when the enabled device feature permits it.
        let sample_shading = std::env::var("FLUXEL_RHI_MSAA_SAMPLE_SHADING").as_deref() == Ok("1")
            && d.capabilities()
                .supports_feature(OptionalFeature::MultisampledShading);
        let msaa = if sample_shading {
            MultisampleState::new(samples).with_sample_shading(0.25)
        } else {
            MultisampleState::new(samples)
        };
        let pipe = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(
                        d,
                        &artifact(ShaderStage::Vertex, "vs_main", vs(), [8; 32]),
                    )
                    .await?,
                    interface,
                )
                .with_fragment(
                    common::shader::create_shader(
                        d,
                        &artifact(ShaderStage::Fragment, "fs_main", fs(), [9; 32]),
                    )
                    .await?,
                )
                .with_vertex_input(vertex_input())
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Back),
                )
                .with_depth_stencil(
                    DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                        DepthState::new(CompareFunction::LessEqual).with_write_enabled(true),
                    ),
                )
                .with_multisample(msaa)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(frame_format)),
            )
            .await?;
        let vb = d.create_buffer(
            &BufferDescriptor::new(
                model.vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("08_msaa voyager vertices"),
        )?;
        let ib = d.create_buffer(
            &BufferDescriptor::new(
                model.indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label("08_msaa voyager indices"),
        )?;
        let ubo = [uniform(d, 0)?, uniform(d, 1)?];
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        let vertex_len = model.vertices.len() as u64;
        let index_len = model.indices.len() as u64;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            vb.clone(),
            0,
            model.vertices,
        ))?)?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            ib.clone(),
            0,
            model.indices,
        ))?)?;
        let mut views = Vec::with_capacity(model.images.len());
        for (i, image) in model.images.iter().enumerate() {
            let decoded = decode_png(image)?;
            let (texture, view) = upload_png_mips(d, &mut r, i, decoded)?;
            let _ = texture;
            views.push(view);
        }
        let sampler_desc = SamplerDescriptor::new()
            .with_address_modes(
                AddressMode::MirrorRepeat,
                AddressMode::MirrorRepeat,
                AddressMode::MirrorRepeat,
            )
            .with_filters(FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
            .with_lod_clamp(0., f32::MAX);
        let sampler = d.create_sampler(
            if d.capabilities()
                .supports_feature(OptionalFeature::SamplerAnisotropy)
            {
                &sampler_desc.with_max_anisotropy(8)
            } else {
                &sampler_desc
            },
        )?;
        let fallback = views
            .first()
            .ok_or_else(|| {
                RhiError::new(RhiErrorKind::InvalidUsage, "voyager.gltf contains no image")
            })?
            .clone();
        let mut material_views: Vec<TextureView> = model
            .materials
            .iter()
            .map(|m| {
                m.base_color_texture
                    .and_then(|t| model.textures.get(t))
                    .and_then(|t| views.get(t.image_index))
                    .cloned()
                    .unwrap_or_else(|| fallback.clone())
            })
            .collect();
        if material_views.is_empty() {
            material_views.push(fallback);
        }
        let groups = [
            groups(d, &layout, &ubo[0], &material_views, &sampler)?,
            groups(d, &layout, &ubo[1], &material_views, &sampler)?,
        ];
        let draws = model
            .primitives
            .iter()
            .map(|p| Draw {
                first: p.first_index,
                count: p.index_count,
                group: p.material_index.unwrap_or(0).min(material_views.len() - 1),
            })
            .collect();
        let mut plan = SubmissionPlanBuilder::new(d);
        plan.add_batch(lane, vec![r.finish()?])?;
        let receipt = d.submit(plan.build()?)?;
        let _ = d.wait_completion(receipt.completion()).await?;
        let (color, depth) = targets(d, extent, frame_format, samples)?;
        Ok(Self {
            pipe,
            groups,
            ubo,
            vertex: BufferBinding::new(vb, BufferRange::new(0, vertex_len)),
            index: BufferBinding::new(ib, BufferRange::new(0, index_len)),
            draws,
            color,
            depth,
            extent,
            samples,
            next: 0,
            done: [None, None],
        })
    }
    fn resize(&mut self, d: &Device, e: Extent3d) -> RhiResult<()> {
        let (c, z) = targets(d, e, self.color.format(), self.samples)?;
        self.color = c;
        self.depth = z;
        self.extent = e;
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
        };
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            self.ubo[n].clone(),
            0,
            Uniform::new(self.extent).bytes(),
        ))?)?;
        {
            let mut q = r.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Texture(self.color.clone()),
                            load: LoadOp::Clear(ColorClearValue::Float([1., 1., 1., 1.])),
                            store: StoreOp::Discard,
                            resolve: Some(ColorAttachmentView::Frame(f.attachment())),
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
            q.set_pipeline(&self.pipe)?;
            q.set_viewport(Viewport::new(
                0.,
                0.,
                self.extent.width as f32,
                self.extent.height as f32,
                0.,
                1.,
            ))?;
            q.set_scissor(Rect::new(0, 0, self.extent.width, self.extent.height))?;
            q.set_vertex_buffer(0, &self.vertex)?;
            q.set_index_buffer(&self.index, IndexFormat::Uint32)?;
            for x in &self.draws {
                q.set_bind_group(BindGroupIndex::new(0), &self.groups[n][x.group], &[])?;
                q.draw_indexed(x.first..x.first + x.count, 0, 0..1)?
            }
            q.end()?
        };
        let mut p = SubmissionPlanBuilder::new(d);
        let point = p.add_batch(lane, vec![r.finish()?])?;
        p.present_after(f, point)?;
        let receipt = d.submit(p.build()?)?;
        self.done[n] = Some(receipt.completion());
        let _ = d
            .wait_present(receipt.presents().first().expect("present").id())
            .await?;
        self.next = (n + 1) % FRAMES;
        Ok(())
    }
}

fn highest_samples(d: &Device, fmt: TextureFormat) -> RhiResult<u32> {
    for count in [64, 32, 16, 8, 4, 2, 1] {
        let c = TextureSupportQuery::new(
            TextureDimension::D2,
            fmt,
            TextureUsage::COLOR_ATTACHMENT,
            count,
        );
        let z = TextureSupportQuery::new(
            TextureDimension::D2,
            TextureFormat::Depth32Float,
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
            count,
        );
        if d.capabilities().texture_support(&c).is_supported()
            && d.capabilities().texture_support(&z).is_supported()
        {
            return Ok(count);
        }
    }
    Err(RhiError::new(
        RhiErrorKind::Unsupported,
        "no common colour/depth sample count",
    ))
}
fn targets(
    d: &Device,
    e: Extent3d,
    fmt: TextureFormat,
    s: u32,
) -> RhiResult<(TextureView, TextureView)> {
    let c = d.create_texture(
        &TextureDescriptor::new_2d(e.width, e.height, fmt, TextureUsage::COLOR_ATTACHMENT)
            .with_sample_count(s),
    )?;
    let z = d.create_texture(
        &TextureDescriptor::new_2d(
            e.width,
            e.height,
            TextureFormat::Depth32Float,
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        )
        .with_sample_count(s),
    )?;
    Ok((
        d.create_texture_view(
            &c,
            &TextureViewDescriptor::whole(&c, TextureViewDimension::D2)?,
        )?,
        d.create_texture_view(
            &z,
            &TextureViewDescriptor::whole(&z, TextureViewDimension::D2)?,
        )?,
    ))
}
fn uniform(d: &Device, n: usize) -> RhiResult<Buffer> {
    d.create_buffer(
        &BufferDescriptor::new(UBO_BYTES, BufferUsage::UNIFORM.union(BufferUsage::COPY_DST))
            .with_label(format!("08_msaa uniform {n}")),
    )
}
fn groups(
    d: &Device,
    l: &BindGroupLayout,
    u: &Buffer,
    v: &[TextureView],
    s: &Sampler,
) -> RhiResult<Vec<BindGroup>> {
    v.iter()
        .map(|x| {
            d.create_bind_group(
                &BindGroupDescriptor::new(l.clone())
                    .with_entry(BindGroupEntry::new(
                        BindingSlotId::new(0),
                        BindingResource::Buffer(BufferBinding::new(
                            u.clone(),
                            BufferRange::new(0, UBO_BYTES),
                        )),
                    ))
                    .with_entry(BindGroupEntry::new(
                        BindingSlotId::new(1),
                        BindingResource::Texture(x.clone()),
                    ))
                    .with_entry(BindGroupEntry::new(
                        BindingSlotId::new(2),
                        BindingResource::Sampler(s.clone()),
                    )),
            )
        })
        .collect()
}

struct Png {
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}
fn decode_png(image: &gltf::EmbeddedImage) -> RhiResult<Png> {
    if image.mime_type != "image/png" {
        return Err(asset(format!(
            "voyager image is {}, expected PNG",
            image.mime_type
        )));
    }
    let mut decoder = png::Decoder::new(Cursor::new(&image.bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| asset(e.to_string()))?;
    let mut bytes = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut bytes)
        .map_err(|e| asset(e.to_string()))?;
    let data = &bytes[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => data.to_vec(),
        png::ColorType::Rgb => data
            .chunks_exact(3)
            .flat_map(|x| [x[0], x[1], x[2], 255])
            .collect(),
        _ => return Err(asset("PNG decoder did not produce RGB/RGBA pixels")),
    };
    Ok(Png {
        w: info.width,
        h: info.height,
        rgba,
    })
}
fn mip_count(mut w: u32, mut h: u32) -> u32 {
    let mut n = 1;
    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        n += 1
    }
    n
}
fn upload_png_mips(
    d: &Device,
    r: &mut fluxel_rhi::api::command::Recorder,
    i: usize,
    p: Png,
) -> RhiResult<(Texture, TextureView)> {
    let levels = mip_count(p.w, p.h);
    let t = d.create_texture(
        &TextureDescriptor::new_2d(
            p.w,
            p.h,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED
                .union(TextureUsage::COPY_DST)
                .union(TextureUsage::COPY_SRC),
        )
        .with_mip_levels(levels)
        .with_label(format!("08_msaa voyager image {i}")),
    )?;
    r.encode_upload(&d.create_texture_upload(TextureUploadDescriptor::new(
        t.clone(),
        TextureSubresourceLayers {
            aspect: TextureAspect::Color,
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
        },
        Origin3d { x: 0, y: 0, z: 0 },
        Extent3d::d2(p.w, p.h),
        HostTexelLayout {
            bytes_per_row: p.w * 4,
            rows_per_image: p.h,
        },
        p.rgba,
    ))?)?;
    for level in 1..levels {
        let sw = (p.w >> (level - 1)).max(1);
        let sh = (p.h >> (level - 1)).max(1);
        let dw = (p.w >> level).max(1);
        let dh = (p.h >> level).max(1);
        r.blit_texture(&TextureBlit {
            src: t.clone(),
            src_subresource: TextureSubresourceLayers {
                aspect: TextureAspect::Color,
                mip_level: level - 1,
                base_layer: 0,
                layer_count: 1,
            },
            src_origin: Origin3d { x: 0, y: 0, z: 0 },
            src_extent: Extent3d::d2(sw, sh),
            dst: t.clone(),
            dst_subresource: TextureSubresourceLayers {
                aspect: TextureAspect::Color,
                mip_level: level,
                base_layer: 0,
                layer_count: 1,
            },
            dst_origin: Origin3d { x: 0, y: 0, z: 0 },
            dst_extent: Extent3d::d2(dw, dh),
            filter: BlitFilter::Linear,
        })?
    }
    let v = d.create_texture_view(
        &t,
        &TextureViewDescriptor::whole(&t, TextureViewDimension::D2)?,
    )?;
    Ok((t, v))
}

struct Uniform {
    p: [f32; 16],
    m: [f32; 16],
    l: [f32; 4],
}
impl Uniform {
    fn new(e: Extent3d) -> Self {
        let a = e.width.max(1) as f32 / e.height.max(1) as f32;
        let f = 1. / 30f32.to_radians().tan();
        let (n, far) = (0.1, 256.);
        let r = (-90f32).to_radians();
        let (c, s) = r.cos_sin();
        let rot = [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.];
        let tr = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 2.5, 2.5, -7.5, 1.,
        ];
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
                far / (n - far),
                -1.,
                0.,
                0.,
                far * n / (n - far),
                0.,
            ],
            m: mm(tr, rot),
            l: [5., -5., 5., 1.],
        }
    }
    fn bytes(self) -> Vec<u8> {
        self.p
            .into_iter()
            .chain(self.m)
            .chain(self.l)
            .flat_map(f32::to_le_bytes)
            .collect()
    }
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
trait CosSin {
    fn cos_sin(self) -> (f32, f32);
}
impl CosSin for f32 {
    fn cos_sin(self) -> (f32, f32) {
        (self.cos(), self.sin())
    }
}
fn vertex_input() -> VertexInputState {
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
fn rr(slot: u32, kind: BindingKind) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(slot),
        kind,
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
fn input(n: u32, c: u8) -> ShaderLocationInterface {
    ShaderLocationInterface {
        interpolation: None,
        ..io(n, c)
    }
}
fn vs() -> ShaderInterface {
    ShaderInterface::new()
        .with_resource(rr(
            0,
            BindingKind::UniformBuffer {
                min_size: UBO_BYTES,
            },
        ))
        .with_input(input(0, 3))
        .with_input(input(1, 3))
        .with_input(input(2, 2))
        .with_input(input(3, 3))
        .with_output(io(0, 3))
        .with_output(io(1, 3))
        .with_output(io(2, 2))
        .with_output(io(3, 3))
        .with_output(io(4, 3))
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
fn artifact(s: ShaderStage, e: &'static str, i: ShaderInterface, h: [u8; 32]) -> ShaderArtifact {
    ShaderArtifact::new(
        s,
        e,
        ShaderCode::Wgsl(Arc::from(SHADER)),
        ShaderAbiVersion { major: 1, minor: 0 },
        i,
        ShaderRequirements::new(),
        ArtifactHash(h),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
fn asset(message: impl Into<String>) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, message.into())
}
