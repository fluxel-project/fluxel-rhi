//! Faithful port of SaschaWillems/Vulkan `examples/computeshader`.
//!
//! The source `vulkan_11_rgba.ktx` appears on the left; a 16x16 storage-texture
//! compute pass writes the selected convolution to the right.

pub mod common;

use std::sync::Arc;

use fluxel_rhi::api::{
    binding::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayout,
        BindGroupLayoutDescriptor, BindingCount, BindingKind, BindingResource, BindingSlot,
        BindingSlotId, SamplerKind, StorageAccess, TextureSampleType,
    },
    command::{
        ColorAttachment, ColorAttachmentView, ColorClearValue, ComputeScopeDescriptor,
        DepthAttachmentMode, DepthStencilAttachment, IndexFormat, LoadOp, RasterScopeDescriptor,
        RecorderDescriptor, Rect, StoreOp, Viewport,
    },
    error::{RhiError, RhiErrorKind, RhiResult},
    format::TextureFormat,
    pipeline::{
        ColorTargetState, ComputePipeline, ComputePipelineDescriptor, DepthState,
        DepthStencilState, PipelineInterfaceDescriptor, PrimitiveState, PrimitiveTopology,
        RasterPipeline, RasterPipelineDescriptor, VertexAttribute, VertexBufferLayout,
        VertexFormat, VertexInputState, VertexStepMode,
    },
    platform::Device,
    presentation::AcquiredFrame,
    resource::{
        AddressMode, Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor,
        BufferUsage, CompareFunction, Extent3d, FilterMode, HostTexelLayout, Origin3d, Sampler,
        SamplerDescriptor, Texture, TextureAspect, TextureDescriptor, TextureSubresourceLayers,
        TextureUploadDescriptor, TextureUsage, TextureView, TextureViewDescriptor,
        TextureViewDimension,
    },
    shader::{
        ArtifactHash, ArtifactProducerVersion, ComputeWorkgroupSize, InterpolationMode,
        InterpolationSampling, ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderInterface,
        ShaderInterpolation, ShaderLocation, ShaderLocationInterface, ShaderNumericType,
        ShaderRequirements, ShaderResourceRequirement, ShaderStage, ShaderStages,
    },
    submission::{CompletionPoint, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder},
};

const FRAMES: usize = 2;
const UBO_BYTES: u64 = 128;
const KTX: &[u8] = include_bytes!("assets/textures/vulkan_11_rgba.ktx");
const KTX_MAGIC: [u8; 12] = [
    0xabu8, b'K', b'T', b'X', b' ', b'1', b'1', 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
];

const GRAPHICS_WGSL: &str = r#"
struct Ubo { projection: mat4x4<f32>, model: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> ubo: Ubo;
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;
struct In { @location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>, };
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, };
@vertex fn vs_main(input: In) -> Out { var out: Out; out.uv=input.uv; out.pos=ubo.projection*ubo.model*vec4<f32>(input.pos,1.0); return out; }
@fragment fn fs_main(input: Out) -> @location(0) vec4<f32> { return textureSample(image,image_sampler,input.uv); }
"#;

// The three entry points exactly retain the C++ kernels and their `rgba8` storage contract.
const COMPUTE_WGSL: &str = r#"
@group(0) @binding(0) var input_image: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1) var result_image: texture_storage_2d<rgba8unorm, write>;
fn avg(p: vec2<i32>) -> f32 { let c=textureLoad(input_image,p).rgb; return (c.r+c.g+c.b)/3.0; }
fn conv(k: array<f32,9>, d: array<f32,9>, denom:f32, offset:f32) -> f32 { var r=0.0; for(var i=0;i<9;i=i+1){r=r+k[i]*d[i];} return clamp(r/denom+offset,0.0,1.0); }
fn gray_data(p: vec2<i32>) -> array<f32,9> { var d:array<f32,9>; var n=0; for(var i=-1;i<2;i=i+1){for(var j=-1;j<2;j=j+1){d[n]=avg(p+vec2<i32>(i,j));n=n+1;}} return d; }
@compute @workgroup_size(16,16,1) fn emboss(@builtin(global_invocation_id) gid:vec3<u32>) { let p=vec2<i32>(gid.xy); let d=gray_data(p); let k=array<f32,9>(-1.,0.,0.,0.,-1.,0.,0.,0.,2.); let v=conv(k,d,1.,.5); textureStore(result_image,p,vec4<f32>(vec3<f32>(v),1.)); }
@compute @workgroup_size(16,16,1) fn edgedetect(@builtin(global_invocation_id) gid:vec3<u32>) { let p=vec2<i32>(gid.xy); let d=gray_data(p); let k=array<f32,9>(-.125,-.125,-.125,-.125,1.,-.125,-.125,-.125,-.125); let v=conv(k,d,.1,0.); textureStore(result_image,p,vec4<f32>(vec3<f32>(v),1.)); }
@compute @workgroup_size(16,16,1) fn sharpen(@builtin(global_invocation_id) gid:vec3<u32>) { let p=vec2<i32>(gid.xy); var r:array<f32,9>;var g:array<f32,9>;var b:array<f32,9>;var n=0; for(var i=-1;i<2;i=i+1){for(var j=-1;j<2;j=j+1){let c=textureLoad(input_image,p+vec2<i32>(i,j)).rgb;r[n]=c.r;g[n]=c.g;b[n]=c.b;n=n+1;}} let k=array<f32,9>(-1.,-1.,-1.,-1.,9.,-1.,-1.,-1.,-1.); textureStore(result_image,p,vec4<f32>(conv(k,r,1.,0.),conv(k,g,1.,0.),conv(k,b,1.,0.),1.)); }
"#;

fn main() {
    if let Err(e) = common::run_example("Compute shader image load/store", create_example()) {
        eprintln!("06_compute: {e}");
        std::process::exit(1)
    }
}
pub fn create_example() -> ComputeExample {
    ComputeExample {
        workload: None,
        lane: None,
    }
}
pub struct ComputeExample {
    workload: Option<ComputeWorkload>,
    lane: Option<SubmissionLaneId>,
}
impl common::Example for ComputeExample {
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
                x.domains().contains(
                    LaneWorkDomains::RASTER
                        .union(LaneWorkDomains::COMPUTE)
                        .union(LaneWorkDomains::COPY),
                )
            })
            .map(|x| x.id())
            .ok_or_else(|| {
                std::io::Error::other("06_compute requires one COPY|COMPUTE|RASTER lane")
            })?;
        let e = c.extent();
        self.workload = Some(common::block_on(ComputeWorkload::new(
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
        let w = self
            .workload
            .as_mut()
            .ok_or_else(|| std::io::Error::other("compute workload is not initialized"))?;
        common::block_on(
            w.render(
                c.device(),
                self.lane
                    .ok_or_else(|| std::io::Error::other("compute lane is not initialized"))?,
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
        if let Some(x) = &mut self.workload {
            x.resize(c.device(), Extent3d::d2(w, h))?
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

struct ComputeWorkload {
    graphics: RasterPipeline,
    compute: [ComputePipeline; 3],
    graphics_groups: [[BindGroup; 2]; FRAMES],
    compute_group: BindGroup,
    uniforms: [Buffer; FRAMES],
    vertex: BufferBinding,
    index: BufferBinding,
    depth: TextureView,
    extent: Extent3d,
    tex_extent: Extent3d,
    filter: usize,
    next: usize,
    done: [Option<CompletionPoint>; FRAMES],
}
impl ComputeWorkload {
    async fn new(
        d: &Device,
        fmt: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let k = parse_ktx(KTX).map_err(|e| RhiError::new(RhiErrorKind::InvalidUsage, e))?;
        let gl = d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
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
        let cl = d.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::COMPUTE,
                BindingKind::StorageTexture {
                    dimension: TextureViewDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    access: StorageAccess::ReadOnly,
                },
            ),
            BindingSlot::new(
                BindingSlotId::new(1),
                ShaderStages::COMPUTE,
                BindingKind::StorageTexture {
                    dimension: TextureViewDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    access: StorageAccess::WriteOnly,
                },
            ),
        ]))?;
        let gi =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![gl.clone()]))?;
        let ci =
            d.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![cl.clone()]))?;
        let graphics = d
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(d, &graphics_artifact(ShaderStage::Vertex))
                        .await?,
                    gi,
                )
                .with_fragment(
                    common::shader::create_shader(d, &graphics_artifact(ShaderStage::Fragment))
                        .await?,
                )
                .with_vertex_input(
                    VertexInputState::new().with_buffer(
                        VertexBufferLayout::new(20, VertexStepMode::Vertex)
                            .with_attribute(VertexAttribute::new(
                                ShaderLocation::new(0),
                                VertexFormat::Float32x3,
                                0,
                            ))
                            .with_attribute(VertexAttribute::new(
                                ShaderLocation::new(1),
                                VertexFormat::Float32x2,
                                12,
                            )),
                    ),
                )
                .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleList))
                .with_depth_stencil(
                    DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                        DepthState::new(CompareFunction::LessEqual).with_write_enabled(true),
                    ),
                )
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(fmt)),
            )
            .await?;
        let compute = [
            d.create_compute_pipeline(&ComputePipelineDescriptor::new(
                common::shader::create_shader(d, &compute_artifact("emboss", 1)).await?,
                ci.clone(),
            ))
            .await?,
            d.create_compute_pipeline(&ComputePipelineDescriptor::new(
                common::shader::create_shader(d, &compute_artifact("edgedetect", 2)).await?,
                ci.clone(),
            ))
            .await?,
            d.create_compute_pipeline(&ComputePipelineDescriptor::new(
                common::shader::create_shader(d, &compute_artifact("sharpen", 3)).await?,
                ci,
            ))
            .await?,
        ];
        let source = d.create_texture(
            &TextureDescriptor::new_2d(
                k.width,
                k.height,
                TextureFormat::Rgba8Unorm,
                TextureUsage::SAMPLED
                    .union(TextureUsage::STORAGE)
                    .union(TextureUsage::COPY_DST),
            )
            .with_mip_levels(k.levels.len() as u32),
        )?;
        let output = d.create_texture(&TextureDescriptor::new_2d(
            k.width,
            k.height,
            TextureFormat::Rgba8Unorm,
            TextureUsage::SAMPLED.union(TextureUsage::STORAGE),
        ))?;
        let sv = d.create_texture_view(
            &source,
            &TextureViewDescriptor::whole(&source, TextureViewDimension::D2)?,
        )?;
        let ov = d.create_texture_view(
            &output,
            &TextureViewDescriptor::whole(&output, TextureViewDimension::D2)?,
        )?;
        let one = TextureViewDescriptor::new(
            TextureViewDimension::D2,
            fluxel_rhi::api::resource::TextureAspects::COLOR,
            0,
            1,
            0,
            1,
        );
        let ss = d.create_texture_view(&source, &one)?;
        let os = d.create_texture_view(&output, &one)?;
        let sampler = d.create_sampler(
            &SamplerDescriptor::new()
                .with_address_modes(
                    AddressMode::ClampToBorder,
                    AddressMode::ClampToBorder,
                    AddressMode::ClampToBorder,
                )
                .with_filters(FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
                .with_lod_clamp(0., 1.),
        )?;
        let uniforms = [
            d.create_buffer(&BufferDescriptor::new(
                UBO_BYTES,
                BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
            ))?,
            d.create_buffer(&BufferDescriptor::new(
                UBO_BYTES,
                BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
            ))?,
        ];
        let graphics_groups = [
            [
                group_graphics(d, &gl, &uniforms[0], &sv, &sampler)?,
                group_graphics(d, &gl, &uniforms[0], &ov, &sampler)?,
            ],
            [
                group_graphics(d, &gl, &uniforms[1], &sv, &sampler)?,
                group_graphics(d, &gl, &uniforms[1], &ov, &sampler)?,
            ],
        ];
        let compute_group = d.create_bind_group(
            &BindGroupDescriptor::new(cl)
                .with_entry(BindGroupEntry::new(
                    BindingSlotId::new(0),
                    BindingResource::Texture(ss),
                ))
                .with_entry(BindGroupEntry::new(
                    BindingSlotId::new(1),
                    BindingResource::Texture(os),
                )),
        )?;
        let v = quad_vertices();
        let i = quad_indices();
        let vb = d.create_buffer(&BufferDescriptor::new(
            v.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let ib = d.create_buffer(&BufferDescriptor::new(
            i.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(vb.clone(), 0, v))?)?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(ib.clone(), 0, i))?)?;
        for (m, bytes) in k.levels.into_iter().enumerate() {
            let w = (k.width >> m).max(1);
            let h = (k.height >> m).max(1);
            r.encode_upload(&d.create_texture_upload(TextureUploadDescriptor::new(
                source.clone(),
                TextureSubresourceLayers {
                    aspect: TextureAspect::Color,
                    mip_level: m as u32,
                    base_layer: 0,
                    layer_count: 1,
                },
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d::d2(w, h),
                HostTexelLayout {
                    bytes_per_row: w * 4,
                    rows_per_image: h,
                },
                bytes,
            ))?)?;
        }
        let mut p = SubmissionPlanBuilder::new(d);
        p.add_batch(lane, vec![r.finish()?])?;
        let receipt = d.submit(p.build()?)?;
        let _ = d.wait_completion(receipt.completion()).await?;
        Ok(Self {
            graphics,
            compute,
            graphics_groups,
            compute_group,
            uniforms,
            vertex: BufferBinding::new(vb, BufferRange::new(0, 80)),
            index: BufferBinding::new(ib, BufferRange::new(0, 24)),
            depth: depth(d, extent)?,
            extent,
            tex_extent: Extent3d::d2(k.width, k.height),
            filter: filter_from_env(),
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
        f: AcquiredFrame,
    ) -> RhiResult<()> {
        let slot = self.next;
        if let Some(x) = self.done[slot].take() {
            let _ = d.wait_completion(x).await?;
        }
        let mut r = d.create_recorder(&RecorderDescriptor::new())?;
        r.encode_upload(&d.create_buffer_upload(BufferUploadDescriptor::new(
            self.uniforms[slot].clone(),
            0,
            ubo(self.extent),
        )?)?)?;
        {
            let mut c = r.begin_compute(
                &ComputeScopeDescriptor::new().with_label("06_compute convolution"),
            )?;
            c.set_pipeline(&self.compute[self.filter])?;
            c.set_bind_group(BindGroupIndex::new(0), &self.compute_group, &[])?;
            c.dispatch(self.tex_extent.width / 16, self.tex_extent.height / 16, 1)?;
            c.end()?;
        }
        {
            let a = f.attachment();
            let mut q = r.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(a),
                            load: LoadOp::Clear(ColorClearValue::Float([0.025, 0.025, 0.025, 1.0])),
                            store: StoreOp::Store,
                            resolve: None,
                            depth_slice: None,
                        },
                    )
                    .with_depth_stencil(DepthStencilAttachment {
                        view: self.depth.clone(),
                        depth: Some(DepthAttachmentMode::ReadWrite {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Discard,
                        }),
                        stencil: None,
                    }),
            )?;
            q.set_pipeline(&self.graphics)?;
            q.set_scissor(Rect::new(0, 0, self.extent.width, self.extent.height))?;
            q.set_vertex_buffer(0, &self.vertex)?;
            q.set_index_buffer(&self.index, IndexFormat::Uint32)?;
            q.set_viewport(Viewport::new(
                0.0,
                0.0,
                self.extent.width as f32 * 0.5,
                self.extent.height as f32,
                0.0,
                1.0,
            ))?;
            q.set_bind_group(BindGroupIndex::new(0), &self.graphics_groups[slot][0], &[])?;
            q.draw_indexed(0..6, 0, 0..1)?;
            q.set_viewport(Viewport::new(
                self.extent.width as f32 * 0.5,
                0.0,
                self.extent.width as f32 * 0.5,
                self.extent.height as f32,
                0.0,
                1.0,
            ))?;
            q.set_bind_group(BindGroupIndex::new(0), &self.graphics_groups[slot][1], &[])?;
            q.draw_indexed(0..6, 0, 0..1)?;
            q.end()?;
        }
        let mut p = SubmissionPlanBuilder::new(d);
        let point = p.add_batch(lane, vec![r.finish()?])?;
        p.present_after(f, point)?;
        let receipt = d.submit(p.build()?)?;
        self.done[slot] = Some(receipt.completion());
        let present = receipt.presents().first().expect("present");
        let _ = d.wait_present(present.id()).await?;
        self.next = (slot + 1) % FRAMES;
        Ok(())
    }
}
fn group_graphics(
    d: &Device,
    l: &BindGroupLayout,
    u: &Buffer,
    t: &TextureView,
    s: &Sampler,
) -> RhiResult<BindGroup> {
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
fn filter_from_env() -> usize {
    match std::env::var("FLUXEL_RHI_COMPUTE_FILTER").as_deref() {
        Ok("edgedetect") => 1,
        Ok("sharpen") => 2,
        _ => 0,
    }
}
fn ubo(e: Extent3d) -> Vec<u8> {
    let a = e.width.max(1) as f32 * 0.5 / e.height.max(1) as f32;
    let x = 1.0 / 30f32.to_radians().tan();
    [
        x / a,
        0.0,
        0.0,
        0.0,
        0.0,
        x,
        0.0,
        0.0,
        0.0,
        0.0,
        256.0 / (1.0 - 256.0),
        -1.0,
        0.0,
        0.0,
        256.0 / (1.0 - 256.0),
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0,
        1.0,
    ]
    .into_iter()
    .flat_map(f32::to_le_bytes)
    .collect()
}
fn quad_vertices() -> Vec<u8> {
    [
        1., 1., 0., 1., 1., -1., 1., 0., 0., 1., -1., -1., 0., 0., 0., 1., -1., 0., 1., 0.,
    ]
    .into_iter()
    .flat_map(f32::to_le_bytes)
    .collect()
}
fn quad_indices() -> Vec<u8> {
    [0u32, 1, 2, 2, 3, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}
fn graphics_artifact(s: ShaderStage) -> ShaderArtifact {
    let mut i = ShaderInterface::new().with_resource(resource(
        0,
        BindingKind::UniformBuffer {
            min_size: UBO_BYTES,
        },
    ));
    if matches!(s, ShaderStage::Vertex) {
        i = i
            .with_input(io(0, 3, None))
            .with_input(io(1, 2, None))
            .with_output(io(0, 2, Some(interp())))
            .with_writes_position(true)
    } else {
        i = i
            .with_resource(resource(
                1,
                BindingKind::SampledTexture {
                    dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                },
            ))
            .with_resource(resource(
                2,
                BindingKind::Sampler {
                    kind: SamplerKind::Filtering,
                },
            ))
            .with_input(io(0, 2, Some(interp())))
            .with_output(io(0, 4, None))
    }
    ShaderArtifact::new(
        s,
        if matches!(s, ShaderStage::Vertex) {
            "vs_main"
        } else {
            "fs_main"
        },
        ShaderCode::Wgsl(Arc::from(GRAPHICS_WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        i,
        ShaderRequirements::new(),
        ArtifactHash(
            [if matches!(s, ShaderStage::Vertex) {
                0x61
            } else {
                0x62
            }; 32],
        ),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
fn compute_artifact(entry: &'static str, n: u8) -> ShaderArtifact {
    ShaderArtifact::new(
        ShaderStage::Compute,
        entry,
        ShaderCode::Wgsl(Arc::from(COMPUTE_WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new()
            .with_compute_workgroup_size(ComputeWorkgroupSize::new(16, 16, 1))
            .with_resource(resource(
                0,
                BindingKind::StorageTexture {
                    dimension: TextureViewDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    access: StorageAccess::ReadOnly,
                },
            ))
            .with_resource(resource(
                1,
                BindingKind::StorageTexture {
                    dimension: TextureViewDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    access: StorageAccess::WriteOnly,
                },
            )),
        ShaderRequirements::new(),
        ArtifactHash([0x70 + n; 32]),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
fn resource(slot: u32, kind: BindingKind) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(slot),
        kind,
        count: BindingCount::One,
    }
}
fn io(n: u32, c: u8, x: Option<ShaderInterpolation>) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(n),
        numeric_type: ShaderNumericType::Float32,
        components: c,
        interpolation: x,
    }
}
fn interp() -> ShaderInterpolation {
    ShaderInterpolation {
        mode: InterpolationMode::Perspective,
        sampling: InterpolationSampling::Center,
    }
}
struct Ktx {
    width: u32,
    height: u32,
    levels: Vec<Vec<u8>>,
}
fn parse_ktx(b: &[u8]) -> Result<Ktx, String> {
    if b.len() < 64 || b[..12] != KTX_MAGIC {
        return Err("vulkan_11_rgba.ktx is not KTX1".into());
    }
    let w = |o: usize| -> Result<u32, String> {
        b.get(o..o + 4)
            .ok_or_else(|| "truncated KTX1".into())
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
    };
    if w(12)? != 0x04030201
        || w(16)? != 0x1401
        || w(20)? != 1
        || w(24)? != 0x1908
        || w(28)? != 0x8058
        || w(32)? != 0x1908
    {
        return Err("KTX must be little-endian RGBA8".into());
    }
    let (width, height, levels, key) = (w(36)?, w(40)?, w(56)?, w(60)?);
    let mut p = 64 + key as usize;
    let mut out = Vec::new();
    for m in 0..levels {
        let n = w(p)? as usize;
        p += 4;
        let end = p.checked_add(n).ok_or_else(|| "KTX overflow".to_string())?;
        let d = b
            .get(p..end)
            .ok_or_else(|| "truncated KTX level".to_string())?;
        let expected = (width >> m).max(1) as usize * (height >> m).max(1) as usize * 4;
        if d.len() != expected {
            return Err("invalid KTX level size".into());
        }
        out.push(d.to_vec());
        p = (end + 3) & !3;
    }
    Ok(Ktx {
        width,
        height,
        levels: out,
    })
}
