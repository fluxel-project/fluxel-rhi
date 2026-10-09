//! Faithful port of SaschaWillems/Vulkan `examples/texture/texture.cpp`.
//!
//! The quad, KTX1 `metalplate01_rgba.ktx` mip chain, camera, descriptor
//! bindings and shader lighting equation are kept from the C++ sample.

pub mod common;

use std::sync::Arc;

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
        ColorTargetState, DepthState, DepthStencilState, PipelineInterfaceDescriptor,
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

const FRAMES_IN_FLIGHT: usize = 2;
// Two mat4 values, a vec4, one scalar, and std140/WGSL struct tail padding.
const UNIFORM_BYTES: u64 = 160;
const KTX_IDENTIFIER: [u8; 12] = [
    0xAB, b'K', b'T', b'X', b' ', b'1', b'1', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];
const METALPLATE_KTX: &[u8] = include_bytes!("assets/textures/metalplate01_rgba.ktx");

// Equivalent to shaders/glsl/texture/{texture.vert,texture.frag}.
const TEXTURE_WGSL: &str = r#"
struct Uniforms { projection: mat4x4<f32>, model: mat4x4<f32>, view_pos: vec4<f32>, lod_bias: f32, };
@group(0) @binding(0) var<uniform> ubo: Uniforms;
@group(0) @binding(1) var sampler_color: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;
struct VertexInput { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) normal: vec3<f32>, };
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) lod_bias: f32, @location(2) normal: vec3<f32>, @location(3) view_vec: vec3<f32>, @location(4) light_vec: vec3<f32>, };
@vertex fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let position = ubo.model * vec4<f32>(input.position, 1.0);
    output.position = ubo.projection * position;
    output.uv = input.uv; output.lod_bias = ubo.lod_bias;
    // `model` is the C++ sample's rigid camera transform, so this equals inverse-transpose.
    output.normal = (ubo.model * vec4<f32>(input.normal, 0.0)).xyz;
    output.light_vec = -position.xyz; output.view_vec = ubo.view_pos.xyz - position.xyz;
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSampleBias(sampler_color, color_sampler, input.uv, input.lod_bias);
    let n = normalize(input.normal); let l = normalize(input.light_vec); let v = normalize(input.view_vec);
    let diffuse = max(dot(n, l), 0.0);
    let specular = pow(max(dot(reflect(-l, n), v), 0.0), 16.0) * color.a;
    return vec4<f32>(diffuse * color.rgb + vec3<f32>(specular), 1.0);
}
"#;

fn main() {
    if let Err(error) = common::run_example("Texture loading", create_example()) {
        eprintln!("02_texture: {error}");
        std::process::exit(1);
    }
}
pub fn create_example() -> TextureExample {
    TextureExample {
        workload: None,
        lane: None,
    }
}
pub struct TextureExample {
    workload: Option<TextureWorkload>,
    lane: Option<SubmissionLaneId>,
}

impl common::Example for TextureExample {
    fn init(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let device = context.device().clone();
        let required = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
        let lane = device
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|candidate| candidate.domains().contains(required))
            .map(|candidate| candidate.id())
            .ok_or_else(|| std::io::Error::other("device has no COPY|RASTER submission lane"))?;
        let format = context.presentation_mut().configuration().format();
        let extent = context.extent();
        self.workload = Some(common::block_on(TextureWorkload::new(
            &device,
            format,
            Extent3d::d2(extent.width, extent.height),
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
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let frame = common::block_on(context.presentation_mut().acquire())?;
        let workload = self
            .workload
            .as_mut()
            .ok_or_else(|| std::io::Error::other("texture workload is not initialized"))?;
        common::block_on(
            workload.render(
                context.device(),
                self.lane
                    .ok_or_else(|| std::io::Error::other("texture lane is not initialized"))?,
                frame,
            ),
        )?;
        Ok(())
    }
    fn resize(
        &mut self,
        context: &mut common::ExampleContext<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(workload) = &mut self.workload {
            workload.resize(context.device(), Extent3d::d2(width, height))?;
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

pub struct TextureWorkload {
    pipeline: RasterPipeline,
    groups: [BindGroup; FRAMES_IN_FLIGHT],
    uniforms: [Buffer; FRAMES_IN_FLIGHT],
    vertex: BufferBinding,
    index: BufferBinding,
    depth: TextureView,
    extent: Extent3d,
    lod_bias: f32,
    next_frame: usize,
    completions: [Option<CompletionPoint>; FRAMES_IN_FLIGHT],
}
impl TextureWorkload {
    pub async fn new(
        device: &Device,
        color_format: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let ktx = parse_rgba8_ktx1(METALPLATE_KTX)
            .map_err(|message| RhiError::new(RhiErrorKind::InvalidUsage, message))?;
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::VERTEX,
                BindingKind::UniformBuffer {
                    min_size: UNIFORM_BYTES,
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
        let interface = device
            .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        let pipeline = device
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(device, &vertex_artifact()).await?,
                    interface,
                )
                .with_label("02_texture raster pipeline")
                .with_fragment(common::shader::create_shader(device, &fragment_artifact()).await?)
                .with_vertex_input(
                    VertexInputState::new().with_buffer(
                        VertexBufferLayout::new(32, VertexStepMode::Vertex)
                            .with_attribute(VertexAttribute::new(
                                ShaderLocation::new(0),
                                VertexFormat::Float32x3,
                                0,
                            ))
                            .with_attribute(VertexAttribute::new(
                                ShaderLocation::new(1),
                                VertexFormat::Float32x2,
                                12,
                            ))
                            .with_attribute(VertexAttribute::new(
                                ShaderLocation::new(2),
                                VertexFormat::Float32x3,
                                20,
                            )),
                    ),
                )
                .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleList))
                .with_depth_stencil(
                    DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                        DepthState::new(CompareFunction::LessEqual).with_write_enabled(true),
                    ),
                )
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(color_format)),
            )
            .await?;
        let vertices = vertex_bytes();
        let indices = index_bytes();
        let vertex_buffer = device.create_buffer(
            &BufferDescriptor::new(
                vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("02_texture quad vertices"),
        )?;
        let index_buffer = device.create_buffer(
            &BufferDescriptor::new(
                indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label("02_texture quad indices"),
        )?;
        let max_lod_bias = ktx.levels.len() as f32;
        let lod_bias = std::env::var("FLUXEL_RHI_TEXTURE_LOD_BIAS")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
            .clamp(0.0, max_lod_bias);
        let texture = device.create_texture(
            &TextureDescriptor::new_2d(
                ktx.width,
                ktx.height,
                TextureFormat::Rgba8Unorm,
                TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
            )
            .with_mip_levels(ktx.levels.len() as u32)
            .with_label("02_texture metalplate01_rgba.ktx"),
        )?;
        let texture_view = device.create_texture_view(
            &texture,
            &TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)?,
        )?;
        let max_anisotropy = if device
            .capabilities()
            .supports_feature(OptionalFeature::SamplerAnisotropy)
        {
            device
                .capabilities()
                .limit(LimitKey::MaxSamplerAnisotropy)
                .unwrap_or(1)
                .clamp(1, u16::MAX as u64) as u16
        } else {
            1
        };
        let sampler = device.create_sampler(
            &SamplerDescriptor::new()
                .with_label("02_texture KTX sampler")
                .with_address_modes(
                    AddressMode::Repeat,
                    AddressMode::Repeat,
                    AddressMode::Repeat,
                )
                .with_filters(FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
                .with_lod_clamp(0.0, ktx.levels.len() as f32)
                .with_max_anisotropy(max_anisotropy),
        )?;
        let uniforms = [
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("02_texture uniforms frame 0"),
            )?,
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("02_texture uniforms frame 1"),
            )?,
        ];
        let groups = [
            create_group(device, &layout, &uniforms[0], &texture_view, &sampler)?,
            create_group(device, &layout, &uniforms[1], &texture_view, &sampler)?,
        ];
        let depth = create_depth_view(device, extent)?;
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            vertex_buffer.clone(),
            0,
            vertices,
        ))?)?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            index_buffer.clone(),
            0,
            indices,
        ))?)?;
        for (level, bytes) in ktx.levels.into_iter().enumerate() {
            let width = (ktx.width >> level).max(1);
            let height = (ktx.height >> level).max(1);
            recorder.encode_upload(&device.create_texture_upload(
                TextureUploadDescriptor::new(
                    texture.clone(),
                    TextureSubresourceLayers {
                        aspect: TextureAspect::Color,
                        mip_level: level as u32,
                        base_layer: 0,
                        layer_count: 1,
                    },
                    Origin3d { x: 0, y: 0, z: 0 },
                    Extent3d::d2(width, height),
                    HostTexelLayout {
                        bytes_per_row: width * 4,
                        rows_per_image: height,
                    },
                    bytes,
                ),
            )?)?;
        }
        let mut plan = SubmissionPlanBuilder::new(device);
        plan.add_batch(lane, vec![recorder.finish()?])?;
        let receipt = device.submit(plan.build()?)?;
        let _ = device.wait_completion(receipt.completion()).await?;
        Ok(Self {
            pipeline,
            groups,
            uniforms,
            vertex: BufferBinding::new(vertex_buffer, BufferRange::new(0, 128)),
            index: BufferBinding::new(index_buffer, BufferRange::new(0, 24)),
            depth,
            extent,
            lod_bias,
            next_frame: 0,
            completions: [None, None],
        })
    }
    pub fn resize(&mut self, device: &Device, extent: Extent3d) -> RhiResult<()> {
        self.depth = create_depth_view(device, extent)?;
        self.extent = extent;
        Ok(())
    }
    pub async fn render(
        &mut self,
        device: &Device,
        lane: SubmissionLaneId,
        frame: AcquiredFrame,
    ) -> RhiResult<()> {
        let slot = self.next_frame;
        if let Some(completion) = self.completions[slot].take() {
            let _ = device.wait_completion(completion).await?;
        }
        let upload = device.create_buffer_upload(BufferUploadDescriptor::new(
            self.uniforms[slot].clone(),
            0,
            TextureUniforms::reference_for_extent(self.extent, self.lod_bias).to_bytes(),
        ))?;
        let attachment = frame.attachment();
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&upload)?;
        {
            let mut raster = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_label("02_texture raster")
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(attachment),
                            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.2, 1.0])),
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
            raster.set_pipeline(&self.pipeline)?;
            raster.set_viewport(Viewport::new(
                0.0,
                0.0,
                self.extent.width as f32,
                self.extent.height as f32,
                0.0,
                1.0,
            ))?;
            raster.set_scissor(Rect::new(0, 0, self.extent.width, self.extent.height))?;
            raster.set_bind_group(BindGroupIndex::new(0), &self.groups[slot], &[])?;
            raster.set_vertex_buffer(0, &self.vertex)?;
            raster.set_index_buffer(&self.index, IndexFormat::Uint32)?;
            raster.draw_indexed(0..6, 0, 0..1)?;
            raster.end()?;
        }
        let mut plan = SubmissionPlanBuilder::new(device);
        let point = plan.add_batch(lane, vec![recorder.finish()?])?;
        plan.present_after(frame, point)?;
        let receipt = device.submit(plan.build()?)?;
        self.completions[slot] = Some(receipt.completion());
        let present = receipt
            .presents()
            .first()
            .expect("one present_after call must yield one receipt");
        let _ = device.wait_present(present.id()).await?;
        self.next_frame = (slot + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }
}
fn create_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform: &Buffer,
    texture: &TextureView,
    sampler: &Sampler,
) -> RhiResult<BindGroup> {
    device.create_bind_group(
        &BindGroupDescriptor::new(layout.clone())
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(0),
                BindingResource::Buffer(BufferBinding::new(
                    uniform.clone(),
                    BufferRange::new(0, UNIFORM_BYTES),
                )),
            ))
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(1),
                BindingResource::Texture(texture.clone()),
            ))
            .with_entry(BindGroupEntry::new(
                BindingSlotId::new(2),
                BindingResource::Sampler(sampler.clone()),
            )),
    )
}
fn create_depth_view(device: &Device, extent: Extent3d) -> RhiResult<TextureView> {
    let texture = device.create_texture(&TextureDescriptor::new_2d(
        extent.width,
        extent.height,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ))?;
    device.create_texture_view(
        &texture,
        &TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)?,
    )
}

#[derive(Clone, Copy)]
struct TextureUniforms {
    projection: [f32; 16],
    model: [f32; 16],
    view_pos: [f32; 4],
    lod_bias: f32,
}
impl TextureUniforms {
    fn reference_for_extent(extent: Extent3d, lod_bias: f32) -> Self {
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        let f = 1.0 / 30.0_f32.to_radians().tan();
        let (s, c) = 15.0_f32.to_radians().sin_cos();
        let near = 0.1_f32;
        let far = 256.0_f32;
        Self {
            projection: [
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
                far / (near - far),
                -1.0,
                0.0,
                0.0,
                far * near / (near - far),
                0.0,
            ],
            model: [
                c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, -2.5, 1.0,
            ],
            view_pos: [0.0, 0.0, 2.5, 0.0],
            lod_bias,
        }
    }
    fn to_bytes(self) -> Vec<u8> {
        self.projection
            .into_iter()
            .chain(self.model)
            .chain(self.view_pos)
            .chain([self.lod_bias, 0.0, 0.0, 0.0])
            .flat_map(f32::to_le_bytes)
            .collect()
    }
}

fn vertex_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Vertex,
        "vs_main",
        ShaderInterface::new()
            .with_resource(resource(
                0,
                BindingKind::UniformBuffer {
                    min_size: UNIFORM_BYTES,
                },
            ))
            .with_input(io(0, 3, None))
            .with_input(io(1, 2, None))
            .with_input(io(2, 3, None))
            .with_output(io(0, 2, Some(interpolation())))
            .with_output(io(1, 1, Some(interpolation())))
            .with_output(io(2, 3, Some(interpolation())))
            .with_output(io(3, 3, Some(interpolation())))
            .with_output(io(4, 3, Some(interpolation())))
            .with_writes_position(true),
        ArtifactHash([0x21; 32]),
    )
}
fn fragment_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Fragment,
        "fs_main",
        ShaderInterface::new()
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
            .with_input(io(0, 2, Some(interpolation())))
            .with_input(io(1, 1, Some(interpolation())))
            .with_input(io(2, 3, Some(interpolation())))
            .with_input(io(3, 3, Some(interpolation())))
            .with_input(io(4, 3, Some(interpolation())))
            .with_output(io(0, 4, None)),
        ArtifactHash([0x22; 32]),
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
fn interpolation() -> ShaderInterpolation {
    ShaderInterpolation {
        mode: InterpolationMode::Perspective,
        sampling: InterpolationSampling::Center,
    }
}
fn io(
    location: u32,
    components: u8,
    interpolation: Option<ShaderInterpolation>,
) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(location),
        numeric_type: ShaderNumericType::Float32,
        components,
        interpolation,
    }
}
fn artifact(
    stage: ShaderStage,
    entry: &'static str,
    interface: ShaderInterface,
    hash: ArtifactHash,
) -> ShaderArtifact {
    ShaderArtifact::new(
        stage,
        entry,
        ShaderCode::Wgsl(Arc::from(TEXTURE_WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        hash,
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
fn vertex_bytes() -> Vec<u8> {
    [
        1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0,
        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    ]
    .into_iter()
    .flat_map(f32::to_le_bytes)
    .collect()
}
fn index_bytes() -> Vec<u8> {
    [0_u32, 1, 2, 2, 3, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}

struct KtxRgba8 {
    width: u32,
    height: u32,
    levels: Vec<Vec<u8>>,
}
fn parse_rgba8_ktx1(bytes: &[u8]) -> Result<KtxRgba8, String> {
    if bytes.len() < 64 || bytes[..12] != KTX_IDENTIFIER {
        return Err("metalplate asset is not a KTX1 file".into());
    }
    let word = |offset: usize| -> Result<u32, String> {
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| "truncated KTX1 header".to_owned())
            .map(|part| u32::from_le_bytes(part.try_into().expect("four-byte KTX word")))
    };
    if word(12)? != 0x0403_0201
        || word(16)? != 0x1401
        || word(20)? != 1
        || word(24)? != 0x1908
        || word(28)? != 0x8058
        || word(32)? != 0x1908
    {
        return Err("metalplate KTX must be little-endian RGBA8".into());
    }
    let (width, height, depth, arrays, faces, levels, key_bytes) = (
        word(36)?,
        word(40)?,
        word(44)?,
        word(48)?,
        word(52)?,
        word(56)?,
        word(60)?,
    );
    if width == 0 || height == 0 || depth != 0 || arrays != 0 || faces != 1 || levels == 0 {
        return Err(
            "metalplate KTX must be a non-array 2D texture with explicit mip levels".into(),
        );
    }
    let mut cursor = 64_usize
        .checked_add(key_bytes as usize)
        .ok_or_else(|| "KTX key/value size overflows".to_owned())?;
    let mut mip_levels = Vec::with_capacity(levels as usize);
    for level in 0..levels {
        let size = bytes
            .get(cursor..cursor + 4)
            .ok_or_else(|| "truncated KTX mip size".to_owned())
            .map(|part| u32::from_le_bytes(part.try_into().expect("four-byte KTX mip size")))?
            as usize;
        cursor += 4;
        let data = bytes
            .get(
                cursor
                    ..cursor
                        .checked_add(size)
                        .ok_or_else(|| "KTX mip size overflows".to_owned())?,
            )
            .ok_or_else(|| "truncated KTX mip payload".to_owned())?;
        let expected = (width >> level).max(1) as usize * (height >> level).max(1) as usize * 4;
        if data.len() != expected {
            return Err(format!(
                "KTX mip {level} has {} bytes, expected {expected}",
                data.len()
            ));
        }
        mip_levels.push(data.to_vec());
        cursor = (cursor + size + 3) & !3;
    }
    Ok(KtxRgba8 {
        width,
        height,
        levels: mip_levels,
    })
}
