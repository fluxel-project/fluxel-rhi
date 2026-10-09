//! Port of SaschaWillems/Vulkan `examples/descriptorindexing`.
//!
//! The scene deliberately uses an unsized descriptor array: thirty-two generated
//! 3x3 RGBA images are selected non-uniformly by a per-vertex face attribute.
//! Binding 2 is the last binding in group 0, as Vulkan requires for a variable
//! descriptor-count binding.

pub mod common;

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

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
    error::RhiResult,
    format::TextureFormat,
    pipeline::{
        ColorTargetState, DepthState, DepthStencilState, PipelineInterfaceDescriptor,
        PrimitiveState, PrimitiveTopology, RasterPipeline, RasterPipelineDescriptor,
        VertexAttribute, VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
    },
    platform::{Device, OptionalFeature},
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

const FRAMES: usize = 2;
const TEXTURES: usize = 32;
const CUBES: usize = 5;
const UNIFORM_BYTES: u64 = 192;
const VERTEX_BYTES: u64 = 24;

const SHADER: &str = r#"
enable wgpu_binding_array;
struct Uniforms { projection: mat4x4<f32>, view: mat4x4<f32>, model: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var nearest_sampler: sampler;
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
struct Vertex { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) texture_index: i32, };
struct Varyings { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) @interpolate(flat) texture_index: i32, };
@vertex fn vs_main(input: Vertex) -> Varyings {
    var out: Varyings;
    out.position = uniforms.projection * uniforms.view * uniforms.model * vec4<f32>(input.position, 1.0);
    out.uv = input.uv;
    out.texture_index = input.texture_index;
    return out;
}
@fragment fn fs_main(input: Varyings) -> @location(0) vec4<f32> {
    // `texture_index` comes from independently rasterized cube faces. Naga emits
    // non-uniform descriptor indexing for this binding-array access on Vulkan.
    return textureSample(textures[u32(input.texture_index)], nearest_sampler, input.uv);
}
"#;

fn main() {
    if let Err(error) = common::run_example("Descriptor indexing", create_example()) {
        eprintln!("14_descriptor_indexing: {error}");
        std::process::exit(1);
    }
}

pub fn create_example() -> Example {
    Example {
        workload: None,
        lane: None,
    }
}

pub struct Example {
    workload: Option<Workload>,
    lane: Option<SubmissionLaneId>,
}

impl common::Example for Example {
    fn init(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let device = context.device().clone();
        let caps = device.capabilities();
        for feature in [
            OptionalFeature::RuntimeSizedBindingArrays,
            OptionalFeature::NonUniformSampledTextureAndStorageBufferIndexing,
        ] {
            if !caps.supports_feature(feature) {
                return Err(format!("14_descriptor_indexing requires {feature:?}").into());
            }
        }
        let lane = caps
            .submission()
            .lanes()
            .iter()
            .find(|lane| {
                lane.domains()
                    .contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY))
            })
            .map(|lane| lane.id())
            .ok_or_else(|| std::io::Error::other("device has no COPY|RASTER submission lane"))?;
        let size = context.extent();
        self.workload = Some(common::block_on(Workload::new(
            &device,
            context.presentation_mut().configuration().format(),
            Extent3d::d2(size.width, size.height),
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
        let workload = self.workload.as_mut().ok_or_else(|| {
            std::io::Error::other("descriptor-indexing workload is not initialized")
        })?;
        let lane = self
            .lane
            .ok_or_else(|| std::io::Error::other("descriptor-indexing lane is not initialized"))?;
        common::block_on(workload.render(context.device(), lane, frame))?;
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

struct Workload {
    pipeline: RasterPipeline,
    groups: [BindGroup; FRAMES],
    uniforms: [Buffer; FRAMES],
    vertex: BufferBinding,
    index: BufferBinding,
    index_count: u32,
    depth: TextureView,
    extent: Extent3d,
    next: usize,
    completions: [Option<CompletionPoint>; FRAMES],
}

impl Workload {
    async fn new(
        device: &Device,
        color_format: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let layout = descriptor_layout(device)?;
        let interface = device
            .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        let pipeline = device
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(device, &vertex_artifact()).await?,
                    interface,
                )
                .with_fragment(common::shader::create_shader(device, &fragment_artifact()).await?)
                .with_vertex_input(
                    VertexInputState::new().with_buffer(
                        VertexBufferLayout::new(VERTEX_BYTES, VertexStepMode::Vertex)
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
                                VertexFormat::Sint32,
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
        let (vertices, indices) = cubes();
        let vertex_buffer = device.create_buffer(
            &BufferDescriptor::new(
                vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("14_descriptor_indexing cube vertices"),
        )?;
        let index_buffer = device.create_buffer(
            &BufferDescriptor::new(
                indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label("14_descriptor_indexing cube indices"),
        )?;
        let sampler = device.create_sampler(
            &SamplerDescriptor::new()
                .with_label("14_descriptor_indexing shared nearest sampler")
                .with_address_modes(
                    AddressMode::Repeat,
                    AddressMode::Repeat,
                    AddressMode::Repeat,
                )
                .with_filters(
                    FilterMode::Nearest,
                    FilterMode::Nearest,
                    FilterMode::Nearest,
                ),
        )?;
        let mut texture_views = Vec::with_capacity(TEXTURES);
        let mut textures = Vec::with_capacity(TEXTURES);
        for i in 0..TEXTURES {
            let texture = device.create_texture(
                &TextureDescriptor::new_2d(
                    3,
                    3,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED.union(TextureUsage::COPY_DST),
                )
                .with_label(format!("14_descriptor_indexing texture {i}")),
            )?;
            texture_views.push(device.create_texture_view(
                &texture,
                &TextureViewDescriptor::whole(&texture, TextureViewDimension::D2)?,
            )?);
            textures.push(texture);
        }
        let uniforms = [
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("14_descriptor_indexing uniforms frame 0"),
            )?,
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("14_descriptor_indexing uniforms frame 1"),
            )?,
        ];
        let groups = [
            group(
                device,
                &layout,
                &uniforms[0],
                &sampler,
                texture_views.clone(),
            )?,
            group(device, &layout, &uniforms[1], &sampler, texture_views)?,
        ];
        let depth = depth(device, extent)?;
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
        for texture in textures {
            // The C++ sample seeds one engine from `std::random_device` for
            // each 3x3 image.
            let mut random = SourceRandom::new();
            recorder.encode_upload(&device.create_texture_upload(
                TextureUploadDescriptor::new(
                    texture,
                    TextureSubresourceLayers {
                        aspect: TextureAspect::Color,
                        mip_level: 0,
                        base_layer: 0,
                        layer_count: 1,
                    },
                    Origin3d { x: 0, y: 0, z: 0 },
                    Extent3d::d2(3, 3),
                    HostTexelLayout {
                        bytes_per_row: 12,
                        rows_per_image: 3,
                    },
                    random_texture(&mut random),
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
            vertex: BufferBinding::new(
                vertex_buffer,
                BufferRange::new(0, (CUBES as u64) * 24 * VERTEX_BYTES),
            ),
            index: BufferBinding::new(index_buffer, BufferRange::new(0, (CUBES as u64) * 36 * 4)),
            index_count: (CUBES * 36) as u32,
            depth,
            extent,
            next: 0,
            completions: [None, None],
        })
    }
    fn resize(&mut self, device: &Device, extent: Extent3d) -> RhiResult<()> {
        self.depth = depth(device, extent)?;
        self.extent = extent;
        Ok(())
    }
    async fn render(
        &mut self,
        device: &Device,
        lane: SubmissionLaneId,
        frame: AcquiredFrame,
    ) -> RhiResult<()> {
        let slot = self.next;
        if let Some(done) = self.completions[slot].take() {
            let _ = device.wait_completion(done).await?;
        }
        let upload = device.create_buffer_upload(BufferUploadDescriptor::new(
            self.uniforms[slot].clone(),
            0,
            Uniforms::for_extent(self.extent).bytes(),
        ))?;
        let attachment = frame.attachment();
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&upload)?;
        {
            let mut raster = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_label("14_descriptor_indexing raster")
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(attachment),
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
            raster.draw_indexed(0..self.index_count, 0, 0..1)?;
            raster.end()?;
        }
        let mut plan = SubmissionPlanBuilder::new(device);
        let point = plan.add_batch(lane, vec![recorder.finish()?])?;
        plan.present_after(frame, point)?;
        let receipt = device.submit(plan.build()?)?;
        self.completions[slot] = Some(receipt.completion());
        let _ = device
            .wait_present(
                receipt
                    .presents()
                    .first()
                    .expect("one present receipt")
                    .id(),
            )
            .await?;
        self.next = (slot + 1) % FRAMES;
        Ok(())
    }
}

fn descriptor_layout(device: &Device) -> RhiResult<BindGroupLayout> {
    device.create_bind_group_layout(
        &BindGroupLayoutDescriptor::new(vec![
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
                BindingKind::Sampler {
                    kind: SamplerKind::Filtering,
                },
            ),
            BindingSlot::new(
                BindingSlotId::new(2),
                ShaderStages::FRAGMENT,
                BindingKind::SampledTexture {
                    dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                },
            )
            .with_count(BindingCount::RuntimeSized),
        ])
        .with_label("14_descriptor_indexing layout"),
    )
}

fn group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform: &Buffer,
    sampler: &Sampler,
    textures: Vec<TextureView>,
) -> RhiResult<BindGroup> {
    device.create_bind_group(&BindGroupDescriptor::new(layout.clone()).with_entries([
        BindGroupEntry::new(
            BindingSlotId::new(0),
            BindingResource::Buffer(BufferBinding::new(
                uniform.clone(),
                BufferRange::new(0, UNIFORM_BYTES),
            )),
        ),
        BindGroupEntry::new(
            BindingSlotId::new(1),
            BindingResource::Sampler(sampler.clone()),
        ),
        BindGroupEntry::new(
            BindingSlotId::new(2),
            BindingResource::TextureArray(textures),
        ),
    ]))
}

fn depth(device: &Device, extent: Extent3d) -> RhiResult<TextureView> {
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
struct Uniforms {
    projection: [f32; 16],
    view: [f32; 16],
    model: [f32; 16],
}
impl Uniforms {
    fn for_extent(extent: Extent3d) -> Self {
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        let f = 1.0 / 22.5_f32.to_radians().tan();
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
            view: [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.819_152, -0.573_576, 0.0, 0.0, 0.573_576, 0.819_152,
                0.0, 0.0, 0.0, -10.0, 1.0,
            ],
            model: [
                0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }
    fn bytes(self) -> Vec<u8> {
        self.projection
            .into_iter()
            .chain(self.view)
            .chain(self.model)
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
                BindingCount::One,
            ))
            .with_input(io(0, ShaderNumericType::Float32, 3, None))
            .with_input(io(1, ShaderNumericType::Float32, 2, None))
            .with_input(io(2, ShaderNumericType::Sint32, 1, None))
            .with_output(io(0, ShaderNumericType::Float32, 2, Some(interpolation())))
            .with_output(io(
                1,
                ShaderNumericType::Sint32,
                1,
                Some(ShaderInterpolation {
                    mode: InterpolationMode::Flat,
                    sampling: InterpolationSampling::Center,
                }),
            ))
            .with_writes_position(true),
        ArtifactHash([0xE1; 32]),
    )
}
fn fragment_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Fragment,
        "fs_main",
        ShaderInterface::new()
            .with_resource(resource(
                1,
                BindingKind::Sampler {
                    kind: SamplerKind::Filtering,
                },
                BindingCount::One,
            ))
            .with_resource(resource(
                2,
                BindingKind::SampledTexture {
                    dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                },
                BindingCount::RuntimeSized,
            ))
            .with_input(io(0, ShaderNumericType::Float32, 2, Some(interpolation())))
            .with_input(io(
                1,
                ShaderNumericType::Sint32,
                1,
                Some(ShaderInterpolation {
                    mode: InterpolationMode::Flat,
                    sampling: InterpolationSampling::Center,
                }),
            ))
            .with_output(io(0, ShaderNumericType::Float32, 4, None)),
        ArtifactHash([0xE2; 32]),
    )
}
fn resource(slot: u32, kind: BindingKind, count: BindingCount) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(slot),
        kind,
        count,
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
    numeric_type: ShaderNumericType,
    components: u8,
    interpolation: Option<ShaderInterpolation>,
) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(location),
        numeric_type,
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
        ShaderCode::Wgsl(Arc::from(SHADER)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        hash,
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}

fn cubes() -> (Vec<u8>, Vec<u8>) {
    let mut random = SourceRandom::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let index_template: [u32; 36] = [
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    ];
    let faces = [
        [[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]],
        [[1., 1., 1.], [1., 1., -1.], [1., -1., -1.], [1., -1., 1.]],
        [
            [-1., -1., -1.],
            [1., -1., -1.],
            [1., 1., -1.],
            [-1., 1., -1.],
        ],
        [
            [-1., -1., -1.],
            [-1., -1., 1.],
            [-1., 1., 1.],
            [-1., 1., -1.],
        ],
        [[1., 1., 1.], [-1., 1., 1.], [-1., 1., -1.], [1., 1., -1.]],
        [
            [-1., -1., -1.],
            [1., -1., -1.],
            [1., -1., 1.],
            [-1., -1., 1.],
        ],
    ];
    let uvs = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    for cube in 0..CUBES {
        let base = (cube * 24) as u32;
        indices.extend(index_template.into_iter().map(|i| i + base));
        let x = 2.5 * cube as f32 - (CUBES as f32 * 2.5 / 2.0) + 1.25;
        for face in faces {
            let texture = random.below(TEXTURES as u32) as i32;
            for (point, uv) in face.into_iter().zip(uvs) {
                for value in [point[0] + x, point[1], point[2], uv[0], uv[1]] {
                    vertices.extend(value.to_le_bytes());
                }
                vertices.extend(texture.to_le_bytes());
            }
        }
    }
    (
        vertices,
        indices.into_iter().flat_map(u32::to_le_bytes).collect(),
    )
}

fn random_texture(random: &mut SourceRandom) -> Vec<u8> {
    let mut out = Vec::with_capacity(36);
    for _ in 0..9 {
        out.extend([
            random.range(50, 255),
            random.range(50, 255),
            random.range(50, 255),
            255,
        ]);
    }
    out
}
struct SourceRandom(u64);
impl SourceRandom {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|x| x.as_nanos() as u64)
            .unwrap_or(0x9e37_79b9);
        Self(seed ^ 0xa076_1d64_78bd_642f)
    }
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
    }
    fn below(&mut self, upper: u32) -> u32 {
        self.next() % upper
    }
    fn range(&mut self, low: u8, high: u8) -> u8 {
        low + (self.below(u32::from(high - low + 1)) as u8)
    }
}
