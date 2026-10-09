//! Strict port of SaschaWillems/Vulkan `dynamicuniformbuffer`.
//!
//! One dynamically offset uniform-buffer binding addresses 125 aligned model
//! matrices. The source's random rotations, 5³ grid, update order, and coloured
//! cube are kept deliberately explicit.

pub mod common;

use std::{
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use fluxel_rhi::api::{
    binding::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayout,
        BindGroupLayoutDescriptor, BindingCount, BindingKind, BindingResource, BindingSlot,
        BindingSlotId,
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
    platform::{Device, LimitKey},
    presentation::AcquiredFrame,
    resource::{
        Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor, BufferUsage,
        CompareFunction, Extent3d, TextureDescriptor, TextureUsage, TextureView,
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
const OBJECTS_PER_AXIS: u32 = 5;
const OBJECT_COUNT: u32 = OBJECTS_PER_AXIS * OBJECTS_PER_AXIS * OBJECTS_PER_AXIS;
const MODEL_BYTES: u64 = 64;
const VIEW_BYTES: u64 = 128;

const WGSL: &str = r#"
struct UboVS { projection: mat4x4<f32>, view: mat4x4<f32>, };
struct UboDynamic { model: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> ubo_vs: UboVS;
@group(0) @binding(1) var<uniform> ubo_dynamic: UboDynamic;
struct VertexIn { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32>, };
struct VertexOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32>, };
@vertex fn vs_main(input: VertexIn) -> VertexOut {
  var output: VertexOut;
  output.pos = ubo_vs.projection * ubo_vs.view * ubo_dynamic.model * vec4<f32>(input.pos, 1.0);
  output.color = input.color;
  return output;
}
@fragment fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color, 1.0);
}
"#;

fn main() {
    if let Err(error) = common::run_example("Dynamic uniform buffers", create_example()) {
        eprintln!("04_dynamic_uniform_buffer: {error}");
        std::process::exit(1);
    }
}

pub fn create_example() -> DynamicUniformExample {
    DynamicUniformExample {
        workload: None,
        lane: None,
        last_frame: None,
    }
}

pub struct DynamicUniformExample {
    workload: Option<Workload>,
    lane: Option<SubmissionLaneId>,
    last_frame: Option<Instant>,
}

impl common::Example for DynamicUniformExample {
    fn init(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let device = context.device().clone();
        let lane = device
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|lane| {
                lane.domains()
                    .contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY))
            })
            .map(|lane| lane.id())
            .ok_or_else(|| std::io::Error::other("no COPY|RASTER lane"))?;
        self.workload = Some(common::block_on(Workload::new(
            &device,
            context.presentation_mut().configuration().format(),
            Extent3d::d2(context.extent().width, context.extent().height),
            lane,
        ))?);
        self.lane = Some(lane);
        self.last_frame = Some(Instant::now());
        Ok(())
    }

    fn update(
        &mut self,
        _: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = Instant::now();
        let delta = now
            .checked_duration_since(self.last_frame.unwrap_or(now))
            .unwrap_or_default()
            .as_secs_f32();
        self.last_frame = Some(now);
        if let Some(workload) = &mut self.workload {
            workload.update_models(delta);
        }
        Ok(())
    }

    fn render(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let frame = common::block_on(context.presentation_mut().acquire())?;
        common::block_on(
            self.workload
                .as_mut()
                .ok_or_else(|| std::io::Error::other("workload not initialized"))?
                .render(
                    context.device(),
                    self.lane.expect("lane set during init"),
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
            workload.extent = Extent3d::d2(width, height);
            workload.depth = depth(context.device(), workload.extent)?;
        }
        Ok(())
    }

    fn device_lost(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.workload = None;
        self.lane = None;
        self.last_frame = None;
        Ok(())
    }

    fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.device_lost()
    }
}

struct Workload {
    pipeline: RasterPipeline,
    groups: [BindGroup; FRAMES],
    vertex: BufferBinding,
    index: BufferBinding,
    view: [Buffer; FRAMES],
    models: [Buffer; FRAMES],
    rotations: [[f32; 3]; OBJECT_COUNT as usize],
    rotation_speeds: [[f32; 3]; OBJECT_COUNT as usize],
    matrices: [[f32; 16]; OBJECT_COUNT as usize],
    dynamic_alignment: u64,
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
        let min_uniform_alignment = device
            .capabilities()
            .limit(LimitKey::MinUniformBufferOffsetAlignment)
            .ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::Unsupported,
                    "adapter did not publish MinUniformBufferOffsetAlignment",
                )
            })?;
        let dynamic_alignment = align_up(MODEL_BYTES, min_uniform_alignment);
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::VERTEX,
                BindingKind::UniformBuffer {
                    min_size: VIEW_BYTES,
                },
            ),
            BindingSlot::new(
                BindingSlotId::new(1),
                ShaderStages::VERTEX,
                BindingKind::UniformBuffer {
                    min_size: MODEL_BYTES,
                },
            )
            .with_dynamic_offset(true),
        ]))?;
        let interface = device
            .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        let depth_state = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(true));
        let pipeline = device
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(device, &vertex_artifact()).await?,
                    interface,
                )
                .with_fragment(common::shader::create_shader(device, &fragment_artifact()).await?)
                .with_vertex_input(
                    VertexInputState::new().with_buffer(
                        VertexBufferLayout::new(24, VertexStepMode::Vertex)
                            .with_attribute(VertexAttribute::new(
                                ShaderLocation::new(0),
                                VertexFormat::Float32x3,
                                0,
                            ))
                            .with_attribute(VertexAttribute::new(
                                ShaderLocation::new(1),
                                VertexFormat::Float32x3,
                                12,
                            )),
                    ),
                )
                .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleList))
                .with_depth_stencil(depth_state)
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(color_format)),
            )
            .await?;

        let vertices = cube_vertices();
        let indices = cube_indices();
        let vertex_buffer = device.create_buffer(
            &BufferDescriptor::new(
                vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("04 dynamic UBO cube vertices"),
        )?;
        let index_buffer = device.create_buffer(
            &BufferDescriptor::new(
                indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label("04 dynamic UBO cube indices"),
        )?;
        let view = [
            uniform_buffer(device, VIEW_BYTES, "view 0")?,
            uniform_buffer(device, VIEW_BYTES, "view 1")?,
        ];
        let dynamic_size = dynamic_alignment * u64::from(OBJECT_COUNT);
        let models = [
            uniform_buffer(device, dynamic_size, "dynamic models 0")?,
            uniform_buffer(device, dynamic_size, "dynamic models 1")?,
        ];
        let groups = [
            dynamic_group(device, &layout, &view[0], &models[0], dynamic_alignment)?,
            dynamic_group(device, &layout, &view[1], &models[1], dynamic_alignment)?,
        ];

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
        let mut plan = SubmissionPlanBuilder::new(device);
        plan.add_batch(lane, vec![recorder.finish()?])?;
        let receipt = device.submit(plan.build()?)?;
        let _ = device.wait_completion(receipt.completion()).await?;

        let (rotations, rotation_speeds) = rotations();
        let mut workload = Self {
            pipeline,
            groups,
            vertex: BufferBinding::new(vertex_buffer, BufferRange::new(0, 8 * 24)),
            index: BufferBinding::new(index_buffer, BufferRange::new(0, 36 * 4)),
            view,
            models,
            rotations,
            rotation_speeds,
            matrices: [[0.0; 16]; OBJECT_COUNT as usize],
            dynamic_alignment,
            depth: depth(device, extent)?,
            extent,
            next: 0,
            completions: [None, None],
        };
        // C++ writes the same initial random rotations before its first draw.
        workload.update_models(0.0);
        Ok(workload)
    }

    fn update_models(&mut self, frame_timer: f32) {
        for x in 0..OBJECTS_PER_AXIS {
            for y in 0..OBJECTS_PER_AXIS {
                for z in 0..OBJECTS_PER_AXIS {
                    let index = (x * OBJECTS_PER_AXIS * OBJECTS_PER_AXIS + y * OBJECTS_PER_AXIS + z)
                        as usize;
                    for component in 0..3 {
                        self.rotations[index][component] +=
                            frame_timer * self.rotation_speeds[index][component];
                    }
                    let position = [
                        -((OBJECTS_PER_AXIS as f32 * 5.0) / 2.0) + 2.5 + x as f32 * 5.0,
                        -((OBJECTS_PER_AXIS as f32 * 5.0) / 2.0) + 2.5 + y as f32 * 5.0,
                        -((OBJECTS_PER_AXIS as f32 * 5.0) / 2.0) + 2.5 + z as f32 * 5.0,
                    ];
                    let rotation = self.rotations[index];
                    self.matrices[index] = mm(
                        mm(
                            mm(translation(position), rotate(rotation[0], [1.0, 1.0, 0.0])),
                            rotate(rotation[1], [0.0, 1.0, 0.0]),
                        ),
                        rotate(rotation[2], [0.0, 0.0, 1.0]),
                    );
                }
            }
        }
    }

    async fn render(
        &mut self,
        device: &Device,
        lane: SubmissionLaneId,
        frame: AcquiredFrame,
    ) -> RhiResult<()> {
        let slot = self.next;
        if let Some(completion) = self.completions[slot].take() {
            let _ = device.wait_completion(completion).await?;
        }
        let extent = frame.attachment().extent();
        if extent.width == 0 || extent.height == 0 {
            return Ok(());
        }

        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            self.view[slot].clone(),
            0,
            camera_bytes(extent),
        ))?)?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            self.models[slot].clone(),
            0,
            dynamic_model_bytes(&self.matrices, self.dynamic_alignment),
        ))?)?;
        {
            let mut raster = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(frame.attachment()),
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
                extent.width as f32,
                extent.height as f32,
                0.0,
                1.0,
            ))?;
            raster.set_scissor(Rect::new(0, 0, extent.width, extent.height))?;
            raster.set_vertex_buffer(0, &self.vertex)?;
            raster.set_index_buffer(&self.index, IndexFormat::Uint32)?;
            for object in 0..OBJECT_COUNT {
                raster.set_bind_group(
                    BindGroupIndex::new(0),
                    &self.groups[slot],
                    &[u32::try_from(u64::from(object) * self.dynamic_alignment)
                        .expect("dynamic buffer is smaller than u32 offset space")],
                )?;
                raster.draw_indexed(0..36, 0, 0..1)?;
            }
            raster.end()?;
        }
        let mut plan = SubmissionPlanBuilder::new(device);
        let point = plan.add_batch(lane, vec![recorder.finish()?])?;
        plan.present_after(frame, point)?;
        let receipt = device.submit(plan.build()?)?;
        self.completions[slot] = Some(receipt.completion());
        let _ = device.wait_present(receipt.presents()[0].id()).await?;
        self.next = (slot + 1) % FRAMES;
        Ok(())
    }
}

fn uniform_buffer(device: &Device, bytes: u64, name: &str) -> RhiResult<Buffer> {
    device.create_buffer(
        &BufferDescriptor::new(bytes, BufferUsage::UNIFORM.union(BufferUsage::COPY_DST))
            .with_label(format!("04 {name}")),
    )
}

fn dynamic_group(
    device: &Device,
    layout: &BindGroupLayout,
    view: &Buffer,
    models: &Buffer,
    dynamic_alignment: u64,
) -> RhiResult<BindGroup> {
    device.create_bind_group(&BindGroupDescriptor::new(layout.clone()).with_entries([
        BindGroupEntry::new(
            BindingSlotId::new(0),
            BindingResource::Buffer(BufferBinding::new(
                view.clone(),
                BufferRange::new(0, VIEW_BYTES),
            )),
        ),
        // C++ overrides VkDescriptorBufferInfo::range with dynamicAlignment.
        BindGroupEntry::new(
            BindingSlotId::new(1),
            BindingResource::Buffer(BufferBinding::new(
                models.clone(),
                BufferRange::new(0, dynamic_alignment),
            )),
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

fn vertex_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Vertex,
        "vs_main",
        ShaderInterface::new()
            .with_resource(resource(0, VIEW_BYTES))
            .with_resource(resource(1, MODEL_BYTES))
            .with_input(vertex_input(0))
            .with_input(vertex_input(1))
            .with_output(varying(0))
            .with_writes_position(true),
        [4; 32],
    )
}
fn fragment_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Fragment,
        "fs_main",
        ShaderInterface::new()
            .with_input(varying(0))
            .with_output(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 4,
                interpolation: None,
            }),
        [5; 32],
    )
}
fn resource(slot: u32, min_size: u64) -> ShaderResourceRequirement {
    ShaderResourceRequirement {
        group: BindGroupIndex::new(0),
        slot: BindingSlotId::new(slot),
        kind: BindingKind::UniformBuffer { min_size },
        count: BindingCount::One,
    }
}
fn vertex_input(location: u32) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(location),
        numeric_type: ShaderNumericType::Float32,
        components: 3,
        interpolation: None,
    }
}
fn varying(location: u32) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(location),
        numeric_type: ShaderNumericType::Float32,
        components: 3,
        interpolation: Some(ShaderInterpolation {
            mode: InterpolationMode::Perspective,
            sampling: InterpolationSampling::Center,
        }),
    }
}
fn artifact(
    stage: ShaderStage,
    entry: &str,
    interface: ShaderInterface,
    hash: [u8; 32],
) -> ShaderArtifact {
    ShaderArtifact::new(
        stage,
        entry,
        ShaderCode::Wgsl(Arc::from(WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        ArtifactHash(hash),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}

fn cube_vertices() -> Vec<u8> {
    const VERTICES: [f32; 48] = [
        -1., -1., 1., 1., 0., 0., 1., -1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., -1., 1., 1., 0.,
        0., 0., -1., -1., -1., 1., 0., 0., 1., -1., -1., 0., 1., 0., 1., 1., -1., 0., 0., 1., -1.,
        1., -1., 0., 0., 0.,
    ];
    VERTICES.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn cube_indices() -> Vec<u8> {
    const INDICES: [u32; 36] = [
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4,
        3, 2, 6, 6, 7, 3,
    ];
    INDICES.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn rotations() -> (
    [[f32; 3]; OBJECT_COUNT as usize],
    [[f32; 3]; OBJECT_COUNT as usize],
) {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as u32)
        .unwrap_or(0);
    let mut rng = Rng(seed);
    let mut rotations = [[0.0; 3]; OBJECT_COUNT as usize];
    let mut speeds = [[0.0; 3]; OBJECT_COUNT as usize];
    for index in 0..OBJECT_COUNT as usize {
        rotations[index] = [
            rng.normal() * std::f32::consts::TAU,
            rng.normal() * std::f32::consts::TAU,
            rng.normal() * std::f32::consts::TAU,
        ];
        speeds[index] = [rng.normal(), rng.normal(), rng.normal()];
    }
    (rotations, speeds)
}

struct Rng(u32);
impl Rng {
    fn uniform_open(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) as f32 + 0.5) / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform_open();
        -1.0 + radius * angle.cos()
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    value.div_ceil(alignment) * alignment
}

fn camera_bytes(extent: Extent3d) -> Vec<u8> {
    let focal = 1.0 / 30f32.to_radians().tan();
    let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
    let projection = [
        focal / aspect,
        0.,
        0.,
        0.,
        0.,
        focal,
        0.,
        0.,
        0.,
        0.,
        256. / (0.1 - 256.0),
        -1.,
        0.,
        0.,
        25.6 / (0.1 - 256.0),
        0.,
    ];
    let view = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., -30., 1.,
    ];
    projection
        .into_iter()
        .chain(view)
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn dynamic_model_bytes(
    matrices: &[[f32; 16]; OBJECT_COUNT as usize],
    dynamic_alignment: u64,
) -> Vec<u8> {
    let mut bytes = vec![0; (dynamic_alignment * u64::from(OBJECT_COUNT)) as usize];
    for (index, matrix) in matrices.iter().enumerate() {
        let start = index * dynamic_alignment as usize;
        for (component, value) in matrix.iter().enumerate() {
            bytes[start + component * 4..start + component * 4 + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn translation(position: [f32; 3]) -> [f32; 16] {
    [
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        position[0],
        position[1],
        position[2],
        1.,
    ]
}
fn rotate(angle: f32, axis: [f32; 3]) -> [f32; 16] {
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let [x, y, z] = [axis[0] / length, axis[1] / length, axis[2] / length];
    let (sine, cosine) = angle.sin_cos();
    let inverse = 1.0 - cosine;
    [
        cosine + x * x * inverse,
        y * x * inverse + z * sine,
        z * x * inverse - y * sine,
        0.,
        x * y * inverse - z * sine,
        cosine + y * y * inverse,
        z * y * inverse + x * sine,
        0.,
        x * z * inverse + y * sine,
        y * z * inverse - x * sine,
        cosine + z * z * inverse,
        0.,
        0.,
        0.,
        0.,
        1.,
    ]
}
fn mm(left: [f32; 16], right: [f32; 16]) -> [f32; 16] {
    let mut output = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4)
                .map(|index| left[index * 4 + row] * right[column * 4 + index])
                .sum();
        }
    }
    output
}
