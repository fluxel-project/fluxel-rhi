//! Faithful port of SaschaWillems/Vulkan `examples/pushconstants`.
//!
//! The source scene uses the embedded `sphere.gltf`, one 192-byte camera UBO,
//! and one vertex-stage `color + position` push-constant record for each of
//! sixteen indexed sphere draws.

pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
        ColorTargetState, CullMode, DepthState, DepthStencilState, FrontFace, ImmediateRange,
        PipelineInterfaceDescriptor, PrimitiveState, PrimitiveTopology, RasterPipeline,
        RasterPipelineDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat,
        VertexInputState, VertexStepMode,
    },
    platform::{Device, LimitKey, OptionalFeature},
    presentation::AcquiredFrame,
    resource::{
        Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor, BufferUsage,
        CompareFunction, Extent3d, TextureDescriptor, TextureUsage, TextureView,
        TextureViewDescriptor, TextureViewDimension,
    },
    shader::{
        ArtifactHash, ArtifactProducerVersion, InterpolationMode, InterpolationSampling,
        ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderImmediateRequirement, ShaderInterface,
        ShaderInterpolation, ShaderLocation, ShaderLocationInterface, ShaderNumericType,
        ShaderRequirements, ShaderResourceRequirement, ShaderStage, ShaderStages,
    },
    submission::{CompletionPoint, LaneWorkDomains, SubmissionLaneId, SubmissionPlanBuilder},
};
use gltf::{LoadOptions, VKGLTF_VERTEX_STRIDE, load_embedded_model};

const FRAMES_IN_FLIGHT: usize = 2;
const SPHERE_COUNT: usize = 16;
const PUSH_BYTES: u32 = 32;
const UNIFORM_BYTES: u64 = 192;
const SPHERE_GLTF: &[u8] = include_bytes!("assets/models/sphere.gltf");

// Direct WGSL transcription of shaders/glsl/pushconstants/{pushconstants.vert,frag}.
const PUSH_CONSTANTS_WGSL: &str = r#"
struct Uniforms {
    projection: mat4x4<f32>,
    model: mat4x4<f32>,
    view: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> ubo: Uniforms;

struct PushConsts { color: vec4<f32>, position: vec4<f32>, };
var<push_constant> push_consts: PushConsts;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.color = input.color.rgb * push_consts.color.rgb;
    let loc_pos = (ubo.model * vec4<f32>(input.position, 1.0)).xyz;
    let world_pos = loc_pos + push_consts.position.xyz;
    output.position = ubo.projection * ubo.view * vec4<f32>(world_pos, 1.0);
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
"#;

fn main() {
    if let Err(error) = common::run_example("Push constants", create_example()) {
        eprintln!("05_push_constants: {error}");
        std::process::exit(1);
    }
}

pub fn create_example() -> PushConstantsExample {
    PushConstantsExample {
        workload: None,
        lane: None,
    }
}

pub struct PushConstantsExample {
    workload: Option<PushConstantsWorkload>,
    lane: Option<SubmissionLaneId>,
}

impl common::Example for PushConstantsExample {
    fn init(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let device = context.device().clone();
        let capabilities = device.capabilities();
        if !capabilities.supports_feature(OptionalFeature::Immediates) {
            return Err("05_push_constants requires native immediate data support".into());
        }
        if capabilities.limit(LimitKey::MaxImmediateSize).unwrap_or(0) < u64::from(PUSH_BYTES) {
            return Err(format!("05_push_constants requires {PUSH_BYTES} immediate bytes").into());
        }
        let required = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
        let lane = capabilities
            .submission()
            .lanes()
            .iter()
            .find(|candidate| candidate.domains().contains(required))
            .map(|candidate| candidate.id())
            .ok_or_else(|| std::io::Error::other("device has no COPY|RASTER submission lane"))?;
        let extent = context.extent();
        self.workload = Some(common::block_on(PushConstantsWorkload::new(
            &device,
            context.presentation_mut().configuration().format(),
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
            .ok_or_else(|| std::io::Error::other("push-constant workload is not initialized"))?;
        common::block_on(
            workload.render(
                context.device(),
                self.lane.ok_or_else(|| {
                    std::io::Error::other("push-constant lane is not initialized")
                })?,
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

#[derive(Clone, Copy)]
struct PushData {
    color: [f32; 4],
    position: [f32; 4],
}

impl PushData {
    fn to_bytes(self) -> [u8; PUSH_BYTES as usize] {
        let mut result = [0; PUSH_BYTES as usize];
        for (index, value) in self.color.into_iter().chain(self.position).enumerate() {
            result[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        result
    }
}

/// Source-equivalent setup: randomized RGB in [0.1, 1.0], and sixteen points
/// distributed around a radius-3.5 circle in the XY plane.
fn setup_spheres() -> [PushData; SPHERE_COUNT] {
    let mut random = SourceRandom::new();
    std::array::from_fn(|index| {
        let angle = (index as f32 * 360.0 / SPHERE_COUNT as f32).to_radians();
        let (sin, cos) = angle.sin_cos();
        PushData {
            color: [
                random.uniform_01_to_1(),
                random.uniform_01_to_1(),
                random.uniform_01_to_1(),
                1.0,
            ],
            position: [sin * 3.5, cos * 3.5, 0.0, 1.0],
        }
    })
}

/// The C++ sample seeds `std::default_random_engine` from `std::random_device`.
/// Rust has no standard equivalent, so this preserves its startup-random color
/// range and call order with a local nondeterministic seed.
struct SourceRandom(u64);
impl SourceRandom {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0x9e37_79b9_7f4a_7c15);
        Self(seed ^ 0xa076_1d64_78bd_642f)
    }
    fn uniform_01_to_1(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let value = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32;
        0.1 + 0.9 * (value as f32 / u32::MAX as f32)
    }
}

pub struct PushConstantsWorkload {
    pipeline: RasterPipeline,
    groups: [BindGroup; FRAMES_IN_FLIGHT],
    uniforms: [Buffer; FRAMES_IN_FLIGHT],
    vertex: BufferBinding,
    index: BufferBinding,
    index_count: u32,
    depth: TextureView,
    extent: Extent3d,
    spheres: [PushData; SPHERE_COUNT],
    next_frame: usize,
    completions: [Option<CompletionPoint>; FRAMES_IN_FLIGHT],
}

impl PushConstantsWorkload {
    async fn new(
        device: &Device,
        color_format: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let model = load_embedded_model(SPHERE_GLTF, LoadOptions::CPP_PORT)
            .map_err(|error| RhiError::new(RhiErrorKind::InvalidUsage, error.to_string()))?;
        let index_count = (model.indices.len() / std::mem::size_of::<u32>()) as u32;
        if index_count == 0 {
            return Err(RhiError::new(
                RhiErrorKind::InvalidUsage,
                "sphere.gltf has no indices",
            ));
        }
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::VERTEX,
                BindingKind::UniformBuffer {
                    min_size: UNIFORM_BYTES,
                },
            ),
        ]))?;
        let interface = device.create_pipeline_interface(
            &PipelineInterfaceDescriptor::new(vec![layout.clone()])
                .with_immediate_range(ImmediateRange::new(0, PUSH_BYTES, ShaderStages::VERTEX)),
        )?;
        let pipeline = device
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(device, &vertex_artifact()).await?,
                    interface,
                )
                .with_label("05_push_constants pipeline")
                .with_fragment(common::shader::create_shader(device, &fragment_artifact()).await?)
                .with_vertex_input(
                    VertexInputState::new().with_buffer(
                        VertexBufferLayout::new(
                            VKGLTF_VERTEX_STRIDE as u64,
                            VertexStepMode::Vertex,
                        )
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
                            VertexFormat::Float32x4,
                            32,
                        )),
                    ),
                )
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Back)
                        .with_front_face(FrontFace::Ccw),
                )
                .with_depth_stencil(
                    DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                        DepthState::new(CompareFunction::LessEqual).with_write_enabled(true),
                    ),
                )
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(color_format)),
            )
            .await?;
        let vertex_buffer = device.create_buffer(
            &BufferDescriptor::new(
                model.vertices.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("05_push_constants sphere vertices"),
        )?;
        let index_buffer = device.create_buffer(
            &BufferDescriptor::new(
                model.indices.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label("05_push_constants sphere indices"),
        )?;
        let uniforms = [
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("05_push_constants uniforms frame 0"),
            )?,
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("05_push_constants uniforms frame 1"),
            )?,
        ];
        let groups = [
            create_group(device, &layout, &uniforms[0])?,
            create_group(device, &layout, &uniforms[1])?,
        ];
        let depth = create_depth_view(device, extent)?;
        let vertex_len = model.vertices.len() as u64;
        let index_len = model.indices.len() as u64;
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            vertex_buffer.clone(),
            0,
            model.vertices,
        ))?)?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            index_buffer.clone(),
            0,
            model.indices,
        ))?)?;
        let mut plan = SubmissionPlanBuilder::new(device);
        plan.add_batch(lane, vec![recorder.finish()?])?;
        let receipt = device.submit(plan.build()?)?;
        let _ = device.wait_completion(receipt.completion()).await?;
        Ok(Self {
            pipeline,
            groups,
            uniforms,
            vertex: BufferBinding::new(vertex_buffer, BufferRange::new(0, vertex_len)),
            index: BufferBinding::new(index_buffer, BufferRange::new(0, index_len)),
            index_count,
            depth,
            extent,
            spheres: setup_spheres(),
            next_frame: 0,
            completions: [None, None],
        })
    }

    fn resize(&mut self, device: &Device, extent: Extent3d) -> RhiResult<()> {
        self.depth = create_depth_view(device, extent)?;
        self.extent = extent;
        Ok(())
    }

    async fn render(
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
            PushConstantsUniforms::reference_for_extent(self.extent).to_bytes(),
        ))?;
        let attachment = frame.attachment();
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&upload)?;
        {
            let mut raster = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_label("05_push_constants raster")
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
            for sphere in self.spheres {
                raster.set_immediates(0, &sphere.to_bytes())?;
                raster.draw_indexed(0..self.index_count, 0, 0..1)?;
            }
            raster.end()?;
        }
        let mut plan = SubmissionPlanBuilder::new(device);
        let point = plan.add_batch(lane, vec![recorder.finish()?])?;
        plan.present_after(frame, point)?;
        let receipt = device.submit(plan.build()?)?;
        self.completions[slot] = Some(receipt.completion());
        let present = receipt.presents().first().expect("one present receipt");
        let _ = device.wait_present(present.id()).await?;
        self.next_frame = (slot + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }
}

fn create_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform: &Buffer,
) -> RhiResult<BindGroup> {
    device.create_bind_group(&BindGroupDescriptor::new(layout.clone()).with_entry(
        BindGroupEntry::new(
            BindingSlotId::new(0),
            BindingResource::Buffer(BufferBinding::new(
                uniform.clone(),
                BufferRange::new(0, UNIFORM_BYTES),
            )),
        ),
    ))
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
struct PushConstantsUniforms {
    projection: [f32; 16],
    model: [f32; 16],
    view: [f32; 16],
}
impl PushConstantsUniforms {
    fn reference_for_extent(extent: Extent3d) -> Self {
        let aspect = extent.width.max(1) as f32 / extent.height.max(1) as f32;
        let f = 1.0 / 30.0_f32.to_radians().tan();
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
                0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            view: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -10.0, 1.0,
            ],
        }
    }
    fn to_bytes(self) -> Vec<u8> {
        self.projection
            .into_iter()
            .chain(self.model)
            .chain(self.view)
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
            .with_immediate_requirement(ShaderImmediateRequirement {
                offset: 0,
                size: PUSH_BYTES,
            })
            .with_input(io(0, 3, None))
            .with_input(io(1, 3, None))
            .with_input(io(2, 4, None))
            .with_output(io(0, 3, Some(interpolation())))
            .with_writes_position(true),
        ArtifactHash([0x51; 32]),
    )
}
fn fragment_artifact() -> ShaderArtifact {
    artifact(
        ShaderStage::Fragment,
        "fs_main",
        ShaderInterface::new()
            .with_input(io(0, 3, Some(interpolation())))
            .with_output(io(0, 4, None)),
        ArtifactHash([0x52; 32]),
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
        ShaderCode::Wgsl(Arc::from(PUSH_CONSTANTS_WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        hash,
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
