//! SaschaWillems `triangle` port through the shared native example framework.

pub mod common;

fn main() {
    if let Err(error) = common::run_example("Basic indexed triangle", create_example()) {
        eprintln!("01_triangle: {error}");
        std::process::exit(1);
    }
}

pub fn create_example() -> TriangleExample {
    TriangleExample {
        workload: None,
        lane: None,
    }
}

pub struct TriangleExample {
    workload: Option<TriangleWorkload>,
    lane: Option<fluxel_rhi::api::submission::SubmissionLaneId>,
}

impl common::Example for TriangleExample {
    fn init(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use fluxel_rhi::api::submission::LaneWorkDomains;

        let device = context.device().clone();
        let required = LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY);
        let lane = device
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|lane| lane.domains().contains(required))
            .map(|lane| lane.id())
            .ok_or_else(|| std::io::Error::other("device has no COPY|RASTER submission lane"))?;
        let format = context.presentation_mut().configuration().format();
        let session_extent = context.extent();
        let extent = Extent3d::d2(session_extent.width, session_extent.height);
        let hashes = TriangleShaderHashes {
            // Stable example artifact identities. They are only used as local
            // content keys; the source and interface remain the authoritative
            // shader program passed to RHI.
            vertex: fluxel_rhi::api::shader::ArtifactHash([0x01; 32]),
            fragment: fluxel_rhi::api::shader::ArtifactHash([0x02; 32]),
        };
        self.workload = Some(common::block_on(TriangleWorkload::new(
            &device, format, extent, hashes, lane,
        ))?);
        self.lane = Some(lane);
        Ok(())
    }

    fn update(
        &mut self,
        _context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    fn render(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let workload = self
            .workload
            .as_mut()
            .ok_or_else(|| std::io::Error::other("triangle workload is not initialized"))?;
        let lane = self
            .lane
            .ok_or_else(|| std::io::Error::other("triangle lane is not initialized"))?;
        let device = context.device().clone();
        let frame = common::block_on(context.presentation_mut().acquire())?;
        let extent = frame.attachment().extent();
        let matrices = TriangleMatrices::reference_for_extent(extent)
            .ok_or_else(|| std::io::Error::other("presentation extent is zero"))?;
        common::block_on(workload.render(&device, lane, frame, matrices))?;
        Ok(())
    }

    fn resize(
        &mut self,
        context: &mut common::ExampleContext<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(workload) = self.workload.as_mut() {
            workload.resize(
                context.device(),
                fluxel_rhi::api::resource::Extent3d::d2(width, height),
            )?;
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

use std::sync::Arc;

use fluxel_rhi::api::binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor, BindingKind,
    BindingResource, BindingSlot, BindingSlotId,
};
use fluxel_rhi::api::command::{
    ColorAttachment, ColorAttachmentView, ColorClearValue, DepthAttachmentMode,
    DepthStencilAttachment, IndexFormat, LoadOp, RasterScopeDescriptor, RecorderDescriptor, Rect,
    StoreOp, Viewport,
};
use fluxel_rhi::api::error::RhiResult;
use fluxel_rhi::api::format::TextureFormat;
use fluxel_rhi::api::pipeline::{
    ColorTargetState, DepthState, DepthStencilState, PipelineInterfaceDescriptor, PrimitiveState,
    PrimitiveTopology, RasterPipeline, RasterPipelineDescriptor, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
};
use fluxel_rhi::api::platform::Device;
use fluxel_rhi::api::presentation::AcquiredFrame;
use fluxel_rhi::api::resource::{
    BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor, BufferUsage, Extent3d,
    TextureDescriptor, TextureUsage, TextureViewDescriptor, TextureViewDimension,
};
use fluxel_rhi::api::shader::{
    ArtifactHash, ArtifactProducerVersion, InterpolationMode, InterpolationSampling,
    ShaderAbiVersion, ShaderArtifact, ShaderCode, ShaderInterface, ShaderInterpolation,
    ShaderLocation, ShaderLocationInterface, ShaderNumericType, ShaderRequirements,
    ShaderResourceRequirement, ShaderStage, ShaderStages,
};
use fluxel_rhi::api::submission::{CompletionPoint, SubmissionLaneId, SubmissionPlanBuilder};

const UNIFORM_BYTES: u64 = 3 * 16 * 4;
const FRAMES_IN_FLIGHT: usize = 2;

/// WGSL equivalent of the reference GLSL shaders.
///
/// The matrix member order matches the C++ `ShaderData`: projection, model,
/// view. WGSL matrices are column-major, as are the source GLSL `mat4`s.
const TRIANGLE_WGSL: &str = r#"
struct Uniforms {
    projection: mat4x4<f32>,
    model: mat4x4<f32>,
    view: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.projection * uniforms.view * uniforms.model * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
"#;

/// Hashes supplied by the example artifact-generation step.
///
/// `ShaderArtifact::content_hash` covers the complete artifact contract, not
/// only WGSL text. The public RHI deliberately has no public artifact hasher, so
/// this source accepts generated values rather than inventing an incompatible
/// local hash algorithm.
#[derive(Clone, Copy)]
pub struct TriangleShaderHashes {
    /// Content hash for the `vs_main` artifact.
    pub vertex: ArtifactHash,
    /// Content hash for the `fs_main` artifact.
    pub fragment: ArtifactHash,
}

/// The C++ sample's three column-major `mat4` uniform members.
#[derive(Clone, Copy)]
pub struct TriangleMatrices {
    /// Perspective projection matrix.
    pub projection: [f32; 16],
    /// Model matrix. The reference sample passes identity.
    pub model: [f32; 16],
    /// View matrix from the reference sample's look-at camera.
    pub view: [f32; 16],
}

impl TriangleMatrices {
    /// Builds the reference camera state for a non-zero presentation extent.
    ///
    /// This transcribes `camera.setPerspective(60°, aspect, 1, 256)`, position
    /// `(0, 0, -2.5)`, zero rotation, and identity model from the C++ sample.
    /// The result is `None` while a host surface is zero-sized or suspended.
    pub fn reference_for_extent(extent: Extent3d) -> Option<Self> {
        if extent.width == 0 || extent.height == 0 {
            return None;
        }
        let f = 1.0 / 30.0_f32.to_radians().tan();
        let aspect = extent.width as f32 / extent.height as f32;
        let near = 1.0;
        let far = 256.0;
        Some(Self {
            // GLM configured with GLM_FORCE_DEPTH_ZERO_TO_ONE, column-major.
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
                (far * near) / (near - far),
                0.0,
            ],
            model: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            // lookat camera with zero rotation: glm::translate(identity, position).
            view: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -2.5, 1.0,
            ],
        })
    }

    /// Serializes the three WGSL `mat4x4<f32>` members in declaration order.
    pub fn to_bytes(self) -> Vec<u8> {
        self.projection
            .into_iter()
            .chain(self.model)
            .chain(self.view)
            .flat_map(f32::to_le_bytes)
            .collect()
    }
}

/// The host chooses a raster-capable lane from `Device::capabilities()`.
///
/// Keeping that selection in the host framework permits one executable layout
/// on desktop, browser, Android, and Apple hosts without backend names here.
pub struct TriangleWorkload {
    pipeline: RasterPipeline,
    vertex: BufferBinding,
    index: BufferBinding,
    uniform_groups: [fluxel_rhi::api::binding::BindGroup; FRAMES_IN_FLIGHT],
    uniforms: [fluxel_rhi::api::resource::Buffer; FRAMES_IN_FLIGHT],
    depth: fluxel_rhi::api::resource::TextureView,
    extent: Extent3d,
    next_frame: usize,
    completions: [Option<CompletionPoint>; FRAMES_IN_FLIGHT],
}

impl TriangleWorkload {
    /// Creates portable persistent state for one presentation extent and format.
    pub async fn new(
        device: &Device,
        color_format: TextureFormat,
        extent: Extent3d,
        hashes: TriangleShaderHashes,
        upload_lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::VERTEX,
                BindingKind::UniformBuffer {
                    min_size: UNIFORM_BYTES,
                },
            ),
        ]))?;
        let interface = device
            .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        let vertex_shader =
            common::shader::create_shader(device, &vertex_artifact(hashes.vertex)).await?;
        let fragment_shader =
            common::shader::create_shader(device, &fragment_artifact(hashes.fragment)).await?;

        let vertex_input = VertexInputState::new().with_buffer(
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
        );
        let pipeline = device
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(vertex_shader, interface)
                    .with_fragment(fragment_shader)
                    .with_vertex_input(vertex_input)
                    .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleList))
                    .with_depth_stencil(
                        DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                            DepthState::new(fluxel_rhi::api::resource::CompareFunction::LessEqual)
                                .with_write_enabled(true),
                        ),
                    )
                    .with_color_target(ShaderLocation::new(0), ColorTargetState::new(color_format)),
            )
            .await?;

        let vertex_bytes = vertex_bytes();
        let index_bytes = index_bytes();
        let vertex_buffer = device.create_buffer(
            &BufferDescriptor::new(
                vertex_bytes.len() as u64,
                BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            )
            .with_label("01_triangle vertex buffer"),
        )?;
        let index_buffer = device.create_buffer(
            &BufferDescriptor::new(
                index_bytes.len() as u64,
                BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            )
            .with_label("01_triangle index buffer"),
        )?;

        let uniforms = [
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("01_triangle uniforms frame 0"),
            )?,
            device.create_buffer(
                &BufferDescriptor::new(
                    UNIFORM_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                )
                .with_label("01_triangle uniforms frame 1"),
            )?,
        ];
        let uniform_groups = [
            create_uniform_group(device, &layout, &uniforms[0])?,
            create_uniform_group(device, &layout, &uniforms[1])?,
        ];
        let depth = create_depth_view(device, extent)?;

        // This is the C++ sample's staging/upload phase.  Waiting before the
        // temporary upload jobs disappear preserves its completion lifetime.
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            vertex_buffer.clone(),
            0,
            vertex_bytes,
        ))?)?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            index_buffer.clone(),
            0,
            index_bytes,
        ))?)?;
        let work = recorder.finish()?;
        let mut plan = SubmissionPlanBuilder::new(device);
        plan.add_batch(upload_lane, vec![work])?;
        let receipt = device.submit(plan.build()?)?;
        let _ = device.wait_completion(receipt.completion()).await?;

        Ok(Self {
            pipeline,
            vertex: BufferBinding::new(vertex_buffer, BufferRange::new(0, 72)),
            index: BufferBinding::new(index_buffer, BufferRange::new(0, 12)),
            uniform_groups,
            uniforms,
            depth,
            extent,
            next_frame: 0,
            completions: [None, None],
        })
    }

    /// Recreates the depth attachment after the host has reconfigured surface
    /// presentation. The graphics pipeline remains valid because its color format
    /// is unchanged; a format change requires a new workload.
    pub fn resize(&mut self, device: &Device, extent: Extent3d) -> RhiResult<()> {
        self.depth = create_depth_view(device, extent)?;
        self.extent = extent;
        Ok(())
    }

    /// Records one C++-equivalent frame and presents its acquired drawable.
    ///
    /// `lane` must accept both raster and copy work because this frame updates
    /// its uniform buffer then consumes it in the same ordered submission.
    pub async fn render(
        &mut self,
        device: &Device,
        lane: SubmissionLaneId,
        frame: AcquiredFrame,
        matrices: TriangleMatrices,
    ) -> RhiResult<()> {
        #[cfg(target_os = "android")]
        eprintln!("FLUXEL_WORKLOAD_BEGIN");
        let attachment = frame.attachment();

        let slot = self.next_frame;
        if let Some(completion) = self.completions[slot].take() {
            let _ = device.wait_completion(completion).await?;
        }

        let upload = device.create_buffer_upload(BufferUploadDescriptor::new(
            self.uniforms[slot].clone(),
            0,
            matrices.to_bytes(),
        ))?;
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&upload)?;
        {
            let mut raster = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_label("01_triangle raster")
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
            raster.set_bind_group(BindGroupIndex::new(0), &self.uniform_groups[slot], &[])?;
            raster.set_vertex_buffer(0, &self.vertex)?;
            raster.set_index_buffer(&self.index, IndexFormat::Uint32)?;
            raster.draw_indexed(0..3, 0, 0..1)?;
            raster.end()?;
        }
        let work = recorder.finish()?;
        let mut plan = SubmissionPlanBuilder::new(device);
        let point = plan.add_batch(lane, vec![work])?;
        plan.present_after(frame, point)?;
        let receipt = device.submit(plan.build()?)?;
        #[cfg(target_os = "android")]
        eprintln!("FLUXEL_AFTER_SUBMIT");
        self.completions[slot] = Some(receipt.completion());
        let present = receipt
            .presents()
            .first()
            .expect("one present_after call must yield one present receipt");
        let _ = device.wait_present(present.id()).await?;
        #[cfg(target_os = "android")]
        eprintln!("FLUXEL_AFTER_WAIT_PRESENT");
        self.next_frame = (slot + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }
}

fn create_uniform_group(
    device: &Device,
    layout: &fluxel_rhi::api::binding::BindGroupLayout,
    uniform: &fluxel_rhi::api::resource::Buffer,
) -> RhiResult<fluxel_rhi::api::binding::BindGroup> {
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

fn create_depth_view(
    device: &Device,
    extent: Extent3d,
) -> RhiResult<fluxel_rhi::api::resource::TextureView> {
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

fn vertex_artifact(hash: ArtifactHash) -> ShaderArtifact {
    ShaderArtifact::new(
        ShaderStage::Vertex,
        "vs_main",
        ShaderCode::Wgsl(Arc::from(TRIANGLE_WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new()
            .with_resource(ShaderResourceRequirement {
                group: BindGroupIndex::new(0),
                slot: BindingSlotId::new(0),
                kind: BindingKind::UniformBuffer {
                    min_size: UNIFORM_BYTES,
                },
                count: fluxel_rhi::api::binding::BindingCount::One,
            })
            .with_input(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 3,
                interpolation: None,
            })
            .with_input(ShaderLocationInterface {
                location: ShaderLocation::new(1),
                numeric_type: ShaderNumericType::Float32,
                components: 3,
                interpolation: None,
            })
            .with_output(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 3,
                interpolation: Some(ShaderInterpolation {
                    mode: InterpolationMode::Perspective,
                    sampling: InterpolationSampling::Center,
                }),
            })
            .with_writes_position(true),
        ShaderRequirements::new(),
        hash,
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}

fn fragment_artifact(hash: ArtifactHash) -> ShaderArtifact {
    ShaderArtifact::new(
        ShaderStage::Fragment,
        "fs_main",
        ShaderCode::Wgsl(Arc::from(TRIANGLE_WGSL)),
        ShaderAbiVersion { major: 1, minor: 0 },
        ShaderInterface::new()
            .with_input(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 3,
                interpolation: Some(ShaderInterpolation {
                    mode: InterpolationMode::Perspective,
                    sampling: InterpolationSampling::Center,
                }),
            })
            .with_output(ShaderLocationInterface {
                location: ShaderLocation::new(0),
                numeric_type: ShaderNumericType::Float32,
                components: 4,
                interpolation: None,
            }),
        ShaderRequirements::new(),
        hash,
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}

fn vertex_bytes() -> Vec<u8> {
    let values: [f32; 18] = [
        1.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0,
    ];
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn index_bytes() -> Vec<u8> {
    [0u32, 1, 2]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}
