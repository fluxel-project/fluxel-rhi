//! SaschaWillems/Vulkan `examples/multiview` port.
//!
//! Renders the room to both layers selected by view mask `0b11`, then displays
//! each eye in one half of the swapchain image using the original barrel warp.
pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use fluxel_rhi::api::{
    binding::*,
    command::*,
    error::{RhiError, RhiErrorKind, RhiResult},
    format::TextureFormat,
    pipeline::*,
    platform::{Device, OptionalFeature},
    presentation::AcquiredFrame,
    resource::*,
    shader::*,
    submission::*,
};
use std::sync::Arc;

const FRAMES: usize = 2;
const UBO_SIZE: u64 = 288;
const ROOM: &[u8] = include_bytes!("assets/models/sampleroom.gltf");

// WGSL's uniform struct trailing alignment produces the C++ UniformData size:
// 2 projection matrices, 2 model-view matrices, lightPos, distortionAlpha.
const ROOM_WGSL: &str = r#"
struct U { projection: array<mat4x4<f32>,2>, modelview: array<mat4x4<f32>,2>, light_pos: vec4<f32>, distortion_alpha: f32, };
@group(0) @binding(0) var<uniform> ubo: U;
struct I { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) color: vec3<f32>, };
struct O { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) color: vec3<f32>, @location(2) view_vec: vec3<f32>, @location(3) light_vec: vec3<f32>, };
@vertex fn vs_main(input: I, @builtin(view_index) view_index: u32) -> O {
  var output: O; let mv=ubo.modelview[view_index]; output.color=input.color;
  output.normal=mat3x3<f32>(mv[0].xyz,mv[1].xyz,mv[2].xyz)*input.normal;
  let world=mv*vec4<f32>(input.position,1.0);
  output.light_vec=(mv*ubo.light_pos).xyz-world.xyz; output.view_vec=-world.xyz;
  output.position=ubo.projection[view_index]*world; return output;
}
@fragment fn fs_main(input: O) -> @location(0) vec4<f32> {
  let n=normalize(input.normal); let l=normalize(input.light_vec); let v=normalize(input.view_vec);
  let r=reflect(-l,n); let ambient=vec3<f32>(0.1); let diffuse=max(dot(n,l),0.0)*vec3<f32>(1.0);
  let specular=pow(max(dot(r,v),0.0),16.0)*vec3<f32>(0.75);
  return vec4<f32>((ambient+diffuse)*input.color+specular,1.0);
}"#;

const DISPLAY_PREFIX: &str = r#"
struct U { projection: array<mat4x4<f32>,2>, modelview: array<mat4x4<f32>,2>, light_pos: vec4<f32>, distortion_alpha: f32, };
@group(0) @binding(0) var<uniform> ubo: U;
@group(0) @binding(1) var sampler_view: texture_2d_array<f32>;
@group(0) @binding(2) var view_sampler: sampler;
struct O { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, };
@vertex fn vs_main(@builtin(vertex_index) vertex_index: u32) -> O {
  var output: O; output.uv=vec2<f32>(f32((vertex_index<<1u)&2u),f32(vertex_index&2u));
  output.position=vec4<f32>(output.uv*2.0-1.0,0.0,1.0); return output;
}
@fragment fn fs_main(input: O) -> @location(0) vec4<f32> {
  let p1=2.0*input.uv-1.0; let p2=(p1/(1.0-ubo.distortion_alpha*length(p1))+1.0)*0.5;
  let inside=p2.x>=0.0 && p2.x<=1.0 && p2.y>=0.0 && p2.y<=1.0;
  return select(vec4<f32>(0.0),textureSample(sampler_view,view_sampler,p2,VIEW_LAYER),inside);
}"#;

fn main() {
    if let Err(error) = common::run_example("Multiview rendering", create_example()) {
        eprintln!("15_multiview: {error}");
        std::process::exit(1);
    }
}
pub fn create_example() -> Example {
    Example {
        work: None,
        lane: None,
    }
}
pub struct Example {
    work: Option<Work>,
    lane: Option<SubmissionLaneId>,
}

impl common::Example for Example {
    fn init(
        &mut self,
        context: &mut common::ExampleContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let device = context.device().clone();
        if !device
            .capabilities()
            .supports_feature(OptionalFeature::Multiview)
        {
            return Err(std::io::Error::other("15_multiview requires Multiview").into());
        }
        let lane = device
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|lane| {
                lane.domains()
                    .contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY))
            })
            .ok_or_else(|| std::io::Error::other("15_multiview needs a COPY|RASTER lane"))?
            .id();
        self.work = Some(common::block_on(Work::new(
            &device,
            context.presentation_mut().configuration().format(),
            Extent3d::d2(context.extent().width, context.extent().height),
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
        common::block_on(self.work.as_mut().expect("initialized").render(
            context.device(),
            self.lane.expect("initialized"),
            frame,
        ))?;
        Ok(())
    }
    fn resize(
        &mut self,
        context: &mut common::ExampleContext<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.work
            .as_mut()
            .expect("initialized")
            .resize(context.device(), Extent3d::d2(width, height))?;
        Ok(())
    }
    fn device_lost(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.work = None;
        Ok(())
    }
    fn close(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.device_lost()
    }
}

struct Work {
    room: RasterPipeline,
    display: [RasterPipeline; 2],
    scene_groups: [BindGroup; FRAMES],
    display_groups: [BindGroup; FRAMES],
    uniforms: [Buffer; FRAMES],
    display_layout: BindGroupLayout,
    sampler: Sampler,
    vertices: BufferBinding,
    indices: BufferBinding,
    color: TextureView,
    multiview_depth: TextureView,
    frame_depth: TextureView,
    color_format: TextureFormat,
    extent: Extent3d,
    frame: usize,
    complete: [Option<CompletionPoint>; FRAMES],
}
impl Work {
    async fn new(
        device: &Device,
        format: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let model =
            gltf::load_embedded_model(ROOM, gltf::LoadOptions::CPP_PORT).map_err(example_error)?;
        let scene_layout = layout(device, false)?;
        let display_layout = layout(device, true)?;
        let scene_interface =
            device.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![
                scene_layout.clone(),
            ]))?;
        let display_interface =
            device.create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![
                display_layout.clone(),
            ]))?;
        let depth = DepthStencilState::new(TextureFormat::Depth32Float)
            .with_depth(DepthState::new(CompareFunction::LessEqual).with_write_enabled(true));
        let room = pipeline(
            device,
            scene_interface,
            ROOM_WGSL,
            format,
            depth.clone(),
            CullMode::Back,
            true,
            0,
        )
        .await?;
        let display = [
            pipeline(
                device,
                display_interface.clone(),
                &display_shader(0),
                format,
                depth.clone(),
                CullMode::Front,
                false,
                2,
            )
            .await?,
            pipeline(
                device,
                display_interface,
                &display_shader(1),
                format,
                depth,
                CullMode::Front,
                false,
                4,
            )
            .await?,
        ];
        let (color, multiview_depth) = multiview_attachments(device, extent, format)?;
        let frame_depth = frame_depth(device, extent)?;
        let sampler = device.create_sampler(
            &SamplerDescriptor::new()
                .with_address_modes(
                    AddressMode::ClampToEdge,
                    AddressMode::ClampToEdge,
                    AddressMode::ClampToEdge,
                )
                .with_filters(FilterMode::Nearest, FilterMode::Nearest, FilterMode::Linear),
        )?;
        let uniforms = [uniform_buffer(device)?, uniform_buffer(device)?];
        let scene_groups = [
            bind_group(device, &scene_layout, &uniforms[0], None)?,
            bind_group(device, &scene_layout, &uniforms[1], None)?,
        ];
        let display_groups = [
            bind_group(
                device,
                &display_layout,
                &uniforms[0],
                Some((&color, &sampler)),
            )?,
            bind_group(
                device,
                &display_layout,
                &uniforms[1],
                Some((&color, &sampler)),
            )?,
        ];
        let vertex_len = model.vertices.len() as u64;
        let index_len = model.indices.len() as u64;
        let vertices = device.create_buffer(&BufferDescriptor::new(
            vertex_len,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let indices = device.create_buffer(&BufferDescriptor::new(
            index_len,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            vertices.clone(),
            0,
            model.vertices,
        ))?)?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            indices.clone(),
            0,
            model.indices,
        ))?)?;
        let mut plan = SubmissionPlanBuilder::new(device);
        plan.add_batch(lane, vec![recorder.finish()?])?;
        let upload = device.submit(plan.build()?)?;
        let _ = device.wait_completion(upload.completion()).await?;
        Ok(Self {
            room,
            display,
            scene_groups,
            display_groups,
            uniforms,
            display_layout,
            sampler,
            vertices: BufferBinding::new(vertices, BufferRange::new(0, vertex_len)),
            indices: BufferBinding::new(indices, BufferRange::new(0, index_len)),
            color,
            multiview_depth,
            frame_depth,
            color_format: format,
            extent,
            frame: 0,
            complete: [None, None],
        })
    }
    fn resize(&mut self, device: &Device, extent: Extent3d) -> RhiResult<()> {
        let (color, depth) = multiview_attachments(device, extent, self.color_format)?;
        self.color = color;
        self.multiview_depth = depth;
        self.frame_depth = frame_depth(device, extent)?;
        self.extent = extent;
        self.display_groups = [
            bind_group(
                device,
                &self.display_layout,
                &self.uniforms[0],
                Some((&self.color, &self.sampler)),
            )?,
            bind_group(
                device,
                &self.display_layout,
                &self.uniforms[1],
                Some((&self.color, &self.sampler)),
            )?,
        ];
        Ok(())
    }
    async fn render(
        &mut self,
        device: &Device,
        lane: SubmissionLaneId,
        frame: AcquiredFrame,
    ) -> RhiResult<()> {
        let slot = self.frame;
        if let Some(point) = self.complete[slot].take() {
            let _ = device.wait_completion(point).await?;
        }
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            self.uniforms[slot].clone(),
            0,
            uniform_data(self.extent),
        )?)?)?;
        {
            let mut pass = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Texture(self.color.clone()),
                            load: LoadOp::Clear(ColorClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                            store: StoreOp::Store,
                            resolve: None,
                            depth_slice: None,
                        },
                    )
                    .with_depth_stencil(DepthStencilAttachment {
                        view: self.multiview_depth.clone(),
                        depth: Some(DepthAttachmentMode::ReadWrite {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil: None,
                    }),
            )?;
            full_viewport(&mut pass, self.extent)?;
            pass.set_pipeline(&self.room)?;
            pass.set_bind_group(BindGroupIndex::new(0), &self.scene_groups[slot], &[])?;
            pass.set_vertex_buffer(0, &self.vertices)?;
            pass.set_index_buffer(&self.indices, IndexFormat::Uint32)?;
            pass.draw_indexed(0..(self.indices.range.size / 4) as u32, 0, 0..1)?;
            pass.end()?;
        }
        {
            let mut pass = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(frame.attachment()),
                            load: LoadOp::Clear(ColorClearValue::Float([0.025, 0.025, 0.025, 1.0])),
                            store: StoreOp::Store,
                            resolve: None,
                            depth_slice: None,
                        },
                    )
                    .with_depth_stencil(DepthStencilAttachment {
                        view: self.frame_depth.clone(),
                        depth: Some(DepthAttachmentMode::ReadWrite {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Discard,
                        }),
                        stencil: None,
                    }),
            )?;
            pass.set_bind_group(BindGroupIndex::new(0), &self.display_groups[slot], &[])?;
            let half = self.extent.width / 2;
            pass.set_viewport(Viewport::new(
                0.0,
                0.0,
                half as f32,
                self.extent.height as f32,
                0.0,
                1.0,
            ))?;
            pass.set_scissor(Rect::new(0, 0, half, self.extent.height))?;
            pass.set_pipeline(&self.display[0])?;
            pass.draw(0..3, 0..1)?;
            pass.set_viewport(Viewport::new(
                half as f32,
                0.0,
                half as f32,
                self.extent.height as f32,
                0.0,
                1.0,
            ))?;
            pass.set_scissor(Rect::new(half, 0, half, self.extent.height))?;
            pass.set_pipeline(&self.display[1])?;
            pass.draw(0..3, 0..1)?;
            pass.end()?;
        }
        let mut plan = SubmissionPlanBuilder::new(device);
        let batch = plan.add_batch(lane, vec![recorder.finish()?])?;
        plan.present_after(frame, batch)?;
        let submitted = device.submit(plan.build()?)?;
        self.complete[slot] = Some(submitted.completion());
        let _ = device.wait_present(submitted.presents()[0].id()).await?;
        self.frame = (slot + 1) % FRAMES;
        Ok(())
    }
}

fn multiview_attachments(
    device: &Device,
    extent: Extent3d,
    format: TextureFormat,
) -> RhiResult<(TextureView, TextureView)> {
    let color = device.create_texture(
        &TextureDescriptor::new_2d(
            extent.width,
            extent.height,
            format,
            TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::SAMPLED),
        )
        .with_array_layers(2),
    )?;
    let depth = device.create_texture(
        &TextureDescriptor::new_2d(
            extent.width,
            extent.height,
            TextureFormat::Depth32Float,
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        )
        .with_array_layers(2),
    )?;
    Ok((
        device.create_texture_view(
            &color,
            &TextureViewDescriptor::whole(&color, TextureViewDimension::D2Array)?,
        )?,
        device.create_texture_view(
            &depth,
            &TextureViewDescriptor::whole(&depth, TextureViewDimension::D2Array)?,
        )?,
    ))
}
fn frame_depth(device: &Device, extent: Extent3d) -> RhiResult<TextureView> {
    let depth = device.create_texture(&TextureDescriptor::new_2d(
        extent.width,
        extent.height,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ))?;
    device.create_texture_view(
        &depth,
        &TextureViewDescriptor::whole(&depth, TextureViewDimension::D2)?,
    )
}
fn uniform_buffer(device: &Device) -> RhiResult<Buffer> {
    device.create_buffer(&BufferDescriptor::new(
        UBO_SIZE,
        BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
    ))
}
fn layout(device: &Device, display: bool) -> RhiResult<BindGroupLayout> {
    let mut slots = vec![BindingSlot::new(
        BindingSlotId::new(0),
        if display {
            ShaderStages::FRAGMENT
        } else {
            ShaderStages::VERTEX
        },
        BindingKind::UniformBuffer { min_size: UBO_SIZE },
    )];
    if display {
        slots.push(BindingSlot::new(
            BindingSlotId::new(1),
            ShaderStages::FRAGMENT,
            BindingKind::SampledTexture {
                dimension: TextureViewDimension::D2Array,
                sample_type: TextureSampleType::Float,
                multisampled: false,
            },
        ));
        slots.push(BindingSlot::new(
            BindingSlotId::new(2),
            ShaderStages::FRAGMENT,
            BindingKind::Sampler {
                kind: SamplerKind::Filtering,
            },
        ));
    }
    device.create_bind_group_layout(&BindGroupLayoutDescriptor::new(slots))
}
fn bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform: &Buffer,
    image: Option<(&TextureView, &Sampler)>,
) -> RhiResult<BindGroup> {
    let mut entries = vec![BindGroupEntry::new(
        BindingSlotId::new(0),
        BindingResource::Buffer(BufferBinding::new(
            uniform.clone(),
            BufferRange::new(0, UBO_SIZE),
        )),
    )];
    if let Some((view, sampler)) = image {
        entries.push(BindGroupEntry::new(
            BindingSlotId::new(1),
            BindingResource::Texture(view.clone()),
        ));
        entries.push(BindGroupEntry::new(
            BindingSlotId::new(2),
            BindingResource::Sampler(sampler.clone()),
        ));
    }
    device.create_bind_group(&BindGroupDescriptor::new(layout.clone()).with_entries(entries))
}
async fn pipeline(
    device: &Device,
    interface: PipelineInterface,
    source: &str,
    format: TextureFormat,
    depth: DepthStencilState,
    cull: CullMode,
    scene: bool,
    hash: u8,
) -> RhiResult<RasterPipeline> {
    let vertex = common::shader::create_shader(
        device,
        &artifact(
            ShaderStage::Vertex,
            "vs_main",
            source,
            vertex_interface(scene),
            [hash; 32],
        ),
    )
    .await?;
    let fragment = common::shader::create_shader(
        device,
        &artifact(
            ShaderStage::Fragment,
            "fs_main",
            source,
            fragment_interface(scene),
            [hash.wrapping_add(1); 32],
        ),
    )
    .await?;
    let mut descriptor = RasterPipelineDescriptor::new(vertex, interface)
        .with_fragment(fragment)
        .with_primitive(PrimitiveState::new(PrimitiveTopology::TriangleList).with_cull_mode(cull))
        .with_depth_stencil(depth)
        .with_color_target(ShaderLocation::new(0), ColorTargetState::new(format));
    if scene {
        descriptor = descriptor
            .with_vertex_input(
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
                            VertexFormat::Float32x3,
                            32,
                        )),
                ),
            )
            .with_multiview_mask(0b11);
    }
    device.create_raster_pipeline(&descriptor).await
}
fn display_shader(layer: u32) -> String {
    format!("const VIEW_LAYER: i32 = {layer};\n{DISPLAY_PREFIX}")
}
fn artifact(
    stage: ShaderStage,
    entry: &str,
    source: &str,
    interface: ShaderInterface,
    hash: [u8; 32],
) -> ShaderArtifact {
    ShaderArtifact::new(
        stage,
        entry,
        ShaderCode::Wgsl(Arc::from(source)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        ArtifactHash(hash),
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
fn location(location: u32, components: u8, input: bool) -> ShaderLocationInterface {
    ShaderLocationInterface {
        location: ShaderLocation::new(location),
        numeric_type: ShaderNumericType::Float32,
        components,
        interpolation: if input {
            None
        } else {
            Some(ShaderInterpolation {
                mode: InterpolationMode::Perspective,
                sampling: InterpolationSampling::Center,
            })
        },
    }
}
fn vertex_interface(scene: bool) -> ShaderInterface {
    if scene {
        ShaderInterface::new()
            .with_resource(resource(
                0,
                BindingKind::UniformBuffer { min_size: UBO_SIZE },
            ))
            .with_input(location(0, 3, true))
            .with_input(location(1, 3, true))
            .with_input(location(2, 3, true))
            .with_output(location(0, 3, false))
            .with_output(location(1, 3, false))
            .with_output(location(2, 3, false))
            .with_output(location(3, 3, false))
            .with_writes_position(true)
    } else {
        ShaderInterface::new()
            .with_output(location(0, 2, false))
            .with_writes_position(true)
    }
}
fn fragment_interface(scene: bool) -> ShaderInterface {
    let interface = ShaderInterface::new().with_output(ShaderLocationInterface {
        interpolation: None,
        ..location(0, 4, false)
    });
    if scene {
        interface
            .with_input(location(0, 3, false))
            .with_input(location(1, 3, false))
            .with_input(location(2, 3, false))
            .with_input(location(3, 3, false))
    } else {
        interface
            .with_resource(resource(
                0,
                BindingKind::UniformBuffer { min_size: UBO_SIZE },
            ))
            .with_resource(resource(
                1,
                BindingKind::SampledTexture {
                    dimension: TextureViewDimension::D2Array,
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
            .with_input(location(0, 2, false))
    }
}
fn full_viewport(pass: &mut RasterScope<'_>, extent: Extent3d) -> RhiResult<()> {
    pass.set_viewport(Viewport::new(
        0.0,
        0.0,
        extent.width as f32,
        extent.height as f32,
        0.0,
        1.0,
    ))?;
    pass.set_scissor(Rect::new(0, 0, extent.width, extent.height))
}
fn example_error(error: impl ToString) -> RhiError {
    RhiError::new(RhiErrorKind::InvalidUsage, error.to_string())
}
fn uniform_data(extent: Extent3d) -> Vec<u8> {
    let aspect = extent.width as f32 * 0.5 / extent.height.max(1) as f32;
    let (near, far) = (0.1, 256.0);
    let half_height = near * 45.0_f32.to_radians().tan();
    let eye_offset = 0.5 * 0.08 * (near / 0.5);
    let projection = [
        frustum(
            -aspect * half_height - eye_offset,
            aspect * half_height - eye_offset,
            -half_height,
            half_height,
            near,
            far,
        ),
        frustum(
            -aspect * half_height + eye_offset,
            aspect * half_height + eye_offset,
            -half_height,
            half_height,
            near,
            far,
        ),
    ];
    let rotate = rotate_y(90.0_f32.to_radians());
    let modelview = [
        multiply(rotate, translate([7.0, 3.2, 0.04])),
        multiply(rotate, translate([7.0, 3.2, -0.04])),
    ];
    projection
        .into_iter()
        .flatten()
        .chain(modelview.into_iter().flatten())
        .chain([-2.5, -3.5, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}
fn frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> [f32; 16] {
    [
        2.0 * n / (r - l),
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 * n / (t - b),
        0.0,
        0.0,
        (r + l) / (r - l),
        (t + b) / (t - b),
        f / (n - f),
        -1.0,
        0.0,
        0.0,
        f * n / (n - f),
        0.0,
    ]
}
fn translate(p: [f32; 3]) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, p[0], p[1], p[2], 1.0,
    ]
}
fn rotate_y(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [
        c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}
fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut output = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4)
                .map(|index| a[index * 4 + row] * b[column * 4 + index])
                .sum();
        }
    }
    output
}
