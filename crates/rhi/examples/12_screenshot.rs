//! Port of SaschaWillems/Vulkan `examples/screenshot`.
//!
//! Renders the original Chinese dragon, reads the acquired frame after the draw,
//! and writes the C++ example's binary P6 `screenshot.ppm` image. Set
//! `FLUXEL_RHI_SCREENSHOT=1` to request the capture at startup; the shared host
//! currently has no button input callback.

pub mod common;
#[path = "common/gltf.rs"]
mod gltf;

use std::{fs, sync::Arc};

use fluxel_rhi::api::{
    binding::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupIndex, BindGroupLayoutDescriptor,
        BindingCount, BindingKind, BindingResource, BindingSlot, BindingSlotId,
    },
    command::{
        ColorAttachment, ColorAttachmentView, ColorClearValue, DepthAttachmentMode,
        DepthStencilAttachment, IndexFormat, LoadOp, RasterScopeDescriptor, RecorderDescriptor,
        Rect, StoreOp, Viewport,
    },
    error::{RhiError, RhiErrorKind, RhiResult},
    format::TextureFormat,
    identity::Label,
    pipeline::{
        ColorTargetState, CullMode, DepthState, DepthStencilState, PipelineInterfaceDescriptor,
        PrimitiveState, PrimitiveTopology, RasterPipeline, RasterPipelineDescriptor,
        VertexAttribute, VertexBufferLayout, VertexFormat, VertexInputState, VertexStepMode,
    },
    platform::Device,
    presentation::AcquiredFrame,
    resource::{
        Buffer, BufferBinding, BufferDescriptor, BufferRange, BufferUploadDescriptor, BufferUsage,
        CompareFunction, Extent3d, ReadbackRequest, ReadbackViewData, TextureDescriptor,
        TextureUsage, TextureView, TextureViewDescriptor, TextureViewDimension,
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
const UBO_BYTES: u64 = 192;
const DRAGON: &[u8] = include_bytes!("assets/models/chinesedragon.gltf");

const SHADER: &str = r#"
struct U { projection:mat4x4<f32>, view:mat4x4<f32>, model:mat4x4<f32>, };
@group(0) @binding(0) var<uniform> u:U;
struct Input { @location(0) pos:vec3<f32>, @location(1) normal:vec3<f32>, @location(2) color:vec3<f32>, };
struct Output { @builtin(position) pos:vec4<f32>, @location(0) normal:vec3<f32>, @location(1) color:vec3<f32>, @location(2) view_vec:vec3<f32>, @location(3) light_vec:vec3<f32>, };
@vertex fn vs_main(i:Input)->Output {
    var o:Output;
    o.color=i.color;
    o.pos=u.projection*u.view*u.model*vec4<f32>(i.pos,1.0);
    let p=u.view*u.model*vec4<f32>(i.pos,1.0);
    o.normal=mat3x3<f32>(u.model[0].xyz,u.model[1].xyz,u.model[2].xyz)*i.normal;
    o.light_vec=vec3<f32>(1.0,-1.0,1.0)-p.xyz;
    o.view_vec=-p.xyz;
    return o;
}
@fragment fn fs_main(i:Output)->@location(0) vec4<f32> {
    let n=normalize(i.normal);let l=normalize(i.light_vec);let v=normalize(i.view_vec);
    let r=reflect(-l,n);
    let diffuse=max(dot(n,l),0.0);
    let specular=pow(max(dot(r,v),0.0),16.0)*vec3<f32>(0.75);
    return vec4<f32>((vec3<f32>(0.1)+vec3<f32>(diffuse))*i.color+specular,1.0);
}
"#;

fn main() {
    if let Err(error) = common::run_example("Saving framebuffer to screenshot", create_example()) {
        eprintln!("12_screenshot: {error}");
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
    work: Option<Workload>,
    lane: Option<SubmissionLaneId>,
}

impl common::Example for Example {
    fn presentation_usage(&self) -> TextureUsage {
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::COPY_SRC)
    }

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
            .find(|candidate| {
                candidate
                    .domains()
                    .contains(LaneWorkDomains::RASTER.union(LaneWorkDomains::COPY))
            })
            .map(|candidate| candidate.id())
            .ok_or_else(|| std::io::Error::other("12_screenshot requires COPY|RASTER lane"))?;
        let extent = context.extent();
        let format = context.presentation_mut().configuration().format();
        self.work = Some(common::block_on(Workload::new(
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
        common::block_on(
            self.work
                .as_mut()
                .ok_or_else(|| std::io::Error::other("not initialized"))?
                .render(
                    context.device(),
                    self.lane.ok_or_else(|| std::io::Error::other("no lane"))?,
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
        if let Some(work) = &mut self.work {
            work.resize(context.device(), Extent3d::d2(width, height))?;
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
    pipeline: RasterPipeline,
    groups: [BindGroup; FRAMES],
    uniforms: [Buffer; FRAMES],
    vertex: BufferBinding,
    index: BufferBinding,
    draws: Vec<(u32, u32)>,
    depth: TextureView,
    extent: Extent3d,
    format: TextureFormat,
    requested: bool,
    saved: bool,
    next: usize,
    done: [Option<CompletionPoint>; FRAMES],
}

impl Workload {
    async fn new(
        device: &Device,
        format: TextureFormat,
        extent: Extent3d,
        lane: SubmissionLaneId,
    ) -> RhiResult<Self> {
        let model = gltf::load_embedded_model(DRAGON, gltf::LoadOptions::CPP_PORT)
            .map_err(|error| RhiError::new(RhiErrorKind::InvalidUsage, error.to_string()))?;
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor::new(vec![
            BindingSlot::new(
                BindingSlotId::new(0),
                ShaderStages::VERTEX,
                BindingKind::UniformBuffer {
                    min_size: UBO_BYTES,
                },
            ),
        ]))?;
        let interface = device
            .create_pipeline_interface(&PipelineInterfaceDescriptor::new(vec![layout.clone()]))?;
        let pipeline = device
            .create_raster_pipeline(
                &RasterPipelineDescriptor::new(
                    common::shader::create_shader(device, &artifact(ShaderStage::Vertex)).await?,
                    interface,
                )
                .with_fragment(
                    common::shader::create_shader(device, &artifact(ShaderStage::Fragment)).await?,
                )
                .with_vertex_input(
                    VertexInputState::new().with_buffer(
                        VertexBufferLayout::new(
                            gltf::VKGLTF_VERTEX_STRIDE as u64,
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
                            VertexFormat::Float32x3,
                            32,
                        )),
                    ),
                )
                .with_primitive(
                    PrimitiveState::new(PrimitiveTopology::TriangleList)
                        .with_cull_mode(CullMode::Back),
                )
                .with_depth_stencil(
                    DepthStencilState::new(TextureFormat::Depth32Float).with_depth(
                        DepthState::new(CompareFunction::LessEqual).with_write_enabled(true),
                    ),
                )
                .with_color_target(ShaderLocation::new(0), ColorTargetState::new(format)),
            )
            .await?;
        let vertex_len = model.vertices.len() as u64;
        let index_len = model.indices.len() as u64;
        let vertex = device.create_buffer(&BufferDescriptor::new(
            vertex_len,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = device.create_buffer(&BufferDescriptor::new(
            index_len,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let draws = model
            .primitives
            .iter()
            .map(|p| (p.first_index, p.index_count))
            .collect();
        let uniforms = [0, 1]
            .map(|_| {
                device.create_buffer(&BufferDescriptor::new(
                    UBO_BYTES,
                    BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                ))
            })
            .into_iter()
            .collect::<RhiResult<Vec<_>>>()?;
        let uniforms: [Buffer; FRAMES] = uniforms
            .try_into()
            .map_err(|_| RhiError::new(RhiErrorKind::BackendFailure, "uniform allocation count"))?;
        let group = |slot: usize| {
            device.create_bind_group(&BindGroupDescriptor::new(layout.clone()).with_entry(
                BindGroupEntry::new(
                    BindingSlotId::new(0),
                    BindingResource::Buffer(BufferBinding::new(
                        uniforms[slot].clone(),
                        BufferRange::new(0, UBO_BYTES),
                    )),
                ),
            ))
        };
        let groups = [group(0)?, group(1)?];
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            vertex.clone(),
            0,
            model.vertices,
        ))?)?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            index.clone(),
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
            vertex: BufferBinding::new(vertex, BufferRange::new(0, vertex_len)),
            index: BufferBinding::new(index, BufferRange::new(0, index_len)),
            draws,
            depth: depth_view(device, extent)?,
            extent,
            format,
            requested: std::env::var("FLUXEL_RHI_SCREENSHOT").as_deref() == Ok("1"),
            saved: false,
            next: 0,
            done: [None, None],
        })
    }

    fn resize(&mut self, device: &Device, extent: Extent3d) -> RhiResult<()> {
        self.depth = depth_view(device, extent)?;
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
        if let Some(completion) = self.done[slot].take() {
            let _ = device.wait_completion(completion).await?;
        }
        let mut recorder = device.create_recorder(&RecorderDescriptor::new())?;
        recorder.encode_upload(&device.create_buffer_upload(BufferUploadDescriptor::new(
            self.uniforms[slot].clone(),
            0,
            uniform_bytes(self.extent),
        ))?)?;
        let frame_attachment = frame.attachment();
        {
            let mut raster = recorder.begin_raster(
                &RasterScopeDescriptor::new()
                    .with_color(
                        ShaderLocation::new(0),
                        ColorAttachment {
                            view: ColorAttachmentView::Frame(frame_attachment.clone()),
                            load: LoadOp::Clear(ColorClearValue::Float([0.025, 0.025, 0.025, 1.])),
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
            raster.set_pipeline(&self.pipeline)?;
            raster.set_viewport(Viewport::new(
                0.,
                0.,
                self.extent.width as f32,
                self.extent.height as f32,
                0.,
                1.,
            ))?;
            raster.set_scissor(Rect::new(0, 0, self.extent.width, self.extent.height))?;
            raster.set_bind_group(BindGroupIndex::new(0), &self.groups[slot], &[])?;
            raster.set_vertex_buffer(0, &self.vertex)?;
            raster.set_index_buffer(&self.index, IndexFormat::Uint32)?;
            for &(first, count) in &self.draws {
                raster.draw_indexed(first..first + count, 0, 0..1)?;
            }
            raster.end()?;
        }
        let ticket = if self.saved || !self.requested {
            None
        } else {
            Some(recorder.encode_readback(ReadbackRequest::Frame {
                label: Label(Some("12_screenshot acquired frame".into())),
                src: frame_attachment,
            })?)
        };
        let mut plan = SubmissionPlanBuilder::new(device);
        let point = plan.add_batch(lane, vec![recorder.finish()?])?;
        plan.present_after(frame, point)?;
        let receipt = device.submit(plan.build()?)?;
        self.done[slot] = Some(receipt.completion());
        if let Some(ticket) = ticket {
            let _ = device.wait_completion(receipt.completion()).await?;
            let read = ticket.try_read()?.ok_or_else(|| {
                RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "frame readback did not complete",
                )
            })?;
            if let ReadbackViewData::Texture { bytes, layout } = read.data() {
                write_ppm(
                    "screenshot.ppm",
                    bytes,
                    layout.bytes_per_row,
                    self.extent,
                    self.format,
                )
                .map_err(|error| RhiError::new(RhiErrorKind::BackendFailure, error.to_string()))?;
                self.saved = true;
            } else {
                return Err(RhiError::new(
                    RhiErrorKind::BackendFailure,
                    "frame readback has no pixel layout",
                ));
            }
        }
        let _ = device
            .wait_present(receipt.presents().first().expect("present").id())
            .await?;
        self.next = (slot + 1) % FRAMES;
        Ok(())
    }
}

fn depth_view(device: &Device, extent: Extent3d) -> RhiResult<TextureView> {
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

fn uniform_bytes(extent: Extent3d) -> Vec<u8> {
    let near = 0.1;
    let far = 512.;
    let f = 1. / 30f32.to_radians().tan();
    let a = extent.width.max(1) as f32 / extent.height.max(1) as f32;
    let projection = [
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
        far / (near - far),
        -1.,
        0.,
        0.,
        far * near / (near - far),
        0.,
    ];
    let (sx, cx) = (-25f32).to_radians().sin_cos();
    let (sy, cy) = (23.75f32).to_radians().sin_cos();
    let rx = [
        1., 0., 0., 0., 0., cx, sx, 0., 0., -sx, cx, 0., 0., 0., 0., 1.,
    ];
    let ry = [
        cy, 0., -sy, 0., 0., 1., 0., 0., sy, 0., cy, 0., 0., 0., 0., 1.,
    ];
    let translate = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., -3., 1.,
    ];
    let view = mm(mm(translate, rx), ry);
    let model = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ];
    projection
        .into_iter()
        .chain(view)
        .chain(model)
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn mm(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}

fn write_ppm(
    path: &str,
    bytes: &[u8],
    row_pitch: u32,
    extent: Extent3d,
    format: TextureFormat,
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let swizzle = match format {
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => true,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => false,
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "screenshot requires BGRA8 or RGBA8 frame",
            ));
        }
    };
    let row_len = (extent.width as usize)
        .checked_mul(4)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "row size overflow"))?;
    if (row_pitch as usize) < row_len {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "readback row pitch is too small",
        ));
    }
    let mut ppm = format!("P6\n{}\n{}\n255\n", extent.width, extent.height).into_bytes();
    for y in 0..extent.height as usize {
        let start = y
            .checked_mul(row_pitch as usize)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "row offset overflow"))?;
        let row = bytes
            .get(start..start + row_len)
            .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, "truncated frame readback"))?;
        for pixel in row.chunks_exact(4) {
            if swizzle {
                ppm.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
            } else {
                ppm.extend_from_slice(&pixel[..3]);
            }
        }
    }
    fs::write(path, ppm)
}

fn artifact(stage: ShaderStage) -> ShaderArtifact {
    let i = || ShaderInterpolation {
        mode: InterpolationMode::Perspective,
        sampling: InterpolationSampling::Center,
    };
    let io = |location, components, interpolation| ShaderLocationInterface {
        location: ShaderLocation::new(location),
        numeric_type: ShaderNumericType::Float32,
        components,
        interpolation,
    };
    let mut interface = ShaderInterface::new();
    if stage == ShaderStage::Vertex {
        interface = interface
            .with_resource(ShaderResourceRequirement {
                group: BindGroupIndex::new(0),
                slot: BindingSlotId::new(0),
                kind: BindingKind::UniformBuffer {
                    min_size: UBO_BYTES,
                },
                count: BindingCount::One,
            })
            .with_input(io(0, 3, None))
            .with_input(io(1, 3, None))
            .with_input(io(2, 3, None))
            .with_output(io(0, 3, Some(i())))
            .with_output(io(1, 3, Some(i())))
            .with_output(io(2, 3, Some(i())))
            .with_output(io(3, 3, Some(i())))
            .with_writes_position(true);
    } else {
        interface = interface
            .with_input(io(0, 3, Some(i())))
            .with_input(io(1, 3, Some(i())))
            .with_input(io(2, 3, Some(i())))
            .with_input(io(3, 3, Some(i())))
            .with_output(io(0, 4, None));
    }
    ShaderArtifact::new(
        stage,
        if stage == ShaderStage::Vertex {
            "vs_main"
        } else {
            "fs_main"
        },
        ShaderCode::Wgsl(Arc::from(SHADER)),
        ShaderAbiVersion { major: 1, minor: 0 },
        interface,
        ShaderRequirements::new(),
        ArtifactHash(if stage == ShaderStage::Vertex {
            [12; 32]
        } else {
            [13; 32]
        }),
        ArtifactProducerVersion { major: 1, minor: 0 },
    )
}
