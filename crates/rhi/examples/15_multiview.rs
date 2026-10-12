//! Typed multiview rendering into a two-layer color attachment.
//! The first pass uses view mask `0b11`; the second samples the two layers into
//! left/right halves of the acquired surface.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
use fluxel_rhi::api::*;
use std::sync::Arc;
const SCENE_VS: &[u8] = include_bytes!("assets/15_multiview/scene_vs.dxil");
const SCENE_PS: &[u8] = include_bytes!("assets/15_multiview/scene_ps.dxil");
const DISPLAY_VS: &[u8] = include_bytes!("assets/15_multiview/display_vs.dxil");
const DISPLAY_PS: &[u8] = include_bytes!("assets/15_multiview/display_ps.dxil");
const ROOM: &[u8] = include_bytes!("assets/models/sampleroom.gltf");

pub trait MultiviewBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + CommandBackend
    + RasterBackend
    + DeferredBackend
{
}
impl<T> MultiviewBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + CommandBackend
        + RasterBackend
        + DeferredBackend
{
}
pub fn require_multiview<B: Backend>(device: &Device<B>, views: u32) -> Result<()> {
    if !device.capabilities().multiview {
        return Err(Error::Unsupported("multiview"));
    }
    if views == 0 || views > device.capabilities().limits.max_multiview_views {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub struct Multiview<B: MultiviewBackend> {
    lane: LaneId,
    vertices: Buffer<B>,
    indices: Buffer<B>,
    vertex_len: u64,
    index_len: u64,
    index_count: u32,
    uniforms: Buffer<B>,
    scene_group: BindGroup<B>,
    scene: RasterPipeline<B>,
    display: RasterPipeline<B>,
    views: Texture<B>,
    view: TextureView<B>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
    group: BindGroup<B>,
    format: TextureFormat,
    extent: [u32; 2],
}
impl<B: MultiviewBackend> Multiview<B> {
    pub async fn new(
        device: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        require_multiview(device, 2)?;
        let room = gltf::load_embedded_model(ROOM, gltf::LoadOptions::CPP_PORT)
            .map_err(|error| Error::Backend(error.to_string()))?;
        let vertices = device.create_buffer(&BufferDesc {
            label: Some("15 sampleroom vertices".into()),
            size: room.vertices.len() as u64,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let indices = device.create_buffer(&BufferDesc {
            label: Some("15 sampleroom indices".into()),
            size: room.indices.len() as u64,
            usage: BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let uniforms = device.create_buffer(&BufferDesc {
            label: Some("15 two-eye transforms".into()),
            size: 256,
            usage: BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let (views, view) = layered_target(device, format, extent)?;
        let (depth, depth_view) = layered_depth(device, extent)?;
        let scene_bind_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("15 scene transform layout".into()),
            entries: vec![binding(
                0,
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 128,
                },
            )],
        })?;
        let scene_group = device.create_bind_group(&BindGroupDesc {
            label: Some("15 scene transform group"),
            layout: &scene_bind_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &uniforms,
                    range: 0..256,
                },
            }],
        })?;
        let scene_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 scene layout"),
            groups: &[&scene_bind_layout],
            push_constants: &[],
        })?;
        let scene = pipeline(
            device,
            scene_layout,
            format,
            SCENE_VS,
            SCENE_PS,
            true,
            true,
            vec![VertexBufferLayout {
                stride: 96,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32(3),
                    },
                    VertexAttribute {
                        location: 1,
                        offset: 12,
                        format: VertexFormat::Float32(3),
                    },
                    VertexAttribute {
                        location: 2,
                        offset: 24,
                        format: VertexFormat::Float32(2),
                    },
                    VertexAttribute {
                        location: 3,
                        offset: 32,
                        format: VertexFormat::Float32(3),
                    },
                ],
            }],
        )
        .await?;
        let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("15 display bindings".into()),
            entries: vec![
                binding(
                    0,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                ),
                binding(
                    1,
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                ),
            ],
        })?;
        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("15 display sampler".into()),
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let entries = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let group = device.create_bind_group(&BindGroupDesc {
            label: Some("15 display group"),
            layout: &layout,
            entries: &entries,
        })?;
        let display_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 display layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let display = pipeline(
            device,
            display_layout,
            format,
            DISPLAY_VS,
            DISPLAY_PS,
            false,
            false,
            vec![],
        )
        .await?;
        Ok(Self {
            lane,
            vertices,
            indices,
            vertex_len: room.vertices.len() as u64,
            index_len: room.indices.len() as u64,
            index_count: (room.indices.len() / 4) as u32,
            uniforms,
            scene_group,
            scene,
            display,
            views,
            view,
            _depth: depth,
            depth_view,
            group,
            format,
            extent,
        })
    }
    pub fn record(&self, device: &Device<B>, image: &SurfaceImage<B>) -> Result<RecordedWork<B>> {
        let mut encoder = device.encoder(self.lane)?;
        let room = gltf::load_embedded_model(ROOM, gltf::LoadOptions::CPP_PORT)
            .map_err(|error| Error::Backend(error.to_string()))?;
        encoder.write_buffer(&self.vertices, 0, &room.vertices)?;
        encoder.write_buffer(&self.indices, 0, &room.indices)?;
        encoder.write_buffer(&self.uniforms, 0, &two_eye_matrices(self.extent))?;
        encoder.transition_resources(&[transition(
            &self.views,
            ImageLayout::Undefined,
            ImageLayout::ColorAttachment,
            TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
        )])?;
        let scene_color = [Some(ColorAttachment {
            view: AttachmentView::Texture(&self.view),
            resolve_target: None,
            // Diagnostic: a visible layer clear separates a failed scene draw
            // from a failed array-SRV/display pass. Restore the scene clear
            // after the capture establishes which boundary is at fault.
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 0.0])),
            store: StoreOp::Store,
        })];
        let mut scene = encoder.begin_raster(&RasterPassDesc {
            label: Some("15 layered multiview scene"),
            colors: &scene_color,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&self.depth_view),
                depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
            render_area: Rect {
                x: 0,
                y: 0,
                width: self.extent[0],
                height: self.extent[1],
            },
            multiview_mask: Some(0b11),
            timestamp_writes: None,
            secondary_inheritance: None,
        })?;
        scene.set_pipeline(&self.scene)?;
        scene.set_bind_group(0, &self.scene_group, &[])?;
        scene.set_vertex_buffer(0, &self.vertices, 0..self.vertex_len)?;
        scene.set_index_buffer(&self.indices, 0..self.index_len, IndexFormat::Uint32)?;
        scene.draw_indexed(0..self.index_count, 0, 0..1)?;
        scene.finish()?;
        encoder.transition_resources(&[transition(
            &self.views,
            ImageLayout::ColorAttachment,
            ImageLayout::ShaderReadOnly,
            TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
        )])?;
        let surface = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut display = encoder.begin_raster(&RasterPassDesc {
            label: Some("15 two-eye display"),
            colors: &surface,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: self.extent[0],
                height: self.extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            secondary_inheritance: None,
        })?;
        display.set_pipeline(&self.display)?;
        display.set_bind_group(0, &self.group, &[])?;
        display.draw(0..3, 0..1)?;
        display.finish()?;
        encoder.finish()
    }
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }
    pub fn format(&self) -> TextureFormat {
        self.format
    }
}
fn layered_target<B: MultiviewBackend>(
    device: &Device<B>,
    format: TextureFormat,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("15 two eye attachment".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 2,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::TEXTURE_BINDING),
        view_formats: vec![],
    })?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format,
            dimension: TextureViewDimension::D2Array,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..2,
        },
    )?;
    Ok((texture, view))
}
fn layered_depth<B: MultiviewBackend>(
    device: &Device<B>,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("15 two eye depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 2,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2Array,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..2,
        },
    )?;
    Ok((texture, view))
}
fn binding(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Fragment],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
fn transition<'a, B: MultiviewBackend>(
    texture: &'a Texture<B>,
    from: ImageLayout,
    to: ImageLayout,
    before: TextureUses,
    after: TextureUses,
) -> ResourceTransition<'a, B> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..2,
        },
        usage: StateTransition {
            from: before,
            to: after,
        },
        layout: Some((from, to)),
        ownership: None,
    })
}
async fn pipeline<B: MultiviewBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    vs: &[u8],
    ps: &[u8],
    multiview: bool,
    depth: bool,
    vertex_buffers: Vec<VertexBufferLayout>,
) -> Result<RasterPipeline<B>> {
    let vs = device
        .shader(ShaderDesc {
            label: Some("15 VS".into()),
            format: ShaderFormat::Dxil,
            bytes: Arc::from(vs),
            entries: vec![ShaderEntryPoint {
                name: "vs_main".into(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
            }],
            required_features: vec![],
        })
        .await?;
    let ps = device
        .shader(ShaderDesc {
            label: Some("15 PS".into()),
            format: ShaderFormat::Dxil,
            bytes: Arc::from(ps),
            entries: vec![ShaderEntryPoint {
                name: "ps_main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![],
        })
        .await?;
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some("15 raster pipeline".into()),
            layout,
            vertex: PipelineStage {
                module: vs,
                entry: "vs_main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: ps,
                entry: "ps_main".into(),
                constants: vec![],
            }),
            vertex_buffers,
            color_targets: vec![Some(ColorTargetState {
                format,
                blend: None,
                write_mask: 15,
            })],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth.then_some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil_front: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Keep,
                },
                stencil_back: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Keep,
                },
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            multisample: MultisampleState {
                count: 1,
                mask: u64::from(u32::MAX),
                alpha_to_coverage_enabled: false,
            },
            multiview: multiview.then_some(2),
            cache: None,
        })
        .await
}
fn triangle() -> [u8; 72] {
    let values = [
        0.0_f32, 0.55, 0.0, 1.0, 0.2, 0.1, -0.55, -0.5, 0.0, 0.1, 1.0, 0.25, 0.55, -0.5, 0.0, 0.2,
        0.35, 1.0,
    ];
    let mut bytes = [0; 72];
    for (index, value) in values.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
fn two_eye_matrices(extent: [u32; 2]) -> [u8; 256] {
    // Same stereo frusta and first-person camera as the C++ multiview sample:
    // 90° FOV, near/far 0.1/256, 0.08 eye separation, camera (7, 3.2, 0)
    // rotated +90° around Y. Matrices are column-major for HLSL `mul(M, v)`.
    let aspect = extent[0].max(1) as f32 * 0.5 / extent[1].max(1) as f32;
    let near = 0.1;
    let far = 256.0;
    let half_width = near; // tan(90° / 2) * near
    let eye_shift = 0.08 * (near / 0.5) * 0.5;
    let projection = |left: f32, right: f32| {
        [
            2.0 * near / (right - left),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            (right + left) / (right - left),
            0.0,
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            far * near / (near - far),
            0.0,
        ]
    };
    let multiply = |a: [f32; 16], b: [f32; 16]| {
        let mut out = [0.0; 16];
        for column in 0..4 {
            for row in 0..4 {
                out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
            }
        }
        out
    };
    let modelview = |z: f32| {
        // `rotY(90°) * translate(7, 3.2, z)` from Camera::updateViewMatrix.
        [
            0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, z, 3.2, -7.0, 1.0,
        ]
    };
    let left = multiply(
        projection(
            -aspect * half_width - eye_shift,
            aspect * half_width - eye_shift,
        ),
        modelview(0.04),
    );
    let right = multiply(
        projection(
            -aspect * half_width + eye_shift,
            aspect * half_width + eye_shift,
        ),
        modelview(-0.04),
    );
    let mut bytes = [0; 256];
    for (index, value) in left.into_iter().chain(right).enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
#[cfg(windows)]
fn main() {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let result = common::run_example(
        common::windows_api::Dx12Host,
        "Multiview rendering",
        config.clone(),
        move |device, _, image, extent| {
            require_multiview(device, 2)?;
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            // The attachment has an explicit Undefined/COMMON initial
            // transition, so rebuild it each frame rather than pretending a
            // device-wide state tracker knows its previous layout.
            let example = common::block_on(Multiview::new(device, lane, config.format, extent))?;
            let work = example.record(device, &image)?;
            common::submit_and_present(device, lane, work, image)
        },
    );
    if let Err(error) = result {
        eprintln!("15_multiview: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("15_multiview requires Windows and a backend with multiview support");
}
