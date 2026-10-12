//! Typed multiview rendering into a two-layer color attachment.
//! The first pass uses view mask `0b11`; the second samples the two layers into
//! left/right halves of the acquired surface.
mod common;
use fluxel_rhi::api::*;
use std::sync::Arc;
const SCENE_VS: &[u8] = include_bytes!("assets/15_multiview/scene_vs.dxil");
const SCENE_PS: &[u8] = include_bytes!("assets/15_multiview/scene_ps.dxil");
const DISPLAY_VS: &[u8] = include_bytes!("assets/15_multiview/display_vs.dxil");
const DISPLAY_PS: &[u8] = include_bytes!("assets/15_multiview/display_ps.dxil");

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
    scene: RasterPipeline<B>,
    display: RasterPipeline<B>,
    views: Texture<B>,
    view: TextureView<B>,
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
        let vertices = device.create_buffer(&BufferDesc {
            label: Some("15 room triangle".into()),
            size: 72,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let (views, view) = layered_target(device, format, extent)?;
        let scene_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 scene layout"),
            groups: &[],
            push_constants: &[],
        })?;
        let scene = pipeline(
            device,
            scene_layout,
            format,
            SCENE_VS,
            SCENE_PS,
            true,
            vec![VertexBufferLayout {
                stride: 24,
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
            vec![],
        )
        .await?;
        Ok(Self {
            lane,
            vertices,
            scene,
            display,
            views,
            view,
            group,
            format,
            extent,
        })
    }
    pub fn record(&self, device: &Device<B>, image: &SurfaceImage<B>) -> Result<RecordedWork<B>> {
        let mut encoder = device.encoder(self.lane)?;
        encoder.write_buffer(&self.vertices, 0, &triangle())?;
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
            // The DX12 texture allocation declares this optimized clear value.
            // Keeping the pass clear equal lets the driver use fast clear.
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 0.0])),
            store: StoreOp::Store,
        })];
        let mut scene = encoder.begin_raster(&RasterPassDesc {
            label: Some("15 layered multiview scene"),
            colors: &scene_color,
            depth_stencil: None,
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
        scene.set_vertex_buffer(0, &self.vertices, 0..72)?;
        scene.draw(0..3, 0..1)?;
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
            depth_stencil: None,
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
