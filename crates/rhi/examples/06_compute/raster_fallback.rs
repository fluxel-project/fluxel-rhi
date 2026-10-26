//! Immediate WebGL2 raster path for the compute-filter comparison.
//!
//! This path keeps the source image and filter visible without claiming a
//! compute stage or storage-texture capability. Each frame is issued directly
//! to the browser owner thread; no CPU command stream is retained or replayed.

use std::sync::Arc;

use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;
use fluxel_rhi::gl::{Gl, GlInstance};

use super::{compute, utils};

const DISPLAY_WGSL: &[u8] = include_bytes!("../assets/06_compute/display.wgsl");
const FILTER_WGSL: &[u8] = include_bytes!("../assets/06_compute/filters_raster.wgsl");
const FILTER_ENTRIES: [&str; 3] = ["emboss_fs", "edgedetect_fs", "sharpen_fs"];

struct RasterWorkload {
    source: Texture<Gl>,
    group: BindGroup<Gl>,
    display_pipeline: RasterPipeline<Gl>,
    pipelines: [RasterPipeline<Gl>; 3],
    vertex: Buffer<Gl>,
    index: Buffer<Gl>,
    pixels: Option<Vec<u8>>,
    source_extent: [u32; 2],
    display_extent: Option<[u32; 2]>,
}

impl RasterWorkload {
    async fn new(device: &Device<Gl>, format: TextureFormat) -> Result<Self> {
        let (width, height, pixels) = compute::load_fallback_image()?;
        let source = device.create_texture(&TextureDesc {
            label: Some("06 WebGL2 filter source".into()),
            dimension: TextureDimension::D2,
            extent: [width, height, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST.union(TextureUsage::TEXTURE_BINDING),
            view_formats: vec![],
        })?;
        let source_view = device.create_view(
            &source,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("06 WebGL2 filter sampler".into()),
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("06 WebGL2 filter bindings".into()),
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })?;
        let entries = [
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&source_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let group = device.create_bind_group(&BindGroupDesc {
            label: Some("06 WebGL2 filter source"),
            layout: &layout,
            entries: &entries,
        })?;
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("06 WebGL2 raster filter layout".into()),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let vertex_shader = device
            .shader(ShaderDesc {
                label: Some("06 shared display vertex WGSL".into()),
                format: ShaderFormat::Wgsl,
                bytes: Arc::from(DISPLAY_WGSL),
                entries: vec![ShaderEntryPoint {
                    name: "vs_main".into(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await?;
        let display_fragment = device
            .shader(ShaderDesc {
                label: Some("06 source display WGSL".into()),
                format: ShaderFormat::Wgsl,
                bytes: Arc::from(DISPLAY_WGSL),
                entries: vec![ShaderEntryPoint {
                    name: "ps_main".into(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await?;
        let display_pipeline = raster_pipeline(
            device,
            pipeline_layout.clone(),
            vertex_shader.clone(),
            display_fragment,
            format,
            "06 source display",
            "ps_main",
        )
        .await?;
        let mut pipelines = Vec::with_capacity(FILTER_ENTRIES.len());
        for entry in FILTER_ENTRIES {
            let fragment = device
                .shader(ShaderDesc {
                    label: Some(format!("06 {entry} WGSL")),
                    format: ShaderFormat::Wgsl,
                    bytes: Arc::from(FILTER_WGSL),
                    entries: vec![ShaderEntryPoint {
                        name: entry.into(),
                        stage: ShaderStage::Fragment,
                        workgroup_size: None,
                    }],
                    required_features: vec![],
                })
                .await?;
            pipelines.push(
                raster_pipeline(
                    device,
                    pipeline_layout.clone(),
                    vertex_shader.clone(),
                    fragment,
                    format,
                    "06 raster filter fallback",
                    entry,
                )
                .await?,
            );
        }
        let vertex = device.create_buffer(&buffer(
            "06 WebGL2 display vertices",
            80,
            BufferUsage::VERTEX,
        ))?;
        let index =
            device.create_buffer(&buffer("06 WebGL2 display indices", 12, BufferUsage::INDEX))?;
        Ok(Self {
            source,
            group,
            display_pipeline,
            pipelines: pipelines.try_into().map_err(|_| Error::InvalidState)?,
            vertex,
            index,
            pixels: Some(pixels),
            source_extent: [width, height],
            display_extent: None,
        })
    }

    fn encode(
        &mut self,
        encoder: &mut CommandEncoder<Gl>,
        image: &SurfaceImage<Gl>,
        extent: [u32; 2],
    ) -> Result<()> {
        if let Some(pixels) = self.pixels.as_deref() {
            encoder
                .write_texture(
                    TextureRegion {
                        texture: &self.source,
                        mip: 0,
                        origin: [0; 3],
                        extent: [self.source_extent[0], self.source_extent[1], 1],
                        aspect: TextureAspects::COLOR,
                    },
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: self.source_extent[0] * 4,
                        rows_per_image: self.source_extent[1],
                    },
                    pixels,
                )
                .map_err(|error| Error::Backend(format!("06 texture upload: {error:?}")))?;
            encoder
                .transition_resources(&[transition(&self.source)])
                .map_err(|error| Error::Backend(format!("06 texture transition: {error:?}")))?;
        }
        if self.display_extent != Some(extent) {
            encoder
                .write_buffer(&self.vertex, 0, &quad(extent))
                .map_err(|error| Error::Backend(format!("06 display vertex upload: {error:?}")))?;
            encoder
                .write_buffer(&self.index, 0, &[0, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 0])
                .map_err(|error| Error::Backend(format!("06 display index upload: {error:?}")))?;
            self.display_extent = Some(extent);
        }
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.015, 0.015, 0.02, 1.0])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder
            .begin_raster(&RasterPassDesc {
                label: Some("06 WebGL2 source and raster filter"),
                colors: &colors,
                depth_stencil: None,
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: extent[0],
                    height: extent[1],
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .map_err(|error| Error::Backend(format!("06 begin raster: {error:?}")))?;
        pass.set_pipeline(&self.display_pipeline)
            .map_err(|error| Error::Backend(format!("06 source pipeline: {error:?}")))?;
        let split = extent[0] / 2;
        draw_half(
            &mut pass,
            &self.display_pipeline,
            &self.group,
            &self.vertex,
            &self.index,
            0,
            split,
            extent,
        )
        .map_err(|error| Error::Backend(format!("06 source draw: {error:?}")))?;
        draw_half(
            &mut pass,
            &self.pipelines[0],
            &self.group,
            &self.vertex,
            &self.index,
            split,
            extent[0] - split,
            extent,
        )
        .map_err(|error| Error::Backend(format!("06 filtered draw: {error:?}")))?;
        pass.finish()
            .map_err(|error| Error::Backend(format!("06 finish raster: {error:?}")))
    }

    fn commit_frame(&mut self) {
        self.pixels = None;
    }
}

pub(super) async fn run(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let extent = [canvas.width(), canvas.height()];
    if extent.contains(&0) {
        return Err(Error::SurfaceLost);
    }
    let native = GlInstance::from_webgl2_canvas(canvas)?;
    // SAFETY: the browser future stays on its JavaScript owner thread and the
    // native instance retains the canvas context owner for every issued call.
    let instance = unsafe { Instance::<Gl>::from_native(InstanceId::allocate()?, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or(Error::Unsupported("WebGL2 adapter"))?;
    let device = instance
        .open(DeviceRequest {
            adapter: adapter.id,
            required_features: vec![],
        })
        .await?;
    if device.capabilities().compute || device.capabilities().storage_textures {
        return Err(Error::InvalidState);
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::Unsupported("WebGL2 graphics lane"))?;
    let format = TextureFormat::Rgba8Unorm;
    let mut surface = Gl::surface(&device)?;
    surface
        .configure(&SurfaceConfig {
            extent,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await?;
    let mut workload = RasterWorkload::new(&device, format)
        .await
        .map_err(|error| Error::Backend(format!("06 WebGL2 raster setup: {error:?}")))?;
    let scope = utils::GpuScope::new(&device);
    for _ in 0..frames {
        let image = surface.acquire().await?;
        let (_, outcome) = utils::execute_frame(
            &device,
            lane,
            image,
            CompletionRequest::End,
            |commands, image| {
                workload
                    .encode(commands, image, extent)
                    .map_err(|error| Error::Backend(format!("06 WebGL2 raster frame: {error:?}")))
            },
        )?;
        workload.commit_frame();
        let point = utils::execution_end_checkpoint(&outcome)?;
        device.wait_completion(&point).await?;
    }
    scope.finish().await;
    Ok(())
}

async fn raster_pipeline(
    device: &Device<Gl>,
    layout: PipelineLayout<Gl>,
    vertex: Shader<Gl>,
    fragment: Shader<Gl>,
    format: TextureFormat,
    label: &'static str,
    entry: &'static str,
) -> Result<RasterPipeline<Gl>> {
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some(label.into()),
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs_main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: entry.into(),
                constants: vec![],
            }),
            vertex_buffers: vec![VertexBufferLayout {
                stride: 20,
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
                        format: VertexFormat::Float32(2),
                    },
                ],
            }],
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
            multiview: None,
            cache: None,
        })
        .await
}

fn draw_half(
    pass: &mut RasterPass<'_, Gl>,
    pipeline: &RasterPipeline<Gl>,
    group: &BindGroup<Gl>,
    vertex: &Buffer<Gl>,
    index: &Buffer<Gl>,
    x: u32,
    width: u32,
    extent: [u32; 2],
) -> Result<()> {
    if width == 0 || extent[1] == 0 {
        return Ok(());
    }
    pass.set_pipeline(pipeline)?;
    // WebGL2 stores the vertex and index bindings in the pipeline's VAO, so
    // each display/filter pipeline must receive its own immediate bindings.
    pass.set_vertex_buffer(0, vertex, 0..80)?;
    pass.set_index_buffer(index, 0..12, IndexFormat::Uint16)?;
    pass.set_bind_group(0, group, &[])?;
    pass.viewport(Viewport {
        x: x as f32,
        y: 0.0,
        width: width as f32,
        height: extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    })?;
    pass.scissor(Rect {
        x,
        y: 0,
        width,
        height: extent[1],
    })?;
    pass.draw_indexed(0..6, 0, 0..1)
}

fn transition(texture: &Texture<Gl>) -> ResourceTransition<'_, Gl> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            to: TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
        },
        layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
        ownership: None,
    })
}

fn buffer(label: &str, size: u64, role: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage: role.union(BufferUsage::COPY_DST),
        memory: MemoryClass::Device,
    }
}

fn quad(extent: [u32; 2]) -> [u8; 80] {
    let y = 0.866_025_4_f32;
    let x = y * (extent[1] as f32 * 2.0 / extent[0] as f32);
    let vertices: [f32; 20] = [
        -x, -y, 0., 0., 1., x, -y, 0., 1., 1., x, y, 0., 1., 0., -x, y, 0., 0., 0.,
    ];
    let mut bytes = [0; 80];
    for (index, value) in vertices.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
