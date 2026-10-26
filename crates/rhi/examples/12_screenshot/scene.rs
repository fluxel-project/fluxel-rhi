//! The C++ screenshot scene: one pretransformed Chinese-dragon draw.

use super::super::{ScreenshotWorkloadBackend, gltf};
use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;

const DRAGON: &[u8] = include_bytes!("../assets/models/chinesedragon.gltf");
const VS: &[u8] = include_bytes!("../assets/12_screenshot/screenshot_vs.dxil");
const PS: &[u8] = include_bytes!("../assets/12_screenshot/screenshot_ps.dxil");
const VS_SPIRV: &[u8] = include_bytes!("../assets/12_screenshot/screenshot.vs.spv");
const PS_SPIRV: &[u8] = include_bytes!("../assets/12_screenshot/screenshot.ps.spv");
const WGSL: &[u8] = include_bytes!("../assets/12_screenshot/screenshot.wgsl");

pub(crate) struct DragonScene<B: ScreenshotWorkloadBackend> {
    vertices: Buffer<B>,
    indices: Buffer<B>,
    transform: Buffer<B>,
    group: BindGroup<B>,
    target_depth: Texture<B>,
    target_depth_view: TextureView<B>,
    surface_depth: Texture<B>,
    surface_depth_view: TextureView<B>,
    target_pipeline: RasterPipeline<B>,
    surface_pipeline: RasterPipeline<B>,
    vertex_len: u64,
    index_len: u64,
    index_count: u32,
    source_vertices: Vec<u8>,
    source_indices: Vec<u8>,
    uploaded: bool,
}

impl<B: ScreenshotWorkloadBackend> DragonScene<B> {
    pub(crate) async fn new(
        device: &Device<B>,
        extent: [u32; 2],
        surface_format: TextureFormat,
    ) -> Result<Self> {
        // Fluxel's Vulkan route uses a negative viewport height to implement
        // its WebGPU-style presentation axis.  Applying the C++ loader's
        // `FlipY` as well would reflect the mesh twice.
        let model =
            gltf::load_embedded_model(DRAGON, gltf::LoadOptions::CPP_PORT.with_flip_y(false))
                .map_err(|error| Error::Backend(error.to_string()))?;
        let vertices = buffer(
            device,
            "12 dragon vertices",
            model.vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        )?;
        let indices = buffer(
            device,
            "12 dragon indices",
            model.indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        )?;
        let transform = buffer(
            device,
            "12 screenshot transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        )?;
        let group_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("12 screenshot transform layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 192,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let group = device.create_bind_group(&BindGroupDesc {
            label: Some("12 screenshot transform group".into()),
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &transform,
                    range: 0..256,
                },
            }],
        })?;
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("12 screenshot pipeline layout"),
            groups: &[&group_layout],
            push_constants: &[],
        })?;
        let target_pipeline =
            pipeline(device, pipeline_layout.clone(), TextureFormat::Rgba8Unorm).await?;
        let surface_pipeline = pipeline(device, pipeline_layout, surface_format).await?;
        let (target_depth, target_depth_view) =
            depth(device, extent, "12 screenshot target depth")?;
        let (surface_depth, surface_depth_view) =
            depth(device, extent, "12 screenshot surface depth")?;
        Ok(Self {
            vertices,
            indices,
            transform,
            group,
            target_depth,
            target_depth_view,
            surface_depth,
            surface_depth_view,
            target_pipeline,
            surface_pipeline,
            vertex_len: model.vertices.len() as u64,
            index_len: model.indices.len() as u64,
            index_count: model.index_count(),
            source_vertices: model.vertices,
            source_indices: model.indices,
            uploaded: false,
        })
    }

    pub(crate) fn record_target(
        &mut self,
        encoder: &mut CommandEncoder<B>,
        color: &TextureView<B>,
        extent: [u32; 2],
    ) -> Result<()> {
        self.upload_and_transform(encoder, extent)?;
        self.record(
            encoder,
            AttachmentView::Texture(color),
            &self.target_depth,
            &self.target_depth_view,
            &self.target_pipeline,
            extent,
            "12 screenshot readback scene",
        )
    }

    pub(crate) fn record_surface(
        &self,
        encoder: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<()> {
        self.record(
            encoder,
            AttachmentView::Surface(image),
            &self.surface_depth,
            &self.surface_depth_view,
            &self.surface_pipeline,
            extent,
            "12 screenshot presentation scene",
        )
    }

    pub(crate) fn mark_uploaded(&mut self) {
        self.uploaded = true;
    }

    fn upload_and_transform(
        &mut self,
        encoder: &mut CommandEncoder<B>,
        extent: [u32; 2],
    ) -> Result<()> {
        if !self.uploaded {
            encoder.write_buffer(&self.vertices, 0, &self.source_vertices)?;
            encoder.write_buffer(&self.indices, 0, &self.source_indices)?;
        }
        encoder.write_buffer(&self.transform, 0, &camera_bytes(extent))
    }

    fn record<'a>(
        &'a self,
        encoder: &mut CommandEncoder<B>,
        color: AttachmentView<'a, B>,
        depth_texture: &'a Texture<B>,
        depth: &'a TextureView<B>,
        pipeline: &'a RasterPipeline<B>,
        extent: [u32; 2],
        label: &'static str,
    ) -> Result<()> {
        let colors = [Some(ColorAttachment {
            view: color,
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.025, 1.0])),
            store: StoreOp::Store,
        })];
        encoder
            .transition_resources(&[super::super::utils::discarded_depth(depth_texture, 0..1)])?;
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some(label),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(depth),
                depth: Some((LoadOp::Clear(1.0), StoreOp::Discard)),
                stencil: None,
                depth_read_only: false,
                stencil_read_only: false,
            }),
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
        })?;
        pass.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        pass.set_pipeline(pipeline)?;
        pass.set_bind_group(0, &self.group, &[])?;
        pass.set_vertex_buffer(0, &self.vertices, 0..self.vertex_len)?;
        pass.set_index_buffer(&self.indices, 0..self.index_len, IndexFormat::Uint32)?;
        pass.draw_indexed(0..self.index_count, 0, 0..1)?;
        pass.finish()
    }
}

fn buffer<B: ScreenshotWorkloadBackend>(
    device: &Device<B>,
    label: &'static str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    device.create_buffer(&BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}

fn depth<B: ScreenshotWorkloadBackend>(
    device: &Device<B>,
    extent: [u32; 2],
    label: &'static str,
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
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
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((texture, view))
}

async fn pipeline<B: ScreenshotWorkloadBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
) -> Result<RasterPipeline<B>> {
    let vertex = super::super::utils::native_shader_with_wgsl(
        device,
        "12 screenshot vertex",
        "vs_main",
        ShaderStage::Vertex,
        VS,
        VS_SPIRV,
        WGSL,
    )
    .await?;
    let fragment = super::super::utils::native_shader_with_wgsl(
        device,
        "12 screenshot fragment",
        "ps_main",
        ShaderStage::Fragment,
        PS,
        PS_SPIRV,
        WGSL,
    )
    .await?;
    let stencil = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some("12 screenshot dragon pipeline".into()),
            layout,
            vertex: PipelineStage {
                module: vertex,
                entry: "vs_main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: "ps_main".into(),
                constants: vec![],
            }),
            vertex_buffers: vec![VertexBufferLayout {
                stride: gltf::VKGLTF_VERTEX_STRIDE as u64,
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
                        offset: 32,
                        format: VertexFormat::Float32(4),
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
                cull_mode: CullMode::Back,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil_front: stencil,
                stencil_back: stencil,
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
            multiview: None,
            cache: None,
        })
        .await
}

fn camera_bytes(extent: [u32; 2]) -> [u8; 256] {
    // screenshot.cpp's CPU struct is `projection, model, view`, while its
    // GLSL block is `projection, view, model`. Preserve that ABI: shader
    // `view` receives identity and shader `model` receives the camera.
    let aspect = extent[0] as f32 / extent[1] as f32;
    let f = 1.0 / 30f32.to_radians().tan();
    let near = 0.1;
    let far = 512.0;
    let projection = [
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
    ];
    let model = identity();
    let view = multiply(
        translation([0.0, 0.0, -3.0]),
        multiply(
            rotation_x(25.0f32.to_radians()),
            rotation_y(23.75f32.to_radians()),
        ),
    );
    let mut bytes = [0; 256];
    for (index, value) in projection.into_iter().chain(model).chain(view).enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn translation(position: [f32; 3]) -> [f32; 16] {
    let mut output = identity();
    output[12..15].copy_from_slice(&position);
    output
}

fn rotation_x(angle: f32) -> [f32; 16] {
    let (sine, cosine) = angle.sin_cos();
    [
        1.0, 0.0, 0.0, 0.0, 0.0, cosine, sine, 0.0, 0.0, -sine, cosine, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn rotation_y(angle: f32) -> [f32; 16] {
    let (sine, cosine) = angle.sin_cos();
    [
        cosine, 0.0, -sine, 0.0, 0.0, 1.0, 0.0, 0.0, sine, 0.0, cosine, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn multiply(left: [f32; 16], right: [f32; 16]) -> [f32; 16] {
    let mut output = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4)
                .map(|index| left[index * 4 + row] * right[column * 4 + index])
                .sum();
        }
    }
    output
}
