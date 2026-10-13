//! WebGPU-only sequential stereo fallback for example 15.
//!
//! WebGPU cannot claim Core multiview support, so this module records one
//! ordinary raster pass per eye and retains the shared texture-array display.

use super::*;

/// WebGPU has no Core multiview capability. This fallback uses two ordinary
/// native raster passes, one per array layer, then retains the display pass.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
struct WebGpuStereoFallback {
    lane: LaneId,
    vertices: Buffer<fluxel_rhi::webgpu::WebGpu>,
    indices: Buffer<fluxel_rhi::webgpu::WebGpu>,
    vertex_bytes: Arc<[u8]>,
    index_bytes: Arc<[u8]>,
    uploaded: bool,
    vertex_len: u64,
    index_len: u64,
    index_count: u32,
    eye_uniforms: [Buffer<fluxel_rhi::webgpu::WebGpu>; 2],
    eye_groups: [BindGroup<fluxel_rhi::webgpu::WebGpu>; 2],
    scene: RasterPipeline<fluxel_rhi::webgpu::WebGpu>,
    display: RasterPipeline<fluxel_rhi::webgpu::WebGpu>,
    _views: Texture<fluxel_rhi::webgpu::WebGpu>,
    scene_views: [TextureView<fluxel_rhi::webgpu::WebGpu>; 2],
    _depth: Texture<fluxel_rhi::webgpu::WebGpu>,
    depth_views: [TextureView<fluxel_rhi::webgpu::WebGpu>; 2],
    display_group: BindGroup<fluxel_rhi::webgpu::WebGpu>,
    extent: [u32; 2],
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
impl WebGpuStereoFallback {
    async fn new(
        device: &Device<fluxel_rhi::webgpu::WebGpu>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let room = gltf::load_embedded_model(ROOM, gltf::LoadOptions::CPP_PORT)
            .map_err(|error| Error::Backend(error.to_string()))?;
        let vertices = device.create_buffer(&BufferDesc {
            label: Some("15 WebGPU sampleroom vertices".into()),
            size: room.vertices.len() as u64,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let indices = device.create_buffer(&BufferDesc {
            label: Some("15 WebGPU sampleroom indices".into()),
            size: room.indices.len() as u64,
            usage: BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let make_eye_uniform = |eye| {
            device.create_buffer(&BufferDesc {
                label: Some(format!("15 WebGPU eye {eye} transform")),
                size: 144,
                usage: BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
        };
        let eye_uniforms = [make_eye_uniform(0)?, make_eye_uniform(1)?];
        let (views, array_view, scene_views) = webgpu_layered_target(device, format, extent)?;
        let (depth, depth_views) = webgpu_layered_depth(device, extent)?;
        let scene_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("15 WebGPU eye transform layout".into()),
            entries: vec![binding(
                0,
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 144,
                },
                ShaderStage::Vertex,
            )],
        })?;
        let eye_group = |eye| {
            device.create_bind_group(&BindGroupDesc {
                label: Some("15 WebGPU eye transform group"),
                layout: &scene_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &eye_uniforms[eye],
                        range: 0..144,
                    },
                }],
            })
        };
        let eye_groups = [eye_group(0)?, eye_group(1)?];
        let scene_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 WebGPU scene layout"),
            groups: &[&scene_layout],
            push_constants: &[],
        })?;
        let scene = webgpu_pipeline(
            device,
            scene_pipeline_layout,
            format,
            WEBGPU_SCENE_WGSL,
            true,
            vec![scene_vertex_layout()],
            "15 WebGPU stereo scene",
        )
        .await?;
        let display_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("15 WebGPU display bindings".into()),
            entries: vec![
                binding(
                    0,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    ShaderStage::Fragment,
                ),
                binding(
                    1,
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    ShaderStage::Fragment,
                ),
            ],
        })?;
        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("15 WebGPU display sampler".into()),
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let display_group = device.create_bind_group(&BindGroupDesc {
            label: Some("15 WebGPU display group"),
            layout: &display_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&array_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        })?;
        let display_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 WebGPU display layout"),
            groups: &[&display_layout],
            push_constants: &[],
        })?;
        let display = webgpu_pipeline(
            device,
            display_pipeline_layout,
            format,
            WEBGPU_DISPLAY_WGSL,
            false,
            vec![],
            "15 WebGPU stereo display",
        )
        .await?;
        let vertex_len = room.vertices.len() as u64;
        let index_len = room.indices.len() as u64;
        let index_count = (room.indices.len() / 4) as u32;
        Ok(Self {
            lane,
            vertices,
            indices,
            vertex_bytes: Arc::from(room.vertices),
            index_bytes: Arc::from(room.indices),
            uploaded: false,
            vertex_len,
            index_len,
            index_count,
            eye_uniforms,
            eye_groups,
            scene,
            display,
            _views: views,
            scene_views,
            _depth: depth,
            depth_views,
            display_group,
            extent,
        })
    }

    fn record(
        &mut self,
        device: &Device<fluxel_rhi::webgpu::WebGpu>,
        image: &SurfaceImage<fluxel_rhi::webgpu::WebGpu>,
    ) -> Result<(RecordedWork<fluxel_rhi::webgpu::WebGpu>, bool)> {
        let mut encoder = device.encoder(self.lane)?;
        let first_upload = !self.uploaded;
        if first_upload {
            encoder.write_buffer(&self.vertices, 0, &self.vertex_bytes)?;
            encoder.write_buffer(&self.indices, 0, &self.index_bytes)?;
            for (buffer, bytes) in self
                .eye_uniforms
                .iter()
                .zip(webgpu_eye_matrices(self.extent))
            {
                encoder.write_buffer(buffer, 0, &bytes)?;
            }
        }
        for eye in 0..2 {
            let colors = [Some(ColorAttachment {
                view: AttachmentView::Texture(&self.scene_views[eye]),
                resolve_target: None,
                load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 0.0])),
                store: StoreOp::Store,
            })];
            let mut pass = encoder.begin_raster(&RasterPassDesc {
                label: Some("15 WebGPU sequential stereo eye"),
                colors: &colors,
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&self.depth_views[eye]),
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
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })?;
            raster_area(&mut pass, self.extent)?;
            pass.set_pipeline(&self.scene)?;
            pass.set_bind_group(0, &self.eye_groups[eye], &[])?;
            pass.set_vertex_buffer(0, &self.vertices, 0..self.vertex_len)?;
            pass.set_index_buffer(&self.indices, 0..self.index_len, IndexFormat::Uint32)?;
            pass.draw_indexed(0..self.index_count, 0, 0..1)?;
            pass.finish()?;
        }
        let surface = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            store: StoreOp::Store,
        })];
        let mut display = encoder.begin_raster(&RasterPassDesc {
            label: Some("15 WebGPU two-eye display"),
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
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        raster_area(&mut display, self.extent)?;
        display.set_pipeline(&self.display)?;
        display.set_bind_group(0, &self.display_group, &[])?;
        display.draw(0..3, 0..1)?;
        display.finish()?;
        Ok((encoder.finish()?, first_upload))
    }

    fn mark_uploaded(&mut self) {
        self.uploaded = true;
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn webgpu_layered_target(
    device: &Device<fluxel_rhi::webgpu::WebGpu>,
    format: TextureFormat,
    extent: [u32; 2],
) -> Result<(
    Texture<fluxel_rhi::webgpu::WebGpu>,
    TextureView<fluxel_rhi::webgpu::WebGpu>,
    [TextureView<fluxel_rhi::webgpu::WebGpu>; 2],
)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("15 WebGPU two-eye attachment".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 2,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::TEXTURE_BINDING),
        view_formats: vec![],
    })?;
    let array = device.create_view(
        &texture,
        &TextureViewDesc {
            format,
            dimension: TextureViewDimension::D2Array,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..2,
        },
    )?;
    let layer = |index| {
        device.create_view(
            &texture,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: index..index + 1,
            },
        )
    };
    let layers = [layer(0)?, layer(1)?];
    Ok((texture, array, layers))
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn webgpu_layered_depth(
    device: &Device<fluxel_rhi::webgpu::WebGpu>,
    extent: [u32; 2],
) -> Result<(
    Texture<fluxel_rhi::webgpu::WebGpu>,
    [TextureView<fluxel_rhi::webgpu::WebGpu>; 2],
)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("15 WebGPU two-eye depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 2,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let layer = |index| {
        device.create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: index..index + 1,
            },
        )
    };
    let layers = [layer(0)?, layer(1)?];
    Ok((texture, layers))
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn scene_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
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
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn raster_area(
    pass: &mut RasterPass<'_, fluxel_rhi::webgpu::WebGpu>,
    extent: [u32; 2],
) -> Result<()> {
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
    })
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
async fn webgpu_pipeline(
    device: &Device<fluxel_rhi::webgpu::WebGpu>,
    layout: PipelineLayout<fluxel_rhi::webgpu::WebGpu>,
    format: TextureFormat,
    wgsl: &[u8],
    depth: bool,
    vertex_buffers: Vec<VertexBufferLayout>,
    label: &'static str,
) -> Result<RasterPipeline<fluxel_rhi::webgpu::WebGpu>> {
    let vertex = common::native_shader_with_wgsl(
        device,
        label,
        "vs_main",
        ShaderStage::Vertex,
        SCENE_VS,
        SCENE_VS_SPIRV,
        wgsl,
    )
    .await?;
    let fragment = common::native_shader_with_wgsl(
        device,
        label,
        "ps_main",
        ShaderStage::Fragment,
        SCENE_PS,
        SCENE_PS_SPIRV,
        wgsl,
    )
    .await?;
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
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
                stencil_front: face,
                stencil_back: face,
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

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn webgpu_eye_matrices(extent: [u32; 2]) -> [[u8; 144]; 2] {
    let all = two_eye_matrices(extent);
    std::array::from_fn(|eye| {
        let mut bytes = [0; 144];
        bytes[..64].copy_from_slice(&all[eye * 64..eye * 64 + 64]);
        bytes[64..128].copy_from_slice(&all[128 + eye * 64..192 + eye * 64]);
        bytes[128..144].copy_from_slice(&all[256..272]);
        bytes
    })
}

/// Browser entry for the sequential two-eye path. `multiview` remains false.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_multiview(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};
    let run = async {
        if frames == 0 {
            return Err(Error::InvalidRange);
        }
        let device = WebGpuInstance::new()?.open_default().await?;
        if device.capabilities().multiview {
            return Err(Error::InvalidState);
        }
        let lane = device
            .lanes()
            .iter()
            .find(|lane| lane.class == LaneClass::Graphics)
            .map(|lane| lane.id)
            .ok_or(Error::Unsupported("WebGPU graphics lane"))?;
        let extent = [canvas.width(), canvas.height()];
        let mut surface = WebGpu::surface_from_canvas(&device, canvas)?;
        let format = surface
            .capabilities()?
            .formats
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("WebGPU canvas format"))?;
        surface
            .configure(&SurfaceConfig {
                extent,
                format,
                present_mode: PresentMode::Fifo,
                alpha_mode: CompositeAlphaMode::Opaque,
                color_space: ColorSpace::Srgb,
                usage: SurfaceUsage::RENDER_ATTACHMENT,
                maximum_frame_latency: 2,
            })
            .await?;
        let mut demo = WebGpuStereoFallback::new(&device, lane, format, extent).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let (work, uploaded) = demo.record(&device, &image)?;
            common::submit_and_present_async(&device, lane, work, image).await?;
            if uploaded {
                demo.mark_uploaded();
            }
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}
