//! Sequential stereo fallback for backends without native multiview.
//!
//! Each eye is rendered by an ordinary pass into its own 2D color/depth
//! targets, then the shared WGSL display pass applies the side-by-side mapping.

use super::*;

/// Two ordinary per-eye passes preserve the visible stereo workload without
/// claiming that the backend supports native multiview.
pub(crate) struct StereoFallback<B: super::WgslStereoBackend> {
    vertices: Buffer<B>,
    indices: Buffer<B>,
    vertex_bytes: Arc<[u8]>,
    index_bytes: Arc<[u8]>,
    uploaded: bool,
    vertex_len: u64,
    index_len: u64,
    index_count: u32,
    eye_uniforms: [Buffer<B>; 2],
    eye_groups: [BindGroup<B>; 2],
    scene: RasterPipeline<B>,
    display: RasterPipeline<B>,
    _views: [Texture<B>; 2],
    scene_views: [TextureView<B>; 2],
    _depth: [Texture<B>; 2],
    depth_views: [TextureView<B>; 2],
    display_groups: [BindGroup<B>; 2],
    extent: [u32; 2],
}

impl<B: super::WgslStereoBackend> StereoFallback<B> {
    pub(crate) async fn new(
        device: &Device<B>,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let room = gltf::load_embedded_model(ROOM, gltf::LoadOptions::CPP_PORT)
            .map_err(|error| Error::Backend(error.to_string()))?;
        let vertices = device.create_buffer(&BufferDesc {
            label: Some("15 stereo sampleroom vertices".into()),
            size: room.vertices.len() as u64,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let indices = device.create_buffer(&BufferDesc {
            label: Some("15 stereo sampleroom indices".into()),
            size: room.indices.len() as u64,
            usage: BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let make_eye_uniform = |eye| {
            device.create_buffer(&BufferDesc {
                label: Some(format!("15 stereo eye {eye} transform")),
                size: 144,
                usage: BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
        };
        let eye_uniforms = [make_eye_uniform(0)?, make_eye_uniform(1)?];
        let (views, scene_views) = eye_color_targets(device, format, extent)?;
        let (depth, depth_views) = eye_depth_targets(device, extent)?;
        let scene_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("15 stereo eye transform layout".into()),
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
                label: Some("15 stereo eye transform group"),
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
            label: Some("15 stereo scene layout"),
            groups: &[&scene_layout],
            push_constants: &[],
        })?;
        let scene = stereo_pipeline(
            device,
            scene_pipeline_layout,
            format,
            FALLBACK_SCENE_WGSL,
            true,
            vec![scene_vertex_layout()],
            "15 stereo scene",
        )
        .await?;
        let make_display_layout = |label: &'static str| {
            device.create_bind_group_layout(&BindGroupLayoutDesc {
                label: Some(label.into()),
                entries: vec![
                    binding(
                        0,
                        BindingType::SampledTexture {
                            sample: TextureSampleType::Float,
                            dimension: TextureViewDimension::D2,
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
            })
        };
        let display_layouts = [
            make_display_layout("15 left-eye display bindings")?,
            make_display_layout("15 right-eye display bindings")?,
        ];
        let make_display_group = |eye| -> Result<BindGroup<B>> {
            let sampler = device.create_sampler(&SamplerDesc {
                label: Some(format!("15 eye {eye} display sampler")),
                address: [AddressMode::ClampToEdge; 3],
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mip_filter: FilterMode::Linear,
                lod: 0.0..1.0,
                anisotropy: 1,
                compare: None,
                border: None,
            })?;
            device.create_bind_group(&BindGroupDesc {
                label: Some("15 eye display group"),
                layout: &display_layouts[eye],
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&scene_views[eye]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&sampler),
                    },
                ],
            })
        };
        let display_groups = [make_display_group(0)?, make_display_group(1)?];
        let display_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 two-group display layout".into()),
            groups: &[&display_layouts[0], &display_layouts[1]],
            push_constants: &[],
        })?;
        let display = stereo_pipeline(
            device,
            display_pipeline_layout,
            format,
            FALLBACK_DISPLAY_WGSL,
            false,
            vec![],
            "15 stereo display",
        )
        .await?;
        let vertex_len = room.vertices.len() as u64;
        let index_len = room.indices.len() as u64;
        let index_count = (room.indices.len() / 4) as u32;
        Ok(Self {
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
            display_groups,
            extent,
        })
    }

    pub(crate) fn encode(
        &mut self,
        encoder: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
    ) -> Result<bool> {
        let first_upload = !self.uploaded;
        if first_upload {
            encoder.write_buffer(&self.vertices, 0, &self.vertex_bytes)?;
            encoder.write_buffer(&self.indices, 0, &self.index_bytes)?;
            for (buffer, bytes) in self
                .eye_uniforms
                .iter()
                .zip(stereo_eye_matrices(self.extent))
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
                label: Some("15 sequential stereo eye"),
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
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        raster_area(&mut display, self.extent)?;
        display.set_pipeline(&self.display)?;
        display.set_bind_group(0, &self.display_groups[0], &[])?;
        display.set_bind_group(1, &self.display_groups[1], &[])?;
        display.draw(0..3, 0..1)?;
        display.finish()?;
        Ok(first_upload)
    }

    pub(crate) fn mark_uploaded(&mut self) {
        self.uploaded = true;
    }

    pub(crate) fn extent(&self) -> [u32; 2] {
        self.extent
    }
}

fn eye_color_targets<B: super::WgslStereoBackend>(
    device: &Device<B>,
    format: TextureFormat,
    extent: [u32; 2],
) -> Result<([Texture<B>; 2], [TextureView<B>; 2])> {
    let make = |eye| -> Result<(Texture<B>, TextureView<B>)> {
        let texture = device.create_texture(&TextureDesc {
            label: Some(format!("15 stereo eye {eye} color")),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
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
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        Ok((texture, view))
    };
    let (left, left_view) = make(0)?;
    let (right, right_view) = make(1)?;
    Ok(([left, right], [left_view, right_view]))
}

fn eye_depth_targets<B: super::WgslStereoBackend>(
    device: &Device<B>,
    extent: [u32; 2],
) -> Result<([Texture<B>; 2], [TextureView<B>; 2])> {
    let make = |eye| -> Result<(Texture<B>, TextureView<B>)> {
        let texture = device.create_texture(&TextureDesc {
            label: Some(format!("15 stereo eye {eye} depth")),
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
    };
    let (left, left_view) = make(0)?;
    let (right, right_view) = make(1)?;
    Ok(([left, right], [left_view, right_view]))
}

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

fn raster_area<B: super::WgslStereoBackend>(
    pass: &mut RasterPass<'_, B>,
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

async fn stereo_pipeline<B: super::WgslStereoBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    wgsl: &[u8],
    depth: bool,
    vertex_buffers: Vec<VertexBufferLayout>,
    label: &'static str,
) -> Result<RasterPipeline<B>> {
    let make_wgsl_shader = |entry: &'static str, stage| ShaderDesc {
        label: Some(label.into()),
        format: ShaderFormat::Wgsl,
        bytes: Arc::from(wgsl),
        entries: vec![ShaderEntryPoint {
            name: entry.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vertex = device
        .shader(make_wgsl_shader("vs_main", ShaderStage::Vertex))
        .await?;
    let fragment = device
        .shader(make_wgsl_shader("ps_main", ShaderStage::Fragment))
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

fn stereo_eye_matrices(extent: [u32; 2]) -> [[u8; 144]; 2] {
    let all = two_eye_matrices(extent);
    std::array::from_fn(|eye| {
        let mut bytes = [0; 144];
        bytes[..64].copy_from_slice(&all[eye * 64..eye * 64 + 64]);
        bytes[64..128].copy_from_slice(&all[128 + eye * 64..192 + eye * 64]);
        bytes[128..144].copy_from_slice(&all[256..272]);
        bytes
    })
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2.rs"]
mod webgl2;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[path = "webgpu.rs"]
mod webgpu;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub use webgl2::run_webgl2_multiview;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub use webgpu::run_webgpu_multiview;
