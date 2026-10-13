//! WebGPU primary-pass fallback for example 13.
//!
//! WebGPU has no secondary command buffers, so this module records the C++
//! workload sequentially while preserving its draw order and dynamic uniforms.
use super::*;

/// Browser-only execution of the C++ workload.  WebGPU deliberately exposes
/// no secondary command buffers, so it records the same star draw followed by
/// all 512 UFO draws in source order into one primary render pass.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
struct WebGpuSequential {
    lane: LaneId,
    vertices: Buffer<fluxel_rhi::webgpu::WebGpu>,
    indices: Buffer<fluxel_rhi::webgpu::WebGpu>,
    transforms: Buffer<fluxel_rhi::webgpu::WebGpu>,
    star_vertices: Buffer<fluxel_rhi::webgpu::WebGpu>,
    star_indices: Buffer<fluxel_rhi::webgpu::WebGpu>,
    star_uniform: Buffer<fluxel_rhi::webgpu::WebGpu>,
    group: BindGroup<fluxel_rhi::webgpu::WebGpu>,
    star_group: BindGroup<fluxel_rhi::webgpu::WebGpu>,
    ufo: RasterPipeline<fluxel_rhi::webgpu::WebGpu>,
    stars: RasterPipeline<fluxel_rhi::webgpu::WebGpu>,
    source: [Vec<u8>; 4],
    lens: [u64; 4],
    counts: [u32; 2],
    uploaded: bool,
    frame: u32,
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
impl WebGpuSequential {
    async fn new(
        d: &Device<fluxel_rhi::webgpu::WebGpu>,
        lane: LaneId,
        format: TextureFormat,
    ) -> Result<Self> {
        let source = gltf::LoadOptions::CPP_PORT.with_flip_y(false);
        let u =
            gltf::load_embedded_model(UFO, source).map_err(|x| Error::Backend(x.to_string()))?;
        let s = gltf::load_embedded_model(STAR_SPHERE, source)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let make = |name: &str, n: u64, usage: BufferUsage| {
            d.create_buffer(&BufferDesc {
                label: Some(name.into()),
                size: n,
                usage: usage.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
        };
        let vertices = make(
            "13 webgpu UFO vertices",
            u.vertices.len() as u64,
            BufferUsage::VERTEX,
        )?;
        let indices = make(
            "13 webgpu UFO indices",
            u.indices.len() as u64,
            BufferUsage::INDEX,
        )?;
        let star_vertices = make(
            "13 webgpu star vertices",
            s.vertices.len() as u64,
            BufferUsage::VERTEX,
        )?;
        let star_indices = make(
            "13 webgpu star indices",
            s.indices.len() as u64,
            BufferUsage::INDEX,
        )?;
        let transforms = make(
            "13 webgpu UFO transforms",
            u64::from(UFO_COUNT) * UNIFORM,
            BufferUsage::UNIFORM,
        )?;
        let star_uniform = make("13 webgpu star transform", UNIFORM, BufferUsage::UNIFORM)?;
        let ufo_bindings = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("13 webgpu UFO transforms".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: true,
                    min_size: 80,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let star_bindings = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("13 webgpu stars transform".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 64,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("13 webgpu UFO group".into()),
            layout: &ufo_bindings,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &transforms,
                    range: 0..UNIFORM,
                },
            }],
        })?;
        let star_group = d.create_bind_group(&BindGroupDesc {
            label: Some("13 webgpu stars group".into()),
            layout: &star_bindings,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &star_uniform,
                    range: 0..UNIFORM,
                },
            }],
        })?;
        let ufo_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("13 webgpu UFO layout".into()),
            groups: &[&ufo_bindings],
            push_constants: &[],
        })?;
        let star_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("13 webgpu stars layout".into()),
            groups: &[&star_bindings],
            push_constants: &[],
        })?;
        let ufo = pipeline_wgsl(
            d,
            "13 webgpu UFO",
            ufo_layout,
            format,
            UFO_WGSL,
            CullMode::Front,
            vec![mesh_layout()],
        )
        .await?;
        let stars = pipeline_wgsl(
            d,
            "13 webgpu stars",
            star_layout,
            format,
            STARS_WGSL,
            CullMode::Back,
            vec![VertexBufferLayout {
                stride: STRIDE,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32(3),
                }],
            }],
        )
        .await?;
        let lens = [
            u.vertices.len() as u64,
            u.indices.len() as u64,
            s.vertices.len() as u64,
            s.indices.len() as u64,
        ];
        let counts = [u.index_count(), s.index_count()];
        Ok(Self {
            lane,
            vertices,
            indices,
            transforms,
            star_vertices,
            star_indices,
            star_uniform,
            group,
            star_group,
            ufo,
            stars,
            source: [u.vertices, u.indices, s.vertices, s.indices],
            lens,
            counts,
            uploaded: false,
            frame: 0,
        })
    }
    fn record(
        &mut self,
        d: &Device<fluxel_rhi::webgpu::WebGpu>,
        image: &SurfaceImage<fluxel_rhi::webgpu::WebGpu>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<fluxel_rhi::webgpu::WebGpu>> {
        let mut e = d.encoder(self.lane)?;
        if !self.uploaded {
            e.write_buffer(&self.vertices, 0, &self.source[0])?;
            e.write_buffer(&self.indices, 0, &self.source[1])?;
            e.write_buffer(&self.star_vertices, 0, &self.source[2])?;
            e.write_buffer(&self.star_indices, 0, &self.source[3])?;
        }
        e.write_buffer(&self.transforms, 0, &transforms(extent, self.frame))?;
        e.write_buffer(&self.star_uniform, 0, &matrix_bytes(star_matrix(extent)))?;
        self.frame = self.frame.wrapping_add(1);
        let (depth, view) = depth_webgpu(d, extent)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("13 webgpu sequential primary pass"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&view),
                depth: Some((LoadOp::Clear(1.), StoreOp::Discard)),
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
        p.viewport(Viewport {
            x: 0.,
            y: 0.,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.,
            max_depth: 1.,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        p.set_pipeline(&self.stars)?;
        p.set_bind_group(0, &self.star_group, &[])?;
        p.set_vertex_buffer(0, &self.star_vertices, 0..self.lens[2])?;
        p.set_index_buffer(&self.star_indices, 0..self.lens[3], IndexFormat::Uint32)?;
        p.draw_indexed(0..self.counts[1], 0, 0..1)?;
        p.set_pipeline(&self.ufo)?;
        p.set_vertex_buffer(0, &self.vertices, 0..self.lens[0])?;
        p.set_index_buffer(&self.indices, 0..self.lens[1], IndexFormat::Uint32)?;
        for object in 0..UFO_COUNT {
            p.set_bind_group(0, &self.group, &[object * UNIFORM as u32])?;
            p.draw_indexed(0..self.counts[0], 0, 0..1)?;
        }
        p.finish()?;
        let _ = depth;
        let work = e.finish()?;
        self.uploaded = true;
        Ok(work)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn mesh_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        stride: STRIDE,
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
fn depth_webgpu(
    d: &Device<fluxel_rhi::webgpu::WebGpu>,
    extent: [u32; 2],
) -> Result<(
    Texture<fluxel_rhi::webgpu::WebGpu>,
    TextureView<fluxel_rhi::webgpu::WebGpu>,
)> {
    let texture = d.create_texture(&TextureDesc {
        label: Some("13 webgpu depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = d.create_view(
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
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn disabled_stencil_webgpu() -> StencilFaceState {
    StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    }
}

async fn pipeline_wgsl(
    d: &Device<fluxel_rhi::webgpu::WebGpu>,
    name: &str,
    layout: PipelineLayout<fluxel_rhi::webgpu::WebGpu>,
    format: TextureFormat,
    wgsl: &[u8],
    cull_mode: CullMode,
    vertex_buffers: Vec<VertexBufferLayout>,
) -> Result<RasterPipeline<fluxel_rhi::webgpu::WebGpu>> {
    let shader =
        common::native_shader_with_wgsl(d, name, "vs_main", ShaderStage::Vertex, &[], &[], wgsl)
            .await?;
    let fragment = common::native_shader_with_wgsl(
        d,
        &format!("{name} fragment"),
        "ps_main",
        ShaderStage::Fragment,
        &[],
        &[],
        wgsl,
    )
    .await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some(name.into()),
        layout,
        vertex: PipelineStage {
            module: shader,
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
            cull_mode,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: name != "13 webgpu stars",
            depth_compare: CompareFunction::LessEqual,
            stencil_front: disabled_stencil_webgpu(),
            stencil_back: disabled_stencil_webgpu(),
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.,
            depth_bias_clamp: 0.,
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
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_multithreaded_recording(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};
    async fn run(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
        let d = WebGpuInstance::new()?.open_default().await?;
        let lane = d
            .lanes()
            .iter()
            .find(|x| x.class == LaneClass::Graphics)
            .map(|x| x.id)
            .ok_or(Error::Unsupported("WebGPU graphics lane"))?;
        let extent = [canvas.width(), canvas.height()];
        let mut surface = WebGpu::surface_from_canvas(&d, canvas)?;
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
        let mut scene = WebGpuSequential::new(&d, lane, format).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let work = scene.record(&d, &image, extent)?;
            common::submit_and_present_async(&d, lane, work, image).await?;
        }
        Ok(())
    }
    run(canvas, frames)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("{e:?}")))
}
