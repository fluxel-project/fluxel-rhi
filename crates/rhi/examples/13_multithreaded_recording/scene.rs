//! The C++ multi-threaded UFO scene, recorded as inherited native secondary work.
//! The primary uploads shared input, while eight workers create 512 indexed draws.
//! The workload does not schedule workers outside the secondary-recording
//! contract, nor does it own the host surface or resource retirement.
#[path = "../utils/gltf.rs"]
mod gltf;
#[path = "math.rs"]
mod math;
#[path = "../utils/mod.rs"]
mod utils;
use fluxel_rhi::api::*;
use math::*;

const UFO: &[u8] = include_bytes!("../assets/models/retroufo_red_lowpoly.gltf");
const STAR_SPHERE: &[u8] = include_bytes!("../assets/models/sphere.gltf");
const VS: &[u8] = include_bytes!("../assets/13_multithreaded_recording/ufo_vs.dxil");
const PS: &[u8] = include_bytes!("../assets/13_multithreaded_recording/ufo_ps.dxil");
const STAR_VS: &[u8] = include_bytes!("../assets/13_multithreaded_recording/stars_vs.dxil");
const STAR_PS: &[u8] = include_bytes!("../assets/13_multithreaded_recording/stars_ps.dxil");
const VS_SPIRV: &[u8] = include_bytes!("../assets/13_multithreaded_recording/ufo_vs.spv");
const PS_SPIRV: &[u8] = include_bytes!("../assets/13_multithreaded_recording/ufo_ps.spv");
const STAR_VS_SPIRV: &[u8] = include_bytes!("../assets/13_multithreaded_recording/stars_vs.spv");
const STAR_PS_SPIRV: &[u8] = include_bytes!("../assets/13_multithreaded_recording/stars_ps.spv");
const UFO_WGSL: &[u8] = include_bytes!("../assets/13_multithreaded_recording/ufo.wgsl");
const STARS_WGSL: &[u8] = include_bytes!("../assets/13_multithreaded_recording/stars.wgsl");
const UFO_COUNT: u32 = 512;
const WORKERS: u32 = 8;
const STRIDE: u64 = 96;
const UNIFORM: u64 = 256;

pub trait SecondaryExampleBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + CommandBackend
    + RasterBackend
    + DeferredBackend
    + SecondaryBackend
{
}
impl<T> SecondaryExampleBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + CommandBackend
        + RasterBackend
        + DeferredBackend
        + SecondaryBackend
{
}
pub trait SequentialExampleBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + CommandBackend + RasterBackend
{
}
impl<T> SequentialExampleBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + CommandBackend + RasterBackend
{
}
pub fn require_parallel_secondary_recording<B: Backend>(device: &Device<B>) -> Result<()> {
    if device.capabilities().secondary_commands {
        Ok(())
    } else {
        Err(Error::Unsupported("secondary raster commands"))
    }
}

pub struct Multithreaded<B: SecondaryExampleBackend> {
    lane: LaneId,
    vertices: Buffer<B>,
    indices: Buffer<B>,
    transforms: Buffer<B>,
    star_vertices: Buffer<B>,
    star_indices: Buffer<B>,
    vertex_len: u64,
    index_len: u64,
    index_count: u32,
    star_vertex_len: u64,
    star_index_len: u64,
    star_index_count: u32,
    source_vertices: Vec<u8>,
    source_indices: Vec<u8>,
    source_star_vertices: Vec<u8>,
    source_star_indices: Vec<u8>,
    uploaded: bool,
    group: BindGroup<B>,
    ufo: RasterPipeline<B>,
    stars: RasterPipeline<B>,
    inheritance: SecondaryRasterDesc,
    frame: u32,
    // The host waits for each frame before recording the next one.
    depth: Option<([u32; 2], Texture<B>, TextureView<B>)>,
    packets: Vec<SecondaryRasterWork<B>>,
}
impl<B: SecondaryExampleBackend> Multithreaded<B> {
    pub async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        require_parallel_secondary_recording(d)?;
        // The C++ sample flips model Y before its positive Vulkan viewport.
        // Fluxel has WebGPU-style Y-up NDC, realized by the Vulkan backend's
        // negative viewport, so applying that model flip again inverts the
        // visible UFOs.  The star shader reflects procedural UVW separately.
        let source = gltf::LoadOptions::CPP_PORT.with_flip_y(false);
        let model =
            gltf::load_embedded_model(UFO, source).map_err(|x| Error::Backend(x.to_string()))?;
        let stars = gltf::load_embedded_model(STAR_SPHERE, source)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let vertices = buffer(
            d,
            "13 UFO vertices",
            model.vertices.len() as u64,
            BufferUsage::VERTEX,
        )?;
        let indices = buffer(
            d,
            "13 UFO indices",
            model.indices.len() as u64,
            BufferUsage::INDEX,
        )?;
        let transforms = buffer(
            d,
            "13 UFO transforms",
            u64::from(UFO_COUNT) * UNIFORM,
            BufferUsage::UNIFORM,
        )?;
        let star_vertices = buffer(
            d,
            "13 star sphere vertices",
            stars.vertices.len() as u64,
            BufferUsage::VERTEX,
        )?;
        let star_indices = buffer(
            d,
            "13 star sphere indices",
            stars.indices.len() as u64,
            BufferUsage::INDEX,
        )?;
        let bindings = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("13 transforms".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: true,
                    min_size: 64,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("13 UFO transforms"),
            layout: &bindings,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &transforms,
                    range: 0..UNIFORM,
                },
            }],
        })?;
        let ufo_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("13 UFO layout"),
            groups: &[&bindings],
            push_constants: &[],
        })?;
        let star_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("13 stars layout"),
            groups: &[],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Vertex],
                range: 0..64,
            }],
        })?;
        Ok(Self {
            lane,
            vertex_len: model.vertices.len() as u64,
            index_len: model.indices.len() as u64,
            index_count: model.index_count(),
            star_vertex_len: stars.vertices.len() as u64,
            star_index_len: stars.indices.len() as u64,
            star_index_count: stars.index_count(),
            source_vertices: model.vertices,
            source_indices: model.indices,
            source_star_vertices: stars.vertices,
            source_star_indices: stars.indices,
            uploaded: false,
            vertices,
            indices,
            transforms,
            star_vertices,
            star_indices,
            group,
            ufo: pipeline(
                d,
                "13 UFO",
                ufo_layout,
                format,
                VS,
                VS_SPIRV,
                UFO_WGSL,
                PS,
                PS_SPIRV,
                UFO_WGSL,
                // This imported mesh omits the C++ loader's FlipY because the
                // RHI uses Y-up NDC.  With Fluxel's projection and Vulkan's
                // negative viewport that reverses its source-facing winding;
                // use the equivalent front cull for this ported asset.
                CullMode::Front,
                vec![VertexBufferLayout {
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
                }],
            )
            .await?,
            stars: pipeline(
                d,
                "13 stars",
                star_layout,
                format,
                STAR_VS,
                STAR_VS_SPIRV,
                STARS_WGSL,
                STAR_PS,
                STAR_PS_SPIRV,
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
            .await?,
            inheritance: SecondaryRasterDesc {
                color_formats: vec![Some(format)],
                depth_stencil_format: Some(TextureFormat::Depth32Float),
                samples: 1,
                multiview_mask: None,
            },
            frame: 0,
            depth: None,
            packets: Vec::new(),
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<(RecordedWork<B>, bool)>
    where
        B::NativeDevice: Send + Sync,
        B::NativeRasterPipeline: Send + Sync,
        B::NativeBuffer: Send + Sync,
        B::NativeBindGroup: Send + Sync,
        B::NativeSecondaryWork: Send,
    {
        let domain = d
            .secondary()
            .ok_or(Error::Unsupported("secondary raster commands"))?;
        let mut packets = Vec::with_capacity((UFO_COUNT + 1) as usize);
        let mut stars = domain.begin(&self.inheritance)?;
        stars.set_pipeline(&self.stars)?;
        area(&mut stars, extent)?;
        stars.push_constants(StageMask::VERTEX, 0, &matrix_bytes(star_matrix(extent)))?;
        stars.set_vertex_buffer(0, &self.star_vertices, 0..self.star_vertex_len)?;
        stars.set_index_buffer(
            &self.star_indices,
            0..self.star_index_len,
            IndexFormat::Uint32,
        )?;
        stars.draw_indexed(0..self.star_index_count, 0, 0..1)?;
        packets.push(domain.finish(stars)?);
        let count = UFO_COUNT / WORKERS;
        std::thread::scope(|scope| -> Result<()> {
            let mut jobs = Vec::with_capacity(WORKERS as usize);
            for worker in 0..WORKERS {
                let d = d.clone();
                let pipeline = self.ufo.clone();
                let group = self.group.clone();
                let vertices = self.vertices.clone();
                let indices = self.indices.clone();
                let inheritance = self.inheritance.clone();
                let vertex_len = self.vertex_len;
                let index_len = self.index_len;
                let index_count = self.index_count;
                jobs.push(scope.spawn(move || -> Result<Vec<SecondaryRasterWork<B>>> {
                    let domain = d
                        .secondary()
                        .ok_or(Error::Unsupported("secondary raster commands"))?;
                    let first = worker * count;
                    let last = if worker + 1 == WORKERS {
                        UFO_COUNT
                    } else {
                        first + count
                    };
                    let mut work = Vec::with_capacity((last - first) as usize);
                    for object in first..last {
                        let mut e = domain.begin(&inheritance)?;
                        e.set_pipeline(&pipeline)?;
                        e.set_bind_group(0, &group, &[object * UNIFORM as u32])?;
                        area(&mut e, extent)?;
                        e.set_vertex_buffer(0, &vertices, 0..vertex_len)?;
                        e.set_index_buffer(&indices, 0..index_len, IndexFormat::Uint32)?;
                        e.draw_indexed(0..index_count, 0, 0..1)?;
                        work.push(domain.finish(e)?);
                    }
                    Ok(work)
                }));
            }
            for job in jobs {
                packets.extend(job.join().map_err(|_| {
                    Error::Backend("13 secondary recording worker panicked".into())
                })??);
            }
            Ok(())
        })?;
        let mut e = d.encoder(self.lane)?;
        let upload_geometry = !self.uploaded;
        if upload_geometry {
            e.write_buffer(&self.vertices, 0, &self.source_vertices)?;
            e.write_buffer(&self.indices, 0, &self.source_indices)?;
            e.write_buffer(&self.star_vertices, 0, &self.source_star_vertices)?;
            e.write_buffer(&self.star_indices, 0, &self.source_star_indices)?;
        }
        e.write_buffer(&self.transforms, 0, &transforms(extent, self.frame))?;
        self.frame = self.frame.wrapping_add(1);
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        if self
            .depth
            .as_ref()
            .is_none_or(|(size, _, _)| *size != extent)
        {
            let (texture, view) = depth(d, extent)?;
            self.depth = Some((extent, texture, view));
        }
        let depth_view = &self.depth.as_ref().ok_or(Error::InvalidState)?.2;
        let mut pass = e.begin_raster(&RasterPassDesc {
            label: Some("13 primary UFO bundle executor"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(depth_view),
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
            secondary_inheritance: Some(self.inheritance.clone()),
        })?;
        for packet in &packets {
            domain.execute(&mut pass, packet)?;
        }
        pass.finish()?;
        let work = e.finish()?;
        // Retain secondary work and its attachment through the host completion wait.
        self.packets = packets;
        Ok((work, upload_geometry))
    }

    pub fn mark_uploaded(&mut self) {
        self.uploaded = true;
    }
}

fn buffer<B: SecondaryExampleBackend>(
    d: &Device<B>,
    label: &str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    d.create_buffer(&BufferDesc {
        label: Some(label.into()),
        size,
        usage: usage.union(BufferUsage::COPY_DST),
        memory: MemoryClass::Device,
    })
}

#[path = "sequential.rs"]
mod sequential;
pub use sequential::Sequential;
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
pub use sequential::run_gl_multithreaded_recording;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub use sequential::run_webgl2_multithreaded_recording;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub use sequential::run_webgpu_multithreaded_recording;

fn depth<B: ResourceBackend>(
    d: &Device<B>,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = d.create_texture(&TextureDesc {
        label: Some("13 depth".into()),
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

fn disabled_stencil() -> StencilFaceState {
    StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    }
}
fn area<B: SecondaryExampleBackend>(e: &mut SecondaryEncoder<B>, x: [u32; 2]) -> Result<()> {
    e.viewport(Viewport {
        x: 0.,
        y: 0.,
        width: x[0] as f32,
        height: x[1] as f32,
        min_depth: 0.,
        max_depth: 1.,
    })?;
    e.scissor(Rect {
        x: 0,
        y: 0,
        width: x[0],
        height: x[1],
    })
}
async fn pipeline<B: SecondaryExampleBackend>(
    d: &Device<B>,
    name: &str,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    vs_dxil: &[u8],
    vs_spirv: &[u8],
    vs_wgsl: &[u8],
    ps_dxil: &[u8],
    ps_spirv: &[u8],
    ps_wgsl: &[u8],
    cull_mode: CullMode,
    vertex_buffers: Vec<VertexBufferLayout>,
) -> Result<RasterPipeline<B>> {
    let vs = utils::native_shader_with_wgsl(
        d,
        &format!("{name} vs"),
        "vs_main",
        ShaderStage::Vertex,
        vs_dxil,
        vs_spirv,
        vs_wgsl,
    )
    .await
    .map_err(|error| {
        eprintln!("{name} vertex shader creation: {error:?}");
        error
    })?;
    let ps = utils::native_shader_with_wgsl(
        d,
        &format!("{name} ps"),
        "ps_main",
        ShaderStage::Fragment,
        ps_dxil,
        ps_spirv,
        ps_wgsl,
    )
    .await
    .map_err(|error| {
        eprintln!("{name} fragment shader creation: {error:?}");
        error
    })?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some(name.into()),
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
            cull_mode,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: name != "13 stars",
            depth_compare: CompareFunction::LessEqual,
            stencil_front: disabled_stencil(),
            stencil_back: disabled_stencil(),
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
    .map_err(|error| {
        eprintln!("{name} raster pipeline creation: {error:?}");
        error
    })
}

#[path = "host.rs"]
mod host;

/// Selects and runs the native backend host.
pub fn run() {
    host::run();
}
