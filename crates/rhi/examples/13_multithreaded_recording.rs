//! The C++ multi-threaded UFO scene, recorded as inherited native secondary work.
//! The primary uploads shared input, while eight workers create 512 indexed draws.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
use fluxel_rhi::api::*;

const UFO: &[u8] = include_bytes!("assets/models/retroufo_red_lowpoly.gltf");
const STAR_SPHERE: &[u8] = include_bytes!("assets/models/sphere.gltf");
const VS: &[u8] = include_bytes!("assets/13_multithreaded_recording/ufo_vs.dxil");
const PS: &[u8] = include_bytes!("assets/13_multithreaded_recording/ufo_ps.dxil");
const STAR_VS: &[u8] = include_bytes!("assets/13_multithreaded_recording/stars_vs.dxil");
const STAR_PS: &[u8] = include_bytes!("assets/13_multithreaded_recording/stars_ps.dxil");
const VS_SPIRV: &[u8] = include_bytes!("assets/13_multithreaded_recording/ufo_vs.spv");
const PS_SPIRV: &[u8] = include_bytes!("assets/13_multithreaded_recording/ufo_ps.spv");
const STAR_VS_SPIRV: &[u8] = include_bytes!("assets/13_multithreaded_recording/stars_vs.spv");
const STAR_PS_SPIRV: &[u8] = include_bytes!("assets/13_multithreaded_recording/stars_ps.spv");
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
                PS,
                PS_SPIRV,
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
                STAR_PS,
                STAR_PS_SPIRV,
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
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<RecordedWork<B>>
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
        let (depth, depth_view) = depth(d, extent)?;
        let mut pass = e.begin_raster(&RasterPassDesc {
            label: Some("13 primary UFO bundle executor"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&depth_view),
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
            secondary_inheritance: Some(self.inheritance.clone()),
        })?;
        for packet in &packets {
            domain.execute(&mut pass, packet)?;
        }
        pass.finish()?;
        // The view retains the texture until native recording has retained the
        // attachment.  Keep the local binding alive through `finish` as well.
        let _ = depth;
        let work = e.finish()?;
        // Do not mark the immutable geometry resident until its upload encoder
        // has successfully finished.  A recoverable recording error must
        // cause the next attempt to upload it again.
        if upload_geometry {
            self.uploaded = true;
        }
        Ok(work)
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
fn depth<B: SecondaryExampleBackend>(
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
    ps_dxil: &[u8],
    ps_spirv: &[u8],
    cull_mode: CullMode,
    vertex_buffers: Vec<VertexBufferLayout>,
) -> Result<RasterPipeline<B>> {
    let vs = common::native_shader(
        d,
        &format!("{name} vs"),
        "vs_main",
        ShaderStage::Vertex,
        vs_dxil,
        vs_spirv,
    )
    .await?;
    let ps = common::native_shader(
        d,
        &format!("{name} ps"),
        "ps_main",
        ShaderStage::Fragment,
        ps_dxil,
        ps_spirv,
    )
    .await?;
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
}
fn transforms(extent: [u32; 2], frame: u32) -> Vec<u8> {
    let mut out = vec![0; UFO_COUNT as usize * UNIFORM as usize];
    let p = perspective(
        60f32.to_radians(),
        extent[0].max(1) as f32 / extent[1].max(1) as f32,
        0.1,
        256.,
    );
    let view = translate(0., 0., 32.5);
    let time = frame as f32 * 0.012;
    for i in 0..UFO_COUNT as usize {
        // Same spherical placement and per-object animation distribution as
        // `multithreading.cpp`, with a stable seed so worker scheduling never
        // changes the picture. Native clipping rejects the far hemisphere;
        // depth testing orders all remaining packets independently of threads.
        let n = i as f32 + 0.5;
        let theta = core::f32::consts::TAU * hash01(n * 12.9898);
        let phi = (1.0 - 2.0 * hash01(n * 78.233)).acos();
        let x = phi.sin() * theta.cos() * 35.0;
        let z = phi.cos() * 35.0;
        let cycle = hash01(n * 37.719) + time * 0.15;
        let direction = if hash01(n * 19.19) < 0.5 { -1.0 } else { 1.0 };
        let rotation_speed_degrees = (2.0 + 4.0 * hash01(n * 41.31)) * direction;
        let y = (core::f32::consts::TAU * cycle).sin() * 2.5;
        let yaw = core::f32::consts::TAU * hash01(n * 91.17)
            + (time * 2.5 * rotation_speed_degrees).to_radians();
        let spin = core::f32::consts::TAU * cycle * direction;
        let size = 0.75 + hash01(n * 53.11) * 0.5;
        let m = multiply(
            translate(x, y, z),
            multiply(
                rotate_x(-(core::f32::consts::TAU * cycle).sin() * 0.25 * direction),
                multiply(
                    // This is Rx(tilt) × Ry(yaw) × Ry(deltaT spin), matching
                    // the native sample's GLM call order.
                    rotate_y(yaw + spin),
                    scale(size),
                ),
            ),
        );
        let record = &mut out[i * UNIFORM as usize..];
        matrix(record, multiply(p, multiply(view, m)));
        let tint = instance_tint(i as u32);
        for (component, value) in tint.into_iter().enumerate() {
            record[64 + component * 4..68 + component * 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}
fn star_matrix(extent: [u32; 2]) -> [f32; 16] {
    let projection = perspective(
        60f32.to_radians(),
        extent[0].max(1) as f32 / extent[1].max(1) as f32,
        0.1,
        256.0,
    );
    // C++ uses projection × view with the camera translation cleared, then
    // scales its unit sphere by two. This camera has no rotation.
    multiply(projection, scale(2.0))
}
fn matrix_bytes(value: [f32; 16]) -> [u8; 64] {
    let mut bytes = [0; 64];
    matrix(&mut bytes, value);
    bytes
}
fn instance_tint(index: u32) -> [f32; 4] {
    // C++ samples three independent uniform components for every object.
    // Deriving all three from stable salts preserves that distribution without
    // coupling colours to worker execution order.
    let n = index as f32 + 0.5;
    [
        hash01(n * 17.131),
        hash01(n * 43.723),
        hash01(n * 97.409),
        1.0,
    ]
}
fn hash01(value: f32) -> f32 {
    (value.sin() * 43_758.547).fract().abs()
}
fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1. / (fov * 0.5).tan();
    [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        far / (far - near),
        1.,
        0.,
        0.,
        -near * far / (far - near),
        0.,
    ]
}
fn translate(x: f32, y: f32, z: f32) -> [f32; 16] {
    [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., x, y, z, 1.]
}
fn scale(s: f32) -> [f32; 16] {
    [s, 0., 0., 0., 0., s, 0., 0., 0., 0., s, 0., 0., 0., 0., 1.]
}
fn rotate_y(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.]
}
fn rotate_x(a: f32) -> [f32; 16] {
    let (s, c) = a.sin_cos();
    [1., 0., 0., 0., 0., c, s, 0., 0., -s, c, 0., 0., 0., 0., 1.]
}
fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut r = [0.; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[c * 4 + k]).sum();
        }
    }
    r
}
fn matrix(bytes: &mut [u8], m: [f32; 16]) {
    for (i, v) in m.into_iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}

#[cfg(windows)]
fn run<B: SecondaryExampleBackend, H: common::ExampleHost<B>>(host: H) -> Result<()>
where
    B::NativeDevice: Send + Sync,
    B::NativeRasterPipeline: Send + Sync,
    B::NativeBuffer: Send + Sync,
    B::NativeBindGroup: Send + Sync,
    B::NativeSecondaryWork: Send,
{
    let cfg = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut scene: Option<Multithreaded<B>> = None;
    common::run_example(
        host,
        "Multi threaded UFO command buffers",
        cfg.clone(),
        move |d, _, image, extent| {
            let lane = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene.is_none() {
                scene = Some(common::block_on(Multithreaded::new(d, lane, cfg.format))?)
            }
            let work = scene
                .as_mut()
                .ok_or(Error::InvalidState)?
                .record(d, &image, extent)?;
            common::submit_and_present(d, lane, work, image)
        },
    )
}

#[cfg(windows)]
fn main() {
    let result = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(e) = result {
        eprintln!("13_multithreaded_recording: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!(
        "13_multithreaded_recording requires Windows and a backend with secondary raster commands"
    )
}
