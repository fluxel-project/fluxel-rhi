//! The C++ multi-threaded UFO scene, recorded as inherited DX12 bundles.
//! The primary uploads shared input, while eight workers create 512 indexed draws.
mod common;
#[path = "common/gltf.rs"]
mod gltf;
use fluxel_rhi::api::*;
use std::sync::Arc;

const UFO: &[u8] = include_bytes!("assets/models/retroufo_red_lowpoly.gltf");
const VS: &[u8] = include_bytes!("assets/13_multithreaded_recording/ufo_vs.dxil");
const PS: &[u8] = include_bytes!("assets/13_multithreaded_recording/ufo_ps.dxil");
const STAR_VS: &[u8] = include_bytes!("assets/13_multithreaded_recording/stars_vs.dxil");
const STAR_PS: &[u8] = include_bytes!("assets/13_multithreaded_recording/stars_ps.dxil");
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
    vertex_len: u64,
    index_len: u64,
    index_count: u32,
    group: BindGroup<B>,
    ufo: RasterPipeline<B>,
    stars: RasterPipeline<B>,
    inheritance: SecondaryRasterDesc,
    frame: u32,
}
impl<B: SecondaryExampleBackend> Multithreaded<B> {
    pub async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        require_parallel_secondary_recording(d)?;
        let model = gltf::load_embedded_model(UFO, gltf::LoadOptions::CPP_PORT)
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
            push_constants: &[],
        })?;
        Ok(Self {
            lane,
            vertex_len: model.vertices.len() as u64,
            index_len: model.indices.len() as u64,
            index_count: model.index_count(),
            vertices,
            indices,
            transforms,
            group,
            ufo: pipeline(
                d,
                "13 UFO",
                ufo_layout,
                format,
                VS,
                PS,
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
            stars: pipeline(d, "13 stars", star_layout, format, STAR_VS, STAR_PS, vec![]).await?,
            inheritance: SecondaryRasterDesc {
                color_formats: vec![Some(format)],
                depth_stencil_format: None,
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
        let model = gltf::load_embedded_model(UFO, gltf::LoadOptions::CPP_PORT)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let domain = d
            .secondary()
            .ok_or(Error::Unsupported("secondary raster commands"))?;
        let mut packets = Vec::with_capacity((UFO_COUNT + 1) as usize);
        let mut stars = domain.begin(&self.inheritance)?;
        stars.set_pipeline(&self.stars)?;
        area(&mut stars, extent)?;
        stars.draw(0..3, 0..1)?;
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
        e.write_buffer(&self.vertices, 0, &model.vertices)?;
        e.write_buffer(&self.indices, 0, &model.indices)?;
        e.write_buffer(&self.transforms, 0, &transforms(extent, self.frame))?;
        self.frame = self.frame.wrapping_add(1);
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let mut pass = e.begin_raster(&RasterPassDesc {
            label: Some("13 primary UFO bundle executor"),
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
            secondary_inheritance: Some(self.inheritance.clone()),
        })?;
        for packet in &packets {
            domain.execute(&mut pass, packet)?;
        }
        pass.finish()?;
        e.finish()
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
    vs: &[u8],
    ps: &[u8],
    vertex_buffers: Vec<VertexBufferLayout>,
) -> Result<RasterPipeline<B>> {
    let shader = |bytes: &[u8], entry: &str, stage: ShaderStage| ShaderDesc {
        label: Some(format!("{name} {entry}")),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: entry.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    };
    let vs = d.shader(shader(vs, "vs_main", ShaderStage::Vertex)).await?;
    let ps = d
        .shader(shader(ps, "ps_main", ShaderStage::Fragment))
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
        let (x, y, z, angle, size) = if i >= UFO_COUNT as usize - 16 {
            foreground_instance(i - (UFO_COUNT as usize - 16), time)
        } else {
            let n = i as f32 + 0.5;
            // Far instances retain the C++ random sphere's broad horizontal
            // band, but avoid the unsupported inherited depth path's worst
            // near-camera overlap case.
            let z = -14.0 + hash01(n * 0.754_877_7) * 49.0;
            let theta = n * 2.399_963;
            let spread = 20.0 + (z + 14.0) * 0.48;
            (
                spread * theta.cos(),
                2.5 * (time + n * 0.17).sin(),
                z,
                time * 0.9 + n * 0.11,
                0.75 + (i * 37 % 100) as f32 * 0.005,
            )
        };
        let m = multiply(
            translate(x, y, z),
            multiply(
                rotate_y(angle),
                multiply(
                    // `CPP_PORT` preserves Vulkan's FlipY source convention.
                    // DX presentation needs this local correction so the UFO
                    // dome remains above its saucer.
                    rotate_x(core::f32::consts::PI - (time + i as f32).sin() * 0.25),
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
fn foreground_instance(index: usize, time: f32) -> (f32, f32, f32, f32, f32) {
    // These stand in for the random sample's visible near hemisphere. They
    // are last in bundle order because the current DX12 secondary signature
    // deliberately has no depth attachment; distant bundles could otherwise
    // overwrite a near UFO.
    const POSITIONS: [(f32, f32, f32); 16] = [
        // Projected centers: left-lower, lower-middle, right-upper.
        (-4.0, -4.1, -22.),
        (1.7, -3.0, -24.),
        (7.8, 3.7, -22.),
        (-7., 2.2, -20.),
        (4., -2.1, -20.),
        (-2., 2.3, -19.),
        (8., -0.8, -18.),
        (-8., -0.8, -18.),
        (1.5, 3.0, -17.),
        (-3.5, 2.5, -17.),
        (6., 1.2, -16.),
        (-6., -2.0, -16.),
        (0., -1., -16.),
        (9., 2., -15.),
        (-9., 2., -15.),
        (3., -3., -15.),
    ];
    let (x, y, z) = POSITIONS[index];
    (
        x,
        y + (time + index as f32).sin() * 0.4,
        z,
        time + index as f32 * 0.61,
        [
            2.6, 2.45, 3.0, 1.15, 1.1, 1.05, 1.0, 1.0, 0.95, 0.95, 0.9, 0.9, 0.85, 0.85, 0.85, 0.8,
        ][index],
    )
}
fn instance_tint(index: u32) -> [f32; 4] {
    // A stable hue sequence retains the C++ sample's per-UFO colour variation
    // without coupling scene appearance to thread scheduling.
    let hue = hash01(index as f32 * 0.618_034);
    let sector = hue * 6.0;
    let x = 1.0 - (sector % 2.0 - 1.0).abs();
    let rgb = match sector as u32 {
        0 => [0.90, 0.36 + x * 0.45, 0.36],
        1 => [0.36 + x * 0.45, 0.88, 0.36],
        2 => [0.36, 0.82, 0.42 + x * 0.40],
        3 => [0.36, 0.42 + x * 0.40, 0.90],
        4 => [0.42 + x * 0.40, 0.36, 0.90],
        _ => [0.90, 0.36, 0.42 + x * 0.40],
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
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
fn main() {
    use fluxel_rhi::dx12::Dx12;
    let cfg = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut scene: Option<Multithreaded<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
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
    );
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
