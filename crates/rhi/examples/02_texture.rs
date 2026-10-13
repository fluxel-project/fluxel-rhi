//! Textured indexed quad using the statically typed RHI.
//!
//! `TexturePixels` carries decoded KTX texels; the workload records a real
//! texture upload and textured indexed draw.  The native host supplies the
//! acquired surface and submits the returned work.
mod common;
use fluxel_rhi::api::*;
const VS: &[u8] = include_bytes!("assets/02_texture/texture.vs.dxil");
const PS: &[u8] = include_bytes!("assets/02_texture/texture.ps.dxil");
const VS_SPV: &[u8] = include_bytes!("assets/02_texture/texture.vs.spv");
const PS_SPV: &[u8] = include_bytes!("assets/02_texture/texture.ps.spv");
const WGSL: &[u8] = include_bytes!("assets/02_texture/texture.wgsl");
pub trait TextureBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> TextureBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
#[derive(Clone, Copy)]
pub struct TexturePixels<'a> {
    pub rgba: &'a [u8],
    pub width: u32,
    pub height: u32,
}
pub struct TextureExample<B: TextureBackend> {
    vertex: Buffer<B>,
    index: Buffer<B>,
    uniform: Buffer<B>,
    texture: Texture<B>,
    group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    lane: LaneId,
}
impl<B: TextureBackend> TextureExample<B> {
    pub async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        p: TexturePixels<'_>,
    ) -> Result<Self> {
        if p.width == 0 || p.height == 0 || p.rgba.len() != p.width as usize * p.height as usize * 4
        {
            return Err(Error::InvalidRange);
        }
        let td = TextureDesc {
            label: Some("02 KTX texture".into()),
            dimension: TextureDimension::D2,
            extent: [p.width, p.height, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING.union(TextureUsage::COPY_DST),
            view_formats: vec![],
        };
        match d.texture_support(&td) {
            Support::Supported(s) if s.filterable => (),
            Support::Supported(_) => return Err(Error::Unsupported("filterable rgba8 texture")),
            Support::Unsupported(x) => return Err(Error::Unsupported(x)),
        };
        let vertex = d.create_buffer(&buf(
            "02 vertices",
            128,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = d.create_buffer(&buf(
            "02 indices",
            12,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let uniform = d.create_buffer(&buf(
            "02 transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let texture = d.create_texture(&td)?;
        let view = d.create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let sampler = d.create_sampler(&SamplerDesc {
            label: Some("02 sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let l = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("02 bindings".into()),
            entries: vec![
                entry(
                    0,
                    vec![ShaderStage::Vertex],
                    BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 256,
                    },
                ),
                entry(
                    1,
                    vec![ShaderStage::Fragment],
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    vec![ShaderStage::Fragment],
                    BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                ),
            ],
        })?;
        let es = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &uniform,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("02 group"),
            layout: &l,
            entries: &es,
        })?;
        let pl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("02 layout"),
            groups: &[&l],
            push_constants: &[],
        })?;
        let pipeline = pipeline(
            d,
            pl,
            format,
            vec![VertexBufferLayout {
                stride: 32,
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
                    VertexAttribute {
                        location: 2,
                        offset: 20,
                        format: VertexFormat::Float32(3),
                    },
                ],
            }],
            VS,
            PS,
        )
        .await?;
        Ok(Self {
            vertex,
            index,
            uniform,
            texture,
            group,
            pipeline,
            lane,
        })
    }
    pub fn record(
        &self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        p: TexturePixels<'_>,
    ) -> Result<RecordedWork<B>> {
        let mut e = d.encoder(self.lane)?;
        e.write_buffer(&self.vertex, 0, &quad())?;
        e.write_buffer(&self.index, 0, &[0, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 0])?;
        e.write_buffer(&self.uniform, 0, &texture_transform(extent))?;
        e.write_texture(
            TextureRegion {
                texture: &self.texture,
                mip: 0,
                origin: [0; 3],
                extent: [p.width, p.height, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: p.width * 4,
                rows_per_image: p.height,
            },
            p.rgba,
        )?;
        e.transition_resources(&[texture_transition(
            &self.texture,
            ImageLayout::Common,
            ImageLayout::ShaderReadOnly,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
        )])?;
        let c = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.02, 0.02, 0.02, 1.0])),
            store: StoreOp::Store,
        })];
        let mut r = e.begin_raster(&RasterPassDesc {
            label: Some("02 texture"),
            colors: &c,
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
        })?;
        r.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        r.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        r.set_pipeline(&self.pipeline)?;
        r.set_bind_group(0, &self.group, &[])?;
        r.set_vertex_buffer(0, &self.vertex, 0..128)?;
        r.set_index_buffer(&self.index, 0..12, IndexFormat::Uint16)?;
        r.draw_indexed(0..6, 0, 0..1)?;
        r.finish()?;
        e.transition_resources(&[texture_transition(
            &self.texture,
            ImageLayout::ShaderReadOnly,
            ImageLayout::Common,
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
            TextureUses {
                stages: StageMask::ALL_COMMANDS,
                accesses: AccessMask::EMPTY,
            },
        )])?;
        e.finish()
    }
}
fn texture_transition<B: TextureBackend>(
    texture: &Texture<B>,
    from: ImageLayout,
    to: ImageLayout,
    before: TextureUses,
    after: TextureUses,
) -> ResourceTransition<'_, B> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: before,
            to: after,
        },
        layout: Some((from, to)),
        ownership: None,
    })
}
fn buf(label: &str, size: u64, usage: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    }
}
fn entry(binding: u32, visibility: Vec<ShaderStage>, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
async fn pipeline<B: TextureBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    v: Vec<VertexBufferLayout>,
    vs: &[u8],
    ps: &[u8],
) -> Result<RasterPipeline<B>> {
    let a =
        common::native_shader_with_wgsl(d, "vs", "vs_main", ShaderStage::Vertex, vs, VS_SPV, WGSL)
            .await?;
    let b = common::native_shader_with_wgsl(
        d,
        "ps",
        "fs_main",
        ShaderStage::Fragment,
        ps,
        PS_SPV,
        WGSL,
    )
    .await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some("02 pipeline".into()),
        layout,
        vertex: PipelineStage {
            module: a,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: b,
            entry: "fs_main".into(),
            constants: vec![],
        }),
        vertex_buffers: v,
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
            // D3D12 sample masks are 32-bit; the API pipeline lowerer rejects
            // a wider public mask instead of truncating it.
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
    .await
}
fn quad() -> [u8; 128] {
    // Same position / UV / normal records as the C++ textured quad.
    let x: [f32; 32] = [
        1., 1., 0., 1., 1., 0., 0., 1., -1., 1., 0., 0., 1., 0., 0., 1., -1., -1., 0., 0., 0., 0.,
        0., 1., 1., -1., 0., 1., 0., 0., 0., 1.,
    ];
    let mut b = [0; 128];
    for (i, v) in x.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
fn texture_transform(extent: [u32; 2]) -> [u8; 256] {
    // Match the source's right-handed look-at camera: z=-2.5, a 15° yaw and
    // a 60° perspective. WebGPU/DX12 still use a [0, 1] depth range; only
    // the example's world/view convention is right-handed.
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / 30f32.to_radians().tan();
    let projection = [
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
        256. / -255.9,
        -1.,
        0.,
        0.,
        -25.6 / 255.9,
        0.,
    ];
    let angle = 15f32.to_radians();
    let (s, c) = angle.sin_cos();
    let rotation = [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.];
    let view = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., -2.5, 1.,
    ];
    let model = multiply(view, rotation);
    let mut bytes = [0; 256];
    write_matrix(&mut bytes, 0, projection);
    write_matrix(&mut bytes, 64, model);
    // `Camera::viewPos` from the source camera at (0, 0, -2.5).
    for (index, value) in [0.0_f32, 0.0, 2.5, 0.0, 0.0].into_iter().enumerate() {
        bytes[128 + index * 4..132 + index * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut output = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    output
}

fn write_matrix(bytes: &mut [u8], offset: usize, matrix: [f32; 16]) {
    for (index, value) in matrix.into_iter().enumerate() {
        bytes[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(windows)]
fn load_ktx() -> Result<(Vec<u8>, u32, u32)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/textures/metalplate01_rgba.ktx"
    );
    let bytes = std::fs::read(path).map_err(|error| Error::Backend(format!("{path}: {error}")))?;
    parse_ktx(&bytes)
}

#[cfg(target_arch = "wasm32")]
fn load_ktx() -> Result<(Vec<u8>, u32, u32)> {
    parse_ktx(include_bytes!("assets/textures/metalplate01_rgba.ktx"))
}

fn parse_ktx(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    const MAGIC: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";
    if bytes.len() < 68 || &bytes[..12] != MAGIC || &bytes[12..16] != b"\x01\x02\x03\x04" {
        return Err(Error::Unsupported("KTX1 little-endian RGBA8 texture"));
    }
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let (width, height) = (word(36), word(40));
    if word(24) != 0x1908 || word(28) != 0x8058 || width == 0 || height == 0 {
        return Err(Error::Unsupported("KTX1 RGBA8 texture"));
    }
    let offset = 64usize
        .checked_add(word(60) as usize)
        .ok_or(Error::InvalidRange)?;
    let size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::InvalidRange)?;
    if offset
        .checked_add(4 + size)
        .is_none_or(|end| end > bytes.len())
        || word(offset) as usize != size
    {
        return Err(Error::InvalidRange);
    }
    Ok((bytes[offset + 4..offset + 4 + size].to_vec(), width, height))
}

/// Draws the textured example into a host-owned browser canvas.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_webgpu_texture(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> std::result::Result<(), wasm_bindgen::JsValue> {
    use fluxel_rhi::webgpu::{WebGpu, WebGpuInstance};

    let run = async {
        if frames == 0 {
            return Err(Error::InvalidRange);
        }
        let (rgba, width, height) = load_ktx()?;
        let pixels = TexturePixels {
            rgba: &rgba,
            width,
            height,
        };
        let device = WebGpuInstance::new()?.open_default().await?;
        let lane = device.lanes()[0].id;
        let extent = [canvas.width(), canvas.height()];
        let mut surface = WebGpu::surface_from_canvas(&device, canvas)?;
        let format = surface.capabilities()?.formats[0];
        let config = SurfaceConfig {
            extent,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 2,
        };
        surface.configure(&config).await?;
        let example = TextureExample::<WebGpu>::new(&device, lane, format, pixels).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let work = example.record(&device, &image, extent, pixels)?;
            common::submit_and_present_async(&device, lane, work, image).await?;
        }
        Ok(())
    };
    run.await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&format!("{error:?}")))
}

#[cfg(windows)]
fn run<B: TextureBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let (rgba, width, height) = match load_ktx() {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut example: Option<TextureExample<B>> = None;
    common::run_example(
        host,
        "Textured quad",
        config.clone(),
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            let pixels = TexturePixels {
                rgba: &rgba,
                width,
                height,
            };
            if example.is_none() {
                example = Some(common::block_on(TextureExample::new(
                    device,
                    lane,
                    config.format,
                    pixels,
                ))?);
            }
            let work = example
                .as_ref()
                .ok_or(Error::InvalidState)?
                .record(device, &image, extent, pixels)?;
            common::submit_and_present(device, lane, work, image)
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
    if let Err(error) = result {
        eprintln!("02_texture: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("02_texture requires Windows and DX12");
}
