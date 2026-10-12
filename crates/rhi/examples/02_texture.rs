//! Textured indexed quad using the statically typed RHI.
//!
//! `TexturePixels` carries decoded KTX texels; the workload records a real
//! texture upload and textured indexed draw.  The native host supplies the
//! acquired surface and submits the returned work.
mod common;
use fluxel_rhi::api::*;
use std::sync::Arc;
const VS: &[u8] = include_bytes!("assets/02_texture/texture.vs.dxil");
const PS: &[u8] = include_bytes!("assets/02_texture/texture.ps.dxil");
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
            80,
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
                        min_size: 64,
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
        e.write_buffer(&self.uniform, 0, &identity())?;
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
        r.set_vertex_buffer(0, &self.vertex, 0..80)?;
        r.set_index_buffer(&self.index, 0..12, IndexFormat::Uint16)?;
        r.draw_indexed(0..6, 0, 0..1)?;
        r.finish()?;
        e.finish()
    }
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
    let a = d
        .shader(ShaderDesc {
            label: Some("vs".into()),
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
    let b = d
        .shader(ShaderDesc {
            label: Some("ps".into()),
            format: ShaderFormat::Dxil,
            bytes: Arc::from(ps),
            entries: vec![ShaderEntryPoint {
                name: "fs_main".into(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
            }],
            required_features: vec![],
        })
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
fn quad() -> [u8; 80] {
    let x: [f32; 20] = [
        -1., -1., 0., 0., 1., 1., -1., 0., 1., 1., 1., 1., 0., 1., 0., -1., 1., 0., 0., 0.,
    ];
    let mut b = [0; 80];
    for (i, v) in x.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
fn identity() -> [u8; 64] {
    let mut b = [0; 64];
    for i in 0..4 {
        b[i * 20..i * 20 + 4].copy_from_slice(&1f32.to_ne_bytes())
    }
    b
}

#[cfg(windows)]
fn load_ktx() -> Result<(Vec<u8>, u32, u32)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/textures/metalplate01_rgba.ktx"
    );
    let bytes = std::fs::read(path).map_err(|error| Error::Backend(format!("{path}: {error}")))?;
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

#[cfg(windows)]
fn main() {
    use fluxel_rhi::dx12::Dx12;
    let (rgba, width, height) = match load_ktx() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("02_texture: {error:?}");
            std::process::exit(1);
        }
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
    let mut example: Option<TextureExample<Dx12>> = None;
    let result = common::run_example(
        common::windows_api::Dx12Host,
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
    );
    if let Err(error) = result {
        eprintln!("02_texture: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("02_texture requires Windows and DX12");
}
