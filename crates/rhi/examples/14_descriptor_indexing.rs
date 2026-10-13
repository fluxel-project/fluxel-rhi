//! Runtime-sized, non-uniformly indexed 32-texture cube scene using RHI.
//!
//! This module first proves the required descriptor capabilities, then owns
//! scene resources and recording. The host remains responsible for surface
//! lifetime, submission policy, and completion-driven retirement.
mod common;
use fluxel_rhi::api::*;
const TEXTURES: usize = 32;
const CUBES: usize = 5;
const VERTEX_BYTES: u64 = 24;
const VERTICES_PER_CUBE: u32 = 24;
const INDICES_PER_CUBE: u32 = 36;
const VS: &[u8] = include_bytes!("assets/14_descriptor_indexing/descriptor_indexing.vs.dxil");
const PS: &[u8] = include_bytes!("assets/14_descriptor_indexing/descriptor_indexing.ps.dxil");
const VS_SPIRV: &[u8] = include_bytes!("assets/14_descriptor_indexing/descriptor_indexing.vs.spv");
const PS_SPIRV: &[u8] = include_bytes!("assets/14_descriptor_indexing/descriptor_indexing.ps.spv");
const WGSL: &[u8] = include_bytes!("assets/14_descriptor_indexing/descriptor_indexing.wgsl");
pub trait DescriptorIndexingBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + DeferredBackend
{
}
impl<T> DescriptorIndexingBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + DeferredBackend
{
}
pub fn require_descriptor_indexing<B: Backend>(d: &Device<B>) -> Result<()> {
    let c = d.capabilities();
    if !c.binding_arrays {
        return Err(Error::Unsupported("binding arrays"));
    }
    if !c.runtime_sized_binding_arrays {
        return Err(Error::Unsupported("runtime-sized binding arrays"));
    }
    if !c.nonuniform_sampled_texture_and_storage_buffer_indexing {
        return Err(Error::Unsupported("non-uniform sampled-texture indexing"));
    }
    Ok(())
}
pub fn descriptor_indexing_layout<B: Backend + ResourceBackend>(
    d: &Device<B>,
) -> Result<BindGroupLayout<B>> {
    require_descriptor_indexing(d)?;
    d.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("14 runtime texture array".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 192,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::RuntimeSized,
                partially_bound: false,
                nonuniform_indexing: true,
            },
        ],
    })
}
pub struct DescriptorIndexing<B: DescriptorIndexingBackend> {
    lane: LaneId,
    vertex: Buffer<B>,
    index: Buffer<B>,
    uniform: Buffer<B>,
    textures: Vec<Texture<B>>,
    depth: Texture<B>,
    depth_view: TextureView<B>,
    depth_extent: [u32; 2],
    group: BindGroup<B>,
    pipeline: RasterPipeline<B>,
    uploaded: bool,
}
impl<B: DescriptorIndexingBackend> DescriptorIndexing<B> {
    pub async fn new(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        // Standard WebGPU has fixed texture-array layers, but no runtime-sized
        // descriptor array. Keep the actual capability false and use a visual
        // equivalent 32-layer texture for the browser demo.
        #[cfg(not(target_arch = "wasm32"))]
        require_descriptor_indexing(d)?;
        let vertex = d.create_buffer(&buffer(
            "14 cube vertices",
            VERTEX_BYTES * u64::from(VERTICES_PER_CUBE) * CUBES as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let index = d.create_buffer(&buffer(
            "14 cube indices",
            u64::from(INDICES_PER_CUBE) * CUBES as u64 * 4,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let uniform = d.create_buffer(&buffer(
            "14 matrices",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        #[cfg(not(target_arch = "wasm32"))]
        let layout = descriptor_indexing_layout(d)?;
        #[cfg(target_arch = "wasm32")]
        let layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("14 WebGPU fixed texture layers".into()),
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Vertex],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 192,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Fragment],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })?;
        let sampler = d.create_sampler(&SamplerDesc {
            label: Some("14 nearest sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let (mut textures, mut views) =
            (Vec::with_capacity(TEXTURES), Vec::with_capacity(TEXTURES));
        for i in 0..if cfg!(target_arch = "wasm32") {
            1
        } else {
            TEXTURES
        } {
            let t = d.create_texture(&TextureDesc {
                label: Some(format!("14 image {i}")),
                dimension: TextureDimension::D2,
                extent: [3, 3, 1],
                array_layers: if cfg!(target_arch = "wasm32") {
                    TEXTURES as u32
                } else {
                    1
                },
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::TEXTURE_BINDING.union(TextureUsage::COPY_DST),
                view_formats: vec![],
            })?;
            views.push(d.create_view(
                &t,
                &TextureViewDesc {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: if cfg!(target_arch = "wasm32") {
                        TextureViewDimension::D2Array
                    } else {
                        TextureViewDimension::D2
                    },
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..if cfg!(target_arch = "wasm32") {
                        TEXTURES as u32
                    } else {
                        1
                    },
                },
            )?);
            textures.push(t)
        }
        let entries = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &uniform,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: if cfg!(target_arch = "wasm32") {
                    BindingResource::TextureView(&views[0])
                } else {
                    BindingResource::Array(
                        views
                            .iter()
                            .map(|v| Some(BindingResource::TextureView(v)))
                            .collect(),
                    )
                },
            },
        ];
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("14 textures"),
            layout: &layout,
            entries: &entries,
        })?;
        let pl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("14 layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let pipeline = pipeline(d, pl, format).await?;
        let (depth, depth_view) = depth_target(d, extent)?;
        Ok(Self {
            lane,
            vertex,
            index,
            uniform,
            textures,
            depth,
            depth_view,
            depth_extent: extent,
            group,
            pipeline,
            uploaded: false,
        })
    }
    pub fn record(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
        time: f32,
    ) -> Result<(RecordedWork<B>, bool)> {
        if self.depth_extent != extent {
            let (depth, depth_view) = depth_target(d, extent)?;
            self.depth = depth;
            self.depth_view = depth_view;
            self.depth_extent = extent;
        }
        let mut e = d.encoder(self.lane)?;
        let upload_geometry = !self.uploaded;
        if upload_geometry {
            e.write_buffer(&self.vertex, 0, &cube_vertices())?;
            e.write_buffer(&self.index, 0, &cube_indices())?;
            for i in 0..TEXTURES {
                let t = &self.textures[if cfg!(target_arch = "wasm32") { 0 } else { i }];
                let p = texture_pixels(i as u8);
                e.write_texture(
                    TextureRegion {
                        texture: t,
                        mip: 0,
                        origin: [
                            0,
                            0,
                            if cfg!(target_arch = "wasm32") {
                                i as u32
                            } else {
                                0
                            },
                        ],
                        extent: [3, 3, 1],
                        aspect: TextureAspects::COLOR,
                    },
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: 12,
                        rows_per_image: 3,
                    },
                    &p,
                )?
            }
        }
        e.write_buffer(&self.uniform, 0, &matrices(extent, time))?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.025, 0.025, 0.025, 1.])),
            store: StoreOp::Store,
        })];
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("14 descriptor indexed cubes"),
            colors: &colors,
            depth_stencil: Some(DepthStencilAttachment {
                view: AttachmentView::Texture(&self.depth_view),
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
        p.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        p.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        p.set_pipeline(&self.pipeline)?;
        p.set_bind_group(0, &self.group, &[])?;
        p.set_vertex_buffer(
            0,
            &self.vertex,
            0..VERTEX_BYTES * u64::from(VERTICES_PER_CUBE) * CUBES as u64,
        )?;
        p.set_index_buffer(
            &self.index,
            0..u64::from(INDICES_PER_CUBE) * CUBES as u64 * 4,
            IndexFormat::Uint32,
        )?;
        p.draw_indexed(0..INDICES_PER_CUBE * CUBES as u32, 0, 0..1)?;
        p.finish()?;
        let work = e.finish()?;
        Ok((work, upload_geometry))
    }

    fn mark_uploaded(&mut self) {
        self.uploaded = true;
    }
}
#[path = "descriptor_indexing/support.rs"]
mod support;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub use support::run_webgpu_descriptor_indexing;
use support::*;

#[cfg(windows)]
fn run<B: DescriptorIndexingBackend, H: common::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let start = std::time::Instant::now();
    let mut scene: Option<DescriptorIndexing<B>> = None;
    common::run_example(
        host,
        "Descriptor indexing",
        config.clone(),
        move |d, _, i, e| {
            let lane = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene.is_none() {
                scene = Some(common::block_on(DescriptorIndexing::new(
                    d,
                    lane,
                    config.format,
                    e,
                ))?)
            }
            let (w, uploaded_geometry) = scene.as_mut().ok_or(Error::InvalidState)?.record(
                d,
                &i,
                e,
                start.elapsed().as_secs_f32(),
            )?;
            common::submit_and_present(d, lane, w, i)?;
            if uploaded_geometry {
                scene.as_mut().ok_or(Error::InvalidState)?.mark_uploaded();
            }
            Ok(())
        },
    )
}
#[cfg(windows)]
fn main() {
    let r = match common::windows_api::selected_backend() {
        Ok(common::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(common::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(common::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(common::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(e) = r {
        eprintln!("14_descriptor_indexing: {e:?}");
        std::process::exit(1)
    }
}
#[cfg(not(windows))]
fn main() {
    eprintln!("14_descriptor_indexing requires Windows and runtime descriptor indexing")
}
