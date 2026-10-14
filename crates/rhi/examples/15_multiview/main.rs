//! Typed multiview rendering into a two-layer color attachment.
//! The first pass uses view mask `0b11`; the second samples the two layers into
//! left/right halves of the acquired surface.
//! Scene resources and pass recording live here; the caller owns surface lease
//! handoff, submission, and lifetime decisions after recording.
#[path = "../utils/gltf.rs"]
mod gltf;
#[path = "../utils/mod.rs"]
mod utils;
use fluxel_rhi::api::*;
use std::sync::Arc;
const SCENE_VS: &[u8] = include_bytes!("../assets/15_multiview/scene_vs.dxil");
const SCENE_PS: &[u8] = include_bytes!("../assets/15_multiview/scene_ps.dxil");
const SCENE_VS_SPIRV: &[u8] = include_bytes!("../assets/15_multiview/scene_vs.spv");
const SCENE_PS_SPIRV: &[u8] = include_bytes!("../assets/15_multiview/scene_ps.spv");
const DISPLAY_VS: &[u8] = include_bytes!("../assets/15_multiview/display_vs.dxil");
const DISPLAY_PS: &[u8] = include_bytes!("../assets/15_multiview/display_ps.dxil");
const DISPLAY_VS_SPIRV: &[u8] = include_bytes!("../assets/15_multiview/display_vs.spv");
const DISPLAY_PS_SPIRV: &[u8] = include_bytes!("../assets/15_multiview/display_ps.spv");
const ROOM: &[u8] = include_bytes!("../assets/models/sampleroom.gltf");
const FALLBACK_SCENE_WGSL: &[u8] = include_bytes!("../assets/15_multiview/scene.wgsl");
const FALLBACK_DISPLAY_WGSL: &[u8] = include_bytes!("../assets/15_multiview/display.wgsl");

pub trait StereoBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + CommandBackend + RasterBackend
{
}
impl<T> StereoBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + CommandBackend + RasterBackend
{
}

/// Backends that use the sequential stereo fallback and compile its WGSL source.
pub trait WgslStereoBackend: StereoBackend {}

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
impl WgslStereoBackend for fluxel_rhi::gl::Gl {}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
impl WgslStereoBackend for fluxel_rhi::webgpu::WebGpu {}

pub trait MultiviewBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + CommandBackend
    + RasterBackend
    + DeferredBackend
{
}
impl<T> MultiviewBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + CommandBackend
        + RasterBackend
        + DeferredBackend
{
}
pub fn require_multiview<B: Backend>(device: &Device<B>, views: u32) -> Result<()> {
    if !device.capabilities().multiview {
        return Err(Error::Unsupported("multiview"));
    }
    if views == 0 || views > device.capabilities().limits.max_multiview_views {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

pub struct Multiview<B: MultiviewBackend> {
    lane: LaneId,
    vertices: Buffer<B>,
    indices: Buffer<B>,
    vertex_bytes: Arc<[u8]>,
    index_bytes: Arc<[u8]>,
    uploaded: bool,
    vertex_len: u64,
    index_len: u64,
    index_count: u32,
    uniforms: Buffer<B>,
    scene_group: BindGroup<B>,
    scene: RasterPipeline<B>,
    display: RasterPipeline<B>,
    views: Texture<B>,
    view: TextureView<B>,
    _depth: Texture<B>,
    depth_view: TextureView<B>,
    group: BindGroup<B>,
    format: TextureFormat,
    extent: [u32; 2],
}
impl<B: MultiviewBackend> Multiview<B> {
    pub async fn new(
        device: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        require_multiview(device, 2)?;
        // Match the C++ sample's glTF preprocessing, including FlipY. The
        // backend's viewport convention is applied after vertex processing.
        let room = gltf::load_embedded_model(ROOM, gltf::LoadOptions::CPP_PORT)
            .map_err(|error| Error::Backend(error.to_string()))?;
        let vertices = device.create_buffer(&BufferDesc {
            label: Some("15 sampleroom vertices".into()),
            size: room.vertices.len() as u64,
            usage: BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let indices = device.create_buffer(&BufferDesc {
            label: Some("15 sampleroom indices".into()),
            size: room.indices.len() as u64,
            usage: BufferUsage::INDEX.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let uniforms = device.create_buffer(&BufferDesc {
            label: Some("15 two-eye transforms".into()),
            // D3D12 CBV descriptors round their range to 256 bytes.  Keep
            // the portable payload at 272 bytes but allocate its native-safe
            // 512-byte envelope.
            size: 512,
            usage: BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
            memory: MemoryClass::Device,
        })?;
        let (views, view) = layered_target(device, format, extent)?;
        let (depth, depth_view) = layered_depth(device, extent)?;
        let scene_bind_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("15 scene transform layout".into()),
            entries: vec![binding(
                0,
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 272,
                },
                ShaderStage::Vertex,
            )],
        })?;
        let scene_group = device.create_bind_group(&BindGroupDesc {
            label: Some("15 scene transform group"),
            layout: &scene_bind_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &uniforms,
                    range: 0..512,
                },
            }],
        })?;
        let scene_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 scene layout"),
            groups: &[&scene_bind_layout],
            push_constants: &[],
        })?;
        let scene = pipeline(
            device,
            scene_layout,
            format,
            SCENE_VS,
            SCENE_PS,
            SCENE_VS_SPIRV,
            SCENE_PS_SPIRV,
            true,
            true,
            vec![VertexBufferLayout {
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
            }],
        )
        .await?;
        let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("15 display bindings".into()),
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
            label: Some("15 display sampler".into()),
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Linear,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let entries = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let group = device.create_bind_group(&BindGroupDesc {
            label: Some("15 display group"),
            layout: &layout,
            entries: &entries,
        })?;
        let display_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("15 display layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let display = pipeline(
            device,
            display_layout,
            format,
            DISPLAY_VS,
            DISPLAY_PS,
            DISPLAY_VS_SPIRV,
            DISPLAY_PS_SPIRV,
            false,
            false,
            vec![],
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
            uniforms,
            scene_group,
            scene,
            display,
            views,
            view,
            _depth: depth,
            depth_view,
            group,
            format,
            extent,
        })
    }
    pub fn record(
        &mut self,
        device: &Device<B>,
        image: &SurfaceImage<B>,
    ) -> Result<(RecordedWork<B>, bool)> {
        let mut encoder = device.encoder(self.lane)?;
        let uploads_geometry = !self.uploaded;
        if uploads_geometry {
            encoder.write_buffer(&self.vertices, 0, &self.vertex_bytes)?;
            encoder.write_buffer(&self.indices, 0, &self.index_bytes)?;
        }
        encoder.write_buffer(&self.uniforms, 0, &two_eye_matrices(self.extent))?;
        // This attachment survives between frames. After the first accepted
        // submission, the display pass leaves both layers shader-readable.
        let (previous_layout, previous_uses) = if self.uploaded {
            (
                ImageLayout::ShaderReadOnly,
                TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            )
        } else {
            (
                ImageLayout::Undefined,
                TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
            )
        };
        encoder.transition_resources(&[transition(
            &self.views,
            previous_layout,
            ImageLayout::ColorAttachment,
            previous_uses,
            TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
        )])?;
        let scene_color = [Some(ColorAttachment {
            view: AttachmentView::Texture(&self.view),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0, 0.0, 0.0, 0.0])),
            store: StoreOp::Store,
        })];
        let mut scene = encoder.begin_raster(&RasterPassDesc {
            label: Some("15 layered multiview scene"),
            colors: &scene_color,
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
                width: self.extent[0],
                height: self.extent[1],
            },
            multiview_mask: Some(0b11),
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        scene.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent[0] as f32,
            height: self.extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        scene.scissor(Rect {
            x: 0,
            y: 0,
            width: self.extent[0],
            height: self.extent[1],
        })?;
        scene.set_pipeline(&self.scene)?;
        scene.set_bind_group(0, &self.scene_group, &[])?;
        scene.set_vertex_buffer(0, &self.vertices, 0..self.vertex_len)?;
        scene.set_index_buffer(&self.indices, 0..self.index_len, IndexFormat::Uint32)?;
        scene.draw_indexed(0..self.index_count, 0, 0..1)?;
        scene.finish()?;
        encoder.transition_resources(&[transition(
            &self.views,
            ImageLayout::ColorAttachment,
            ImageLayout::ShaderReadOnly,
            TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
            TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::SHADER_READ,
            },
        )])?;
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
        display.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent[0] as f32,
            height: self.extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        display.scissor(Rect {
            x: 0,
            y: 0,
            width: self.extent[0],
            height: self.extent[1],
        })?;
        display.set_pipeline(&self.display)?;
        display.set_bind_group(0, &self.group, &[])?;
        display.draw(0..3, 0..1)?;
        display.finish()?;
        let work = encoder.finish()?;
        Ok((work, uploads_geometry))
    }
    fn mark_uploaded(&mut self) {
        self.uploaded = true;
    }
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }
    pub fn format(&self) -> TextureFormat {
        self.format
    }
}
mod resources;
use resources::*;

mod per_eye;
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub use per_eye::run_webgpu_multiview;
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
mod immediate;

#[cfg(windows)]
fn run<B: MultiviewBackend, H: utils::ExampleHost<B>>(host: H) -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 2,
    };
    let mut scene: Option<Multiview<B>> = None;
    utils::run_example(
        host,
        "Multiview rendering",
        config.clone(),
        move |device, _, image, extent| {
            require_multiview(device, 2)?;
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("graphics lane"))?;
            if scene
                .as_ref()
                .is_none_or(|example| example.extent() != extent)
            {
                scene = Some(utils::block_on(Multiview::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let example = scene.as_mut().ok_or(Error::InvalidState)?;
            let (work, uploaded_geometry) = example.record(device, &image)?;
            utils::submit_and_present(device, lane, work, image)?;
            if uploaded_geometry {
                example.mark_uploaded();
            }
            Ok(())
        },
    )
}
#[cfg(windows)]
fn main() {
    #[cfg(feature = "native-gl-wgl")]
    if std::env::args()
        .skip_while(|arg| arg != "--backend")
        .nth(1)
        .is_some_and(|backend| backend == "gl")
    {
        if let Err(error) = immediate::run() {
            eprintln!("15_multiview: {error:?}");
            std::process::exit(1);
        }
        return;
    }
    let result: Result<()> = match utils::windows_api::selected_backend() {
        #[cfg(feature = "dx12")]
        Ok(utils::windows_api::SelectedBackend::Dx12) => {
            run::<fluxel_rhi::dx12::Dx12, _>(utils::windows_api::Dx12Host)
        }
        #[cfg(feature = "vulkan")]
        Ok(utils::windows_api::SelectedBackend::Vulkan) => {
            run::<fluxel_rhi::vulkan::Vulkan, _>(utils::windows_api::VulkanHost)
        }
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        eprintln!("15_multiview: {error:?}");
        std::process::exit(1);
    }
}
#[cfg(all(target_os = "linux", feature = "native-gles-egl"))]
fn main() {
    if let Err(error) = immediate::run() {
        eprintln!("15_multiview: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
fn main() {
    eprintln!("15_multiview requires Windows and a backend with multiview support");
}
