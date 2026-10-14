//! Sequential primary-pass fallback for backends without secondary commands.
//!
//! WebGPU and immediate GL issue the same C++ scene in source order. This path
//! preserves draw order and per-object transforms without inventing CPU replay.
use super::*;

/// C++ workload executed sequentially when secondary recording is unavailable.
pub(super) struct Sequential<B: SequentialExampleBackend> {
    vertices: Buffer<B>,
    indices: Buffer<B>,
    transforms: Buffer<B>,
    star_vertices: Buffer<B>,
    star_indices: Buffer<B>,
    star_uniform: Buffer<B>,
    group: BindGroup<B>,
    star_group: BindGroup<B>,
    ufo: RasterPipeline<B>,
    stars: RasterPipeline<B>,
    source: [Vec<u8>; 4],
    lens: [u64; 4],
    counts: [u32; 2],
    uploaded: bool,
    frame: u32,
}

impl<B: SequentialExampleBackend> Sequential<B> {
    pub(super) async fn new(d: &Device<B>, format: TextureFormat) -> Result<Self> {
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
            "13 sequential UFO vertices",
            u.vertices.len() as u64,
            BufferUsage::VERTEX,
        )?;
        let indices = make(
            "13 sequential UFO indices",
            u.indices.len() as u64,
            BufferUsage::INDEX,
        )?;
        let star_vertices = make(
            "13 sequential star vertices",
            s.vertices.len() as u64,
            BufferUsage::VERTEX,
        )?;
        let star_indices = make(
            "13 sequential star indices",
            s.indices.len() as u64,
            BufferUsage::INDEX,
        )?;
        let transforms = make(
            "13 sequential UFO transforms",
            u64::from(UFO_COUNT) * UNIFORM,
            BufferUsage::UNIFORM,
        )?;
        let star_uniform = make(
            "13 sequential star transform",
            UNIFORM,
            BufferUsage::UNIFORM,
        )?;
        let ufo_bindings = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("13 sequential UFO transforms".into()),
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
            label: Some("13 sequential stars transform".into()),
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
            label: Some("13 sequential UFO group".into()),
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
            label: Some("13 sequential stars group".into()),
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
            label: Some("13 sequential UFO layout".into()),
            groups: &[&ufo_bindings],
            push_constants: &[],
        })?;
        let star_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("13 sequential stars layout".into()),
            groups: &[&star_bindings],
            push_constants: &[],
        })?;
        let ufo = pipeline_wgsl(
            d,
            "13 sequential UFO",
            ufo_layout,
            format,
            UFO_WGSL,
            CullMode::Front,
            vec![mesh_layout()],
        )
        .await?;
        let stars = pipeline_wgsl(
            d,
            "13 sequential stars",
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
    pub(super) fn encode(
        &mut self,
        d: &Device<B>,
        image: &SurfaceImage<B>,
        e: &mut CommandEncoder<B>,
        extent: [u32; 2],
    ) -> Result<bool> {
        let uploaded_geometry = !self.uploaded;
        if uploaded_geometry {
            e.write_buffer(&self.vertices, 0, &self.source[0])?;
            e.write_buffer(&self.indices, 0, &self.source[1])?;
            e.write_buffer(&self.star_vertices, 0, &self.source[2])?;
            e.write_buffer(&self.star_indices, 0, &self.source[3])?;
        }
        e.write_buffer(&self.transforms, 0, &transforms(extent, self.frame))?;
        e.write_buffer(&self.star_uniform, 0, &matrix_bytes(star_matrix(extent)))?;
        self.frame = self.frame.wrapping_add(1);
        let (depth, view) = depth(d, extent)?;
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0., 0., 0., 1.])),
            store: StoreOp::Store,
        })];
        let mut p = e.begin_raster(&RasterPassDesc {
            label: Some("13 sequential primary pass"),
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
        Ok(uploaded_geometry)
    }

    pub(super) fn mark_uploaded(&mut self) {
        self.uploaded = true;
    }
}

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
fn depth<B: ResourceBackend>(
    d: &Device<B>,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = d.create_texture(&TextureDesc {
        label: Some("13 sequential depth".into()),
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

async fn pipeline_wgsl<B: SequentialExampleBackend>(
    d: &Device<B>,
    name: &str,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    wgsl: &[u8],
    cull_mode: CullMode,
    vertex_buffers: Vec<VertexBufferLayout>,
) -> Result<RasterPipeline<B>> {
    let shader =
        utils::native_shader_with_wgsl(d, name, "vs_main", ShaderStage::Vertex, &[], &[], wgsl)
            .await?;
    let fragment = utils::native_shader_with_wgsl(
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
            depth_write_enabled: name != "13 sequential stars",
            depth_compare: CompareFunction::LessEqual,
            stencil_front: disabled_stencil(),
            stencil_back: disabled_stencil(),
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
        let mut scene = Sequential::<WebGpu>::new(&d, format).await?;
        for _ in 0..frames {
            let image = surface.acquire().await?;
            let mut encoder = d.encoder(lane)?;
            let uploaded_geometry = scene.encode(&d, &image, &mut encoder, extent)?;
            let work = encoder.finish()?;
            if uploaded_geometry {
                scene.mark_uploaded();
            }
            utils::submit_and_present_async(&d, lane, work, image).await?;
        }
        Ok(())
    }
    run(canvas, frames)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("{e:?}")))
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub fn run_gl_multithreaded_recording() -> Result<()> {
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    use fluxel_rhi::gl::Gl;

    static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);
    fn outcome_error(outcome: ImmediateRunOutcome<Gl>) -> Error {
        match outcome {
            ImmediateRunOutcome::Issued(_) => Error::InvalidState,
            ImmediateRunOutcome::Failed { error, .. }
            | ImmediateRunOutcome::Indeterminate { error, .. } => error,
        }
    }

    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let scene = Rc::new(RefCell::new(None::<Sequential<Gl>>));
    let dropped = Rc::clone(&scene);
    utils::gl_api::run_wgl(
        "Sequential UFO recording fallback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if scene.borrow().is_none() {
                let next = utils::block_on(Sequential::<Gl>::new(device, config.format))?;
                *scene.borrow_mut() = Some(next);
            }
            let ordinal = NEXT_PLAN
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| Error::OutOfMemory)?;
            // SAFETY: this single-threaded WGL callback owns the context and
            // issues both plan steps consecutively on its graphics lane.
            let plan = unsafe {
                SubmissionPlan::from_compiled(
                    PlanId {
                        device: device.id(),
                        ordinal,
                    },
                    Arc::from([
                        StepDesc {
                            lane,
                            kind: StepKind::Commands,
                            waits: vec![],
                        },
                        StepDesc {
                            lane,
                            kind: StepKind::Present,
                            waits: vec![],
                        },
                    ]),
                )
            };
            // SAFETY: immediate encoding and presentation happen on the
            // current WGL context before the callback returns.
            let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
            let mut commands = session.begin_commands().map_err(outcome_error)?;
            let uploaded_geometry = scene
                .borrow_mut()
                .as_mut()
                .ok_or(Error::InvalidState)?
                .encode(device, &image, commands.commands(), extent)?;
            match commands.close() {
                ImmediateRunOutcome::Issued(_) => {}
                outcome => return Err(outcome_error(outcome)),
            }
            if uploaded_geometry {
                scene
                    .borrow_mut()
                    .as_mut()
                    .ok_or(Error::InvalidState)?
                    .mark_uploaded();
            }
            match session.present(PresentBinding::new(image)) {
                ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
                ImmediatePresentResult::Run(outcome) => return Err(outcome_error(outcome)),
                ImmediatePresentResult::Rejected { error, .. } => return Err(error),
            }
            let point = utils::immediate_end_checkpoint(session.finish())?;
            utils::block_on(device.wait_completion(&point))
        },
    )
}
