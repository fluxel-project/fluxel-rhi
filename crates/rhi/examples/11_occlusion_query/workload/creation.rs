//! Resource and pipeline construction for the occlusion-query workload.

use super::*;

impl<B: OcclusionWorkloadBackend> Workload<B> {
    pub async fn new(d: &Device<B>, lane: LaneId, format: TextureFormat) -> Result<Self> {
        Self::new_mode(d, lane, format, true).await
    }

    /// Creates the scene-only mode used when a backend has no occlusion-query
    /// facility. It draws the same geometry but never fabricates query results.
    pub async fn new_visual_fallback(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
    ) -> Result<Self> {
        Self::new_mode(d, lane, format, false).await
    }

    async fn new_mode(
        d: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        enable_queries: bool,
    ) -> Result<Self> {
        let q = if enable_queries {
            Some(
                d.queries(QueryKind::Occlusion)
                    .ok_or(Error::Unsupported("occlusion queries"))?
                    .create(&QueryDesc {
                        kind: QueryKind::Occlusion,
                        count: 2,
                    })?,
            )
        } else {
            None
        };
        let make = |name: &str, n: usize, usage: BufferUsage| {
            d.create_buffer(&BufferDesc {
                label: Some(name.into()),
                size: n as u64,
                usage: usage.union(BufferUsage::COPY_DST),
                memory: MemoryClass::Device,
            })
        };
        // C++ uses FlipY with a positive-height Vulkan viewport. Fluxel's
        // WebGPU-compatible viewport has negative native height, so retain
        // the same final orientation by cancelling that loader-space flip.
        let options = gltf::LoadOptions::CPP_PORT.with_flip_y(false);
        let plane =
            gltf::load_embedded_model(PLANE, options).map_err(|x| Error::Backend(x.to_string()))?;
        let teapot = gltf::load_embedded_model(TEAPOT, options)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let sphere = gltf::load_embedded_model(SPHERE, options)
            .map_err(|x| Error::Backend(x.to_string()))?;
        let plane_vertices = make(
            "11 occluder vertices",
            plane.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let plane_indices = make(
            "11 occluder indices",
            plane.indices.len(),
            BufferUsage::INDEX,
        )?;
        let teapot_vertices = make(
            "11 teapot vertices",
            teapot.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let teapot_indices = make(
            "11 teapot indices",
            teapot.indices.len(),
            BufferUsage::INDEX,
        )?;
        let sphere_vertices = make(
            "11 sphere vertices",
            sphere.vertices.len(),
            BufferUsage::VERTEX,
        )?;
        let sphere_indices = make(
            "11 sphere indices",
            sphere.indices.len(),
            BufferUsage::INDEX,
        )?;
        let plane_uniform = make("11 occluder transform", 256, BufferUsage::UNIFORM)?;
        let teapot_uniform = make("11 teapot transform", 256, BufferUsage::UNIFORM)?;
        let sphere_uniform = make("11 sphere transform", 256, BufferUsage::UNIFORM)?;
        let vs = utils::native_shader_with_wgsl(
            d,
            "11 VS",
            "vs_main",
            ShaderStage::Vertex,
            VS,
            VS_SPIRV,
            WGSL,
        )
        .await?;
        let simple_ps = utils::native_shader_with_wgsl(
            d,
            "11 simple PS",
            "simple_ps",
            ShaderStage::Fragment,
            SIMPLE_PS,
            SIMPLE_PS_SPIRV,
            WGSL,
        )
        .await?;
        let mesh_ps = utils::native_shader_with_wgsl(
            d,
            "11 mesh PS",
            "mesh_ps",
            ShaderStage::Fragment,
            MESH_PS,
            MESH_PS_SPIRV,
            WGSL,
        )
        .await?;
        let occluder_ps = utils::native_shader_with_wgsl(
            d,
            "11 occluder PS",
            "occluder_ps",
            ShaderStage::Fragment,
            OCCLUDER_PS,
            OCCLUDER_PS_SPIRV,
            WGSL,
        )
        .await?;
        let layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("11 transform layout".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Vertex],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    // WGSL uniform structures have a 16-byte final alignment;
                    // the 240 bytes of fields therefore occupy a 256-byte binding.
                    min_size: 256,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let bind = |buffer: &Buffer<B>, name: &str| {
            d.create_bind_group(&BindGroupDesc {
                label: Some(name.into()),
                layout: &layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer,
                        range: 0..256,
                    },
                }],
            })
        };
        let plane_group = bind(&plane_uniform, "11 occluder group")?;
        let teapot_group = bind(&teapot_uniform, "11 teapot group")?;
        let sphere_group = bind(&sphere_uniform, "11 sphere group")?;
        let l = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("11 layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let face = StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        };
        let pipeline = |label: &str,
                        fragment: Shader<B>,
                        entry: &str,
                        target_format: TextureFormat,
                        blend: Option<BlendState>| RasterPipelineDesc {
            label: Some(label.into()),
            layout: l.clone(),
            vertex: PipelineStage {
                module: vs.clone(),
                entry: "vs_main".into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: fragment,
                entry: entry.into(),
                constants: vec![],
            }),
            vertex_buffers: vec![VertexBufferLayout {
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
            color_targets: vec![Some(ColorTargetState {
                format: target_format,
                blend,
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
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil_front: face,
                stencil_back: face,
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
        };
        let simple_pipeline = d
            .raster_pipeline(pipeline(
                "11 simple pipeline",
                simple_ps,
                "simple_ps",
                TextureFormat::Rgba8Unorm,
                None,
            ))
            .await?;
        let mesh_pipeline = d
            .raster_pipeline(pipeline(
                "11 mesh pipeline",
                mesh_ps,
                "mesh_ps",
                format,
                None,
            ))
            .await?;
        let occluder_pipeline = d
            .raster_pipeline(pipeline(
                "11 occluder pipeline",
                occluder_ps,
                "occluder_ps",
                format,
                Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::SrcColor,
                        dst_factor: BlendFactor::OneMinusSrcColor,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                }),
            ))
            .await?;
        let results = if enable_queries {
            Some(d.create_buffer(&BufferDesc {
                label: Some("11 resolved occlusion results".into()),
                size: 16,
                // WebGPU cannot map a QUERY_RESOLVE buffer directly: MAP_READ
                // buffers may only also carry COPY_DST.  ReadbackBackend copies
                // this resolve target to its MAP_READ staging allocation.
                usage: BufferUsage::QUERY_RESOLVE.union(BufferUsage::COPY_SRC),
                memory: MemoryClass::Device,
            })?)
        } else {
            None
        };
        Ok(Self {
            lane,
            plane_vertices,
            plane_indices,
            teapot_vertices,
            teapot_indices,
            sphere_vertices,
            sphere_indices,
            plane_count: (plane.indices.len() / 4) as u32,
            teapot_count: (teapot.indices.len() / 4) as u32,
            sphere_count: (sphere.indices.len() / 4) as u32,
            plane_len: plane.vertices.len() as u64,
            teapot_len: teapot.vertices.len() as u64,
            sphere_len: sphere.vertices.len() as u64,
            plane_index_len: plane.indices.len() as u64,
            teapot_index_len: teapot.indices.len() as u64,
            sphere_index_len: sphere.indices.len() as u64,
            plane_uniform,
            teapot_uniform,
            sphere_uniform,
            plane_group,
            teapot_group,
            sphere_group,
            simple_pipeline,
            mesh_pipeline,
            occluder_pipeline,
            queries: q,
            results,
            visible: [true; 2],
        })
    }
}
