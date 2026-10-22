//! Initial GPU resources and acceleration structures for the live scene.
use super::*;

impl<B> Live<B>
where
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + ReadbackBackend
        + DeferredBackend
        + ExecutionBackend<Mode = DeferredExecution>,
{
    pub fn new(device: &Device<B>, format: ShaderFormat) -> Result<Self> {
        if !device.capabilities().ray_query {
            return Err(Error::Unsupported("inline ray query"));
        }
        let lane = device
            .lanes()
            .iter()
            .find(|l| l.class == LaneClass::Graphics)
            .ok_or(Error::Unsupported("graphics lane"))?
            .id;
        let ray = device
            .ray()
            .ok_or(Error::Unsupported("acceleration structures"))?;
        let buffer = |size, usage| {
            device.create_buffer(&BufferDesc {
                label: None,
                size,
                usage,
                memory: MemoryClass::Device,
            })
        };
        let meshes = scene::load().map_err(Error::Backend)?;
        let (vertices, normals) = scene::world_triangles(&meshes);
        let vertex_data = bytes(&vertices);
        let normal_data = bytes(&normals);
        let vertex = buffer(
            vertex_data.len() as u64,
            BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST,
        )?;
        let normal = buffer(
            normal_data.len() as u64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        let decoder = png::Decoder::new(std::io::Cursor::new(include_bytes!(
            "../assets/19_ray_query_particles/environment-map-rgba.png"
        )));
        let mut reader = decoder
            .read_info()
            .map_err(|e| Error::Backend(e.to_string()))?;
        let mut environment_data = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut environment_data)
            .map_err(|e| Error::Backend(e.to_string()))?;
        if (info.width, info.height, info.color_type, info.bit_depth)
            != (2048, 1024, png::ColorType::Rgba, png::BitDepth::Eight)
        {
            return Err(Error::InvalidState);
        }
        environment_data.truncate(info.buffer_size());
        let environment = buffer(
            environment_data.len() as u64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        let decoder = png::Decoder::new(std::io::Cursor::new(include_bytes!(
            "../assets/19_ray_query_particles/smoke-particle.png"
        )));
        let mut reader = decoder
            .read_info()
            .map_err(|e| Error::Backend(e.to_string()))?;
        let mut smoke_data = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut smoke_data)
            .map_err(|e| Error::Backend(e.to_string()))?;
        if (info.width, info.height, info.color_type, info.bit_depth)
            != (512, 512, png::ColorType::Rgba, png::BitDepth::Eight)
        {
            return Err(Error::InvalidState);
        }
        smoke_data.truncate(info.buffer_size());
        let smoke_data = crate::texture::particle_mips(smoke_data).map_err(Error::Backend)?;
        let smoke = buffer(
            smoke_data.len() as u64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        let particle = buffer(
            (particles::MAX_PARTICLES * 48) as u64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        let constants = buffer(256, BufferUsage::UNIFORM | BufferUsage::COPY_DST)?;
        let ui_size: u64 = (crate::ui::WIDTH * crate::ui::HEIGHT * 4) as u64;
        let ui_buffer = buffer(ui_size, BufferUsage::STORAGE | BufferUsage::COPY_DST)?;
        let blas_desc = AccelerationDesc {
            input: AccelerationInput::Blas(vec![AccelerationGeometry::Triangles {
                vertices: vertex.clone(),
                vertex_offset: 0,
                vertex_count: (vertices.len() / 3) as u32,
                vertex_format: VertexFormat::Float32(3),
                stride: 12,
                indices: None,
                opaque: true,
            }]),
            fast_trace: true,
            allow_update: false,
        };
        let blas = ray.create_acceleration(&blas_desc)?;
        let bounds_data = bytes(&[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        let bounds = buffer(24, BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST)?;
        let particle_desc = AccelerationDesc {
            input: AccelerationInput::Blas(vec![AccelerationGeometry::Aabbs {
                data: bounds.clone(),
                offset: 0,
                stride: 24,
                count: 1,
                opaque: false,
            }]),
            fast_trace: true,
            allow_update: false,
        };
        let particle_blas = ray.create_acceleration(&particle_desc)?;
        let ps = ray.sizes(&particle_desc)?;
        let pscratch = buffer(ps.scratch_bytes, BufferUsage::ACCEL_SCRATCH)?;
        let quad_data = bytes(&crate::particle_acceleration::quad_vertices(
            &particles::Simulation::new(),
            false,
        ));
        let quad_buffer = buffer(
            quad_data.len() as u64,
            BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST,
        )?;
        let quad_desc = AccelerationDesc {
            input: AccelerationInput::Blas(vec![AccelerationGeometry::Triangles {
                vertices: quad_buffer.clone(),
                vertex_offset: 0,
                vertex_count: (particles::MAX_PARTICLES * 6) as u32,
                vertex_format: VertexFormat::Float32(3),
                stride: 12,
                indices: None,
                opaque: false,
            }]),
            fast_trace: true,
            allow_update: false,
        };
        let quad_blas = ray.create_acceleration(&quad_desc)?;
        let qs = ray.sizes(&quad_desc)?;
        let qscratch = buffer(qs.scratch_bytes, BufferUsage::ACCEL_SCRATCH)?;
        let tlas_desc = AccelerationDesc {
            input: AccelerationInput::Tlas(crate::particle_acceleration::instances(
                &blas,
                &particle_blas,
                &quad_blas,
                &particles::Simulation::new(),
            )),
            fast_trace: true,
            allow_update: false,
        };
        let tlas = ray.create_acceleration(&tlas_desc)?;
        let bs = ray.sizes(&blas_desc)?;
        let ts = ray.sizes(&tlas_desc)?;
        let bscratch = buffer(bs.scratch_bytes, BufferUsage::ACCEL_SCRATCH)?;
        let tscratch = buffer(ts.scratch_bytes, BufferUsage::ACCEL_SCRATCH)?;
        let output = device.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [WIDTH, HEIGHT, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING
                | TextureUsage::COPY_SRC
                | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })?;
        let view = device.create_view(
            &output,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let entry = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: vec![ShaderStage::Compute],
            ty,
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        };
        let storage = |size| BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size: size,
            access: StorageAccess::Read,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(0, BindingType::AccelerationStructure),
                entry(
                    1,
                    BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2,
                        access: StorageAccess::Write,
                    },
                ),
                entry(2, storage(normal_data.len() as u64)),
                entry(3, storage((particles::MAX_PARTICLES * 48) as u64)),
                entry(
                    4,
                    BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 256,
                    },
                ),
                entry(5, storage(environment_data.len() as u64)),
                entry(6, storage(smoke_data.len() as u64)),
                entry(7, storage(ui_size)),
            ],
        })?;
        let group = device.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Acceleration(&tlas),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::StorageBufferView {
                        buffer: &normal,
                        range: 0..normal_data.len() as u64,
                        view: StorageBufferView::Structured { stride: 16 },
                    },
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::StorageBufferView {
                        buffer: &particle,
                        range: 0..(particles::MAX_PARTICLES * 48) as u64,
                        view: StorageBufferView::Structured { stride: 48 },
                    },
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer {
                        buffer: &constants,
                        range: 0..256,
                    },
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::StorageBufferView {
                        buffer: &environment,
                        range: 0..environment_data.len() as u64,
                        view: StorageBufferView::Structured { stride: 4 },
                    },
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::StorageBufferView {
                        buffer: &smoke,
                        range: 0..smoke_data.len() as u64,
                        view: StorageBufferView::Structured { stride: 4 },
                    },
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::StorageBufferView {
                        buffer: &ui_buffer,
                        range: 0..ui_size,
                        view: StorageBufferView::Structured { stride: 4 },
                    },
                },
            ],
        })?;
        let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })?;
        let shader = block_on(device.shader(ShaderDesc {
            label: None,
            format,
            bytes: match format {
                ShaderFormat::Dxil => Arc::from(&include_bytes!("controlled.dxil")[..]),
                ShaderFormat::SpirV => Arc::from(
                    &include_bytes!("../assets/19_ray_query_particles/controlled.spv")[..],
                ),
                _ => return Err(Error::Unsupported("shader format")),
            },
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([8, 8, 1]),
            }],
            required_features: vec![SHADER_FEATURE_RAY_QUERY.into()],
        }))?;
        let compute = device.compute().ok_or(Error::Unsupported("compute"))?;
        let pipeline = block_on(compute.create_pipeline(ComputePipelineDesc {
            label: None,
            layout,
            compute: PipelineStage {
                module: shader,
                entry: "main".into(),
                constants: vec![],
            },
            cache: None,
            required_subgroup_size: None,
        }))?;
        let mut enc = device.encoder(lane)?;
        enc.write_buffer(&vertex, 0, &vertex_data)?;
        enc.write_buffer(&bounds, 0, &bounds_data)?;
        enc.write_buffer(&quad_buffer, 0, &quad_data)?;
        enc.write_buffer(&normal, 0, &normal_data)?;
        enc.write_buffer(&environment, 0, &environment_data)?;
        enc.write_buffer(&smoke, 0, &smoke_data)?;
        let empty = BufferUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        let barrier = |buffer, size, to| {
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer,
                    range: 0..size,
                },
                usage: StateTransition { from: empty, to },
                ownership: None,
            })
        };
        enc.transition_resources(&[
            barrier(
                &quad_buffer,
                quad_data.len() as u64,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_READ,
                },
            ),
            barrier(
                &qscratch,
                qs.scratch_bytes,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
            ),
            barrier(
                &bounds,
                24,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_READ,
                },
            ),
            barrier(
                &pscratch,
                ps.scratch_bytes,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
            ),
            barrier(
                &vertex,
                vertex_data.len() as u64,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_READ,
                },
            ),
            barrier(
                &normal,
                normal_data.len() as u64,
                BufferUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            ),
            barrier(
                &environment,
                environment_data.len() as u64,
                BufferUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            ),
            barrier(
                &smoke,
                smoke_data.len() as u64,
                BufferUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            ),
            barrier(
                &bscratch,
                bs.scratch_bytes,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
            ),
            barrier(
                &tscratch,
                ts.scratch_bytes,
                BufferUses {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
            ),
        ])?;
        ray.build(&mut enc, &blas, &bscratch, &blas_desc)?;
        ray.build(&mut enc, &quad_blas, &qscratch, &quad_desc)?;
        enc.transition_resources(&[ResourceTransition::Acceleration {
            target: &quad_blas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            },
        }])?;
        ray.build(&mut enc, &particle_blas, &pscratch, &particle_desc)?;
        enc.transition_resources(&[ResourceTransition::Acceleration {
            target: &particle_blas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            },
        }])?;
        enc.transition_resources(&[ResourceTransition::Acceleration {
            target: &blas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_READ,
            },
        }])?;
        ray.build(&mut enc, &tlas, &tscratch, &tlas_desc)?;
        enc.transition_resources(&[ResourceTransition::Acceleration {
            target: &tlas,
            before: AccessScope {
                stages: StageMask::ACCELERATION,
                accesses: AccessMask::ACCEL_WRITE,
            },
            after: AccessScope {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::ACCEL_READ,
            },
        }])?;
        submit(device, lane, enc.finish()?, 19000000)?;
        Ok(Self {
            device: device.clone(),
            lane,
            output,
            particle,
            constants,
            ui_buffer,
            pipeline,
            group,
            blas,
            particle_blas,
            quad_blas,
            quad_buffer,
            quad_desc,
            qscratch,
            qscratch_size: qs.scratch_bytes,
            tlas,
            tscratch,
            tscratch_size: ts.scratch_bytes,
            first: true,
            last_beam: false,
            last_view: None,
            frame: 0,
        })
    }
}
