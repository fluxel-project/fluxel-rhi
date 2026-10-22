//! Reusable resources for continuously animated ray-query presentation.
use super::{particles, run, scene};
use fluxel_rhi::api::*;
use run::{HEIGHT, WIDTH, block_on, bytes};
use std::sync::Arc;
pub struct Live<B: ResourceBackend + PipelineBackend> {
    device: Device<B>,
    lane: LaneId,
    output: Texture<B>,
    particle: Buffer<B>,
    constants: Buffer<B>,
    pipeline: ComputePipeline<B>,
    group: BindGroup<B>,
    _blas: AccelerationStructure<B>,
    _tlas: AccelerationStructure<B>,
    first: bool,
    frame: u64,
}
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
        let smoke_data = super::texture::smoke_mips(smoke_data);
        let smoke = buffer(
            smoke_data.len() as u64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        let particle = buffer(
            (particles::MAX_PARTICLES * 48) as u64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
        )?;
        let constants = buffer(256, BufferUsage::UNIFORM | BufferUsage::COPY_DST)?;
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
        let tlas_desc = AccelerationDesc {
            input: AccelerationInput::Tlas(vec![AccelerationInstance {
                blas: blas.clone(),
                transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
                mask: 255,
                custom_index: 0,
                shader_table_offset: 0,
                flags: AccelerationInstanceFlags::NONE,
            }]),
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
            pipeline,
            group,
            _blas: blas,
            _tlas: tlas,
            first: true,
            frame: 0,
        })
    }
    pub fn output(&self) -> &Texture<B> {
        &self.output
    }
    pub fn render(&mut self, sim: &particles::Simulation) -> Result<()> {
        let mut values = Vec::new();
        for p in sim.particles.iter().filter(|p| p.active) {
            values.extend_from_slice(&[
                p.position[0],
                p.position[1],
                p.position[2],
                p.radius,
                p.color[0],
                p.color[1],
                p.color[2],
                p.opacity,
                p.rotation,
                0.,
                0.,
                0.,
            ]);
        }
        let count = values.len() as u32 / 12;
        if values.is_empty() {
            values.resize(12, 0.);
        }
        let data = bytes(&values);
        let mut cb = bytes(&[
            1.987, 4.052, 1.987, 0., -0.66446, -0.34202, -0.66446, 0., 0.7071, 0., -0.7071, 0.,
            -0.24184, 0.93969, -0.24184, 0.,
        ]);
        for n in [WIDTH, HEIGHT, count, 0, 0, 0, 0, 0, 0, 0, 0, 0] {
            cb.extend_from_slice(&n.to_ne_bytes());
        }
        cb.resize(256, 0);
        let read = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_READ,
        };
        let uniform = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::UNIFORM_READ,
        };
        // write_buffer performs its upload and returns to the idle state.
        let idle = BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::EMPTY,
        };
        let barrier = |buffer, size, from, to| {
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer,
                    range: 0..size,
                },
                usage: StateTransition { from, to },
                ownership: None,
            })
        };
        let mut enc = self.device.encoder(self.lane)?;
        if !self.first {
            enc.transition_resources(&[
                barrier(
                    &self.particle,
                    (particles::MAX_PARTICLES * 48) as u64,
                    read,
                    idle,
                ),
                barrier(&self.constants, 256, uniform, idle),
            ])?;
        }
        enc.write_buffer(&self.particle, 0, &data)?;
        enc.write_buffer(&self.constants, 0, &cb)?;
        enc.transition_resources(&[
            barrier(
                &self.particle,
                (particles::MAX_PARTICLES * 48) as u64,
                idle,
                read,
            ),
            barrier(&self.constants, 256, idle, uniform),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &self.output,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 0..1,
                },
                usage: StateTransition {
                    from: if self.first {
                        TextureUses {
                            stages: StageMask::EMPTY,
                            accesses: AccessMask::EMPTY,
                        }
                    } else {
                        TextureUses {
                            stages: StageMask::FRAGMENT,
                            accesses: AccessMask::SHADER_READ,
                        }
                    },
                    to: TextureUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                },
                layout: Some((
                    if self.first {
                        ImageLayout::Undefined
                    } else {
                        ImageLayout::ShaderReadOnly
                    },
                    ImageLayout::General,
                )),
                ownership: None,
            }),
        ])?;
        let compute = self.device.compute().ok_or(Error::InvalidState)?;
        let mut pass = compute.begin(&mut enc)?;
        pass.set_pipeline(&self.pipeline)?;
        pass.set_bind_group(0, &self.group, &[])?;
        pass.dispatch([WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1])?;
        pass.finish()?;
        enc.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &self.output,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::General, ImageLayout::ShaderReadOnly)),
            ownership: None,
        })])?;
        submit(
            &self.device,
            self.lane,
            enc.finish()?,
            19000001 + self.frame,
        )?;
        self.first = false;
        self.frame += 1;
        Ok(())
    }
}
fn submit<B: DeferredBackend + ExecutionBackend<Mode = DeferredExecution>>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    ordinal: u64,
) -> Result<()> {
    // SAFETY: the work uses the same device and lane as this one-step plan.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: resources referenced by work remain alive through the wait below.
    let result = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(receipt) => {
            receipt
                .checkpoints
                .into_iter()
                .next()
                .ok_or(Error::InvalidState)?
                .point
        }
        _ => return Err(Error::InvalidState),
    };
    block_on(device.wait_completion(&point))
}
