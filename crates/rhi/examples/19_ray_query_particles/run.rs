//! Headless GPU execution and readback for three deterministic animation times.
use super::{particles, scene};
use fluxel_rhi::api::*;

use std::{future::Future, sync::Arc};
pub const WIDTH: u32 = 1000;
pub const HEIGHT: u32 = 709;
pub(super) fn block_on<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
pub(super) fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
pub fn run<B>(device: &Device<B>, format: ShaderFormat) -> Result<()>
where
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + ReadbackBackend
        + DeferredBackend
        + ExecutionBackend<Mode = DeferredExecution>,
{
    let times = if std::env::var_os("FLUXEL_RAY_QUERY_FIXED_REFERENCE").is_some() {
        [2., 4., 6.]
    } else {
        [8.363f32, 10.212, 12.228]
    };
    for (index, time) in times.into_iter().enumerate() {
        render(device, format, index, time, None, true)?;
    }
    Ok(())
}
pub fn render<B>(
    device: &Device<B>,
    format: ShaderFormat,
    index: usize,
    time: f32,
    live_simulation: Option<&particles::Simulation>,
    save_readback: bool,
) -> Result<Texture<B>>
where
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + ReadbackBackend
        + DeferredBackend
        + ExecutionBackend<Mode = DeferredExecution>,
{
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
    let smoke_data = super::texture::particle_mips(smoke_data).map_err(Error::Backend)?;
    let smoke = buffer(
        smoke_data.len() as u64,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
    )?;
    let mut offline_simulation = particles::Simulation::new();
    // The comparison mode uses a fixed reference sequence; presentation supplies
    // the simulation advanced with actual elapsed frame times.
    let steps = (time * 60.) as u32;
    if live_simulation.is_none() {
        for _ in 0..steps {
            offline_simulation.advance(1. / 60.);
        }
    }
    let sim = live_simulation.unwrap_or(&offline_simulation);
    if let Some(directory) = std::env::var_os("FLUXEL_RAY_QUERY_STATE_DIR") {
        use std::fmt::Write as _;
        let mut text = String::new();
        for (i, p) in sim.particles.iter().enumerate().filter(|(_, p)| p.active) {
            writeln!(
                text,
                "{} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
                i,
                p.position[0],
                p.position[1],
                p.position[2],
                p.velocity[0],
                p.velocity[1],
                p.velocity[2],
                p.color[0],
                p.color[1],
                p.color[2],
                p.radius,
                p.age,
                p.opacity,
                p.rotation
            )
            .map_err(|e| Error::Backend(e.to_string()))?;
        }
        std::fs::write(
            std::path::Path::new(&directory).join(format!("state-{steps}.txt")),
            text,
        )
        .map_err(|e| Error::Backend(e.to_string()))?;
    }
    let mut particle_data = Vec::new();
    for p in sim.particles.iter().filter(|p| p.active) {
        particle_data.extend_from_slice(&[
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
    let count = particle_data.len() as u32 / 12;
    if particle_data.is_empty() {
        particle_data.resize(12, 0.);
    }
    let particle_data = bytes(&particle_data);
    let particle = buffer(
        particle_data.len() as u64,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
    )?;
    let constants = buffer(256, BufferUsage::UNIFORM | BufferUsage::COPY_DST)?;
    let ui_size: u64 = 4;
    let ui_buffer = buffer(ui_size, BufferUsage::STORAGE | BufferUsage::COPY_DST)?;
    let mut cb = bytes(&[
        1.9867786,
        4.052121,
        1.9867786,
        0.,
        -0.66446304,
        -0.34202015,
        -0.66446304,
        0.,
        0.70710677,
        0.,
        -0.70710677,
        0.,
        -0.24184476,
        0.9396926,
        -0.24184476,
        0.,
    ]);
    for n in [WIDTH, HEIGHT, count, 0, 0, 0, 0, 0, 0, 0, 0, 0] {
        cb.extend_from_slice(&n.to_ne_bytes());
    }
    cb.resize(256, 0);
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
    let quad_data = bytes(&super::particle_acceleration::quad_vertices(sim, false));
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
        input: AccelerationInput::Tlas(super::particle_acceleration::instances(
            &blas,
            &particle_blas,
            &quad_blas,
            sim,
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
            entry(3, storage(particle_data.len() as u64)),
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
                    range: 0..particle_data.len() as u64,
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
            ShaderFormat::SpirV => {
                Arc::from(&include_bytes!("../assets/19_ray_query_particles/controlled.spv")[..])
            }
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
    let pitch = (WIDTH * 4 + 255) & !255;
    let readback = device.create_buffer(&BufferDesc {
        label: None,
        size: u64::from(pitch) * u64::from(HEIGHT),
        usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        memory: MemoryClass::Readback,
    })?;
    let mut enc = device.encoder(lane)?;
    enc.write_buffer(&vertex, 0, &vertex_data)?;
    enc.write_buffer(&bounds, 0, &bounds_data)?;
    enc.write_buffer(&quad_buffer, 0, &quad_data)?;
    enc.write_buffer(&normal, 0, &normal_data)?;
    enc.write_buffer(&particle, 0, &particle_data)?;
    enc.write_buffer(&constants, 0, &cb)?;
    enc.write_buffer(&environment, 0, &environment_data)?;
    enc.write_buffer(&smoke, 0, &smoke_data)?;
    enc.write_buffer(&ui_buffer, 0, &[0; 4])?;
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
            &ui_buffer,
            ui_size,
            BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_READ,
            },
        ),
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
            &particle,
            particle_data.len() as u64,
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
            &constants,
            256,
            BufferUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::UNIFORM_READ,
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
    let range = || Subresources {
        texture: &output,
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
    };
    enc.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: range(),
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::COMPUTE,
                accesses: AccessMask::SHADER_WRITE,
            },
        },
        layout: Some((ImageLayout::Undefined, ImageLayout::General)),
        ownership: None,
    })])?;
    let mut pass = compute.begin(&mut enc)?;
    pass.set_pipeline(&pipeline)?;
    pass.set_bind_group(0, &group, &[])?;
    pass.dispatch([WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1])?;
    pass.finish()?;
    let ticket = if save_readback {
        enc.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::General, ImageLayout::CopySrc)),
            ownership: None,
        })])?;
        enc.texture_to_buffer(
            TextureRegion {
                texture: &output,
                mip: 0,
                origin: [0, 0, 0],
                extent: [WIDTH, HEIGHT, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: pitch,
                rows_per_image: HEIGHT,
            },
        )?;
        let ticket = enc.readback_buffer(&readback, 0..u64::from(pitch) * u64::from(HEIGHT))?;
        enc.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
                to: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::CopySrc, ImageLayout::ShaderReadOnly)),
            ownership: None,
        })])?;
        Some(ticket)
    } else {
        enc.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: range(),
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
        None
    };
    let work = enc.finish()?;
    // SAFETY: the plan has one command step for this device and lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 19000 + index as u64,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the work and resources stay alive through the completion wait.
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
    block_on(device.wait_completion(&point))?;
    if let Some(ticket) = ticket {
        // SAFETY: the completion covers the readback ticket.
        let data = unsafe { block_on(device.read_after(ticket, &point))? }.bytes;
        let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
        for row in data.chunks_exact(pitch as usize).take(HEIGHT as usize) {
            rgba.extend_from_slice(&row[..(WIDTH * 4) as usize]);
        }
        let path = format!("ray-query-particles-{:02}.png", index + 1);
        let file = std::fs::File::create(&path).map_err(|e| Error::Backend(e.to_string()))?;
        let mut png = png::Encoder::new(file, WIDTH, HEIGHT);
        png.set_color(png::ColorType::Rgba);
        png.set_depth(png::BitDepth::Eight);
        let mut writer = png
            .write_header()
            .map_err(|e| Error::Backend(e.to_string()))?;
        writer
            .write_image_data(&rgba)
            .map_err(|e| Error::Backend(e.to_string()))?;
        println!("{path}: t={time:.3}s, {count} particles");
    }
    Ok(output)
}
