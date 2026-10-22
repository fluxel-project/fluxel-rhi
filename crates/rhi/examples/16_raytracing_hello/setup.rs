//! Scene resources and ray pipeline construction.

use super::*;
use std::sync::Arc;

fn buffer<B: ResourceBackend>(
    device: &Device<B>,
    name: &str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    device.create_buffer(&BufferDesc {
        label: Some(name.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}

fn align(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

impl<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
    Hello<B>
{
    pub(super) async fn new(
        device: &Device<B>,
        lane: LaneId,
        extent: [u32; 2],
        format: TextureFormat,
        shader_format: ShaderFormat,
    ) -> Result<Self> {
        let ray = device.ray().ok_or(Error::Unsupported("ray tracing"))?;
        let vertices = buffer(
            device,
            "ray vertices",
            36,
            BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST,
        )?;
        let indices = buffer(
            device,
            "ray indices",
            6,
            BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST,
        )?;
        let blas_desc = AccelerationDesc {
            input: AccelerationInput::Blas(vec![AccelerationGeometry::Triangles {
                vertices: vertices.clone(),
                vertex_offset: 0,
                vertex_count: 3,
                vertex_format: VertexFormat::Float32(3),
                stride: 12,
                indices: Some(AccelerationIndices {
                    buffer: indices.clone(),
                    offset: 0,
                    count: 3,
                    format: IndexFormat::Uint16,
                }),
                opaque: false,
            }]),
            fast_trace: true,
            allow_update: false,
        };
        let blas = ray.create_acceleration(&blas_desc)?;
        let tlas_desc = AccelerationDesc {
            input: AccelerationInput::Tlas(vec![AccelerationInstance {
                blas: blas.clone(),
                transform: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                mask: 1,
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
        let blas_scratch = buffer(
            device,
            "BLAS scratch",
            bs.scratch_bytes,
            BufferUsage::ACCEL_SCRATCH,
        )?;
        let tlas_scratch = buffer(
            device,
            "TLAS scratch",
            ts.scratch_bytes,
            BufferUsage::ACCEL_SCRATCH,
        )?;
        let output = device.create_texture(&TextureDesc {
            label: Some("ray output".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING
                | TextureUsage::TEXTURE_BINDING
                | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })?;
        let row_pitch = extent[0]
            .checked_mul(4)
            .and_then(|n| n.checked_add(255))
            .ok_or(Error::InvalidRange)?
            & !255;
        let readback = device.create_buffer(&BufferDesc {
            label: Some("ray output readback".into()),
            size: u64::from(row_pitch) * u64::from(extent[1]),
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
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
        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::RayGeneration],
                ty: BindingType::AccelerationStructure,
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: vec![ShaderStage::RayGeneration],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2,
                    access: StorageAccess::Write,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            },
        ];
        let group_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("ray resources".into()),
            entries,
        })?;
        let group = device.create_bind_group(&BindGroupDesc {
            label: Some("ray resources"),
            layout: &group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Acceleration(&tlas),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&view),
                },
            ],
        })?;
        let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("ray layout"),
            groups: &[&group_layout],
            push_constants: &[],
        })?;
        let names = [
            ("MyRaygenShader", ShaderStage::RayGeneration),
            ("MyMissShader", ShaderStage::Miss),
            ("MyClosestHitShader", ShaderStage::ClosestHit),
        ];
        let shader = device
            .shader(ShaderDesc {
                label: Some("Microsoft ray library".into()),
                format: shader_format,
                bytes: Arc::from(match shader_format {
                    ShaderFormat::Dxil => {
                        include_bytes!("../assets/16_raytracing_hello/Raytracing_rhi.dxil")
                            .as_slice()
                    }
                    ShaderFormat::SpirV => {
                        include_bytes!("../assets/16_raytracing_hello/Raytracing.spv").as_slice()
                    }
                    _ => return Err(Error::Unsupported("ray shader format")),
                }),
                entries: names
                    .iter()
                    .map(|(name, stage)| ShaderEntryPoint {
                        name: (*name).into(),
                        stage: *stage,
                        workgroup_size: None,
                    })
                    .collect(),
                required_features: vec![],
            })
            .await?;
        let shaders = names
            .iter()
            .map(|(name, _)| PipelineStage {
                module: shader.clone(),
                entry: (*name).into(),
                constants: vec![],
            })
            .collect();
        let group_desc = |general, closest_hit, local_constants| RayGroup {
            general,
            closest_hit,
            any_hit: None,
            intersection: None,
            local_constants,
        };
        let pipeline = ray
            .create_pipeline(RayPipelineDesc {
                layout,
                shaders,
                groups: vec![
                    group_desc(
                        Some(0),
                        None,
                        Some(RayLocalConstants {
                            binding: 0,
                            group: 0,
                            size_bytes: 32,
                        }),
                    ),
                    group_desc(Some(1), None, None),
                    group_desc(None, Some(2), None),
                ],
                max_recursion: 1,
                payload_bytes: 16,
                hit_attribute_bytes: 8,
                cache: None,
            })
            .await?;
        let properties = ray.shader_table_properties()?;
        let stride = align(
            u64::from(properties.identifier_size) + 32,
            u64::from(properties.record_alignment),
        );
        let section = align(stride, u64::from(properties.table_alignment));
        let mut table_bytes = vec![0; (section * 3) as usize];
        for index in 0..3 {
            let id = ray.shader_group_identifier(&pipeline, index)?;
            let start = section as usize * index as usize;
            table_bytes[start..start + id.len()].copy_from_slice(&id);
        }
        for (index, value) in scene::raygen_constants(extent[0], extent[1])
            .iter()
            .enumerate()
        {
            let offset = properties.identifier_size as usize + index * 4;
            table_bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
        let table_buffer = buffer(
            device,
            "shader table",
            section * 3,
            BufferUsage::SHADER_TABLE | BufferUsage::COPY_DST,
        )?;
        let region = |offset| ShaderTableRegion {
            buffer: table_buffer.clone(),
            offset,
            size: stride,
            stride,
        };
        let table = ShaderTable {
            raygen: region(0),
            miss: Some(region(section)),
            hit: Some(region(section * 2)),
            callable: None,
        };
        let display =
            display::Display::new_with_format(device, &view, format, shader_format).await?;
        Ok(Self {
            lane,
            vertices,
            indices,
            blas_desc: Some(blas_desc),
            tlas_desc: Some(tlas_desc),
            blas: Some(blas),
            tlas,
            blas_scratch,
            tlas_scratch,
            scratch_sizes: [bs.scratch_bytes, ts.scratch_bytes],
            output,
            readback,
            row_pitch,
            pipeline,
            group,
            table,
            table_bytes,
            display,
            initialized: false,
            captured: false,
        })
    }
}
