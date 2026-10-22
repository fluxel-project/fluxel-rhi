//! Acceleration structure ownership, size queries, and command recording.
use super::super::{
    DeviceState, Vulkan, VulkanBuffer, VulkanDevice, VulkanEncoder, resource, vk_error,
};
use ash::vk;
use fluxel_rhi_core::api::{self, Error, Result};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct VulkanAcceleration(pub(crate) Arc<AccelerationState>);
pub(crate) struct AccelerationState {
    device: Arc<DeviceState>,
    pub(crate) native: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    pub(crate) top: bool,
    children: Mutex<Vec<VulkanAcceleration>>,
}
impl Drop for AccelerationState {
    fn drop(&mut self) {
        // SAFETY: this final owner outlives all submitted uses and keeps the device alive.
        unsafe {
            self.device
                .ray_loaders
                .as_ref()
                .expect("created with ray loader")
                .acceleration
                .destroy_acceleration_structure(self.native, None);
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}
impl VulkanAcceleration {
    pub(crate) fn address(&self) -> u64 {
        // SAFETY: the structure and enabled loader remain live through this call.
        unsafe {
            self.0
                .device
                .ray_loaders
                .as_ref()
                .expect("created with ray loader")
                .acceleration
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(self.0.native),
                )
        }
    }
}
struct Inputs {
    geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
    counts: Vec<u32>,
    buffers: Vec<VulkanBuffer>,
    top: bool,
    flags: vk::BuildAccelerationStructureFlagsKHR,
}
fn checked_address(
    buffer: &VulkanBuffer,
    offset: u64,
    count: u32,
    stride: u64,
    item: u64,
    align: u64,
) -> Result<u64> {
    if !buffer.0.usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT)
        || count == 0
        || stride < item
        || stride % align != 0
    {
        return Err(Error::InvalidRange);
    }
    let end = u64::from(count - 1)
        .checked_mul(stride)
        .and_then(|x| x.checked_add(item))
        .and_then(|x| x.checked_add(offset))
        .ok_or(Error::InvalidRange)?;
    let address = buffer
        .device_address()?
        .checked_add(offset)
        .ok_or(Error::InvalidRange)?;
    if end > buffer.0.size || address % align != 0 {
        return Err(Error::InvalidRange);
    }
    Ok(address)
}
impl Inputs {
    fn new(desc: &api::AccelerationDesc<Vulkan>, instance_address: u64) -> Result<Self> {
        let mut geometries = Vec::new();
        let mut counts = Vec::new();
        let mut buffers = Vec::new();
        let top = match &desc.input {
            api::AccelerationInput::Tlas(instances) => {
                if instances.is_empty() {
                    return Err(Error::InvalidRange);
                }
                counts.push(u32::try_from(instances.len()).map_err(|_| Error::InvalidRange)?);
                let data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_address,
                    });
                geometries.push(
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data }),
                );
                true
            }
            api::AccelerationInput::Blas(items) => {
                if items.is_empty() {
                    return Err(Error::InvalidRange);
                }
                for item in items {
                    match item {
                        api::AccelerationGeometry::Triangles {
                            vertices,
                            vertex_offset,
                            vertex_count,
                            vertex_format,
                            stride,
                            indices,
                            opaque,
                        } => {
                            let (format, size, alignment) = match vertex_format {
                                api::VertexFormat::Float32(3) => {
                                    (vk::Format::R32G32B32_SFLOAT, 12, 4)
                                }
                                api::VertexFormat::Float32(2) => (vk::Format::R32G32_SFLOAT, 8, 4),
                                api::VertexFormat::Float16(4) => {
                                    (vk::Format::R16G16B16A16_SFLOAT, 8, 2)
                                }
                                api::VertexFormat::Float16(2) => (vk::Format::R16G16_SFLOAT, 4, 2),
                                _ => return Err(Error::Unsupported("Vulkan ray vertex format")),
                            };
                            // SAFETY: Core checked all geometry resource device identities.
                            let vb = unsafe { vertices.native_ref() };
                            // SAFETY: the instance and physical device are retained by the buffer.
                            let properties = unsafe {
                                vb.0.device
                                    .instance
                                    .instance
                                    .get_physical_device_format_properties(
                                        vb.0.device.physical,
                                        format,
                                    )
                            };
                            if !properties.buffer_features.contains(
                                vk::FormatFeatureFlags::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR,
                            ) {
                                return Err(Error::Unsupported("Vulkan ray vertex format"));
                            }
                            let vertex_address = checked_address(
                                vb,
                                *vertex_offset,
                                *vertex_count,
                                *stride,
                                size,
                                alignment,
                            )?;
                            buffers.push(vb.clone());
                            let (itype, index_address, n) = if let Some(indices) = indices {
                                let (itype, width) = match indices.format {
                                    api::IndexFormat::Uint16 => (vk::IndexType::UINT16, 2),
                                    api::IndexFormat::Uint32 => (vk::IndexType::UINT32, 4),
                                };
                                // SAFETY: Core checked the index buffer device identity.
                                let ib = unsafe { indices.buffer.native_ref() };
                                let address = checked_address(
                                    ib,
                                    indices.offset,
                                    indices.count,
                                    width,
                                    width,
                                    width,
                                )?;
                                buffers.push(ib.clone());
                                (itype, address, indices.count)
                            } else {
                                (vk::IndexType::NONE_KHR, 0, *vertex_count)
                            };
                            if n % 3 != 0 {
                                return Err(Error::InvalidRange);
                            }
                            counts.push(n / 3);
                            let triangles =
                                vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                                    .vertex_format(format)
                                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                        device_address: vertex_address,
                                    })
                                    .vertex_stride(*stride)
                                    .max_vertex(vertex_count - 1)
                                    .index_type(itype)
                                    .index_data(vk::DeviceOrHostAddressConstKHR {
                                        device_address: index_address,
                                    });
                            geometries.push(
                                vk::AccelerationStructureGeometryKHR::default()
                                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                                        triangles,
                                    })
                                    .flags(if *opaque {
                                        vk::GeometryFlagsKHR::OPAQUE
                                    } else {
                                        vk::GeometryFlagsKHR::empty()
                                    }),
                            );
                        }
                        api::AccelerationGeometry::Aabbs {
                            data,
                            offset,
                            stride,
                            count,
                            opaque,
                        } => {
                            // SAFETY: Core checked this buffer's device identity.
                            let buffer = unsafe { data.native_ref() };
                            let address = checked_address(buffer, *offset, *count, *stride, 24, 8)?;
                            buffers.push(buffer.clone());
                            counts.push(*count);
                            let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                                .data(vk::DeviceOrHostAddressConstKHR {
                                    device_address: address,
                                })
                                .stride(*stride);
                            geometries.push(
                                vk::AccelerationStructureGeometryKHR::default()
                                    .geometry_type(vk::GeometryTypeKHR::AABBS)
                                    .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
                                    .flags(if *opaque {
                                        vk::GeometryFlagsKHR::OPAQUE
                                    } else {
                                        vk::GeometryFlagsKHR::empty()
                                    }),
                            );
                        }
                    }
                }
                false
            }
        };
        let mut flags = if desc.fast_trace {
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        } else {
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD
        };
        if desc.allow_update {
            flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
        }
        Ok(Self {
            geometries,
            counts,
            buffers,
            top,
            flags,
        })
    }
    fn info(&self) -> vk::AccelerationStructureBuildGeometryInfoKHR<'_> {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(if self.top {
                vk::AccelerationStructureTypeKHR::TOP_LEVEL
            } else {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            })
            .flags(self.flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&self.geometries)
    }
}
pub(crate) fn sizes(
    device: &VulkanDevice,
    desc: &api::AccelerationDesc<Vulkan>,
) -> Result<api::AccelerationSizes> {
    let loader = &device
        .0
        .ray_loaders
        .as_ref()
        .ok_or(Error::Unsupported("Vulkan acceleration structures"))?
        .acceleration;
    let input = Inputs::new(desc, 0)?;
    let support = device.0.ray_support;
    let total = input
        .counts
        .iter()
        .try_fold(0u64, |sum, &count| sum.checked_add(u64::from(count)))
        .ok_or(Error::InvalidRange)?;
    if (input.top && total > support.max_instance_count)
        || (!input.top
            && (input.geometries.len() as u64 > support.max_geometry_count
                || total > support.max_primitive_count))
        || support.scratch_alignment == 0
    {
        return Err(Error::InvalidRange);
    }
    let mut result = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: geometry descriptors and primitive counts have matching lengths and are valid during this call.
    unsafe {
        loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &input.info(),
            &input.counts,
            &mut result,
        )
    };
    if result.acceleration_structure_size == 0 || result.build_scratch_size == 0 {
        return Err(Error::Unsupported("Vulkan acceleration sizes"));
    }
    Ok(api::AccelerationSizes {
        storage_bytes: result.acceleration_structure_size,
        scratch_bytes: result.build_scratch_size.max(result.update_scratch_size),
        scratch_alignment: u64::from(device.0.ray_support.scratch_alignment),
    })
}
pub(crate) fn create(
    device: &VulkanDevice,
    desc: &api::AccelerationDesc<Vulkan>,
) -> Result<VulkanAcceleration> {
    let size = sizes(device, desc)?.storage_bytes;
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: device is live and buffer descriptor is fully initialized.
    let buffer = unsafe { device.0.device.create_buffer(&info, None) }.map_err(vk_error)?;
    // SAFETY: the buffer is live and exclusively owned here.
    let requirements = unsafe { device.0.device.get_buffer_memory_requirements(buffer) };
    let memory = resource::allocate_with_address(
        device,
        requirements,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
    .inspect_err(|_| unsafe { device.0.device.destroy_buffer(buffer, None) })?;
    // SAFETY: allocation satisfies buffer requirements and both handles are private.
    if let Err(e) = unsafe { device.0.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.0.device.destroy_buffer(buffer, None);
            device.0.device.free_memory(memory, None);
        }
        return Err(vk_error(e));
    }
    let top = matches!(desc.input, api::AccelerationInput::Tlas(_));
    let info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer)
        .size(size)
        .ty(if top {
            vk::AccelerationStructureTypeKHR::TOP_LEVEL
        } else {
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
        });
    // SAFETY: owned backing buffer is bound, sized and survives the structure.
    let native = match unsafe {
        device
            .0
            .ray_loaders
            .as_ref()
            .unwrap()
            .acceleration
            .create_acceleration_structure(&info, None)
    } {
        Ok(v) => v,
        Err(e) => {
            unsafe {
                device.0.device.destroy_buffer(buffer, None);
                device.0.device.free_memory(memory, None);
            }
            return Err(vk_error(e));
        }
    };
    Ok(VulkanAcceleration(Arc::new(AccelerationState {
        device: device.0.clone(),
        native,
        buffer,
        memory,
        size,
        top,
        children: Mutex::new(Vec::new()),
    })))
}
pub(crate) fn build(
    encoder: &mut VulkanEncoder,
    dst: &VulkanAcceleration,
    scratch: &VulkanBuffer,
    desc: &api::AccelerationDesc<Vulkan>,
) -> Result<()> {
    if encoder.compute || encoder.raster {
        return Err(Error::InvalidState);
    }
    let sizes = sizes(&encoder.owner, desc)?;
    if !scratch.0.usage.contains(api::BufferUsage::ACCEL_SCRATCH)
        || scratch.0.size < sizes.scratch_bytes
        || dst.0.size < sizes.storage_bytes
        || dst.0.top != matches!(desc.input, api::AccelerationInput::Tlas(_))
    {
        return Err(Error::InvalidRange);
    }
    let scratch_address = scratch.device_address()?;
    if sizes.scratch_alignment == 0 || scratch_address % sizes.scratch_alignment != 0 {
        return Err(Error::InvalidRange);
    }
    let mut instance_address = 0;
    let mut children = Vec::new();
    if let api::AccelerationInput::Tlas(instances) = &desc.input {
        let mut data = Vec::with_capacity(instances.len());
        for instance in instances {
            if instance.custom_index > 0x00ff_ffff || instance.shader_table_offset > 0x00ff_ffff {
                return Err(Error::InvalidRange);
            }
            // SAFETY: Core validated the acceleration structure's device identity.
            let blas = unsafe { instance.blas.native_ref() };
            if blas.0.top {
                return Err(Error::InvalidState);
            }
            let mut flags = vk::GeometryInstanceFlagsKHR::empty();
            for (a, b) in [
                (
                    api::AccelerationInstanceFlags::TRIANGLE_CULL_DISABLE,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
                ),
                (
                    api::AccelerationInstanceFlags::TRIANGLE_FRONT_COUNTER_CLOCKWISE,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FRONT_COUNTERCLOCKWISE,
                ),
                (
                    api::AccelerationInstanceFlags::FORCE_OPAQUE,
                    vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE,
                ),
                (
                    api::AccelerationInstanceFlags::FORCE_NON_OPAQUE,
                    vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE,
                ),
            ] {
                if instance.flags.contains(a) {
                    flags |= b;
                }
            }
            let mut transform = [0.; 12];
            for (i, row) in instance.transform.iter().enumerate() {
                transform[i * 4..i * 4 + 4].copy_from_slice(row);
            }
            data.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: transform },
                instance_custom_index_and_mask: vk::Packed24_8::new(
                    instance.custom_index,
                    instance.mask,
                ),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    instance.shader_table_offset,
                    flags.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas.address(),
                },
            });
            children.push(blas.clone());
        }
        let length = data
            .len()
            .checked_mul(std::mem::size_of::<vk::AccelerationStructureInstanceKHR>())
            .ok_or(Error::InvalidRange)?;
        let upload = resource::create_buffer(
            &encoder.owner,
            &api::BufferDesc {
                label: None,
                size: length as u64,
                usage: api::BufferUsage::ACCEL_BUILD_INPUT,
                memory: api::MemoryClass::Upload,
            },
        )?;
        // SAFETY: this fresh HOST_COHERENT allocation is mapped exclusively;
        // the native instance slice is fully initialized and fits the allocation.
        unsafe {
            let mapped = encoder
                .device
                .device
                .map_memory(
                    upload.0.memory,
                    0,
                    length as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(vk_error)?;
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), mapped.cast::<u8>(), length);
            encoder.device.device.unmap_memory(upload.0.memory);
        }
        instance_address = upload.device_address()?;
        // Host writes finish before queue submission. A HOST->AS barrier makes
        // them available to the build without exposing staging to the caller.
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        // SAFETY: command buffer is recording and barrier has valid scopes.
        unsafe {
            encoder.device.device.cmd_pipeline_barrier(
                encoder.command,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
        encoder.uploads.push(upload);
    }
    let input = Inputs::new(desc, instance_address)?;
    let info = input
        .info()
        .dst_acceleration_structure(dst.0.native)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_address,
        });
    let ranges: Vec<_> = input
        .counts
        .iter()
        .map(|&count| vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count))
        .collect();
    let loader = &encoder
        .device
        .ray_loaders
        .as_ref()
        .ok_or(Error::Unsupported("Vulkan acceleration structures"))?
        .acceleration;
    if !children.is_empty() {
        let mut retained =
            dst.0.children.lock().map_err(|_| {
                Error::Backend("Vulkan acceleration ownership lock poisoned".into())
            })?;
        retained.extend(children.iter().cloned());
    }
    // SAFETY: all geometry addresses and scratch ranges were validated; the
    // caller supplies the required preceding resource transitions. The encoder
    // retains every native input and destination through GPU completion.
    unsafe { loader.cmd_build_acceleration_structures(encoder.command, &[info], &[&ranges]) };
    encoder.buffers.extend(input.buffers);
    encoder.buffers.push(scratch.clone());
    encoder.accelerations.extend(children);
    encoder.accelerations.push(dst.clone());
    Ok(())
}
