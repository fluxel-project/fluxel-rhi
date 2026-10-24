//! Metal acceleration structure construction. Pipeline-style tracing is unavailable.
//!
//! Native objects and their inputs are retained by Metal command buffers using
//! retained references. A TLAS keeps only the BLAS set accepted by its most
//! recent completed build; each in-flight native build owns its own snapshot.
use super::{Metal, MetalBuffer, MetalCommandEncoder, MetalDevice};
use fluxel_rhi_core::api::{self, RayOps};
use std::future::{Ready, ready};

#[cfg(target_vendor = "apple")]
mod vertex;
#[cfg(target_vendor = "apple")]
use vertex::acceleration_vertex_format;

#[cfg(any(target_vendor = "apple", test))]
mod dependencies;
#[cfg(any(target_vendor = "apple", test))]
pub(super) use dependencies::{BuildCompletion, BuildReservation, FrozenDependencies};

pub struct MetalRay;
#[derive(Clone)]
pub struct MetalAcceleration {
    #[cfg(target_vendor = "apple")]
    pub(crate) raw: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLAccelerationStructure>,
    >,
    #[cfg(target_vendor = "apple")]
    device: MetalDevice,
    #[cfg(target_vendor = "apple")]
    storage_bytes: u64,
    #[cfg(target_vendor = "apple")]
    is_tlas: bool,
    #[cfg(target_vendor = "apple")]
    dependencies: std::sync::Arc<FrozenDependencies<MetalAcceleration>>,
}

#[cfg(target_vendor = "apple")]
impl MetalAcceleration {
    pub(crate) fn declare_dependencies(
        &self,
        compute: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        owner: u64,
    ) -> api::Result<()> {
        use objc2_metal::{MTLComputeCommandEncoder as _, MTLResource, MTLResourceUsage};
        if !self.is_tlas {
            return Err(api::Error::InvalidState);
        }
        self.dependencies.with_snapshot(owner, |dependencies| {
            for dependency in dependencies {
                let resource =
                    objc2::runtime::ProtocolObject::<dyn MTLResource>::from_ref(&*dependency.raw);
                compute.useResource_usage(resource, MTLResourceUsage::Read);
            }
        })
    }
    pub(crate) fn declare_render_dependencies(
        &self,
        render: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        owner: u64,
        stages: objc2_metal::MTLRenderStages,
    ) -> api::Result<()> {
        use objc2_metal::{MTLRenderCommandEncoder as _, MTLResource, MTLResourceUsage};
        if !self.is_tlas {
            return Err(api::Error::InvalidState);
        }
        self.dependencies.with_snapshot(owner, |dependencies| {
            for dependency in dependencies {
                let resource =
                    objc2::runtime::ProtocolObject::<dyn MTLResource>::from_ref(&*dependency.raw);
                render.useResource_usage_stages(resource, MTLResourceUsage::Read, stages);
            }
        })
    }
    pub(crate) fn validate_binding(&self, owner: u64) -> api::Result<()> {
        if !self.is_tlas {
            return Err(api::Error::InvalidState);
        }
        self.dependencies.with_snapshot(owner, |_| ())
    }

    /// Snapshot per-instance shader table offsets for the matching TLAS build.
    /// The caller retains this immutable buffer for every command using it.
    pub(crate) fn query_contributions(
        &self,
        owner: u64,
    ) -> api::Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>
    {
        use objc2_metal::{MTLBuffer as _, MTLDevice as _, MTLResourceOptions};
        if !self.is_tlas {
            return Err(api::Error::InvalidState);
        }
        self.dependencies.with_contributions(owner, |values| {
            let size = values
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or(api::Error::InvalidRange)?
                .max(4);
            let buffer = self
                .device
                ._state
                ._device
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .ok_or(api::Error::OutOfMemory)?;
            // SAFETY: the newly allocated shared buffer has room for the complete snapshot.
            unsafe {
                std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, size);
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents().as_ptr().cast::<u32>(),
                    values.len(),
                );
            }
            Ok(buffer)
        })?
    }

    /// Pack each acceleration structure's contribution snapshot after a table
    /// of u32 element offsets, preserving the order of the binding array.
    pub(crate) fn query_contribution_array(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        structures: &[Option<Self>],
        capacity: usize,
        owner: u64,
    ) -> api::Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>
    {
        use objc2_metal::{MTLBuffer as _, MTLDevice as _, MTLResourceOptions};
        if capacity == 0 || structures.len() > capacity {
            return Err(api::Error::InvalidRange);
        }
        let mut packed = vec![0u32; capacity];
        for (index, structure) in structures.iter().enumerate() {
            let Some(structure) = structure else {
                continue;
            };
            if !structure.is_tlas || !std::ptr::eq(device, &*structure.device._state._device) {
                return Err(api::Error::InvalidState);
            }
            structure
                .dependencies
                .with_contributions(owner, |values| {
                    packed[index] =
                        u32::try_from(packed.len()).map_err(|_| api::Error::InvalidRange)?;
                    packed.extend_from_slice(values);
                    if values.is_empty() {
                        packed.push(0);
                    }
                    Ok::<_, api::Error>(())
                })??;
        }
        let size = packed
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(api::Error::InvalidRange)?;
        let buffer = device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .ok_or(api::Error::OutOfMemory)?;
        // SAFETY: this newly created shared allocation covers every packed u32.
        unsafe {
            std::ptr::copy_nonoverlapping(
                packed.as_ptr(),
                buffer.contents().as_ptr().cast::<u32>(),
                packed.len(),
            );
        }
        Ok(buffer)
    }
}

impl RayOps<Metal> for MetalRay {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = Ready<api::Result<()>>;
    fn create_pipeline<'a>(
        _: &'a MetalDevice,
        _: &'a api::RayPipelineDesc<Metal>,
    ) -> Self::PipelineFuture<'a> {
        ready(Err(api::Error::Unsupported("Metal ray pipeline")))
    }
    fn set_pipeline(_: &mut MetalCommandEncoder, _: &()) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal ray pipeline"))
    }
    fn set_bind_group(
        _: &mut MetalCommandEncoder,
        _: u32,
        _: &super::MetalBindGroup,
        _: &[u32],
    ) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal ray pipeline"))
    }
    fn shader_table_properties(_: &MetalDevice) -> api::Result<api::ShaderTableProperties> {
        Err(api::Error::Unsupported("Metal ray pipeline"))
    }
    fn shader_group_identifier(_: &MetalDevice, _: &(), _: u32) -> api::Result<Vec<u8>> {
        Err(api::Error::Unsupported("Metal ray pipeline"))
    }
    fn acceleration_sizes(
        device: &MetalDevice,
        desc: &api::AccelerationDesc<Metal>,
    ) -> api::Result<api::AccelerationSizes> {
        sizes(device, desc)
    }
    fn create_acceleration(
        device: &MetalDevice,
        desc: &api::AccelerationDesc<Metal>,
    ) -> api::Result<MetalAcceleration> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::MTLDevice as _;
            let sizes = sizes(device, desc)?;
            let raw = device
                ._state
                ._device
                .newAccelerationStructureWithSize(sizes.storage_bytes as usize)
                .ok_or(api::Error::OutOfMemory)?;
            Ok(MetalAcceleration {
                raw,
                device: device.clone(),
                storage_bytes: sizes.storage_bytes,
                is_tlas: matches!(desc.input, api::AccelerationInput::Tlas(_)),
                dependencies: Default::default(),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, desc);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn build_acceleration(
        encoder: &mut MetalCommandEncoder,
        dst: &MetalAcceleration,
        scratch: &MetalBuffer,
        input: &api::AccelerationDesc<Metal>,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{
                MTLAccelerationStructureCommandEncoder as _, MTLCommandBuffer as _,
                MTLCommandEncoder as _,
            };
            if encoder.render.is_some() || encoder.compute.is_some() {
                return Err(api::Error::InvalidState);
            }
            let requirements = sizes(&dst.device, input)?;
            if requirements.storage_bytes > dst.storage_bytes
                || scratch.size() < requirements.scratch_bytes
                || !scratch.usage.contains(api::BufferUsage::ACCEL_SCRATCH)
            {
                return Err(api::Error::InvalidRange);
            }
            if dst.is_tlas != matches!(input.input, api::AccelerationInput::Tlas(_)) {
                return Err(api::Error::InvalidState);
            }
            let (descriptor, upload, dependencies) = native_descriptor(&dst.device, input)?;
            let native = encoder
                .command
                .accelerationStructureCommandEncoder()
                .ok_or(api::Error::OutOfMemory)?;
            if dst.is_tlas {
                let snapshot = dependencies.clone();
                let api::AccelerationInput::Tlas(instances) = &input.input else {
                    unreachable!()
                };
                let contributions: Vec<u32> = instances
                    .iter()
                    .map(|instance| instance.shader_table_offset)
                    .collect();
                if let Err(error) =
                    dst.dependencies
                        .freeze(encoder.ray_owner, dependencies, contributions.clone())
                {
                    native.endEncoding();
                    return Err(error);
                }
                encoder
                    .ray_builds
                    .push(std::sync::Arc::new(BuildReservation::new(
                        encoder.ray_owner,
                        dst.dependencies.clone(),
                        snapshot,
                        contributions,
                    )));
            }
            native.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                &dst.raw,
                &descriptor,
                &scratch.raw,
                0,
            );
            native.endEncoding();
            if let Some(upload) = upload {
                encoder.indirect_arguments.push(upload);
            }
            encoder.accelerations.push(dst.clone());
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, dst, scratch, input);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn trace(
        _: &mut MetalCommandEncoder,
        _: &(),
        _: &api::ShaderTable<Metal>,
        _: [u32; 3],
    ) -> api::Result<()> {
        Err(api::Error::Unsupported("Metal ray pipeline"))
    }
}

fn sizes(
    device: &MetalDevice,
    desc: &api::AccelerationDesc<Metal>,
) -> api::Result<api::AccelerationSizes> {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_metal::MTLDevice as _;
        if !device._state.ray_supported {
            return Err(api::Error::Unsupported("Metal ray tracing hardware"));
        }
        let (descriptor, _, _) = native_descriptor(device, desc)?;
        let sizes = device
            ._state
            ._device
            .accelerationStructureSizesWithDescriptor(&descriptor);
        Ok(api::AccelerationSizes {
            storage_bytes: sizes.accelerationStructureSize as u64,
            scratch_bytes: sizes.buildScratchBufferSize as u64,
            scratch_alignment: 256,
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, desc);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(target_vendor = "apple")]
type NativeDescriptor = (
    objc2::rc::Retained<objc2_metal::MTLAccelerationStructureDescriptor>,
    Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    Vec<MetalAcceleration>,
);

#[cfg(target_vendor = "apple")]
fn native_descriptor(
    device: &MetalDevice,
    desc: &api::AccelerationDesc<Metal>,
) -> api::Result<NativeDescriptor> {
    use objc2_foundation::NSArray;
    use objc2_metal::{
        MTLAccelerationStructureInstanceDescriptorType as DescriptorType,
        MTLAccelerationStructureInstanceOptions as Options,
        MTLAccelerationStructureUserIDInstanceDescriptor as Instance, MTLDevice as _,
        MTLInstanceAccelerationStructureDescriptor, MTLPackedFloat3, MTLPackedFloat4x3,
        MTLResourceOptions,
    };
    let api::AccelerationInput::Tlas(instances) = &desc.input else {
        return Ok((
            primitive_descriptor(device, desc)?.into_super(),
            None,
            Vec::new(),
        ));
    };
    let mut records = Vec::with_capacity(instances.len());
    let mut dependencies = Vec::with_capacity(instances.len());
    for (i, instance) in instances.iter().enumerate() {
        // SAFETY: Core checked the BLAS device identity before invoking the SPI.
        let blas = unsafe { instance.blas.native_ref() };
        if blas.is_tlas {
            return Err(api::Error::InvalidState);
        }
        let matrix = instance.transform;
        let columns = std::array::from_fn(|col| MTLPackedFloat3 {
            x: matrix[0][col],
            y: matrix[1][col],
            z: matrix[2][col],
        });
        let f = instance.flags;
        let mut options = Options::None;
        if f.contains(api::AccelerationInstanceFlags::TRIANGLE_CULL_DISABLE) {
            options |= Options::DisableTriangleCulling;
        }
        if f.contains(api::AccelerationInstanceFlags::TRIANGLE_FRONT_COUNTER_CLOCKWISE) {
            options |= Options::TriangleFrontFacingWindingCounterClockwise;
        }
        if f.contains(api::AccelerationInstanceFlags::FORCE_OPAQUE) {
            options |= Options::Opaque;
        }
        if f.contains(api::AccelerationInstanceFlags::FORCE_NON_OPAQUE) {
            options |= Options::NonOpaque;
        }
        records.push(Instance {
            transformationMatrix: MTLPackedFloat4x3 { columns },
            options,
            mask: u32::from(instance.mask),
            intersectionFunctionTableOffset: instance.shader_table_offset,
            accelerationStructureIndex: u32::try_from(i).map_err(|_| api::Error::InvalidRange)?,
            userID: instance.custom_index,
        });
        dependencies.push(blas.clone());
    }
    let bytes = records
        .len()
        .checked_mul(std::mem::size_of::<Instance>())
        .ok_or(api::Error::InvalidRange)?;
    let upload = if records.is_empty() {
        None
    } else {
        // SAFETY: records is nonempty and remains readable for the synchronous copy.
        Some(
            unsafe {
                device._state._device.newBufferWithBytes_length_options(
                    std::ptr::NonNull::new(records.as_mut_ptr().cast())
                        .ok_or(api::Error::InvalidRange)?,
                    bytes,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or(api::Error::OutOfMemory)?,
        )
    };
    let refs: Vec<_> = dependencies.iter().map(|d| &*d.raw).collect();
    let array = NSArray::from_slice(&refs);
    let descriptor = MTLInstanceAccelerationStructureDescriptor::descriptor();
    descriptor.setUsage(acceleration_usage(desc));
    descriptor.setInstanceDescriptorType(DescriptorType::UserID);
    descriptor.setInstanceDescriptorBuffer(upload.as_deref());
    descriptor.setInstanceCount(records.len());
    descriptor.setInstancedAccelerationStructures(Some(&array));
    Ok((descriptor.into_super(), upload, dependencies))
}

#[cfg(target_vendor = "apple")]
fn primitive_descriptor(
    device: &MetalDevice,
    desc: &api::AccelerationDesc<Metal>,
) -> api::Result<objc2::rc::Retained<objc2_metal::MTLPrimitiveAccelerationStructureDescriptor>> {
    use objc2_foundation::NSArray;
    use objc2_metal::{
        MTLAccelerationStructureBoundingBoxGeometryDescriptor,
        MTLAccelerationStructureGeometryDescriptor,
        MTLAccelerationStructureTriangleGeometryDescriptor, MTLIndexType,
        MTLPrimitiveAccelerationStructureDescriptor,
    };
    let api::AccelerationInput::Blas(geometries) = &desc.input else {
        return Err(api::Error::Unsupported(
            "Metal instance acceleration structure",
        ));
    };
    let mut native = Vec::with_capacity(geometries.len());
    for geometry in geometries {
        if let api::AccelerationGeometry::Aabbs {
            data,
            offset,
            stride,
            count,
            opaque,
        } = geometry
        {
            // SAFETY: Core validates that this buffer belongs to the device.
            let buffer = unsafe { data.native_ref() };
            if !buffer.usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT)
                || !valid_aabb_range(buffer.size(), *offset, *stride, *count)
            {
                return Err(api::Error::InvalidRange);
            }
            let geom = MTLAccelerationStructureBoundingBoxGeometryDescriptor::descriptor();
            geom.setBoundingBoxBuffer(Some(&buffer.raw));
            // SAFETY: the buffer range, offset and stride were validated above.
            unsafe {
                geom.setBoundingBoxBufferOffset(*offset as usize);
                geom.setBoundingBoxStride(*stride as usize);
            }
            geom.setBoundingBoxCount(*count as usize);
            geom.setOpaque(*opaque);
            native.push(geom.into_super());
            continue;
        }
        let api::AccelerationGeometry::Triangles {
            vertices,
            vertex_offset,
            vertex_count,
            vertex_format,
            stride,
            indices,
            opaque,
        } = geometry
        else {
            unreachable!("all acceleration geometry variants are handled");
        };
        let (format, format_size, component_alignment) =
            acceleration_vertex_format(*vertex_format)?;
        // SAFETY: Core validates the device identity and the descriptor owns the buffer.
        let vertex = unsafe { vertices.native_ref() };
        if !vertex.usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT)
            || *vertex_count == 0
            || *stride < format_size
            || stride % component_alignment != 0
            || vertex_offset % stride != 0
            || u64::from(vertex_count.saturating_sub(1))
                .checked_mul(*stride)
                .and_then(|v| vertex_offset.checked_add(v))
                .and_then(|v| v.checked_add(format_size))
                .is_none_or(|end| end > vertex.size())
        {
            return Err(api::Error::InvalidRange);
        }
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
        use objc2::runtime::NSObjectProtocol as _;
        use objc2_metal::MTLDevice as _;
        let extended_formats = device
            ._state
            ._device
            .respondsToSelector(objc2::sel!(supportsFamily:))
            && device
                ._state
                ._device
                .supportsFamily(objc2_metal::MTLGPUFamily::Metal3)
            && geom.respondsToSelector(objc2::sel!(setVertexFormat:));
        if extended_formats {
            geom.setVertexFormat(format);
        } else if !matches!(vertex_format, api::VertexFormat::Float32(3)) {
            return Err(api::Error::Unsupported(
                "Metal runtime lacks acceleration vertex formats",
            ));
        }
        geom.setVertexBuffer(Some(&vertex.raw));
        // SAFETY: range and alignment checked above.
        unsafe {
            geom.setVertexBufferOffset(*vertex_offset as usize);
        }
        geom.setVertexStride(*stride as usize);
        geom.setOpaque(*opaque);
        let count = if let Some(indices) = indices {
            // SAFETY: Core validates the index buffer's device identity.
            let index = unsafe { indices.buffer.native_ref() };
            let (width, ty) = match indices.format {
                api::IndexFormat::Uint16 => (2, MTLIndexType::UInt16),
                api::IndexFormat::Uint32 => (4, MTLIndexType::UInt32),
            };
            if !index.usage.contains(api::BufferUsage::ACCEL_BUILD_INPUT)
                || indices.offset % width != 0
                || indices
                    .offset
                    .checked_add(u64::from(indices.count) * width)
                    .is_none_or(|end| end > index.size())
            {
                return Err(api::Error::InvalidRange);
            }
            geom.setIndexBuffer(Some(&index.raw));
            // SAFETY: Core validated alignment and the index range was checked above.
            unsafe {
                geom.setIndexBufferOffset(indices.offset as usize);
            }
            geom.setIndexType(ty);
            indices.count
        } else {
            *vertex_count
        };
        if count == 0 || count % 3 != 0 {
            return Err(api::Error::InvalidRange);
        }
        geom.setTriangleCount((count / 3) as usize);
        native.push(geom.into_super());
    }
    let array = NSArray::<MTLAccelerationStructureGeometryDescriptor>::from_retained_slice(&native);
    let descriptor = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    descriptor.setUsage(acceleration_usage(desc));
    descriptor.setGeometryDescriptors(Some(&array));
    Ok(descriptor)
}

#[cfg(target_vendor = "apple")]
fn acceleration_usage(
    desc: &api::AccelerationDesc<Metal>,
) -> objc2_metal::MTLAccelerationStructureUsage {
    let mut usage = objc2_metal::MTLAccelerationStructureUsage::None;
    if desc.allow_update {
        usage |= objc2_metal::MTLAccelerationStructureUsage::Refit;
    }
    if !desc.fast_trace {
        usage |= objc2_metal::MTLAccelerationStructureUsage::PreferFastBuild;
    }
    usage
}

#[cfg(any(target_vendor = "apple", test))]
fn valid_aabb_range(size: u64, offset: u64, stride: u64, count: u32) -> bool {
    count != 0
        && stride >= 24
        && stride % 4 == 0
        && offset % stride == 0
        && u64::from(count - 1)
            .checked_mul(stride)
            .and_then(|v| offset.checked_add(v))
            .and_then(|v| v.checked_add(24))
            .is_some_and(|end| end <= size)
}

#[cfg(test)]
mod tests {
    use super::valid_aabb_range;

    #[test]
    fn bounding_boxes_require_valid_native_alignment_and_complete_buffer_range() {
        assert!(valid_aabb_range(88, 32, 32, 2));
        assert!(valid_aabb_range(56, 0, 28, 2));
        assert!(!valid_aabb_range(71, 24, 24, 2));
        assert!(!valid_aabb_range(80, 8, 24, 2));
        assert!(!valid_aabb_range(u64::MAX, u64::MAX - 7, 24, u32::MAX));
        assert!(!valid_aabb_range(80, 0, 24, 0));
    }
}
