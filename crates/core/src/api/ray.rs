//! Optional ray-tracing pipeline and acceleration-structure contract.
//!
//! This module carries device-local operations only. Scene scheduling and
//! memory residency decisions remain renderer or backend concerns.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;

mod compaction;
pub use compaction::AccelerationCompactionTicket;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone, Debug)]
pub struct AccelerationSizes {
    pub storage_bytes: u64,
    pub scratch_bytes: u64,
    pub scratch_alignment: u64,
}
/// Buffer layout for triangle positions in an acceleration build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccelerationVertexLayout {
    pub position_bytes: u32,
    pub offset_alignment: u32,
    pub stride_alignment: u32,
    /// Exact byte stride when this position encoding has a fixed stride.
    pub required_stride: Option<u32>,
}
#[derive(Clone)]
pub enum AccelerationGeometry<B: Backend> {
    Triangles {
        vertices: Buffer<B>,
        vertex_offset: u64,
        vertex_count: u32,
        vertex_format: VertexFormat,
        stride: u64,
        indices: Option<AccelerationIndices<B>>,
        opaque: bool,
    },
    /// Triangles with a geometry-to-BLAS transform. At `transform_offset`,
    /// the build-input buffer stores 12 `f32` values in three row-major rows.
    TransformedTriangles {
        vertices: Buffer<B>,
        vertex_offset: u64,
        vertex_count: u32,
        vertex_format: VertexFormat,
        stride: u64,
        indices: Option<AccelerationIndices<B>>,
        opaque: bool,
        transform: Buffer<B>,
        transform_offset: u64,
    },
    Aabbs {
        data: Buffer<B>,
        offset: u64,
        stride: u64,
        count: u32,
        opaque: bool,
    },
}
/// Index data for a triangle geometry. Offsets are in bytes.
#[derive(Clone)]
pub struct AccelerationIndices<B: Backend> {
    pub buffer: Buffer<B>,
    pub offset: u64,
    pub count: u32,
    pub format: IndexFormat,
}
/// Per-instance ray traversal behavior, independent of native API bit values.
///
/// With no flags, the instance uses the default clockwise triangle front face,
/// permits face culling, and preserves each geometry's opaque setting. Forced
/// opacity applies to all geometries in the instance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AccelerationInstanceFlags(pub u32);
impl AccelerationInstanceFlags {
    pub const NONE: Self = Self(0);
    /// Disable triangle face culling for this instance.
    pub const TRIANGLE_CULL_DISABLE: Self = Self(1 << 0);
    /// Treat counterclockwise triangles as front facing for this instance.
    pub const TRIANGLE_FRONT_COUNTER_CLOCKWISE: Self = Self(1 << 1);
    /// Treat all instance geometries as opaque during traversal.
    pub const FORCE_OPAQUE: Self = Self(1 << 2);
    /// Treat all instance geometries as nonopaque during traversal.
    pub const FORCE_NON_OPAQUE: Self = Self(1 << 3);
    const KNOWN: u32 = Self::TRIANGLE_CULL_DISABLE.0
        | Self::TRIANGLE_FRONT_COUNTER_CLOCKWISE.0
        | Self::FORCE_OPAQUE.0
        | Self::FORCE_NON_OPAQUE.0;
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    fn valid(self) -> bool {
        self.0 & !Self::KNOWN == 0
            && !(self.contains(Self::FORCE_OPAQUE) && self.contains(Self::FORCE_NON_OPAQUE))
    }
}
impl std::ops::BitOr for AccelerationInstanceFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for AccelerationInstanceFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
#[derive(Clone)]
pub struct AccelerationInstance<B: Backend> {
    pub blas: AccelerationStructure<B>,
    pub transform: [[f32; 4]; 3],
    pub mask: u8,
    pub custom_index: u32,
    pub shader_table_offset: u32,
    pub flags: AccelerationInstanceFlags,
}
#[derive(Clone)]
pub enum AccelerationInput<B: Backend> {
    Blas(Vec<AccelerationGeometry<B>>),
    Tlas(Vec<AccelerationInstance<B>>),
}
impl<B: Backend> AccelerationInput<B> {
    fn validate(&self, device: DeviceId, limits: &DeviceLimits) -> Result<()> {
        let foreign = match self {
            Self::Blas(geometries) => geometries.iter().any(|geometry| match geometry {
                AccelerationGeometry::Triangles {
                    vertices, indices, ..
                } | AccelerationGeometry::TransformedTriangles { vertices, indices, .. } => {
                    vertices.device != device
                        || indices
                            .as_ref()
                            .is_some_and(|indices| indices.buffer.device != device)
                        || matches!(geometry, AccelerationGeometry::TransformedTriangles { transform, .. } if transform.device != device)
                }
                AccelerationGeometry::Aabbs { data, .. } => data.device != device,
            }),
            Self::Tlas(instances) => instances
                .iter()
                .any(|instance| instance.blas.device != device),
        };
        if foreign {
            return Err(Error::WrongDevice);
        }
        if let Self::Tlas(instances) = self {
            if limits.max_tlas_instance_count != 0
                && instances.len() > limits.max_tlas_instance_count as usize
            {
                return Err(Error::InvalidRange);
            }
            if instances.iter().any(|instance| !instance.flags.valid()) {
                return Err(Error::InvalidState);
            }
        }
        if let Self::Blas(geometries) = self {
            if limits.max_blas_geometry_count != 0
                && geometries.len() > limits.max_blas_geometry_count as usize
            {
                return Err(Error::InvalidRange);
            }
            let mut primitive_count = 0u64;
            for geometry in geometries {
                if limits.max_blas_primitive_count != 0 {
                    let count = match geometry {
                        AccelerationGeometry::Triangles {
                            vertex_count,
                            indices,
                            ..
                        }
                        | AccelerationGeometry::TransformedTriangles {
                            vertex_count,
                            indices,
                            ..
                        } => {
                            indices
                                .as_ref()
                                .map_or(*vertex_count, |indices| indices.count)
                                / 3
                        }
                        AccelerationGeometry::Aabbs { count, .. } => *count,
                    };
                    primitive_count += u64::from(count);
                    if primitive_count > u64::from(limits.max_blas_primitive_count) {
                        return Err(Error::InvalidRange);
                    }
                }
                let valid = match geometry {
                    AccelerationGeometry::Triangles {
                        vertex_offset,
                        vertex_count,
                        stride,
                        indices,
                        ..
                    }
                    | AccelerationGeometry::TransformedTriangles {
                        vertex_offset,
                        vertex_count,
                        stride,
                        indices,
                        ..
                    } => {
                        *vertex_count != 0
                            && *stride != 0
                            && u64::from(*vertex_count - 1)
                                .checked_mul(*stride)
                                .and_then(|n| vertex_offset.checked_add(n))
                                .is_some()
                            && indices.as_ref().is_none_or(|indices| {
                                let width = match indices.format {
                                    IndexFormat::Uint16 => 2,
                                    IndexFormat::Uint32 => 4,
                                };
                                indices.count != 0
                                    && indices.count % 3 == 0
                                    && indices.offset % width == 0
                                    && u64::from(indices.count)
                                        .checked_mul(width)
                                        .and_then(|n| indices.offset.checked_add(n))
                                        .is_some()
                            })
                    }
                    AccelerationGeometry::Aabbs {
                        offset,
                        stride,
                        count,
                        ..
                    } => {
                        *count != 0
                            && *stride >= 24
                            && *stride % 8 == 0
                            && *offset % 8 == 0
                            && u64::from(*count - 1)
                                .checked_mul(*stride)
                                .and_then(|n| offset.checked_add(n))
                                .and_then(|n| n.checked_add(24))
                                .is_some()
                    }
                };
                if !valid {
                    return Err(Error::InvalidRange);
                }
                if let AccelerationGeometry::TransformedTriangles {
                    transform_offset, ..
                } = geometry
                {
                    if transform_offset % 16 != 0 || transform_offset.checked_add(48).is_none() {
                        return Err(Error::InvalidRange);
                    }
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct AccelerationDesc<B: Backend> {
    pub input: AccelerationInput<B>,
    pub fast_trace: bool,
    pub allow_update: bool,
}
/// Destination for a build's native-endian `u64` byte size. The caller keeps
/// the buffer in storage-write state until the build and size write finish.
pub struct AccelerationSizeOutput<'a, B: Backend> {
    pub buffer: &'a Buffer<B>,
    pub offset: u64,
    pub kind: AccelerationSizeKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelerationSizeKind {
    Current,
    /// Requests a compaction-enabled build and its compacted byte size.
    Compacted,
}
#[derive(Clone)]
pub struct RayPipelineDesc<B: Backend> {
    pub layout: PipelineLayout<B>,
    pub shaders: Vec<PipelineStage<B>>,
    pub groups: Vec<RayGroup>,
    /// Maximum trace recursion depth. Zero on a supporting backend permits
    /// ray generation without `TraceRay` calls.
    pub max_recursion: u32,
    pub payload_bytes: u32,
    pub hit_attribute_bytes: u32,
    pub cache: Option<PipelineCache<B>>,
}
#[derive(Clone, Debug)]
pub struct RayGroup {
    /// Inline constants stored immediately after this group's identifier.
    pub local_constants: Option<RayLocalConstants>,
    pub general: Option<u32>,
    pub closest_hit: Option<u32>,
    pub any_hit: Option<u32>,
    pub intersection: Option<u32>,
}
/// One inline block of 32-bit shader constants in a shader record.
#[derive(Clone, Copy, Debug)]
pub struct RayLocalConstants {
    /// Logical uniform binding occupied by this record's constants.
    pub binding: u32,
    /// Logical bind group containing the binding.
    pub group: u32,
    pub size_bytes: u32,
}
#[derive(Clone, Debug)]
pub struct ShaderRecord {
    pub group_index: u32,
    pub local_data: Vec<u8>,
}

/// Device limits for constructing shader records and table sections.
#[derive(Clone, Copy, Debug)]
pub struct ShaderTableProperties {
    pub identifier_size: u32,
    pub record_alignment: u32,
    pub table_alignment: u32,
    pub max_record_stride: u32,
}

/// A contiguous shader table section in a device buffer, measured in bytes.
#[derive(Clone)]
pub struct ShaderTableRegion<B: Backend> {
    pub buffer: Buffer<B>,
    pub offset: u64,
    pub size: u64,
    /// Byte distance between records. On supporting backends a zero stride
    /// in a miss, hit, or callable section selects one record at every index.
    pub stride: u64,
}

/// Ray generation has one record; other sections may be absent.
#[derive(Clone)]
pub struct ShaderTable<B: Backend> {
    pub raygen: ShaderTableRegion<B>,
    pub miss: Option<ShaderTableRegion<B>>,
    pub hit: Option<ShaderTableRegion<B>>,
    pub callable: Option<ShaderTableRegion<B>>,
}

impl<B: Backend> RayOps<B> for Unavailable {
    const AVAILABLE: bool = false;
    type PipelineFuture<'a>
        = std::future::Ready<Result<B::NativeRayPipeline>>
    where
        B: 'a;
    fn create_pipeline<'a>(
        _: &'a B::NativeDevice,
        _: &'a RayPipelineDesc<B>,
    ) -> Self::PipelineFuture<'a> {
        std::future::ready(Err(Error::Unsupported("ray tracing")))
    }
    fn set_pipeline(_: &mut B::NativeEncoder, _: &B::NativeRayPipeline) -> Result<()> {
        Err(Error::Unsupported("ray tracing"))
    }
    fn set_bind_group(
        _: &mut B::NativeEncoder,
        _: u32,
        _: &B::NativeBindGroup,
        _: &[u32],
    ) -> Result<()> {
        Err(Error::Unsupported("ray tracing"))
    }
    fn shader_table_properties(_: &B::NativeDevice) -> Result<ShaderTableProperties> {
        Err(Error::Unsupported("ray tracing"))
    }
    fn shader_group_identifier(
        _: &B::NativeDevice,
        _: &B::NativeRayPipeline,
        _: u32,
    ) -> Result<Vec<u8>> {
        Err(Error::Unsupported("ray tracing"))
    }
    fn acceleration_sizes(
        _: &B::NativeDevice,
        _: &AccelerationDesc<B>,
    ) -> Result<AccelerationSizes> {
        Err(Error::Unsupported("acceleration structure"))
    }
    fn create_acceleration(
        _: &B::NativeDevice,
        _: &AccelerationDesc<B>,
    ) -> Result<B::NativeAcceleration> {
        Err(Error::Unsupported("acceleration structure"))
    }
    fn build_acceleration(
        _: &mut B::NativeEncoder,
        _: &B::NativeAcceleration,
        _: &B::NativeBuffer,
        _: &AccelerationDesc<B>,
    ) -> Result<()> {
        Err(Error::Unsupported("acceleration structure"))
    }
    fn trace(
        _: &mut B::NativeEncoder,
        _: &B::NativeRayPipeline,
        _: &ShaderTable<B>,
        _: [u32; 3],
    ) -> Result<()> {
        Err(Error::Unsupported("ray tracing"))
    }
}
pub struct RayDomain<'a, B: Backend> {
    device: &'a Device<B>,
}
impl<B: Backend> Device<B> {
    pub fn ray(&self) -> Option<RayDomain<'_, B>> {
        ((self.caps.acceleration_structure || self.caps.ray || self.caps.ray_query)
            && B::Ray::AVAILABLE)
            .then_some(RayDomain { device: self })
    }
}
impl<B: Backend> RayDomain<'_, B> {
    /// Checks the descriptor and declared shader groups before native pipeline
    /// creation. The backend validates the linked shader artifacts at creation.
    pub fn pipeline_support(&self, desc: &RayPipelineDesc<B>) -> Result<()> {
        self.validate_pipeline_desc(desc)?;
        B::Ray::pipeline_support(&self.device.native, desc)
    }

    pub async fn create_pipeline(&self, desc: RayPipelineDesc<B>) -> Result<RayPipeline<B>> {
        self.validate_pipeline_desc(&desc)?;
        let native = B::Ray::create_pipeline(&self.device.native, &desc).await?;
        // SAFETY: the SPI created this pipeline for this exact native device.
        let mut pipeline = unsafe { RayPipeline::from_native(self.device.id, native) };
        pipeline.layout = Some(desc.layout);
        Ok(pipeline)
    }

    fn validate_pipeline_desc(&self, desc: &RayPipelineDesc<B>) -> Result<()> {
        self.require_pipeline()?;
        // Shader tables refer to the executable pipeline's shader identities;
        // accepting a foreign module would defer a deterministic ownership
        // error until a backend-specific trace operation.
        if desc.layout.device != self.device.id
            || desc
                .shaders
                .iter()
                .any(|s| s.module.device != self.device.id)
            || desc
                .cache
                .as_ref()
                .is_some_and(|c| c.device != self.device.id)
        {
            return Err(Error::WrongDevice);
        }
        let max_depth = self.device.caps.limits.max_ray_recursion_depth;
        if max_depth != 0 && desc.max_recursion > max_depth {
            return Err(Error::InvalidRange);
        }
        if desc.groups.iter().any(|group| {
            group
                .local_constants
                .is_some_and(|constants| constants.size_bytes == 0 || constants.size_bytes % 4 != 0)
        }) {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }
    /// Selects a ray pipeline on an encoder outside a raster or compute pass.
    pub fn set_pipeline(
        &self,
        encoder: &mut CommandEncoder<B>,
        pipeline: &RayPipeline<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.require_pipeline()?;
        if encoder.device != self.device.id || pipeline.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = encoder.call(|e| B::Ray::set_pipeline(e, &pipeline.native));
        if result.is_ok() {
            encoder.stats.record_pipeline_bind();
        }
        result
    }
    /// Binds resources for the selected ray pipeline using the compute-style binding contract.
    pub fn set_bind_group(
        &self,
        encoder: &mut CommandEncoder<B>,
        slot: u32,
        group: &BindGroup<B>,
        offsets: &[u32],
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.require_pipeline()?;
        if encoder.device != self.device.id || group.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let result = encoder.call(|e| B::Ray::set_bind_group(e, slot, &group.native, offsets));
        if result.is_ok() {
            encoder.stats.record_bind_group_bind();
        }
        result
    }
    pub fn shader_table_properties(&self) -> Result<ShaderTableProperties> {
        self.require_pipeline()?;
        B::Ray::shader_table_properties(&self.device.native)
    }
    /// Returns the opaque identifier for a pipeline group, suitable for a record.
    pub fn shader_group_identifier(
        &self,
        pipeline: &RayPipeline<B>,
        group_index: u32,
    ) -> Result<Vec<u8>> {
        self.require_pipeline()?;
        if pipeline.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        B::Ray::shader_group_identifier(&self.device.native, &pipeline.native, group_index)
    }
    pub fn sizes(&self, desc: &AccelerationDesc<B>) -> Result<AccelerationSizes> {
        self.require_acceleration()?;
        self.validate_acceleration_input(&desc.input)?;
        B::Ray::acceleration_sizes(&self.device.native, desc)
    }
    /// Allocation and scratch bounds for a compaction-enabled build.
    pub fn compactable_sizes(&self, desc: &AccelerationDesc<B>) -> Result<AccelerationSizes> {
        self.require_compaction_size()?;
        self.validate_acceleration_input(&desc.input)?;
        B::Ray::compactable_sizes(&self.device.native, desc)
    }

    /// Allocates storage for a build that requests a compacted size output.
    pub fn create_compactable_acceleration(
        &self,
        desc: &AccelerationDesc<B>,
    ) -> Result<AccelerationStructure<B>> {
        self.require_compaction_size()?;
        self.validate_acceleration_input(&desc.input)?;
        let native = B::Ray::create_compactable_acceleration(&self.device.native, desc)?;
        // SAFETY: the SPI created this acceleration structure for this exact device.
        Ok(unsafe { AccelerationStructure::from_native(self.device.id, native) })
    }
    pub fn triangle_vertex_layout(&self, format: VertexFormat) -> Result<AccelerationVertexLayout> {
        self.require_acceleration()?;
        B::Ray::triangle_vertex_layout(&self.device.native, format)
    }
    pub fn create_acceleration(
        &self,
        desc: &AccelerationDesc<B>,
    ) -> Result<AccelerationStructure<B>> {
        self.require_acceleration()?;
        self.validate_acceleration_input(&desc.input)?;
        let native = B::Ray::create_acceleration(&self.device.native, desc)?;
        // SAFETY: the SPI created this acceleration structure for this device.
        Ok(unsafe { AccelerationStructure::from_native(self.device.id, native) })
    }
    pub fn build(
        &self,
        encoder: &mut CommandEncoder<B>,
        dst: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.require_acceleration()?;
        if encoder.device != self.device.id
            || dst.device != self.device.id
            || scratch.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        self.validate_acceleration_input(&desc.input)?;
        encoder.call(|e| B::Ray::build_acceleration(e, &dst.native, &scratch.native, desc))
    }

    /// Reports whether the device can clone an acceleration structure.
    pub fn supports_acceleration_clone(&self) -> bool {
        self.device.caps.acceleration_structure && B::Ray::ACCELERATION_CLONE_AVAILABLE
    }

    /// Records a clone of a built acceleration structure into separate storage.
    /// The destination must be of the same kind and at least as large as the
    /// source allocation. A cloned TLAS references the same BLAS objects.
    /// The caller provides the required barriers and queue ordering for both
    /// structures. The source build must precede the clone at submission.
    pub fn clone_acceleration(
        &self,
        encoder: &mut CommandEncoder<B>,
        dst: &AccelerationStructure<B>,
        src: &AccelerationStructure<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        if !self.supports_acceleration_clone() {
            return Err(Error::Unsupported("acceleration structure clone"));
        }
        if encoder.device != self.device.id
            || dst.device != self.device.id
            || src.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        encoder.call(|e| B::Ray::clone_acceleration(e, &dst.native, &src.native))
    }
    /// Builds an acceleration structure and writes its selected size in
    /// bytes. For `Compacted`, allocate with `create_compactable_acceleration`
    /// and size scratch with `compactable_sizes`. Read the result after a
    /// covering completion point.
    pub fn build_with_size_output(
        &self,
        encoder: &mut CommandEncoder<B>,
        dst: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
        output: AccelerationSizeOutput<'_, B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        self.record_with_size_output(encoder, dst, scratch, desc, output, false)
    }
    /// Refits a compatible build and writes its selected size in bytes.
    /// Use `Compacted` for a build originally recorded with compaction enabled.
    pub fn update_with_size_output(
        &self,
        encoder: &mut CommandEncoder<B>,
        dst: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
        output: AccelerationSizeOutput<'_, B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        if !self.device.caps.acceleration_structure_update {
            return Err(Error::Unsupported("acceleration structure update"));
        }
        if !desc.allow_update {
            return Err(Error::InvalidState);
        }
        self.record_with_size_output(encoder, dst, scratch, desc, output, true)
    }
    fn record_with_size_output(
        &self,
        encoder: &mut CommandEncoder<B>,
        dst: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
        output: AccelerationSizeOutput<'_, B>,
        refit: bool,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        if !self.device.caps.acceleration_build_size_output {
            return Err(Error::Unsupported("acceleration build size output"));
        }
        if output.kind == AccelerationSizeKind::Compacted {
            self.require_compaction_size()?;
        }
        if encoder.device != self.device.id
            || dst.device != self.device.id
            || scratch.device != self.device.id
            || output.buffer.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        if output.offset % 8 != 0 || output.offset.checked_add(8).is_none() {
            return Err(Error::InvalidRange);
        }
        self.validate_acceleration_input(&desc.input)?;
        encoder.call(|e| {
            B::Ray::build_with_size_output(
                e,
                &dst.native,
                &scratch.native,
                desc,
                (&output.buffer.native, output.offset, output.kind),
                refit,
            )
        })
    }
    /// Records an in-place refit. The execution must order a compatible build
    /// with `allow_update` before this command, and make its writes visible.
    /// A backend may check prior builds at submission to include earlier work
    /// recorded in the same packet. Unsupported geometry returns an error.
    /// BLAS primitive activity must stay fixed. A changed BLAS requires its
    /// referencing TLAS to be updated or rebuilt before further traversal.
    pub fn update(
        &self,
        encoder: &mut CommandEncoder<B>,
        dst: &AccelerationStructure<B>,
        scratch: &Buffer<B>,
        desc: &AccelerationDesc<B>,
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        if !self.device.caps.acceleration_structure_update {
            return Err(Error::Unsupported("acceleration structure update"));
        }
        if encoder.device != self.device.id
            || dst.device != self.device.id
            || scratch.device != self.device.id
        {
            return Err(Error::WrongDevice);
        }
        if !desc.allow_update {
            return Err(Error::InvalidState);
        }
        self.validate_acceleration_input(&desc.input)?;
        encoder.call(|e| B::Ray::update_acceleration(e, &dst.native, &scratch.native, desc))
    }
    fn require_acceleration(&self) -> Result<()> {
        if self.device.caps.acceleration_structure {
            Ok(())
        } else {
            Err(Error::Unsupported("acceleration structures"))
        }
    }
    fn require_compaction_size(&self) -> Result<()> {
        if !self.device.caps.acceleration_compaction_size {
            return Err(Error::Unsupported("acceleration compaction size"));
        }
        self.require_acceleration()
    }
    fn validate_acceleration_input(&self, input: &AccelerationInput<B>) -> Result<()> {
        input.validate(self.device.id, &self.device.caps.limits)?;
        if !self.device.caps.acceleration_geometry_transform
            && matches!(input, AccelerationInput::Blas(geometries) if geometries.iter().any(|g| matches!(g, AccelerationGeometry::TransformedTriangles { .. })))
        {
            return Err(Error::Unsupported("acceleration geometry transform"));
        }
        Ok(())
    }
    /// Checks dispatch dimensions and the shader table layout before recording.
    /// The caller must bind the pipeline and prepare the table records.
    pub fn trace_support(
        &self,
        pipeline: &RayPipeline<B>,
        table: &ShaderTable<B>,
        dims: [u32; 3],
    ) -> Result<()> {
        self.validate_trace(pipeline, table, dims)?;
        B::Ray::trace_support(table, dims)
    }

    pub fn trace(
        &self,
        encoder: &mut CommandEncoder<B>,
        pipeline: &RayPipeline<B>,
        table: &ShaderTable<B>,
        dims: [u32; 3],
    ) -> Result<()>
    where
        B: CommandBackend,
    {
        if encoder.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        self.validate_trace(pipeline, table, dims)?;
        let result = encoder.call(|e| B::Ray::trace(e, &pipeline.native, table, dims));
        if result.is_ok() {
            encoder.stats.record_ray_trace();
        }
        result
    }

    fn validate_trace(
        &self,
        pipeline: &RayPipeline<B>,
        table: &ShaderTable<B>,
        dims: [u32; 3],
    ) -> Result<()> {
        self.require_pipeline()?;
        if pipeline.device != self.device.id {
            return Err(Error::WrongDevice);
        }
        let regions = std::iter::once(&table.raygen)
            .chain(table.miss.iter())
            .chain(table.hit.iter())
            .chain(table.callable.iter());
        for region in regions {
            if region.buffer.device != self.device.id {
                return Err(Error::WrongDevice);
            }
            if region.size == 0
                || (region.stride == 0 && !B::Ray::ZERO_STRIDE_SHADER_TABLES)
                || (region.stride != 0 && region.size % region.stride != 0)
                || region.offset.checked_add(region.size).is_none()
            {
                return Err(Error::InvalidRange);
            }
        }
        if table.raygen.size != table.raygen.stride {
            return Err(Error::InvalidRange);
        }
        let max_count = self.device.caps.limits.max_ray_dispatch_count;
        if dims.contains(&0) {
            return if B::Ray::EMPTY_DISPATCH {
                Ok(())
            } else {
                Err(Error::InvalidRange)
            };
        }
        if max_count != 0
            && dims
                .into_iter()
                .try_fold(1u64, |count, axis| count.checked_mul(u64::from(axis)))
                .is_none_or(|count| count > u64::from(max_count))
        {
            return Err(Error::InvalidRange);
        }
        Ok(())
    }

    fn require_pipeline(&self) -> Result<()> {
        if self.device.caps.ray {
            Ok(())
        } else {
            Err(Error::Unsupported("ray tracing pipeline"))
        }
    }
}
