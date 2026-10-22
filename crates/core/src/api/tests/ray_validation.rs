//! Device provenance checks before invoking the ray backend.
use super::*;
use std::cell::Cell;

thread_local! { static CALLS: Cell<usize> = const { Cell::new(0) }; }

#[test]
fn query_only_device_exposes_acceleration_but_rejects_ray_pipeline() {
    use std::task::{Context, Poll, Waker};
    CALLS.with(|c| c.set(0));
    let id = DeviceId::allocate().unwrap();
    // SAFETY: the mock native handles belong to this mock device.
    let device = unsafe {
        Device::<Mock>::from_native(
            id,
            (),
            DeviceCapabilities {
                ray_query: true,
                acceleration_structure: true,
                ..Default::default()
            },
        )
    };
    let ray = device
        .ray()
        .expect("query devices need acceleration structures");
    let desc = AccelerationDesc {
        input: AccelerationInput::Blas(vec![]),
        fast_trace: false,
        allow_update: false,
    };
    assert!(ray.sizes(&desc).is_ok());
    assert!(ray.create_acceleration(&desc).is_ok());
    assert!(matches!(
        ray.shader_table_properties(),
        Err(Error::Unsupported(_))
    ));
    let pipeline = RayPipelineDesc {
        layout: unsafe { PipelineLayout::<Mock>::from_native(id, ()) },
        shaders: vec![],
        groups: vec![],
        max_recursion: 1,
        payload_bytes: 0,
        hit_attribute_bytes: 0,
        cache: None,
    };
    let mut future = std::pin::pin!(ray.create_pipeline(pipeline));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(matches!(
        future.as_mut().poll(&mut cx),
        Poll::Ready(Err(Error::Unsupported(_)))
    ));
    CALLS.with(|c| assert_eq!(c.get(), 2));
}

#[test]
fn acceleration_capability_is_independent_of_ray_and_query() {
    let id = DeviceId::allocate().unwrap();
    let desc = AccelerationDesc {
        input: AccelerationInput::Blas(vec![]),
        fast_trace: false,
        allow_update: false,
    };
    // SAFETY: the mock handles have the device identity under test.
    let as_only = unsafe {
        Device::<Mock>::from_native(
            id,
            (),
            DeviceCapabilities {
                acceleration_structure: true,
                ..Default::default()
            },
        )
    };
    let ray = as_only.ray().unwrap();
    assert!(ray.sizes(&desc).is_ok());
    assert!(ray.create_acceleration(&desc).is_ok());
    assert!(matches!(
        ray.shader_table_properties(),
        Err(Error::Unsupported(_))
    ));
    // SAFETY: the mock handles have the device identity under test.
    let no_as = unsafe {
        Device::<Mock>::from_native(
            id,
            (),
            DeviceCapabilities {
                ray_query: true,
                ..Default::default()
            },
        )
    };
    assert!(matches!(
        no_as.ray().unwrap().sizes(&desc),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        no_as.ray().unwrap().create_acceleration(&desc),
        Err(Error::Unsupported(_))
    ));
}

struct Mock;
struct MockRay;
impl Backend for Mock {
    type NativeDevice = ();
    type NativeBuffer = ();
    type NativeTexture = ();
    type NativeTextureView = ();
    type NativeSampler = ();
    type NativeBindGroupLayout = ();
    type NativePipelineLayout = ();
    type NativeBindGroup = ();
    type NativeShader = ();
    type NativeRasterPipeline = ();
    type NativeComputePipeline = ();
    type NativeMeshPipeline = ();
    type NativeRayPipeline = ();
    type NativePipelineCache = ();
    type NativeAcceleration = ();
    type NativeSparseAllocation = ();
    type NativeQuerySet = ();
    type NativeEncoder = ();
    type NativeCompletion = ();
    type NativeSurface = ();
    type NativeSurfaceImage = ();
    type NativeUncertainty = ();
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = MockRay;
    type Extras = Unavailable;
}
impl RayOps<Mock> for MockRay {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = std::future::Ready<Result<()>>;
    fn create_pipeline<'a>(_: &'a (), _: &'a RayPipelineDesc<Mock>) -> Self::PipelineFuture<'a> {
        CALLS.with(|c| c.set(c.get() + 1));
        std::future::ready(Ok(()))
    }
    fn set_pipeline(_: &mut (), _: &()) -> Result<()> {
        Ok(())
    }
    fn set_bind_group(_: &mut (), _: u32, _: &(), _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn shader_table_properties(_: &()) -> Result<ShaderTableProperties> {
        Ok(ShaderTableProperties {
            identifier_size: 32,
            record_alignment: 32,
            table_alignment: 64,
            max_record_stride: 4096,
        })
    }
    fn shader_group_identifier(_: &(), _: &(), group: u32) -> Result<Vec<u8>> {
        Ok(vec![group as u8; 32])
    }
    fn acceleration_sizes(_: &(), _: &AccelerationDesc<Mock>) -> Result<AccelerationSizes> {
        CALLS.with(|c| c.set(c.get() + 1));
        Ok(AccelerationSizes {
            storage_bytes: 1,
            scratch_bytes: 1,
            scratch_alignment: 1,
        })
    }
    fn create_acceleration(_: &(), _: &AccelerationDesc<Mock>) -> Result<()> {
        CALLS.with(|c| c.set(c.get() + 1));
        Ok(())
    }
    fn build_acceleration(_: &mut (), _: &(), _: &(), _: &AccelerationDesc<Mock>) -> Result<()> {
        CALLS.with(|c| c.set(c.get() + 1));
        Ok(())
    }
    fn trace(_: &mut (), _: &(), _: &ShaderTable<Mock>, _: [u32; 3]) -> Result<()> {
        CALLS.with(|c| c.set(c.get() + 1));
        Ok(())
    }
}
impl CommandBackend for Mock {
    fn begin_encoder(_: &(), _: LaneId) -> Result<()> {
        Ok(())
    }
    fn transition(_: &mut (), _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        unreachable!()
    }
    fn copy_buffer(_: &mut (), _: &(), _: u64, _: &(), _: u64, _: u64) -> Result<()> {
        unreachable!()
    }
    fn copy_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        unreachable!()
    }
    fn buffer_to_texture(
        _: &mut (),
        _: &(),
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        unreachable!()
    }
    fn texture_to_buffer(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: &(),
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        unreachable!()
    }
    fn write_buffer(_: &mut (), _: &(), _: u64, _: &[u8]) -> Result<()> {
        unreachable!()
    }
    fn write_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        unreachable!()
    }
}

#[test]
fn nested_inputs_are_checked_by_every_ray_entry_before_backend() {
    CALLS.with(|c| c.set(0));
    let id = DeviceId::allocate().unwrap();
    let foreign = DeviceId::allocate().unwrap();
    let caps = DeviceCapabilities {
        ray: true,
        acceleration_structure: true,
        ..Default::default()
    };
    // SAFETY: all unit native handles below belong to the indicated mock device.
    let device = unsafe { Device::<Mock>::from_native(id, (), caps) };
    let buffer = |owner| unsafe { Buffer::<Mock>::from_native(owner, ()) };
    let accel = |owner| unsafe { AccelerationStructure::<Mock>::from_native(owner, ()) };
    let desc = |input| AccelerationDesc {
        input,
        fast_trace: false,
        allow_update: false,
    };
    let inputs = [
        desc(AccelerationInput::Blas(vec![
            AccelerationGeometry::Triangles {
                vertices: buffer(foreign),
                vertex_offset: 0,
                vertex_count: 3,
                vertex_format: VertexFormat::Float32(3),
                stride: 12,
                indices: None,
                opaque: false,
            },
        ])),
        desc(AccelerationInput::Blas(vec![
            AccelerationGeometry::Triangles {
                vertices: buffer(id),
                vertex_offset: 0,
                vertex_count: 3,
                vertex_format: VertexFormat::Float32(3),
                stride: 12,
                indices: Some(AccelerationIndices {
                    buffer: buffer(foreign),
                    offset: 0,
                    count: 3,
                    format: IndexFormat::Uint16,
                }),
                opaque: false,
            },
        ])),
        desc(AccelerationInput::Blas(vec![AccelerationGeometry::Aabbs {
            data: buffer(foreign),
            offset: 0,
            stride: 24,
            count: 1,
            opaque: false,
        }])),
        desc(AccelerationInput::Tlas(vec![AccelerationInstance {
            blas: accel(foreign),
            transform: [[0.0; 4]; 3],
            mask: 0,
            custom_index: 0,
            shader_table_offset: 0,
            flags: AccelerationInstanceFlags::NONE,
        }])),
    ];
    let ray = device.ray().unwrap();
    let mut encoder = CommandEncoder::new(
        &device,
        LaneId {
            device: id,
            ordinal: 0,
        },
    )
    .unwrap();
    let dst = accel(id);
    let scratch = buffer(id);
    for input in &inputs {
        assert!(matches!(ray.sizes(input), Err(Error::WrongDevice)));
        assert!(matches!(
            ray.create_acceleration(input),
            Err(Error::WrongDevice)
        ));
        assert!(matches!(
            ray.build(&mut encoder, &dst, &scratch, input),
            Err(Error::WrongDevice)
        ));
    }
    CALLS.with(|c| assert_eq!(c.get(), 0));
    let valid = desc(AccelerationInput::Blas(vec![
        AccelerationGeometry::Triangles {
            vertices: buffer(id),
            vertex_offset: 0,
            vertex_count: 3,
            vertex_format: VertexFormat::Float32(3),
            stride: 12,
            indices: Some(AccelerationIndices {
                buffer: buffer(id),
                offset: 0,
                count: 3,
                format: IndexFormat::Uint16,
            }),
            opaque: false,
        },
    ]));
    assert!(ray.sizes(&valid).is_ok());
    assert!(ray.create_acceleration(&valid).is_ok());
    assert!(ray.build(&mut encoder, &dst, &scratch, &valid).is_ok());
    CALLS.with(|c| assert_eq!(c.get(), 3));
}

#[test]
fn shader_tables_validate_ranges_and_device_before_trace() {
    CALLS.with(|c| c.set(0));
    let id = DeviceId::allocate().unwrap();
    let other = DeviceId::allocate().unwrap();
    let device = unsafe {
        Device::<Mock>::from_native(
            id,
            (),
            DeviceCapabilities {
                ray: true,
                acceleration_structure: true,
                ..Default::default()
            },
        )
    };
    let mut encoder = CommandEncoder::new(
        &device,
        LaneId {
            device: id,
            ordinal: 0,
        },
    )
    .unwrap();
    let pipeline = unsafe { RayPipeline::<Mock>::from_native(id, ()) };
    let foreign_pipeline = unsafe { RayPipeline::<Mock>::from_native(other, ()) };
    let region = |owner| ShaderTableRegion {
        buffer: unsafe { Buffer::<Mock>::from_native(owner, ()) },
        offset: 0,
        size: 32,
        stride: 32,
    };
    let mut table = ShaderTable {
        raygen: region(id),
        miss: Some(region(id)),
        hit: None,
        callable: None,
    };
    let ray = device.ray().unwrap();
    assert_eq!(
        ray.shader_group_identifier(&pipeline, 2).unwrap(),
        vec![2; 32]
    );
    assert!(matches!(
        ray.shader_group_identifier(&foreign_pipeline, 0),
        Err(Error::WrongDevice)
    ));
    table.miss = Some(region(other));
    assert!(matches!(
        ray.trace(&mut encoder, &pipeline, &table, [1, 1, 1]),
        Err(Error::WrongDevice)
    ));
    table.miss = Some(region(id));
    table.raygen.size = 64;
    assert!(matches!(
        ray.trace(&mut encoder, &pipeline, &table, [1, 1, 1]),
        Err(Error::InvalidRange)
    ));
    table.raygen.size = 32;
    table.raygen.offset = u64::MAX;
    assert!(matches!(
        ray.trace(&mut encoder, &pipeline, &table, [1, 1, 1]),
        Err(Error::InvalidRange)
    ));
    table.raygen.offset = 0;
    assert!(
        ray.trace(&mut encoder, &pipeline, &table, [1, 1, 1])
            .is_ok()
    );
    CALLS.with(|c| assert_eq!(c.get(), 1));
}

#[test]
fn invalid_geometry_ranges_never_reach_backend() {
    CALLS.with(|c| c.set(0));
    let id = DeviceId::allocate().unwrap();
    let device = unsafe {
        Device::<Mock>::from_native(
            id,
            (),
            DeviceCapabilities {
                ray: true,
                acceleration_structure: true,
                ..Default::default()
            },
        )
    };
    let buffer = || unsafe { Buffer::<Mock>::from_native(id, ()) };
    let invalid = [
        AccelerationGeometry::Triangles {
            vertices: buffer(),
            vertex_offset: 0,
            vertex_count: 3,
            vertex_format: VertexFormat::Float32(3),
            stride: 12,
            indices: Some(AccelerationIndices {
                buffer: buffer(),
                offset: 1,
                count: 3,
                format: IndexFormat::Uint16,
            }),
            opaque: true,
        },
        AccelerationGeometry::Aabbs {
            data: buffer(),
            offset: u64::MAX - 7,
            stride: 24,
            count: 2,
            opaque: true,
        },
    ];
    let ray = device.ray().unwrap();
    for geometry in invalid {
        let desc = AccelerationDesc {
            input: AccelerationInput::Blas(vec![geometry]),
            fast_trace: false,
            allow_update: false,
        };
        assert!(matches!(ray.sizes(&desc), Err(Error::InvalidRange)));
    }
    CALLS.with(|c| assert_eq!(c.get(), 0));
}

#[test]
fn local_constants_require_whole_words_before_pipeline_creation() {
    use std::task::{Context, Poll, Waker};
    CALLS.with(|c| c.set(0));
    let id = DeviceId::allocate().unwrap();
    let device = unsafe {
        Device::<Mock>::from_native(
            id,
            (),
            DeviceCapabilities {
                ray: true,
                acceleration_structure: true,
                ..Default::default()
            },
        )
    };
    let ray = device.ray().unwrap();
    let descriptor = |size_bytes| RayPipelineDesc {
        layout: unsafe { PipelineLayout::<Mock>::from_native(id, ()) },
        shaders: vec![],
        groups: vec![RayGroup {
            general: Some(0),
            closest_hit: None,
            any_hit: None,
            intersection: None,
            local_constants: Some(RayLocalConstants {
                binding: 0,
                group: 0,
                size_bytes,
            }),
        }],
        max_recursion: 1,
        payload_bytes: 16,
        hit_attribute_bytes: 8,
        cache: None,
    };
    let mut cx = Context::from_waker(Waker::noop());
    let mut invalid = std::pin::pin!(ray.create_pipeline(descriptor(3)));
    assert!(matches!(
        invalid.as_mut().poll(&mut cx),
        Poll::Ready(Err(Error::InvalidRange))
    ));
    let mut valid = std::pin::pin!(ray.create_pipeline(descriptor(32)));
    assert!(matches!(valid.as_mut().poll(&mut cx), Poll::Ready(Ok(_))));
    CALLS.with(|c| assert_eq!(c.get(), 1));
}

#[test]
fn instance_flags_are_validated_before_native_calls() {
    CALLS.with(|c| c.set(0));
    let id = DeviceId::allocate().unwrap();
    let device = unsafe {
        Device::<Mock>::from_native(
            id,
            (),
            DeviceCapabilities {
                ray: true,
                acceleration_structure: true,
                ..Default::default()
            },
        )
    };
    let desc = |flags| AccelerationDesc {
        input: AccelerationInput::Tlas(vec![AccelerationInstance {
            blas: unsafe { AccelerationStructure::<Mock>::from_native(id, ()) },
            transform: [[0.0; 4]; 3],
            mask: 255,
            custom_index: 0,
            shader_table_offset: 0,
            flags,
        }]),
        fast_trace: false,
        allow_update: false,
    };
    let ray = device.ray().unwrap();
    let mut encoder = CommandEncoder::new(
        &device,
        LaneId {
            device: id,
            ordinal: 0,
        },
    )
    .unwrap();
    let dst = unsafe { AccelerationStructure::<Mock>::from_native(id, ()) };
    let scratch = unsafe { Buffer::<Mock>::from_native(id, ()) };
    for flags in [
        AccelerationInstanceFlags(1 << 16),
        AccelerationInstanceFlags::FORCE_OPAQUE | AccelerationInstanceFlags::FORCE_NON_OPAQUE,
    ] {
        let invalid = desc(flags);
        assert!(matches!(ray.sizes(&invalid), Err(Error::InvalidState)));
        assert!(matches!(
            ray.create_acceleration(&invalid),
            Err(Error::InvalidState)
        ));
        assert!(matches!(
            ray.build(&mut encoder, &dst, &scratch, &invalid),
            Err(Error::InvalidState)
        ));
    }
    let valid = desc(
        AccelerationInstanceFlags::TRIANGLE_CULL_DISABLE
            | AccelerationInstanceFlags::TRIANGLE_FRONT_COUNTER_CLOCKWISE
            | AccelerationInstanceFlags::FORCE_NON_OPAQUE,
    );
    assert!(ray.sizes(&valid).is_ok());
    CALLS.with(|c| assert_eq!(c.get(), 1));
}
