//! Device provenance checks before invoking the ray backend.
use super::*;
use std::cell::Cell;

thread_local! { static CALLS: Cell<usize> = const { Cell::new(0) }; }

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
        unreachable!()
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
    fn trace(_: &mut (), _: &(), _: &(), _: [u32; 3]) -> Result<()> {
        unreachable!()
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
                stride: 12,
                indices: None,
                opaque: false,
            },
        ])),
        desc(AccelerationInput::Blas(vec![
            AccelerationGeometry::Triangles {
                vertices: buffer(id),
                stride: 12,
                indices: Some((buffer(foreign), IndexFormat::Uint16)),
                opaque: false,
            },
        ])),
        desc(AccelerationInput::Blas(vec![AccelerationGeometry::Aabbs {
            data: buffer(foreign),
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
            flags: 0,
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
            stride: 12,
            indices: Some((buffer(id), IndexFormat::Uint16)),
            opaque: false,
        },
    ]));
    assert!(ray.sizes(&valid).is_ok());
    assert!(ray.create_acceleration(&valid).is_ok());
    assert!(ray.build(&mut encoder, &dst, &scratch, &valid).is_ok());
    CALLS.with(|c| assert_eq!(c.get(), 3));
}
