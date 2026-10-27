//! Core admission checks for mesh indirect batch operations.
use super::*;
use std::{cell::Cell, sync::Arc};

thread_local! {
    static MULTI_CALLS: Cell<u32> = const { Cell::new(0) };
    static COUNT_CALLS: Cell<u32> = const { Cell::new(0) };
}

struct MeshMock;
struct MeshMockOps;

impl Backend for MeshMock {
    type NativeDevice = Arc<()>;
    type NativeBuffer = Arc<()>;
    type NativeTexture = Arc<()>;
    type NativeTextureView = Arc<()>;
    type NativeSampler = Arc<()>;
    type NativeBindGroupLayout = Arc<()>;
    type NativePipelineLayout = Arc<()>;
    type NativeBindGroup = Arc<()>;
    type NativeShader = Arc<()>;
    type NativeRasterPipeline = Arc<()>;
    type NativeComputePipeline = Arc<()>;
    type NativeMeshPipeline = Arc<()>;
    type NativeRayPipeline = Arc<()>;
    type NativePipelineCache = Arc<()>;
    type NativeAcceleration = Arc<()>;
    type NativeSparseAllocation = Arc<()>;
    type NativeQuerySet = Arc<()>;
    type NativeEncoder = ();
    type NativeCompletion = Arc<()>;
    type NativeSurface = Arc<()>;
    type NativeSurfaceImage = ();
    type NativeUncertainty = ();
    type Compute = Unavailable;
    type Mesh = MeshMockOps;
    type Ray = Unavailable;
    type Extras = Unavailable;
    type MemoryObservation = Unavailable;
}

impl CommandBackend for MeshMock {
    fn begin_encoder(_: &Arc<()>, _: LaneId) -> Result<()> {
        Ok(())
    }
    fn transition(_: &mut (), _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        Ok(())
    }
    fn copy_buffer(_: &mut (), _: &Arc<()>, _: u64, _: &Arc<()>, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn copy_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn buffer_to_texture(
        _: &mut (),
        _: &Arc<()>,
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: &Arc<()>,
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut (), _: &Arc<()>, _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Ok(())
    }
}

impl RasterBackend for MeshMock {
    fn begin_raster(_: &mut (), _: &RasterPassDesc<'_, Self>) -> Result<()> {
        Ok(())
    }
    fn set_pipeline(_: &mut (), _: &Arc<()>) -> Result<()> {
        Ok(())
    }
    fn set_bind_group(_: &mut (), _: u32, _: &Arc<()>, _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn set_vertex_buffer(_: &mut (), _: u32, _: &Arc<()>, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn set_index_buffer(_: &mut (), _: &Arc<()>, _: u64, _: u64, _: IndexFormat) -> Result<()> {
        Ok(())
    }
    fn viewport(_: &mut (), _: Viewport) -> Result<()> {
        Ok(())
    }
    fn scissor(_: &mut (), _: Rect) -> Result<()> {
        Ok(())
    }
    fn blend_constant(_: &mut (), _: [f32; 4]) -> Result<()> {
        Ok(())
    }
    fn stencil_reference(_: &mut (), _: u32) -> Result<()> {
        Ok(())
    }
    fn draw(_: &mut (), _: Range<u32>, _: Range<u32>) -> Result<()> {
        Ok(())
    }
    fn draw_indexed(_: &mut (), _: Range<u32>, _: i32, _: Range<u32>) -> Result<()> {
        Ok(())
    }
    fn end_raster(_: &mut ()) -> Result<()> {
        Ok(())
    }
    fn abort_raster(_: &mut ()) {}
}

impl MeshOps<MeshMock> for MeshMockOps {
    const AVAILABLE: bool = true;
    type PipelineFuture<'a> = std::future::Ready<Result<Arc<()>>>;
    fn create_pipeline<'a>(
        _: &'a Arc<()>,
        _: &'a MeshPipelineDesc<MeshMock>,
    ) -> Self::PipelineFuture<'a> {
        std::future::ready(Ok(Arc::new(())))
    }
    fn bind_pipeline(_: &mut (), _: &Arc<()>) -> Result<()> {
        Ok(())
    }
    fn dispatch(_: &mut (), _: [u32; 3]) -> Result<()> {
        Ok(())
    }
    fn dispatch_indirect(_: &mut (), _: &Arc<()>, _: u64) -> Result<()> {
        Ok(())
    }
    fn dispatch_indirect_multi(_: &mut (), _: &Arc<()>, _: u64, _: u32) -> Result<()> {
        MULTI_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }
    fn dispatch_indirect_count(
        _: &mut (),
        _: &Arc<()>,
        _: u64,
        _: &Arc<()>,
        _: u64,
        _: u32,
    ) -> Result<()> {
        COUNT_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(())
    }
}

fn mesh_device(multi: bool, count: bool) -> Device<MeshMock> {
    let id = DeviceId::allocate().unwrap();
    let caps = DeviceCapabilities {
        mesh: true,
        mesh_multi_indirect: multi,
        mesh_indirect_count: count,
        limits: DeviceLimits {
            max_mesh_indirect_count: if multi { 2 } else { 0 },
            ..Default::default()
        },
        ..Default::default()
    };
    // SAFETY: the test handles share the allocated device identity.
    unsafe { Device::from_native(id, Arc::new(()), caps) }
}

fn with_pass(device: &Device<MeshMock>, run: impl FnOnce(&mut RasterPass<'_, MeshMock>)) {
    let lane = LaneId {
        device: device.id(),
        ordinal: 0,
    };
    let mut encoder = CommandEncoder::new(device, lane).unwrap();
    // SAFETY: this mock view belongs to this device and remains alive while
    // the pass descriptor is consumed.
    let view = unsafe { TextureView::<MeshMock>::from_native(device.id(), Arc::new(())) };
    let colors = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Load,
        store: StoreOp::Store,
    })];
    let mut pass = encoder
        .begin_raster(&RasterPassDesc {
            label: None,
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })
        .unwrap();
    run(&mut pass);
    pass.finish().unwrap();
}

fn buffer(device: &Device<MeshMock>) -> Buffer<MeshMock> {
    // SAFETY: this mock buffer belongs to this device.
    unsafe { Buffer::from_native(device.id(), Arc::new(())) }
}

#[test]
fn mesh_indirect_admission_rejects_unavailable_capabilities_before_spi() {
    let device = mesh_device(false, false);
    let args = buffer(&device);
    MULTI_CALLS.with(|calls| calls.set(0));
    COUNT_CALLS.with(|calls| calls.set(0));
    with_pass(&device, |pass| {
        let mesh = device.mesh().unwrap();
        assert!(matches!(
            mesh.dispatch_indirect_multi(pass, &args, 0, 1),
            Err(Error::Unsupported("mesh multi-indirect"))
        ));
        assert!(matches!(
            mesh.dispatch_indirect_count(pass, &args, 0, &args, 0, 1),
            Err(Error::Unsupported("mesh indirect count"))
        ));
    });
    MULTI_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    COUNT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}

#[test]
fn mesh_indirect_admission_rejects_foreign_and_invalid_arguments_before_spi() {
    let device = mesh_device(true, true);
    let foreign = mesh_device(true, true);
    let args = buffer(&device);
    let foreign_args = buffer(&foreign);
    let count = buffer(&device);
    MULTI_CALLS.with(|calls| calls.set(0));
    COUNT_CALLS.with(|calls| calls.set(0));
    with_pass(&device, |pass| {
        let mesh = device.mesh().unwrap();
        assert_eq!(
            mesh.dispatch_indirect_multi(pass, &foreign_args, 0, 1),
            Err(Error::WrongDevice)
        );
        assert_eq!(
            mesh.dispatch_indirect_count(pass, &args, 0, &foreign_args, 0, 1),
            Err(Error::WrongDevice)
        );
        assert_eq!(
            mesh.dispatch_indirect_multi(pass, &args, 2, 1),
            Err(Error::InvalidRange)
        );
        assert_eq!(
            mesh.dispatch_indirect_count(pass, &args, 0, &count, 2, 1),
            Err(Error::InvalidRange)
        );
        assert_eq!(
            mesh.dispatch_indirect_multi(pass, &args, 0, 3),
            Err(Error::InvalidRange)
        );
        assert_eq!(
            mesh.dispatch_indirect_count(pass, &args, 0, &count, 0, 3),
            Err(Error::InvalidRange)
        );
    });
    MULTI_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    COUNT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}

#[test]
fn mesh_indirect_zero_count_is_a_core_noop() {
    let device = mesh_device(true, true);
    let args = buffer(&device);
    MULTI_CALLS.with(|calls| calls.set(0));
    COUNT_CALLS.with(|calls| calls.set(0));
    with_pass(&device, |pass| {
        let mesh = device.mesh().unwrap();
        mesh.dispatch_indirect_multi(pass, &args, 0, 0).unwrap();
        mesh.dispatch_indirect_count(pass, &args, 0, &args, 0, 0)
            .unwrap();
    });
    MULTI_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    COUNT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}
