//! Configuration provenance carried by acquired surface leases.

use super::*;
use std::{
    future::{Ready, ready},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

struct State {
    generation: AtomicU64,
    fail_configure: AtomicBool,
    invalidate_on_configure_failure: AtomicBool,
    acquires: AtomicU64,
}

struct Lease {
    state: Arc<State>,
    generation: u64,
}

// SAFETY: the test lease compares its captured generation to the one mutable
// native surface state, and has no external native side effects.
unsafe impl NativeImageLease for Lease {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if self.is_current(generation) {
            Ok(())
        } else {
            Err(Error::SurfaceOutdated)
        }
    }

    fn is_current(&self, generation: u64) -> bool {
        self.generation == generation && self.state.generation.load(Ordering::Acquire) == generation
    }
}

struct SurfaceTest;

impl Backend for SurfaceTest {
    type NativeDevice = Arc<State>;
    type NativeBuffer = ();
    type NativeTexture = ();
    type NativeTextureView = ();
    type NativeVideoFrame = ();
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
    type NativeSurface = Arc<State>;
    type NativeSurfaceImage = Lease;
    type NativeUncertainty = ();
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
    type MemoryObservation = Unavailable;
}

impl SurfaceBackend for SurfaceTest {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = Ready<Result<Lease>>;

    fn generation(surface: &Arc<State>) -> u64 {
        surface.generation.load(Ordering::Acquire)
    }

    fn capabilities(_: &Arc<State>) -> Result<SurfaceCapabilities> {
        Ok(SurfaceCapabilities {
            formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm],
            present_modes: vec![PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            color_spaces: vec![ColorSpace::Srgb],
            usages: SurfaceUsage::RENDER_ATTACHMENT,
            min_extent: [1, 1],
            max_extent: [4096, 4096],
        })
    }

    fn supports_config(_: &Arc<State>, desc: &SurfaceConfig) -> Support<()> {
        if desc.extent.contains(&0)
            || !matches!(
                desc.format,
                TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm
            )
        {
            Support::Unsupported("surface test configuration")
        } else {
            Support::Supported(())
        }
    }

    fn configure<'a>(
        surface: &'a mut Arc<State>,
        _: &'a SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        if surface.fail_configure.load(Ordering::Acquire) {
            if surface
                .invalidate_on_configure_failure
                .load(Ordering::Acquire)
            {
                surface.generation.fetch_add(1, Ordering::AcqRel);
            }
            return ready(Err(Error::InvalidState));
        }
        surface.generation.fetch_add(1, Ordering::AcqRel);
        ready(Ok(()))
    }

    fn acquire<'a>(surface: &'a mut Arc<State>) -> Self::AcquireFuture<'a> {
        surface.acquires.fetch_add(1, Ordering::AcqRel);
        ready(Ok(Lease {
            state: surface.clone(),
            generation: Self::generation(surface),
        }))
    }
}

fn state() -> Arc<State> {
    Arc::new(State {
        generation: AtomicU64::new(0),
        fail_configure: AtomicBool::new(false),
        invalidate_on_configure_failure: AtomicBool::new(false),
        acquires: AtomicU64::new(0),
    })
}

fn configuration(format: TextureFormat, extent: [u32; 2]) -> SurfaceConfig {
    SurfaceConfig {
        extent,
        format,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    }
}

fn poll_ready<T>(future: impl std::future::Future<Output = T>) -> T {
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => panic!("surface test future unexpectedly pending"),
    }
}

fn surface(state: Arc<State>) -> Surface<SurfaceTest> {
    // SAFETY: the test device and surface use the same synthetic native state.
    let device = unsafe {
        Device::<SurfaceTest>::from_native(
            DeviceId::allocate().unwrap(),
            state.clone(),
            DeviceCapabilities::default(),
        )
    };
    // SAFETY: the test surface belongs to the device created above.
    unsafe { Surface::from_native(&device, SurfaceId::allocate().unwrap(), state) }
}

#[test]
fn acquired_lease_carries_exact_successful_configuration() {
    let state = state();
    let mut surface = surface(state);
    let first = configuration(TextureFormat::Rgba8Unorm, [64, 32]);
    poll_ready(surface.configure(&first)).unwrap();
    let old = poll_ready(surface.acquire()).unwrap();
    assert_eq!(old.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(old.extent(), [64, 32]);

    let second = configuration(TextureFormat::Bgra8Unorm, [128, 48]);
    poll_ready(surface.configure(&second)).unwrap();
    let new = poll_ready(surface.acquire()).unwrap();
    assert_eq!(new.format(), TextureFormat::Bgra8Unorm);
    assert_eq!(new.extent(), [128, 48]);
    assert!(!old.is_current());
}

#[test]
fn failed_configure_preserves_last_committed_lease_facts() {
    let state = state();
    let mut surface = surface(state.clone());
    let config = configuration(TextureFormat::Rgba8Unorm, [16, 16]);
    poll_ready(surface.configure(&config)).unwrap();
    state.fail_configure.store(true, Ordering::Release);
    assert_eq!(
        poll_ready(surface.configure(&configuration(TextureFormat::Bgra8Unorm, [32, 32])),),
        Err(Error::InvalidState)
    );
    let image = poll_ready(surface.acquire()).unwrap();
    assert_eq!(image.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(image.extent(), [16, 16]);
}

#[test]
fn failed_configure_that_invalidates_native_state_cannot_mint_old_facts() {
    let state = state();
    let mut surface = surface(state.clone());
    let config = configuration(TextureFormat::Rgba8Unorm, [16, 16]);
    poll_ready(surface.configure(&config)).unwrap();
    state.fail_configure.store(true, Ordering::Release);
    state
        .invalidate_on_configure_failure
        .store(true, Ordering::Release);
    assert_eq!(
        poll_ready(surface.configure(&configuration(TextureFormat::Bgra8Unorm, [32, 32])),),
        Err(Error::InvalidState)
    );

    // The Core cache intentionally retains the last successful facts, but a
    // backend generation transition makes them ineligible for a new lease.
    assert!(matches!(
        poll_ready(surface.acquire()),
        Err(Error::SurfaceOutdated)
    ));
}

#[test]
fn acquire_rejects_native_generation_that_no_longer_matches_config() {
    let state = state();
    let mut surface = surface(state.clone());
    poll_ready(surface.configure(&configuration(TextureFormat::Rgba8Unorm, [16, 16]))).unwrap();
    state.generation.fetch_add(1, Ordering::AcqRel);
    assert!(matches!(
        poll_ready(surface.acquire()),
        Err(Error::SurfaceOutdated)
    ));
    assert_eq!(state.acquires.load(Ordering::Acquire), 1);
}
