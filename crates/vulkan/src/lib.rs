//! Vulkan implementation for Fluxel RHI.

pub use fluxel_rhi_core::api;

/// Stops lowering when a newer portable vocabulary reaches an older backend.
pub(crate) fn unknown_portable_variant() -> ! {
    panic!("the Vulkan backend does not support this portable API variant")
}

/// Opens a dynamically loaded Vulkan provider.
#[cfg(not(target_arch = "wasm32"))]
pub fn create_provider() -> api::error::RhiResult<api::platform::PlatformProvider> {
    use api::platform::provider::{BackendKind, PlatformProvider};

    let instance = fluxel_rhi_core::backend_spi::next_provider_instance();
    let native = backend::vulkan::VulkanProvider::new(instance)?;
    Ok(PlatformProvider::new(
        BackendKind::Vulkan,
        instance,
        Box::new(native),
    ))
}

mod backend {
    #[cfg(test)]
    #[path = "../../tests/common/mod.rs"]
    pub(crate) mod conformance;

    #[cfg(test)]
    #[path = "../../tests/harness/mod.rs"]
    pub(crate) mod test_harness;

    pub(crate) mod vulkan;
}

/// Android NativeActivity-only Vulkan WSI integration bridge.
///
/// This is intentionally hidden from the portable RHI contract: it accepts an
/// `ANativeWindow` owned by Android host glue and returns only JSON evidence.
#[cfg(target_os = "android")]
#[doc(hidden)]
pub mod android_vulkan_wsi {
    use super::api::command::{
        ColorAttachment, ColorAttachmentView, ColorClearValue, LoadOp, RasterScopeDescriptor,
        RecorderDescriptor, StoreOp,
    };
    use super::api::identity::DeviceInstanceId;
    use super::api::platform::provider::AdapterSelection;
    use super::api::platform::request::DeviceRequestDescriptor;
    use super::api::platform::requirements::DeviceRequirements;
    use super::api::platform::{BackendKind, PlatformProvider};
    use super::api::presentation::{
        Extent2d, PresentMode, PresentationConfiguration, PresentationExtent,
        PresentationExtentControl,
    };
    use super::api::submission::{LaneWorkDomains, SubmissionPlanBuilder};
    use crate::backend::vulkan::platform::VulkanProvider;
    use core::future::Future;
    use core::task::{Context, Poll, Waker};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::pin::pin;

    thread_local! {
        static ANDROID_TARGETS: RefCell<HashMap<usize, crate::backend::vulkan::presentation::android::AndroidTargetRegistration>> = RefCell::new(HashMap::new());
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    /// Runs the Android WSI evidence workload for one host-owned native window.
    pub fn run(window: *mut core::ffi::c_void) -> Result<String, String> {
        let identity = DeviceInstanceId::new(0xA11D_0001);
        let native = VulkanProvider::new(identity).map_err(|e| e.to_string())?;
        let target = native
            .register_android_presentation_target(window)
            .map_err(|e| e.to_string())?;
        let registration = native.retain_android_presentation_target(target.id());
        let provider = PlatformProvider::new(BackendKind::Vulkan, identity, Box::new(native));
        let adapters = block_on(provider.enumerate_adapters())
            .map_err(|e| e.to_string())?
            .ok_or("no Vulkan adapters")?;
        let adapter = adapters
            .iter()
            .find(|a| {
                provider
                    .supports_presentation(a.id(), &target)
                    .unwrap_or(false)
            })
            .ok_or("no Android-presentable Vulkan adapter")?;
        let request = DeviceRequestDescriptor::new(
            AdapterSelection::Explicit(adapter.id()),
            DeviceRequirements::new(),
        )
        .require_presentation_target(target.clone());
        let device = block_on(provider.request_device(request)).map_err(|e| e.to_string())?;
        let caps = device
            .presentation_capabilities(&target)
            .map_err(|e| e.to_string())?;
        let mut config =
            PresentationConfiguration::new(*caps.formats().first().ok_or("no present format")?)
                .with_present_mode(PresentMode::Fifo);
        if let PresentationExtentControl::Configurable { min, max } = caps.extent_control() {
            config = config.with_extent(PresentationExtent::Exact(Extent2d {
                width: 64u32.clamp(min.width, max.width),
                height: 64u32.clamp(min.height, max.height),
            }));
        }
        let mut surface =
            block_on(device.configure_presentation(&target, &config)).map_err(|e| e.to_string())?;
        let frame = block_on(surface.acquire()).map_err(|e| e.to_string())?;
        let scope = RasterScopeDescriptor::new().with_color(
            super::api::shader::ShaderLocation::new(0),
            ColorAttachment {
                view: ColorAttachmentView::Frame(frame.attachment()),
                load: LoadOp::Clear(ColorClearValue::Float([0.05, 0.2, 0.4, 1.0])),
                store: StoreOp::Store,
                resolve: None,
                depth_slice: None,
            },
        );
        let mut recorder = device
            .create_recorder(&RecorderDescriptor::new())
            .map_err(|e| e.to_string())?;
        recorder
            .begin_raster(&scope)
            .map_err(|e| e.to_string())?
            .end()
            .map_err(|e| e.to_string())?;
        let work = recorder.finish().map_err(|e| e.to_string())?;
        let lane = device
            .capabilities()
            .submission()
            .lanes()
            .iter()
            .find(|l| l.domains().contains(LaneWorkDomains::RASTER))
            .ok_or("no raster lane")?
            .id();
        let mut plan = SubmissionPlanBuilder::new(&device);
        let point = plan
            .add_batch(lane, vec![work])
            .map_err(|e| e.to_string())?;
        plan.present_after(frame, point)
            .map_err(|e| e.to_string())?;
        let receipt = device
            .submit(plan.build().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let present =
            block_on(device.wait_present(receipt.presents()[0].id())).map_err(|e| e.to_string())?;
        let again = block_on(surface.acquire()).map_err(|e| e.to_string())?;
        block_on(again.abandon()).map_err(|e| e.to_string())?;
        block_on(surface.reconfigure(&config)).map_err(|e| e.to_string())?;
        let final_frame = block_on(surface.acquire()).map_err(|e| e.to_string())?;
        block_on(final_frame.abandon()).map_err(|e| e.to_string())?;
        let evidence = format!(
            "{{\"schema\":\"fluxel.android-vulkan-wsi.v1\",\"acquire\":true,\"raster_clear\":true,\"present\":\"{present:?}\",\"reacquire\":true,\"reconfigure\":true}}"
        );
        ANDROID_TARGETS.with(|targets| {
            targets.borrow_mut().insert(window as usize, registration);
        });
        Ok(evidence)
    }

    /// Called from `ANativeActivityCallbacks::onNativeWindowDestroyed`.
    pub fn on_native_window_destroyed(window: *mut core::ffi::c_void) {
        ANDROID_TARGETS.with(|targets| {
            targets.borrow_mut().remove(&(window as usize));
        });
    }
}
