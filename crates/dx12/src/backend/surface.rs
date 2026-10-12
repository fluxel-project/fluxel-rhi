//! DXGI HWND swapchain implementation for the surface SPI.

use std::{
    future::{Ready, ready},
    sync::{Arc, Mutex},
};

use fluxel_rhi_core::api::{self, Error, Result};
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};
use windows::{
    Win32::{
        Foundation::HWND,
        Graphics::{
            Direct3D12::{
                D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC,
                D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                ID3D12DescriptorHeap, ID3D12Resource,
            },
            Dxgi::Common::{
                DXGI_ALPHA_MODE_IGNORE, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_SAMPLE_DESC,
            },
            Dxgi::{
                CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_PRESENT, DXGI_SCALING_STRETCH,
                DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT,
                DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIFactory4,
                IDXGISwapChain2, IDXGISwapChain3,
            },
        },
    },
    core::Interface,
};

use super::{Dx12, Dx12Device, Dx12Surface, Dx12SurfaceImage};

pub(crate) struct SurfaceState {
    device: Dx12Device,
    hwnd: HWND,
    factory: IDXGIFactory4,
    inner: Mutex<Inner>,
}
struct Inner {
    chain: Option<IDXGISwapChain3>,
    rtv_heap: Option<ID3D12DescriptorHeap>,
    rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    generation: u64,
    acquired: bool,
}
pub(crate) struct ImageLease {
    pub(crate) resource: ID3D12Resource,
    pub(crate) rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    state: Arc<SurfaceState>,
    generation: u64,
    handed: bool,
}

pub(crate) unsafe fn create_surface(
    device: &api::Device<Dx12>,
    handle: RawWindowHandle,
) -> Result<api::Surface<Dx12>> {
    let RawWindowHandle::Win32(Win32WindowHandle { hwnd, .. }) = handle else {
        return Err(Error::Unsupported("a Win32 window handle"));
    };
    let native = unsafe { device.native_ref() }.clone();
    let factory = unsafe { CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) }
        .map_err(error)?;
    let state = Arc::new(SurfaceState {
        device: native,
        hwnd: HWND(hwnd.get() as *mut _),
        factory,
        inner: Mutex::new(Inner {
            chain: None,
            rtv_heap: None,
            rtvs: Vec::new(),
            generation: 0,
            acquired: false,
        }),
    });
    Ok(unsafe {
        api::Surface::from_native(device.id(), api::SurfaceId::allocate()?, Dx12Surface(state))
    })
}
impl Drop for ImageLease {
    fn drop(&mut self) {
        if !self.handed {
            let mut inner = self.state.inner.lock().unwrap_or_else(|p| p.into_inner());
            if inner.generation == self.generation {
                inner.acquired = false;
            }
        }
    }
}

unsafe impl api::NativeImageLease for Dx12SurfaceImage {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if self.0.generation != generation || self.0.handed {
            return Err(Error::SurfaceOutdated);
        }
        let mut inner = self.0.state.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.generation != generation || !inner.acquired {
            return Err(Error::SurfaceOutdated);
        }
        unsafe {
            inner
                .chain
                .as_ref()
                .ok_or(Error::SurfaceLost)?
                .Present(1, DXGI_PRESENT(0))
        }
        .ok()
        .map_err(error)?;
        inner.acquired = false;
        self.0.handed = true;
        Ok(())
    }
    fn is_current(&self, generation: u64) -> bool {
        let inner = self.0.state.inner.lock().unwrap_or_else(|p| p.into_inner());
        !self.0.handed && inner.acquired && inner.generation == generation
    }
}

impl api::SurfaceBackend for Dx12 {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = Ready<Result<Dx12SurfaceImage>>;
    fn generation(surface: &Dx12Surface) -> u64 {
        surface
            .0
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .generation
    }
    fn capabilities(_: &Dx12Surface) -> Result<api::SurfaceCapabilities> {
        Ok(caps())
    }
    fn supports_config(_: &Dx12Surface, d: &api::SurfaceConfig) -> api::Support<()> {
        if d.extent.contains(&0)
            || d.extent.iter().any(|v| *v > 16384)
            || !matches!(
                d.format,
                api::TextureFormat::Bgra8Unorm | api::TextureFormat::Rgba8Unorm
            )
            || !matches!(d.present_mode, api::PresentMode::Fifo)
            || d.usage.0 != api::SurfaceUsage::RENDER_ATTACHMENT.0
            || !matches!(d.alpha_mode, api::CompositeAlphaMode::Opaque)
            || !matches!(d.color_space, api::ColorSpace::Srgb)
            || !(1..=16).contains(&d.maximum_frame_latency)
        {
            api::Support::Unsupported("DXGI flip-model surface configuration")
        } else {
            api::Support::Supported(())
        }
    }
    fn configure<'a>(
        surface: &'a mut Dx12Surface,
        d: &'a api::SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        ready(configure(surface, d))
    }
    fn acquire<'a>(surface: &'a mut Dx12Surface) -> Self::AcquireFuture<'a> {
        let mut i = surface.0.inner.lock().unwrap_or_else(|p| p.into_inner());
        if i.acquired {
            return ready(Err(Error::InvalidState));
        }
        let Some(chain) = i.chain.as_ref() else {
            return ready(Err(Error::SurfaceLost));
        };
        let index = unsafe { chain.GetCurrentBackBufferIndex() };
        let resource = match unsafe { chain.GetBuffer::<ID3D12Resource>(index) } {
            Ok(value) => value,
            Err(e) => return ready(Err(error(e))),
        };
        let Some(rtv) = i.rtvs.get(index as usize).copied() else {
            return ready(Err(Error::SurfaceLost));
        };
        i.acquired = true;
        let generation = i.generation;
        drop(i);
        ready(Ok(Dx12SurfaceImage(ImageLease {
            resource,
            rtv,
            state: surface.0.clone(),
            generation,
            handed: false,
        })))
    }
}

fn configure(s: &Dx12Surface, d: &api::SurfaceConfig) -> Result<()> {
    if !matches!(
        <Dx12 as api::SurfaceBackend>::supports_config(s, d),
        api::Support::Supported(())
    ) {
        return Err(Error::Unsupported("DXGI flip-model surface configuration"));
    }
    let format = if matches!(d.format, api::TextureFormat::Bgra8Unorm) {
        DXGI_FORMAT_B8G8R8A8_UNORM
    } else {
        DXGI_FORMAT_R8G8B8A8_UNORM
    };
    let flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    let mut i = s.0.inner.lock().unwrap_or_else(|p| p.into_inner());
    if i.acquired {
        return Err(Error::InvalidState);
    }
    let completed = unsafe { s.0.device.0.fence.GetCompletedValue() };
    if s.0
        .device
        .0
        .in_flight
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .any(|(value, _)| *value == u64::MAX || *value > completed)
    {
        return Err(Error::InvalidState);
    }
    let resized = if let Some(chain) = i.chain.as_ref() {
        unsafe { chain.ResizeBuffers(2, d.extent[0], d.extent[1], format, flags) }
            .map_err(error)?;
        true
    } else {
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: d.extent[0],
            Height: d.extent[1],
            Format: format,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: DXGI_ALPHA_MODE_IGNORE,
            Flags: flags.0 as u32,
        };
        let chain = unsafe {
            s.0.factory
                .CreateSwapChainForHwnd(&s.0.device.0.queue, s.0.hwnd, &desc, None, None)
        }
        .map_err(error)?;
        i.chain = Some(chain.cast().map_err(error)?);
        false
    };

    // `ResizeBuffers` invalidates every old backbuffer and RTV immediately.
    // A later failure therefore cannot leave the former generation published:
    // discard the native chain and make acquire report SurfaceLost. A later
    // configure is allowed to construct a fresh chain for the same HWND.
    let post_resize = (|| {
        let latency: IDXGISwapChain2 = i
            .chain
            .as_ref()
            .ok_or(Error::SurfaceLost)?
            .cast()
            .map_err(error)?;
        unsafe { latency.SetMaximumFrameLatency(d.maximum_frame_latency) }.map_err(error)?;
        rebuild_rtvs(&s.0.device, &mut i)
    })();
    if let Err(error) = post_resize {
        if resized || i.chain.is_some() {
            invalidate(&mut i);
        }
        return Err(error);
    }
    i.generation += 1;
    Ok(())
}

/// Clears every handle whose validity was tied to the latest resize attempt.
/// Incrementing generation also makes any stale lease fail `is_current` even
/// when it was held across a native failure boundary.
fn invalidate(inner: &mut Inner) {
    inner.rtvs.clear();
    inner.rtv_heap = None;
    inner.chain = None;
    inner.acquired = false;
    inner.generation = inner.generation.wrapping_add(1);
}
fn rebuild_rtvs(device: &Dx12Device, inner: &mut Inner) -> Result<()> {
    let heap = unsafe {
        device
            .0
            .native
            .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
    }
    .map_err(error)?;
    let increment = unsafe {
        device
            .0
            .native
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
    } as usize;
    let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let chain = inner.chain.as_ref().ok_or(Error::SurfaceLost)?;
    let mut rtvs = Vec::with_capacity(2);
    for index in 0..2 {
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + index * increment,
        };
        let resource = unsafe { chain.GetBuffer::<ID3D12Resource>(index as u32) }.map_err(error)?;
        unsafe {
            device
                .0
                .native
                .CreateRenderTargetView(&resource, None, handle)
        };
        rtvs.push(handle);
    }
    inner.rtv_heap = Some(heap);
    inner.rtvs = rtvs;
    Ok(())
}
fn caps() -> api::SurfaceCapabilities {
    api::SurfaceCapabilities {
        formats: vec![
            api::TextureFormat::Bgra8Unorm,
            api::TextureFormat::Rgba8Unorm,
        ],
        present_modes: vec![api::PresentMode::Fifo],
        alpha_modes: vec![api::CompositeAlphaMode::Opaque],
        color_spaces: vec![api::ColorSpace::Srgb],
        usages: api::SurfaceUsage::RENDER_ATTACHMENT,
        min_extent: [1, 1],
        max_extent: [16384, 16384],
    }
}
fn error(e: windows::core::Error) -> Error {
    Error::Backend(format!("DXGI: {e}"))
}
