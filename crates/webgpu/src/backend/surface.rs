//! Canvas configuration and exclusive browser image leases.
//!
//! A surface is bound to one browser `GPUDevice` and owner thread. Configuration
//! increments its generation; at most one current texture lease exists. Dropping
//! an unhanded lease marks the canvas abandoned so a later configuration first
//! retires the browser current texture instead of reusing it.

use std::{
    cell::RefCell,
    future::{Ready, ready},
    rc::Rc,
};

use fluxel_rhi_core::api::{self, Error, Result, Support};
use js_sys::Object;
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

use super::{
    instance::{WebGpu, WebGpuDevice},
    js, resource,
};

#[derive(Clone)]
pub struct WebGpuSurface(Rc<RefCell<SurfaceState>>);

struct SurfaceState {
    device: WebGpuDevice,
    canvas: HtmlCanvasElement,
    context: JsValue,
    generation: u64,
    configured: Option<api::SurfaceConfig>,
    acquired: bool,
    abandoned: bool,
}

/// A single `getCurrentTexture` result and its renderable view.
pub struct WebGpuImageLease {
    surface: WebGpuSurface,
    generation: u64,
    pub(super) view: JsValue,
    pub(super) extent: [u32; 2],
    pub(super) format: api::TextureFormat,
    pub(super) usage: api::SurfaceUsage,
    handed_off: bool,
}

impl WebGpu {
    /// Binds one host-owned browser canvas to this device.
    pub fn surface_from_canvas(
        device: &api::Device<Self>,
        canvas: HtmlCanvasElement,
    ) -> Result<api::Surface<Self>> {
        let context = canvas
            .get_context("webgpu")
            .map_err(|error| Error::Backend(js::message(&error)))?
            .ok_or(Error::Unsupported("GPUCanvasContext"))?;
        // SAFETY: `device` is this backend's typed Core handle. The returned
        // native clone stays in surface state, confined to the browser owner.
        let native = WebGpuSurface(Rc::new(RefCell::new(SurfaceState {
            device: unsafe { device.native_ref() }.clone(),
            canvas,
            context: context.into(),
            generation: 0,
            configured: None,
            acquired: false,
            abandoned: false,
        })));
        // SAFETY: this freshly created native surface is associated exactly
        // once with `device.id()` and has no outstanding image lease.
        Ok(unsafe { api::Surface::from_native(device, api::SurfaceId::allocate()?, native) })
    }
}

// SAFETY: `try_handoff` only succeeds for the matching live generation and
// clears the sole acquired flag. Drop clears an unhanded lease exactly once.
unsafe impl fluxel_rhi_core::backend::NativeImageLease for WebGpuImageLease {
    fn try_handoff(&mut self, generation: u64) -> Result<()> {
        if !self.is_current(generation) {
            return Err(Error::SurfaceOutdated);
        }
        self.handed_off = true;
        self.surface.0.borrow_mut().acquired = false;
        Ok(())
    }

    fn is_current(&self, generation: u64) -> bool {
        let state = self.surface.0.borrow();
        !self.handed_off
            && state.acquired
            && !state.abandoned
            && self.generation == generation
            && state.generation == generation
    }
}

impl Drop for WebGpuImageLease {
    fn drop(&mut self) {
        if !self.handed_off {
            let mut state = self.surface.0.borrow_mut();
            if state.generation == self.generation && state.acquired {
                state.acquired = false;
                // A browser canvas may keep an unpresented current texture.
                // Reconfiguration explicitly retires it before reuse.
                state.abandoned = true;
            }
        }
    }
}

impl fluxel_rhi_core::backend::SurfaceBackend for WebGpu {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = Ready<Result<WebGpuImageLease>>;

    fn generation(surface: &WebGpuSurface) -> u64 {
        surface.0.borrow().generation
    }

    fn capabilities(surface: &WebGpuSurface) -> Result<api::SurfaceCapabilities> {
        let state = surface.0.borrow();
        let gpu = js::browser_gpu().map_err(|error| Error::Backend(js::message(&error)))?;
        let preferred = js::call0(&gpu, "getPreferredCanvasFormat")
            .map_err(|error| Error::Backend(js::message(&error)))?
            .as_string()
            .ok_or(Error::InvalidState)?;
        let format = match preferred.as_str() {
            "bgra8unorm" => api::TextureFormat::Bgra8Unorm,
            "rgba8unorm" => api::TextureFormat::Rgba8Unorm,
            _ => return Err(Error::Unsupported("browser canvas format")),
        };
        Ok(api::SurfaceCapabilities {
            formats: {
                let mut formats = vec![format];
                for candidate in [
                    api::TextureFormat::Bgra8Unorm,
                    api::TextureFormat::Rgba8Unorm,
                    api::TextureFormat::Rgba16Float,
                ] {
                    if !formats.contains(&candidate) {
                        formats.push(candidate);
                    }
                }
                formats
            },
            present_modes: vec![api::PresentMode::Fifo],
            alpha_modes: vec![
                api::CompositeAlphaMode::Opaque,
                api::CompositeAlphaMode::Premultiplied,
            ],
            color_spaces: vec![api::ColorSpace::Srgb, api::ColorSpace::DisplayP3],
            usages: api::SurfaceUsage::RENDER_ATTACHMENT
                | api::SurfaceUsage::COPY_SRC
                | api::SurfaceUsage::COPY_DST
                | api::SurfaceUsage::TEXTURE_BINDING
                | api::SurfaceUsage::STORAGE_BINDING,
            min_extent: [1, 1],
            max_extent: [
                resource::device_limit(&state.device.0.device, "maxTextureDimension2D") as u32,
                resource::device_limit(&state.device.0.device, "maxTextureDimension2D") as u32,
            ],
        })
    }

    fn supports_config(surface: &WebGpuSurface, desc: &api::SurfaceConfig) -> Support<()> {
        let Ok(caps) = Self::capabilities(surface) else {
            return Support::Unsupported("WebGPU canvas capabilities");
        };
        if !caps.formats.contains(&desc.format)
            || desc.present_mode != api::PresentMode::Fifo
            || !caps.alpha_modes.contains(&desc.alpha_mode)
            || !caps.color_spaces.contains(&desc.color_space)
            || desc.usage.0 == 0
            || desc.usage.0 & !caps.usages.0 != 0
            || (desc.usage.0 & api::SurfaceUsage::STORAGE_BINDING.0 != 0
                && desc.format == api::TextureFormat::Bgra8Unorm
                && !js::has_feature(&surface.0.borrow().device.0.device, "bgra8unorm-storage"))
            || desc.extent.contains(&0)
            || desc.extent[0] > caps.max_extent[0]
            || desc.extent[1] > caps.max_extent[1]
        {
            return Support::Unsupported("WebGPU canvas configuration");
        }
        Support::Supported(())
    }

    fn configure<'a>(
        surface: &'a mut WebGpuSurface,
        desc: &'a api::SurfaceConfig,
    ) -> Self::ConfigureFuture<'a> {
        ready(configure(surface, desc))
    }

    fn acquire<'a>(surface: &'a mut WebGpuSurface) -> Self::AcquireFuture<'a> {
        ready(acquire(surface))
    }
}

fn configure(surface: &WebGpuSurface, desc: &api::SurfaceConfig) -> Result<()> {
    if let Support::Unsupported(reason) =
        <WebGpu as fluxel_rhi_core::backend::SurfaceBackend>::supports_config(surface, desc)
    {
        return Err(Error::Unsupported(reason));
    }
    let mut state = surface.0.borrow_mut();
    if state.device.0.lost.get() {
        return Err(Error::DeviceLost);
    }
    if state.acquired {
        return Err(Error::InvalidState);
    }
    if state.canvas.width() != desc.extent[0] || state.canvas.height() != desc.extent[1] {
        return Err(Error::SurfaceOutdated);
    }
    let generation = state.generation.checked_add(1).ok_or(Error::OutOfMemory)?;
    if state.abandoned {
        js::call0(&state.context, "unconfigure")
            .map_err(|error| Error::Backend(js::message(&error)))?;
        state.configured = None;
    }
    let object: JsValue = Object::new().into();
    js::set(&object, "device", &state.device.0.device)
        .map_err(|error| Error::Backend(js::message(&error)))?;
    js::set(
        &object,
        "format",
        &JsValue::from_str(resource::texture_format(desc.format)?),
    )
    .map_err(|error| Error::Backend(js::message(&error)))?;
    let native_usage = [
        (api::SurfaceUsage::RENDER_ATTACHMENT, 16u32),
        (api::SurfaceUsage::COPY_SRC, 1),
        (api::SurfaceUsage::COPY_DST, 2),
        (api::SurfaceUsage::TEXTURE_BINDING, 4),
        (api::SurfaceUsage::STORAGE_BINDING, 8),
    ]
    .into_iter()
    .fold(0u32, |mask, (flag, native)| {
        mask | if desc.usage.0 & flag.0 != 0 {
            native
        } else {
            0
        }
    });
    js::set(
        &object,
        "usage",
        &JsValue::from_f64(f64::from(native_usage)),
    )
    .map_err(|error| Error::Backend(js::message(&error)))?;
    js::set(
        &object,
        "alphaMode",
        &JsValue::from_str(match desc.alpha_mode {
            api::CompositeAlphaMode::Opaque => "opaque",
            api::CompositeAlphaMode::Premultiplied => "premultiplied",
            _ => return Err(Error::Unsupported("WebGPU canvas alpha mode")),
        }),
    )
    .map_err(|error| Error::Backend(js::message(&error)))?;
    js::set(
        &object,
        "colorSpace",
        &JsValue::from_str(match desc.color_space {
            api::ColorSpace::Srgb => "srgb",
            api::ColorSpace::DisplayP3 => "display-p3",
            _ => return Err(Error::Unsupported("WebGPU canvas color space")),
        }),
    )
    .map_err(|error| Error::Backend(js::message(&error)))?;
    js::call1(&state.context, "configure", &object)
        .map_err(|error| Error::Backend(js::message(&error)))?;
    state.configured = Some(desc.clone());
    state.generation = generation;
    state.abandoned = false;
    Ok(())
}

fn acquire(surface: &WebGpuSurface) -> Result<WebGpuImageLease> {
    let mut state = surface.0.borrow_mut();
    if state.device.0.lost.get() {
        return Err(Error::DeviceLost);
    }
    let config = state.configured.as_ref().ok_or(Error::InvalidState)?;
    if state.acquired || state.abandoned {
        return Err(Error::InvalidState);
    }
    if state.canvas.width() != config.extent[0] || state.canvas.height() != config.extent[1] {
        return Err(Error::SurfaceOutdated);
    }
    let texture = js::call0(&state.context, "getCurrentTexture")
        .map_err(|error| Error::Backend(js::message(&error)))?;
    let view = match js::call1(&texture, "createView", &Object::new().into()) {
        Ok(view) => view,
        Err(error) => {
            state.abandoned = true;
            return Err(Error::Backend(js::message(&error)));
        }
    };
    let extent = config.extent;
    let format = config.format;
    let usage = config.usage;
    state.acquired = true;
    Ok(WebGpuImageLease {
        surface: surface.clone(),
        generation: state.generation,
        view,
        extent,
        format,
        usage,
        handed_off: false,
    })
}
