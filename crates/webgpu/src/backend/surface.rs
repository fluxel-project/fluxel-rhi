//! Canvas configuration and exclusive browser image leases.

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
        let native = WebGpuSurface(Rc::new(RefCell::new(SurfaceState {
            device: unsafe { device.native_ref() }.clone(),
            canvas,
            context: context.into(),
            generation: 0,
            configured: None,
            acquired: false,
            abandoned: false,
        })));
        Ok(unsafe { api::Surface::from_native(device.id(), api::SurfaceId::allocate()?, native) })
    }
}

unsafe impl api::NativeImageLease for WebGpuImageLease {
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

impl api::SurfaceBackend for WebGpu {
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
            formats: vec![format],
            present_modes: vec![api::PresentMode::Fifo],
            alpha_modes: vec![
                api::CompositeAlphaMode::Opaque,
                api::CompositeAlphaMode::Premultiplied,
            ],
            color_spaces: vec![api::ColorSpace::Srgb],
            usages: api::SurfaceUsage::RENDER_ATTACHMENT,
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
            || desc.color_space != api::ColorSpace::Srgb
            || desc.usage != api::SurfaceUsage::RENDER_ATTACHMENT
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
    let mut state = surface.0.borrow_mut();
    if state.acquired {
        return Err(Error::InvalidState);
    }
    if state.canvas.width() != desc.extent[0] || state.canvas.height() != desc.extent[1] {
        return Err(Error::SurfaceOutdated);
    }
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
    js::set(&object, "usage", &JsValue::from_f64(16.0))
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
    js::call1(&state.context, "configure", &object)
        .map_err(|error| Error::Backend(js::message(&error)))?;
    let generation = state.generation.checked_add(1).ok_or(Error::OutOfMemory)?;
    state.configured = Some(desc.clone());
    state.generation = generation;
    state.abandoned = false;
    Ok(())
}

fn acquire(surface: &WebGpuSurface) -> Result<WebGpuImageLease> {
    let mut state = surface.0.borrow_mut();
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
    state.acquired = true;
    Ok(WebGpuImageLease {
        surface: surface.clone(),
        generation: state.generation,
        view,
        handed_off: false,
    })
}
