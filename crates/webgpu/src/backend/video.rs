//! WebGPU external-video source ownership and import.
//!
//! The adapter supplies one browser source object. The backend keeps it
//! private, imports an immutable `GPUExternalTexture` snapshot, and lets the
//! browser perform source-format and color conversion.
use fluxel_rhi_core::{
    api::{Error, Result},
    backend::VideoBackend,
};
use wasm_bindgen::JsValue;

use super::{instance::WebGpuDevice, js, resource};

/// Opaque source wrapper used by a platform bridge to transfer a browser video
/// source into the WebGPU backend.
pub struct WebGpuVideoSource {
    pub(super) raw: JsValue,
}

impl WebGpuVideoSource {
    /// Wraps a browser `VideoFrame` or `HTMLVideoElement` source value.
    ///
    /// The backend validates/imports the value when `Device::import_video_frame`
    /// is called. The wrapper takes ownership of this JS reference.
    pub fn from_js_value(source: JsValue) -> Self {
        Self { raw: source }
    }
}

/// Imported `GPUExternalTexture` plus its source root. Bind groups retain the
/// native external texture; this wrapper keeps its source reachable as well.
pub struct WebGpuVideoFrame {
    pub(super) raw: JsValue,
    _source: WebGpuVideoSource,
}

impl WebGpuVideoFrame {
    pub(crate) fn source_root(&self) -> &JsValue {
        &self._source.raw
    }
}

impl VideoBackend for super::WebGpu {
    type NativeVideoSource = WebGpuVideoSource;

    fn import_video_frame(
        device: &WebGpuDevice,
        source: Self::NativeVideoSource,
    ) -> Result<Self::NativeVideoFrame> {
        let descriptor = resource::descriptor();
        resource::field(&descriptor, "source", source.raw.clone())?;
        let raw = js::call1(&device.0.device, "importExternalTexture", &descriptor)
            .map_err(|error| Error::Backend(js::message(&error)))?;
        Ok(WebGpuVideoFrame {
            raw,
            _source: source,
        })
    }
}
