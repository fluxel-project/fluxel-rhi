//! Direct native indirect draw calls; WebGPU has no native multi-draw command.
//!
//! Calls operate on the currently open native raster pass and retain the
//! argument buffer with that encoder; they never expand an indirect draw on CPU.

use fluxel_rhi_core::api::{self, Error, Result};
use wasm_bindgen::JsValue;

use super::{command::WebGpuEncoder, instance::WebGpu, js, resource::WebGpuBuffer};

pub struct WebGpuExtras;

impl api::RasterExtrasOps<WebGpu> for WebGpuExtras {
    const AVAILABLE: bool = true;

    fn draw_indirect(
        encoder: &mut WebGpuEncoder,
        buffer: &WebGpuBuffer,
        offset: u64,
        indexed: bool,
    ) -> Result<()> {
        let size = if indexed { 20 } else { 16 };
        if !buffer.usage.contains(api::BufferUsage::INDIRECT)
            || offset % 4 != 0
            || offset.checked_add(size).is_none_or(|end| end > buffer.size)
        {
            return Err(Error::InvalidRange);
        }
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        js::call(
            pass,
            if indexed {
                "drawIndexedIndirect"
            } else {
                "drawIndirect"
            },
            &[buffer.raw.clone(), JsValue::from_f64(offset as f64)],
        )
        .map_err(|error| Error::Backend(js::message(&error)))?;
        encoder.keep_alive.push(buffer.raw.clone());
        Ok(())
    }

    fn multi_draw(
        _: &mut WebGpuEncoder,
        _: &WebGpuBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("WebGPU multi-draw"))
    }

    fn multi_draw_count(
        _: &mut WebGpuEncoder,
        _: &WebGpuBuffer,
        _: u64,
        _: &WebGpuBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("WebGPU multi-draw count"))
    }

    fn set_push_constants(
        _: &mut WebGpuEncoder,
        _: api::StageMask,
        _: u32,
        _: &[u8],
    ) -> Result<()> {
        Err(Error::Unsupported("WebGPU push constants"))
    }
}
