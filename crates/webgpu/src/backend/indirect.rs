//! Direct native indirect draw calls; WebGPU has no native multi-draw command.
//!
//! Calls operate on the currently open native raster pass and retain the
//! argument buffer with that encoder; each draw is recorded directly on the pass.

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
        encoder: &mut WebGpuEncoder,
        buffer: &WebGpuBuffer,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        let size = if indexed { 20u64 } else { 16u64 };
        if !buffer.usage.contains(api::BufferUsage::INDIRECT)
            || offset % 4 != 0
            || (count != 0 && (u64::from(stride) < size || stride % 4 != 0))
        {
            return Err(Error::InvalidRange);
        }
        if count != 0 {
            let end = offset
                .checked_add(u64::from(count - 1) * u64::from(stride))
                .and_then(|value| value.checked_add(size))
                .ok_or(Error::InvalidRange)?;
            if end > buffer.size {
                return Err(Error::InvalidRange);
            }
        }
        let pass = encoder.pass.as_ref().ok_or(Error::InvalidState)?;
        for index in 0..count {
            let argument_offset = offset + u64::from(index) * u64::from(stride);
            js::call(
                pass,
                if indexed {
                    "drawIndexedIndirect"
                } else {
                    "drawIndirect"
                },
                &[
                    buffer.raw.clone(),
                    JsValue::from_f64(argument_offset as f64),
                ],
            )
            .map_err(|error| Error::Backend(js::message(&error)))?;
        }
        encoder.keep_alive.push(buffer.raw.clone());
        Ok(())
    }

    fn multi_draw_count(
        encoder: &mut WebGpuEncoder,
        args: &WebGpuBuffer,
        args_offset: u64,
        count: &WebGpuBuffer,
        count_offset: u64,
        max_draw_count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        super::command::pass(encoder)?;
        super::indirect_count::validate(
            encoder,
            args,
            args_offset,
            count,
            count_offset,
            max_draw_count,
            stride,
            indexed,
        )?;
        if max_draw_count == 0 {
            return Ok(());
        }
        let suspended = super::pass_split::suspend(encoder)?;
        let chunks = super::indirect_count::prepare(
            encoder,
            args,
            args_offset,
            count,
            count_offset,
            max_draw_count,
            stride,
            indexed,
        )?;
        super::pass_split::resume(encoder, suspended, None, true)?;
        let pass = super::command::pass(encoder)?;
        let size = if indexed { 20u64 } else { 16u64 };
        for (result, draws) in &chunks {
            for index in 0..*draws {
                js::call(
                    pass,
                    if indexed {
                        "drawIndexedIndirect"
                    } else {
                        "drawIndirect"
                    },
                    &[
                        result.clone(),
                        JsValue::from_f64((u64::from(index) * size) as f64),
                    ],
                )
                .map_err(|error| Error::Backend(js::message(&error)))?;
            }
        }
        encoder
            .keep_alive
            .extend(chunks.into_iter().map(|(result, _)| result));
        Ok(())
    }

    fn set_push_constants(
        encoder: &mut WebGpuEncoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()> {
        if stages.0 & !(api::StageMask::VERTEX.0 | api::StageMask::FRAGMENT.0) != 0 {
            return Err(Error::InvalidRange);
        }
        super::immediate::write(super::command::pass(encoder)?, stages, offset, bytes)
    }
}
