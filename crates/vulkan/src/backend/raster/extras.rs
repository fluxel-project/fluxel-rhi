//! Vulkan indirect draws and push-constant lowering.
//!
//! These operations are valid only while the parent raster module has an
//! active dynamic-rendering scope. They validate the native command shape and
//! record it immediately; this module neither owns pass state nor stores a
//! replayable command stream.

use super::*;

pub struct RasterExtras;

impl api::RasterExtrasOps<Vulkan> for RasterExtras {
    const AVAILABLE: bool = true;

    fn draw_indirect(
        encoder: &mut VulkanEncoder,
        arguments: &VulkanBuffer,
        offset: u64,
        indexed: bool,
    ) -> Result<()> {
        validate_indirect(
            encoder,
            arguments,
            offset,
            1,
            indirect_stride(indexed),
            indexed,
            false,
        )?;
        // SAFETY: validation proves that this is a live raster command buffer,
        // the retained indirect buffer has the requested usage and range, and
        // the selected command form agrees with the bound index state.
        unsafe {
            if indexed {
                encoder.device.device.cmd_draw_indexed_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    1,
                    indirect_stride(true),
                );
            } else {
                encoder.device.device.cmd_draw_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    1,
                    indirect_stride(false),
                );
            }
        }
        encoder.buffers.push(arguments.clone());
        Ok(())
    }

    fn multi_draw(
        encoder: &mut VulkanEncoder,
        arguments: &VulkanBuffer,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        validate_indirect(encoder, arguments, offset, count, stride, indexed, true)?;
        // SAFETY: `validate_indirect` checked the native alignment, stride,
        // count limit and buffer extent before recording this command.
        unsafe {
            if indexed {
                encoder.device.device.cmd_draw_indexed_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    count,
                    stride,
                );
            } else {
                encoder.device.device.cmd_draw_indirect(
                    encoder.command,
                    arguments.0.buffer,
                    offset,
                    count,
                    stride,
                );
            }
        }
        encoder.buffers.push(arguments.clone());
        Ok(())
    }

    fn multi_draw_count(
        _: &mut VulkanEncoder,
        _: &VulkanBuffer,
        _: u64,
        _: &VulkanBuffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<()> {
        Err(Error::Unsupported("Vulkan multi draw count"))
    }

    fn set_push_constants(
        encoder: &mut VulkanEncoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<()> {
        let layout = encoder.raster_layout.ok_or(Error::InvalidState)?;
        let push_constants = encoder
            .raster_push_constants
            .as_ref()
            .ok_or(Error::InvalidState)?;
        let native_stages = push_constants.validate(stages, offset, bytes)?;
        // SAFETY: the active pipeline supplied this layout and validation
        // checked the byte range and stage visibility against that layout.
        unsafe {
            encoder.device.device.cmd_push_constants(
                encoder.command,
                layout,
                native_stages,
                offset,
                bytes,
            );
        }
        Ok(())
    }
}

fn indirect_stride(indexed: bool) -> u32 {
    if indexed { 20 } else { 16 }
}

fn validate_indirect(
    encoder: &VulkanEncoder,
    arguments: &VulkanBuffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
    multi: bool,
) -> Result<()> {
    if !encoder.raster
        || encoder.raster_layout.is_none()
        || (indexed && !encoder.raster_index_bound)
        || !arguments.0.usage.contains(api::BufferUsage::INDIRECT)
        || offset % 4 != 0
    {
        return Err(Error::InvalidState);
    }
    let command_size = indirect_stride(indexed);
    if (count != 0 && (stride < command_size || stride % 4 != 0))
        || (multi
            && count > 1
            && (!encoder.device.multi_draw_indirect
                || count > encoder.device.max_draw_indirect_count))
    {
        return Err(Error::InvalidRange);
    }
    let bytes = if count == 0 {
        offset
    } else {
        u64::from(count - 1)
            .checked_mul(u64::from(stride))
            .and_then(|tail| offset.checked_add(tail))
            .and_then(|last| last.checked_add(u64::from(command_size)))
            .ok_or(Error::InvalidRange)?
    };
    if bytes > arguments.0.size {
        return Err(Error::InvalidRange);
    }
    Ok(())
}
