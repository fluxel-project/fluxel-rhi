//! GPU conversion of raw indirect draw records into Metal ICBs.
//!
//! A counted draw ends the active render encoder, converts records through a
//! compute encoder on the same command buffer, then resumes and executes the
//! ICBs. This is direct native lowering, not CPU replay or scheduling.

use crate::backend::{Metal, MetalBuffer, MetalCommandEncoder};
use fluxel_rhi_core::api;
#[cfg(any(target_vendor = "apple", test))]
use fluxel_rhi_core::api::BufferUsage;

pub struct RasterExtras;

impl api::RasterExtrasOps<Metal> for RasterExtras {
    const AVAILABLE: bool = true;
    fn draw_indirect(
        encoder: &mut MetalCommandEncoder,
        args: &MetalBuffer,
        offset: u64,
        indexed: bool,
    ) -> api::Result<()> {
        Self::multi_draw(
            encoder,
            args,
            offset,
            1,
            if indexed { 20 } else { 16 },
            indexed,
        )
    }
    fn multi_draw(
        encoder: &mut MetalCommandEncoder,
        args: &MetalBuffer,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLIndexType, MTLRenderCommandEncoder as _};
            let record = if indexed { 20u64 } else { 16u64 };
            if !args.usage.contains(BufferUsage::INDIRECT)
                || offset % 4 != 0
                || stride % 4 != 0
                || u64::from(stride) < record
                || (count > 0
                    && offset
                        .checked_add(u64::from(count - 1) * u64::from(stride))
                        .and_then(|v| v.checked_add(record))
                        .is_none_or(|end| end > args.size()))
            {
                return Err(api::Error::InvalidRange);
            }
            if !encoder.raster_pipeline_bound || encoder.mesh_threads.is_some() {
                return Err(api::Error::InvalidState);
            }
            let index = if indexed {
                Some(
                    encoder
                        .index_buffer
                        .as_ref()
                        .ok_or(api::Error::InvalidState)?,
                )
            } else {
                None
            };
            if index.is_some_and(|(_, _, format, _)| {
                encoder.strip_index_format.is_some_and(|expected| {
                    std::mem::discriminant(&expected) != std::mem::discriminant(format)
                })
            }) {
                return Err(api::Error::InvalidState);
            }
            super::raster::for_view_batches(encoder, |render| {
                for i in 0..count {
                    let indirect_offset = offset + u64::from(i) * u64::from(stride);
                    // SAFETY: argument ranges and alignment were checked; Core owns the
                    // indirect data contract, and native resources are retained through submission.
                    unsafe {
                        if let Some((buffer, base, format, _)) = index {
                            render.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                            encoder.primitive_type,
                            match format { api::IndexFormat::Uint16 => MTLIndexType::UInt16, api::IndexFormat::Uint32 => MTLIndexType::UInt32 },
                            buffer, *base as usize, &args.raw, indirect_offset as usize,
                        );
                        } else {
                            render.drawPrimitives_indirectBuffer_indirectBufferOffset(
                                encoder.primitive_type,
                                &args.raw,
                                indirect_offset as usize,
                            );
                        }
                    }
                }
            })?;
            encoder.indirect_arguments.push(args.raw.clone());
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, args, offset, count, stride, indexed);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn multi_draw_count(
        encoder: &mut MetalCommandEncoder,
        args: &MetalBuffer,
        args_offset: u64,
        count: &MetalBuffer,
        count_offset: u64,
        max: u32,
        stride: u32,
        indexed: bool,
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            if !valid_recording_state(
                encoder.indirect_count_enabled,
                encoder.raster_pipeline_bound,
                encoder.mesh_threads.is_some(),
                encoder.render.is_some(),
                indexed,
                encoder.index_buffer.is_some(),
            ) {
                return Err(api::Error::InvalidState);
            }
            let index = if indexed {
                let (raw, offset, format, size) = encoder
                    .index_buffer
                    .as_ref()
                    .ok_or(api::Error::InvalidState)?;
                Some((raw.clone(), *offset, *size, *format))
            } else {
                None
            };
            validate(
                args.usage,
                args.size(),
                args_offset,
                count.usage,
                count.size(),
                count_offset,
                max,
                stride,
                indexed,
            )?;
            if max == 0 {
                return Ok(());
            }
            super::raster::resume::end_for_indirect_count(encoder)?;
            let converter = encoder
                .indirect_count_converter
                .as_ref()
                .ok_or(api::Error::Unsupported("Metal indirect draw count"))?;
            let draws = prepare(
                converter,
                &encoder.command,
                args,
                args_offset,
                count,
                count_offset,
                max,
                stride,
                index
                    .as_ref()
                    .map(|(raw, offset, size, format)| (raw, *offset, *size, *format)),
            )?;
            super::raster::resume::resume_after_indirect_count(encoder)?;
            super::raster::for_view_batches(encoder, |render| draws.encode_render(render))?;
            encoder.prepared_draws.push(draws);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (
                encoder,
                args,
                args_offset,
                count,
                count_offset,
                max,
                stride,
                indexed,
            );
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
    fn set_push_constants(
        encoder: &mut MetalCommandEncoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> api::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            super::raster::write_push(encoder, stages, offset, bytes)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (encoder, stages, offset, bytes);
            Err(api::Error::Unsupported("Metal requires an Apple target"))
        }
    }
}

#[cfg(any(target_vendor = "apple", test))]
const ICB_CHUNK: u32 = 16_384;

#[cfg(any(target_vendor = "apple", test))]
fn valid_recording_state(
    enabled: bool,
    pipeline: bool,
    mesh: bool,
    render: bool,
    indexed: bool,
    index_bound: bool,
) -> bool {
    enabled && pipeline && !mesh && render && (!indexed || index_bound)
}

#[cfg(any(target_vendor = "apple", test))]
fn validate(
    args_usage: BufferUsage,
    args_size: u64,
    args_offset: u64,
    count_usage: BufferUsage,
    count_size: u64,
    count_offset: u64,
    max: u32,
    stride: u32,
    indexed: bool,
) -> api::Result<()> {
    let record = if indexed { 20u64 } else { 16u64 };
    if !args_usage.contains(BufferUsage::INDIRECT)
        || !count_usage.contains(BufferUsage::INDIRECT)
        || args_offset % 4 != 0
        || count_offset % 4 != 0
        || count_offset
            .checked_add(4)
            .is_none_or(|end| end > count_size)
        || (max != 0 && (u64::from(stride) < record || stride % 4 != 0))
    {
        return Err(api::Error::InvalidRange);
    }
    let end = if max == 0 {
        Some(args_offset)
    } else {
        u64::from(max - 1)
            .checked_mul(u64::from(stride))
            .and_then(|bytes| args_offset.checked_add(bytes))
            .and_then(|last| last.checked_add(record))
    };
    if end.is_none_or(|end| end > args_size) {
        return Err(api::Error::InvalidRange);
    }
    Ok(())
}

#[cfg(any(target_vendor = "apple", test))]
fn chunks(max: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..max)
        .step_by(ICB_CHUNK as usize)
        .map(move |first| (first, (max - first).min(ICB_CHUNK)))
}

#[cfg(target_vendor = "apple")]
mod native;
#[cfg(target_vendor = "apple")]
pub(in crate::backend) use native::{ConverterCache, PreparedDraws, prepare, probe};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_complete_strided_records_and_count_word() {
        let usage = BufferUsage::INDIRECT;
        assert!(validate(usage, 68, 0, usage, 8, 4, 3, 24, true).is_ok());
        assert!(validate(usage, 67, 0, usage, 8, 4, 3, 24, true).is_err());
        assert!(validate(usage, 68, 0, usage, 7, 4, 3, 24, true).is_err());
        assert!(validate(usage, 68, 0, usage, 8, 4, 3, 18, true).is_err());
        assert!(validate(BufferUsage::COPY_SRC, 68, 0, usage, 8, 4, 3, 24, true).is_err());
    }

    #[test]
    fn partitions_at_metal3_limit_without_losing_or_wrapping_commands() {
        assert_eq!(chunks(0).collect::<Vec<_>>(), vec![]);
        assert_eq!(
            chunks(16_385).collect::<Vec<_>>(),
            vec![(0, 16_384), (16_384, 1)]
        );
        let last = chunks(u32::MAX).last().unwrap();
        assert_eq!(u64::from(last.0) + u64::from(last.1), u64::from(u32::MAX));
    }

    #[test]
    fn indexed_base_vertex_words_preserve_signed_bits() {
        for (value, word) in [(i32::MIN, 0x8000_0000), (-1, 0xffff_ffff), (0, 0), (1, 1)] {
            assert_eq!(u32::from_ne_bytes(value.to_ne_bytes()), word);
            assert_eq!(i32::from_ne_bytes(word.to_ne_bytes()), value);
        }
    }

    #[test]
    fn recording_requires_active_raster_pipeline_render_and_index_when_indexed() {
        assert!(valid_recording_state(true, true, false, true, false, false));
        assert!(valid_recording_state(true, true, false, true, true, true));
        assert!(!valid_recording_state(
            false, true, false, true, false, false
        ));
        assert!(!valid_recording_state(true, true, true, true, false, false));
        assert!(!valid_recording_state(
            true, true, false, false, false, false
        ));
        assert!(!valid_recording_state(true, true, false, true, true, false));
    }
}
